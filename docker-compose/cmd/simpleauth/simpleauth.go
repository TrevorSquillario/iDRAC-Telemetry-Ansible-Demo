// Licensed to You under the Apache License, Version 2.0.

package main

import (
	"bytes"
	"log"
	"os"
	"strconv"
	"strings"
	"time"

	"golang.org/x/crypto/ssh"

	"github.com/dell/iDRAC-Telemetry-Reference-Tools/internal/auth"
	"github.com/dell/iDRAC-Telemetry-Reference-Tools/internal/disc"

	"github.com/dell/iDRAC-Telemetry-Reference-Tools/internal/messagebus/stomp"
)

var configStrings = map[string]string{
	"mbhost": "activemq",
	"mbport": "61613",
}

var authServices map[string]auth.Service

func handleDiscServiceChannel(serviceIn chan *disc.Service, username string, password string, authorizationService *auth.AuthorizationService) {
	for {
		service := <-serviceIn
		authService := new(auth.Service)
		authService.ServiceType = service.ServiceType
		authService.Ip = service.Ip
		authService.HostTags = service.HostTags
		if authService.ServiceType == auth.EC {
			sshconfig := &ssh.ClientConfig{
				User: username,
				Auth: []ssh.AuthMethod{
					ssh.Password(password),
				},
				HostKeyCallback: ssh.InsecureIgnoreHostKey(),
			}
			serviceName := service.Ip
			if strings.Contains(service.Ip, ":") {
				split := strings.Split(service.Ip, ":")
				serviceName = split[0]
			}
			client, err := ssh.Dial("tcp", serviceName+":22", sshconfig)
			if err != nil {
				log.Print("Failed to dial: ", err)
				continue
			}
			session, err := client.NewSession()
			if err != nil {
				log.Print("Failed to create session: ", err)
				continue
			}
			var b bytes.Buffer
			session.Stdout = &b
			if err := session.Run("/usr/bin/hapitest -e"); err != nil {
				log.Print("Failed to run: " + err.Error())
				session.Close()
				continue
			}
			session.Close()
			str := b.String()
			if !strings.Contains(str, "Local  EC Active State   = 1") {
				log.Printf("EC at %s is not active. Skipping...\n", service.Ip)
				continue
			}
			session, err = client.NewSession()
			if err != nil {
				log.Print("Failed to create session: ", err)
				continue
			}
			session.Stdout = &b
			if err := session.Run("/usr/bin/oauthtest token"); err != nil {
				log.Print("Failed to run: " + err.Error())
				session.Close()
				continue
			}
			session.Close()
			str = b.String()
			parts := strings.Split(str, "Local device token : ")
			authService.AuthType = auth.AuthTypeBearerToken
			authService.Auth = make(map[string]string)
			authService.Auth["token"] = strings.TrimSpace(parts[1])
		} else {
			if username == "" {
				//TODO get token
			} else {
				authService.AuthType = auth.AuthTypeUsernamePassword
				authService.Auth = make(map[string]string)
				authService.Auth["username"] = username
				authService.Auth["password"] = password
			}
		}
		//log.Print("Got Service = ", *authService)
		_ = authorizationService.SendService(*authService)
		if authServices == nil {
			authServices = make(map[string]auth.Service)
		}
		authServices[service.Ip] = *authService
	}
}

func getEnvSettings() {
	mbHost := os.Getenv("MESSAGEBUS_HOST")
	if len(mbHost) > 0 {
		configStrings["mbhost"] = mbHost
	}
	mbPort := os.Getenv("MESSAGEBUS_PORT")
	if len(mbPort) > 0 {
		configStrings["mbport"] = mbPort
	}
	username := os.Getenv("USERNAME")
	if len(username) > 0 {
		configStrings["username"] = username
	}
	password := os.Getenv("PASSWORD")
	if len(password) > 0 {
		configStrings["password"] = password
	}
}

func main() {
	//Gather configuration from environment variables
	getEnvSettings()

	discoveryClient := new(disc.DiscoveryClient)
	authorizationService := new(auth.AuthorizationService)

	for {
		stompPort, _ := strconv.Atoi(configStrings["mbport"])
		mb, err := stomp.NewStompMessageBus(configStrings["mbhost"], stompPort)
		if err != nil {
			log.Printf("Could not connect to message bus: %s", err)
			time.Sleep(5 * time.Second)
		} else {
			discoveryClient.Bus = mb
			authorizationService.Bus = mb
			defer mb.Close()
			break
		}
	}
	serviceIn := make(chan *disc.Service, 10)
	commands := make(chan *auth.Command)

	log.Print("Auth Service is initialized")

	discoveryClient.ResendAll()
	go discoveryClient.GetService(serviceIn)
	go handleDiscServiceChannel(serviceIn, configStrings["username"], configStrings["password"], authorizationService)
	go authorizationService.ReceiveCommand(commands) //nolint: errcheck
	for {
		command := <-commands
		log.Printf("in simpleauth, Received command: %s", command.Command)
		switch command.Command {
		case auth.RESEND:
			for _, element := range authServices {
				go authorizationService.SendService(element) //nolint: errcheck
			}
		case auth.TERMINATE:
			os.Exit(0)
		}
	}
}
