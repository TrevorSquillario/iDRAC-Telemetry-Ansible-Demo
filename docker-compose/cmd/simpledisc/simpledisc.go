// Licensed to You under the Apache License, Version 2.0.

package main

import (
	"log"
	"os"
	"strconv"
	"time"

	"github.com/dell/iDRAC-Telemetry-Reference-Tools/internal/disc"
	"github.com/dell/iDRAC-Telemetry-Reference-Tools/internal/messagebus/stomp"

	"gopkg.in/yaml.v3"
)

var configStrings = map[string]string{
	"mbhost": "activemq",
	"mbport": "61613",
}

var services []disc.Service

func getEnvSettings() {
	mbHost := os.Getenv("MESSAGEBUS_HOST")
	if len(mbHost) > 0 {
		configStrings["mbhost"] = mbHost
	}
	mbPort := os.Getenv("MESSAGEBUS_PORT")
	if len(mbPort) > 0 {
		configStrings["mbport"] = mbPort
	}
}

func main() {

	var ServiceConfig disc.ServiceConfig
	yamlData, err := os.ReadFile("config.yaml")

	if err != nil {
		log.Fatal("Fail to read file: %v", err)
	}

	yaml.Unmarshal(yamlData, &ServiceConfig)
	log.Println("Loaded Server Config")

	//Gather configuration from environment variables
	getEnvSettings()

	log.Printf("Services: %+v", ServiceConfig)

	discoveryService := new(disc.DiscoveryService)
	for {
		stompPort, _ := strconv.Atoi(configStrings["mbport"])
		mb, err := stomp.NewStompMessageBus(configStrings["mbhost"], stompPort)
		if err != nil {
			log.Printf("Could not connect to message bus: %s", err)
			time.Sleep(5 * time.Second)
		} else {
			discoveryService.Bus = mb
			defer mb.Close()
			break
		}
	}
	commands := make(chan *disc.Command)

	log.Print("Discovery Service is initialized")

	for _, element := range ServiceConfig.Services {
		go func(elem disc.Service) {
			err := discoveryService.SendService(elem)
			if err != nil {
				log.Printf("Failed sending service %v %v", elem, err)
			}
		}(element)
	}

	go discoveryService.ReceiveCommand(commands)
	for {
		command := <-commands
		log.Printf("in simpledisc Received command: %s", command.Command)
		switch command.Command {
		case disc.RESEND:
			for _, element := range ServiceConfig.Services {
				go func(elem disc.Service) {
					err := discoveryService.SendService(elem)
					if err != nil {
						log.Printf("Failed resending service %v %v", elem, err)
					}
				}(element)
			}
		case disc.TERMINATE:
			os.Exit(0)
		}
	}
}
