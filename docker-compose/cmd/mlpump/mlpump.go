// Licensed to You under the Apache License, Version 2.0.

package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
	"time"
	"sync"
	"net/http"
	"bytes"
	"encoding/json"

	influxdb2 "github.com/influxdata/influxdb-client-go/v2"
	"github.com/influxdata/influxdb-client-go/v2/api"
	//"github.com/influxdata/influxdb-client-go/v2/api/write"

	"github.com/dell/iDRAC-Telemetry-Reference-Tools/internal/databus"
	"github.com/dell/iDRAC-Telemetry-Reference-Tools/internal/messagebus/stomp"
)

var configStrings = map[string]string{
	"mbhost": "activemq",
	"mbport": "61613",
	"URL":    "http://localhost:8086",
}

type MetricReading struct {
	ID                        string
	Context                   string
	Label                     string
	Value                     float64
	System                    string
	Timestamp                 time.Time
	HostName				  string
	HostTags				  string
}

func sendMetricReadingsToMLProcess(data string){
	posturl := "http://ml/send_data"

	body := []byte(data)

	r, err := http.NewRequest("POST", posturl, bytes.NewBuffer(body))
	if err != nil {
		panic(err)
	}

	r.Header.Add("Content-Type", "application/json")

	client := &http.Client{}
	res, err := client.Do(r)
	if err != nil {
		panic(err)
	}

	defer res.Body.Close()

	/*
	post := &Post{}
	derr := json.NewDecoder(res.Body).Decode(post)
	if derr != nil {
		panic(derr)
	}
	*/
	if res.StatusCode != http.StatusOK {
		log.Println("ERROR")
	}

	log.Println("DEBUG: POST return %s", res)
}

func processBatch(list []MetricReading) {
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		// do more complex things here
		data, _ := json.Marshal(list)
		log.Println("DEBUG: processBatch %v", string(data))
		go sendMetricReadingsToMLProcess(string(data))
	}()
	wg.Wait()
	/*
	for _, i := range list {
		x := i
		wg.Add(1)
		go func() {
			defer wg.Done()
			// do more complex things here
			log.Println(x)
		}()
	}
	*/
}

const batchSize = 32

func process(data []MetricReading) {
	for start, end := 0, 0; start <= len(data)-1; start = end {
		end = start + batchSize
		if end > len(data) {
			end = len(data)
		}
		batch := data[start:end]
		processBatch(batch)
	}
	log.Println("DEBUG: done processing all data")
}

func contains(s []string, e string) bool {
	// Check if array contains a specific string
    for _, a := range s {
        if a == e {
            return true
        }
    }
    return false
}

func handleGroups(writeAPI api.WriteAPI, groupsChan chan *databus.DataGroup) {
	log.Printf("DEBUG handleGroups start")
	for group := range groupsChan {
		var valuesToProcess []MetricReading

		log.Printf("DEBUG handleGroups value count: %#v\n", len(group.Values))
		for _, value := range group.Values {
			floatVal, _ := strconv.ParseFloat(value.Value, 64)

			timestamp, err := time.Parse(time.RFC3339, value.Timestamp)
			//fmt.Printf("Value: %#v\n", value)
			if err != nil {
				log.Printf("Error parsing timestamp as RFC3339 for point %s: (%s) %v", value.Context+"_"+value.ID, value.Timestamp, err)
				continue
			}

			//log.Printf("DEBUG: WritePoint group %#v\n", group)
			//log.Printf("DEBUG: WritePoint value %#v\n", value)
			validLabels := strings.Split(configStrings["Labels"], ",")

			if group.Label == "Redfish Metric Report" {
				var r MetricReading
				r.ID = value.ID
				r.Context = value.Context
				r.Label = value.Label
				r.Value = floatVal
				r.System = value.System
				r.Timestamp = timestamp
				r.HostTags = group.HostTags
				r.HostName = group.HostName

				if contains(validLabels, r.Label) {
					//log.Printf("DEBUG MetricReading: %#v\n", r)
					valuesToProcess = append(valuesToProcess, r)
				}
			}

		}

		if len(valuesToProcess) > 0 {
			log.Printf("Sending %s readings to batch process\n", len(valuesToProcess))
			process(valuesToProcess)	
		}
	}
}

func getEnvSettings() {
	// debugging only. leaks potentially sensitive info, so leave this commented
	// unless debugging.
	// fmt.Printf("Environment dump: %#v\n", os.Environ())
	mbHost := os.Getenv("MESSAGEBUS_HOST")
	if len(mbHost) > 0 {
		configStrings["mbhost"] = mbHost
	}
	mbPort := os.Getenv("MESSAGEBUS_PORT")
	if len(mbPort) > 0 {
		configStrings["mbport"] = mbPort
	}
	configStrings["URL"] = os.Getenv("INFLUXDB_URL")
	configStrings["Token"] = os.Getenv("INFLUX_TOKEN")
	configStrings["Org"] = os.Getenv("INFLUX_ORG")
	configStrings["Bucket"] = os.Getenv("INFLUX_BUCKET")
	configStrings["Labels"] = os.Getenv("METRIC_LABELS")
}

func main() {
	ctx := context.Background()

	//Gather configuration from environment variables
	getEnvSettings()

	dbClient := new(databus.DataBusClient)
	stompPort, _ := strconv.Atoi(configStrings["mbport"])
	for {
		mb, err := stomp.NewStompMessageBus(configStrings["mbhost"], stompPort)
		if err != nil {
			log.Printf("Could not connect to message bus: %s", err)
			time.Sleep(5 * time.Second)
		} else {
			dbClient.Bus = mb
			defer mb.Close()
			break
		}
	}

	groupsIn := make(chan *databus.DataGroup, 10)
	dbClient.Subscribe("/ml")
	dbClient.Get("/ml")
	go dbClient.GetGroup(groupsIn, "/ml")

	if configStrings["Token"] == "" {
		log.Fatalf("must specify influx token using INFLUX_TOKEN environment variable")
	}

	for {
		time.Sleep(5 * time.Second)

		client := influxdb2.NewClientWithOptions(
			configStrings["URL"],
			configStrings["Token"],
			influxdb2.DefaultOptions().SetBatchSize(5000),
		)
		writeAPI := client.WriteAPI(configStrings["Org"], configStrings["Bucket"]) // async, non-blocking

		go func(writeAPI api.WriteAPI) {
			for err := range writeAPI.Errors() {
				fmt.Printf("async write error: %s\n", err)
			}
		}(writeAPI)

		// Never print out the token in debug output. print out the length of the string, most common problem is not set at all
		log.Printf("Connected to influx org(%s) bucket(%s) at URL (%s) Token Len(%v)\n", configStrings["Org"], configStrings["Bucket"], configStrings["URL"], strings.Repeat("X", len(configStrings["Token"])))

		timedCtx, cancel := context.WithTimeout(ctx, time.Second)
		defer cancel()
		ok, err := client.Ping(timedCtx)
		cancel()
		if !ok || err != nil {
			log.Printf("influx ping return = (%t): %s\n", ok, err)
			client.Close()
			continue
		}

		log.Printf("influx ping return = (%t)\n", ok)
		defer client.Close()
		handleGroups(writeAPI, groupsIn)
	}
}