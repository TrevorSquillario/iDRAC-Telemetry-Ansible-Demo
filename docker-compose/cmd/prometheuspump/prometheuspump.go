// Licensed to You under the Apache License, Version 2.0.

package main

import (
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"
	"golang.org/x/exp/maps"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"

	"github.com/dell/iDRAC-Telemetry-Reference-Tools/internal/databus"
	"github.com/dell/iDRAC-Telemetry-Reference-Tools/internal/messagebus/stomp"
)

var configStrings = map[string]string{
	"mbhost": "activemq",
	"mbport": "61613",
}

var collectors map[string]map[string]*prometheus.GaugeVec

func getValidMetricName(name string) string {
	// Metric names can contain letters, numbers, underscores, or colons, based on the regular expression [a-zA-Z_:][a-zA-Z0-9_:]*.
	// Reserve colons (:) in the name for calculated or aggregated metrics, such as those produced by rollup rules.

	// Rename Context like 'SMARTData:Disk.Bay.0:Enclosure.Internal.0-1:RAID.SL.3-1' to 'SMARTData'
	colon_match := strings.Index(name, ":")
	var new_name string
	if colon_match > -1 {
		new_name = name[:colon_match]
	} else {
		// Rename Context like 'DIMM.Socket.A2' to 'DIMM'
		period_match := strings.Index(name, ".")
		if period_match > -1 {
			new_name = name[:period_match]
		} else {
			new_name = name
		}
	}
	log.Printf("Renaming Metric from %s to %s at string index %s", name, new_name, colon_match)
	return new_name
}

func getTags(hostTags map[string]string, system string, context string, hostname string) map[string]string {
	defaultTags := map[string]string{
		"ServiceTag": system,
		"FQDD": context,
		"HostName": hostname,
	}
	for key, value := range hostTags {
		defaultTags[key] = value
	}
	log.Printf("DEBUG: getTags value %#v\n", defaultTags)
	return defaultTags
}

func doFQDDGuage(value databus.DataValue, registry *prometheus.Registry, hostTags map[string]string, hostName string) {
	log.Printf("DEBUG: doFQDDGuage value %#v\n", value)
	tags := getTags(hostTags, value.System, value.Context, hostName)
	if collectors["FQDD"] == nil {
		collectors["FQDD"] = make(map[string]*prometheus.GaugeVec)
	}
	if collectors["FQDD"][value.ID] == nil {
		guage := prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Namespace: "PowerEdge",
				Subsystem: getValidMetricName(value.Context),
				Name:      value.ID,
			},
			maps.Keys(tags))
		registry.MustRegister(guage)
		floatVal, err := strconv.ParseFloat(value.Value, 64)
		if err != nil {
			if value.Value == "Up" || value.Value == "Operational" {
				floatVal = 1
			}
		}
		guage.WithLabelValues(maps.Values(tags)...).Set(floatVal)
		collectors["FQDD"][value.ID] = guage
	} else {
		guage := collectors["FQDD"][value.ID]
		floatVal, err := strconv.ParseFloat(value.Value, 64)
		if err != nil {
			if value.Value == "Up" || value.Value == "Operational" {
				floatVal = 1
			}
		}
		guage.WithLabelValues(maps.Values(tags)...).Set(floatVal)
	}
}

func doNonFQDDGuage(value databus.DataValue, registry *prometheus.Registry, hostTags map[string]string, hostName string) {
	log.Printf("DEBUG: doNonFQDDGuage value %#v\n", value)
	value.Context = strings.Replace(value.Context, " ", "", -1)
	tags := getTags(hostTags, value.System, value.Context, hostName)
	if collectors[value.Context] == nil {
		collectors[value.Context] = make(map[string]*prometheus.GaugeVec)
	}
	if collectors[value.Context][value.ID] == nil {
		guage := prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Namespace: "PowerEdge",
				Subsystem: value.Context,
				Name:      value.ID,
			},
			maps.Keys(tags))
		registry.MustRegister(guage)
		floatVal, _ := strconv.ParseFloat(value.Value, 64)
		guage.WithLabelValues(maps.Values(tags)...).Set(floatVal)
		collectors[value.Context][value.ID] = guage
	} else {
		guage := collectors[value.Context][value.ID]
		floatVal, _ := strconv.ParseFloat(value.Value, 64)
		guage.WithLabelValues(maps.Values(tags)...).Set(floatVal)
	}
}

func handleGroups(groupsChan chan *databus.DataGroup, registry *prometheus.Registry) {
	collectors = make(map[string]map[string]*prometheus.GaugeVec)
	for {
		group := <-groupsChan
		for _, value := range group.Values {
			//log.Print("value: ", value)
			if group.Label == "Redfish Metric Report" {
				if strings.Contains(value.Context, ".") {
					doFQDDGuage(value, registry, group.HostTags, group.HostName)
				} else {
					doNonFQDDGuage(value, registry, group.HostTags, group.HostName)
				}
			}
		}
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
}

func main() {

	//Gather configuration from environment variables
	getEnvSettings()

	dbClient := new(databus.DataBusClient)
	//Initialize messagebus
	for {
		stompPort, _ := strconv.Atoi(configStrings["mbport"])
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
	dbClient.Subscribe("/prometheus")
	dbClient.Get("/prometheus")
	go dbClient.GetGroup(groupsIn, "/prometheus")

	registry := prometheus.NewRegistry()
	go handleGroups(groupsIn, registry)

	gatherer := prometheus.Gatherer(registry)
	http.Handle("/metrics", promhttp.HandlerFor(gatherer, promhttp.HandlerOpts{}))
	err := http.ListenAndServe(":2112", nil)
	if err != nil {
		log.Printf("Failed to start webserver %v", err)
	}
}