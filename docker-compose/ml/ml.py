import time
import json
import re
import logging
import os
import sys
import datetime

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv, DataFrame, concat
from influxdb_client import InfluxDBClient, Point
from prophet import Prophet
from fastapi import FastAPI, Response, status
#from fastapi.log import log
from pydantic import BaseModel, Field
from typing import List
from logging.config import dictConfig
from contextlib import ExitStack

logging.getLogger("prophet").setLevel(logging.WARNING)
#logging.getLogger("cmdstanpy").disabled=True
cmdstanpy_logger = logging.getLogger('cmdstanpy')
cmdstanpy_logger.addHandler(logging.NullHandler())
cmdstanpy_logger.propagate = False
cmdstanpy_logger.setLevel(logging.CRITICAL)

log = logging.getLogger('ml')
log.setLevel(logging.INFO)
console_handler = logging.StreamHandler(sys.stdout)
log.addHandler(console_handler)

###
# Query & Prepare Dataset
###

def get_metric_readings(bucket, query_api, service_tag, fqdd, metric_label, end_date):
    query_train = '''from(bucket: "{bucket}")
    |> range(start: -1y,  stop: {end_date})
    |> filter(fn: (r) => r["_measurement"] == "telemetry")
    |> filter(fn: (r) => r["_field"] == "value")
    |> filter(fn: (r) => r["ServiceTag"]  == "{service_tag}" )
    |> filter(fn: (r) => r["FQDD"] == "{fqdd}")
    |> filter(fn: (r) => r["Label"] == "{metric_label}")
    |> pivot(rowKey:["_time"], columnKey: ["Label"], valueColumn: "_value")
    |> drop(columns:["_start", "_stop", "host", "_field", "_measurement"])'''.format(bucket=bucket, 
                                                                                     service_tag=service_tag, 
                                                                                     fqdd=fqdd, 
                                                                                     metric_label=metric_label,
                                                                                     end_date=end_date)

    log.info("QUERY: {}".format(query_train))
    # Get training dataset
    df_train = query_api.query_data_frame(query_train)
    df_train.head()

    log.info("READINGS")
    log.info(df_train.tail(20).to_string())
    #print(df_train.to_string())

    df_train["_time"] = pd.to_datetime(df_train["_time"].astype(str), errors='coerce')
    df_train = df_train.dropna(subset=['_time'])
    df_train['_time'] = df_train['_time'].dt.tz_convert(None)
    df_train = df_train.drop(columns=["result", "table", "FQDD", "HostName", "HostTags", "ServiceTag"])
    df_train = df_train.rename(columns={'SystemUsage CPUUsage': 'y', '_time': 'ds'})
    #df_train = df_train.set_index("ds")
    #df_train.head()

    #print(df_train.head().to_string())

    return df_train

def send_readings_influx(write_api, bucket, df, batch_start, batch_end, service_tag, fqdd, metric_label):
    # ds, trend, yhat, yhat_lower, yhat_upper, fact, anomaly, importance
    df = df.drop(columns=['trend', 'yhat', 'anomaly', 'importance'])
    df = df.rename(columns={
        'ds': '_time', 
        'fact': '_value', 
        'yhat_lower': 'Min', 
        'yhat_upper': 'Max'
        })
    df = df.set_index('_time')
    df['ServiceTag'] = service_tag
    df['FQDD'] = fqdd
    df['Label'] = metric_label

    write_api.write(bucket, record = df, data_frame_measurement_name='anomaly', data_frame_tag_columns=['ServiceTag', 'FQDD', 'Label'])

    '''
    for index, row in df.iterrows():
        p = Point("anomaly") \
            .tag("location", "Prague") \
            .field("temperature", 25.3)
            .tag("ServiceTag", value.System).
            .tag("FQDD", value.Context).
            .tag("Label", value.Label).
            .field("value", floatVal).
            SetTime(timestamp).
    p := write.NewPointWithMeasurement("telemetry").
					AddTag("ServiceTag", value.System).
					AddTag("FQDD", value.Context).
					AddTag("Label", value.Label).
					AddTag("HostTags", group.HostTags).
					AddTag("HostName", group.HostName).
					AddField("MetricID", value.ID).
					AddField("value", floatVal).
					SetTime(timestamp)
    '''

###
# Anomoly Detection
###

def fit_predict_model(dataframe, interval_width = 0.99, changepoint_range = 0.8):
   m = Prophet(daily_seasonality = True, yearly_seasonality = True, weekly_seasonality = True,
               seasonality_mode = 'additive',
               interval_width = interval_width,
               changepoint_range = changepoint_range)
   m = m.fit(dataframe)
   forecast = m.predict(dataframe)
   forecast['fact'] = dataframe['y'].reset_index(drop = True)
   return forecast

def detect_anomalies(forecast):
    forecasted = forecast[['ds','trend', 'yhat', 'yhat_lower', 'yhat_upper', 'fact']].copy()
    
    #log.info("DEBUG: forcasted")
    #log.info(forecasted.shape)
    #log.info(forecasted.dtypes)

    forecasted['anomaly'] = 0
    forecasted['anomaly'] = forecasted['anomaly'].astype('float64')
    forecasted.loc[forecasted['fact'] > forecasted['yhat_upper'], 'anomaly'] = 1
    forecasted.loc[forecasted['fact'] < forecasted['yhat_lower'], 'anomaly'] = -1
    #anomaly importances
    forecasted['importance'] = 0
    forecasted['importance'] = forecasted['importance'].astype('float64')
    forecasted.loc[forecasted['anomaly'] ==1, 'importance'] = (forecasted['fact'] - forecasted['yhat_upper'])/forecast['fact']
    forecasted.loc[forecasted['anomaly'] ==-1, 'importance'] = (forecasted['yhat_lower'] - forecasted['fact'])/forecast['fact']

    #log.info("DEBUG: forcasted")
    #log.info(forecasted.shape)
    #log.info(forecasted.dtypes)

    return forecasted
###
# Serve Requests via FastAPI
###

class MetricReading(BaseModel):
    id: str = Field(alias='ID')
    context: str = Field(alias='Context')
    label: str = Field(alias='Label')
    value: float = Field(alias='Value')
    system: str = Field(alias='System')
    timestamp: datetime.datetime = Field(alias='Timestamp')
    host_name: str = Field(alias='HostName', default=None) 
    host_tags: str = Field(alias='HostTags', default=None) 

token = os.environ['INFLUX_TOKEN']
org = os.environ['INFLUX_ORG']
bucket = os.environ['INFLUX_BUCKET']
influx_url =  os.environ['INFLUXDB_URL']
client = InfluxDBClient(url=influx_url, token=token, org=org)
query_api = client.query_api()
write_api = client.write_api()

app = FastAPI(debug=True)

log.info('API is starting up')

@app.post("/send_data")
async def send_data(items: List[MetricReading]):
    service_tags = list(set([item.system for item in items]))
    fqdds = list(set([item.context for item in items]))
    metric_labels = list(set([item.label for item in items]))
    batch_start = min(item.timestamp for item in items)
    batch_end = max(item.timestamp for item in items)

    for service_tag in service_tags:
        for fqdd in fqdds:
            for metric_label in metric_labels:
                df = get_metric_readings(
                    bucket=bucket, 
                    query_api=query_api, 
                    service_tag=service_tag, 
                    fqdd=fqdd, 
                    metric_label=metric_label,
                    end_date=batch_end.strftime("%Y-%m-%dT%H:%M:%SZ"))
                
                pred = fit_predict_model(df)
                pred = detect_anomalies(pred)
                #log.info(pred.to_string())

                mask = pred['anomaly'] == 1
                anomaly = pred.loc[mask]
                #log.info("Anomalies Found")
                #log.info(anomaly.to_string())

                # Parallelize this
                log.info("DEBUG: anomalies")
                log.info(anomaly.shape)
                log.info(anomaly.dtypes)
                # Filter anomalies by batch
                mask = (anomaly['ds'] >= np.datetime64(batch_start)) & (anomaly['ds'] <= np.datetime64(batch_end))
                anomalies_batch = anomaly.loc[mask]
                anomalies_batch_count = len(anomalies_batch)
                log.info("Anomalies in batch {}".format(anomalies_batch_count))
                log.info(anomalies_batch.to_string())
                if anomalies_batch_count > 0:
                    send_readings_influx(
                        write_api=write_api, 
                        bucket=bucket, 
                        df=anomalies_batch, 
                        batch_start=batch_start, 
                        batch_end=batch_end, 
                        service_tag=service_tag, 
                        fqdd=fqdd, 
                        metric_label=metric_label)
                
    log.info("Service Tags Found: {}".format(service_tags))
    log.info("FQDDs Found: {}".format(fqdds))
    log.info("Labels Found: {}".format(metric_labels))
    log.info("Batch Start: {}".format(batch_start))
    log.info("Batch End: {}".format(batch_end))

    return status.HTTP_200_OK