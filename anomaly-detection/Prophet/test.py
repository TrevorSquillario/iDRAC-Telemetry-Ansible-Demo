import os
import pandas as pd
from influxdb_client import InfluxDBClient
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from prophet import Prophet
from multiprocessing.pool import Pool
from multiprocessing import cpu_count

import logging

logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").disabled=True

token = os.environ['INFLUX_TOKEN']
org = "my-org"
client = InfluxDBClient(url="http://localhost:8086", token=token, org=org)

query_train = '''from(bucket: "my-org-bucket")
  |> range(start: -1w,  stop: now())
  |> filter(fn: (r) => r["_measurement"] == "telemetry")
  |> filter(fn: (r) => r["_field"] == "value")
  |> filter(fn: (r) => r["ServiceTag"]  == "9Z38MH3" )
  |> filter(fn: (r) => r["FQDD"] == "SystemUsage")
  |> filter(fn: (r) => r["Label"] == "SystemUsage CPUUsage")
  |> pivot(rowKey:["_time"], columnKey: ["Label"], valueColumn: "_value")
  |> drop(columns:["_start", "_stop", "host", "_field", "_measurement"])'''

query_test = '''from(bucket: "my-org-bucket")
  |> range(start: 2024-06-06T00:00:00Z ,  stop: 2024-06-06T23:59:00Z)
  |> filter(fn: (r) => r["_measurement"] == "telemetry")
  |> filter(fn: (r) => r["_field"] == "value")
  |> filter(fn: (r) => r["ServiceTag"]  == "9Z38MH3" )
  |> filter(fn: (r) => r["FQDD"] == "SystemUsage")
  |> filter(fn: (r) => r["Label"] == "SystemUsage CPUUsage")
  |> pivot(rowKey:["_time"], columnKey: ["Label"], valueColumn: "_value")
  |> drop(columns:["_start", "_stop", "host", "_field", "_measurement"])'''

query_api = client.query_api()

###
# Query & Prepare Dataset
###

# Get training dataset
df_train = query_api.query_data_frame(query_train)
df_train.head()

#print(df_train.to_string())

df_train["_time"] = pd.to_datetime(df_train["_time"].astype(str), errors='coerce')
df_train = df_train.dropna(subset=['_time'])
df_train['_time'] = df_train['_time'].dt.tz_convert(None)
df_train = df_train.drop(columns=["result", "table", "FQDD", "HostName", "HostTags", "ServiceTag"])
df_train = df_train.rename(columns={'SystemUsage CPUUsage': 'y', '_time': 'ds'})
#df_train = df_train.set_index("ds")
df_train.head()

print(df_train.head().to_string())

###
# Anomoly Detection
###
matplotlib.use('TkAgg')

def fit_predict_model(dataframe, interval_width = 0.99, changepoint_range = 0.8):
   m = Prophet(daily_seasonality = True, yearly_seasonality = True, weekly_seasonality = True,
               seasonality_mode = 'additive',
               interval_width = interval_width,
               changepoint_range = changepoint_range).add_seasonality(name='hourly', period=1/24, fourier_order = 1)
   m = m.fit(dataframe)
   forecast = m.predict(dataframe)
   forecast['fact'] = dataframe['y'].reset_index(drop = True)
   return forecast

pred = fit_predict_model(df_train)

def detect_anomalies(forecast):
  forecasted = forecast[['ds','trend', 'yhat', 'yhat_lower', 'yhat_upper', 'fact']].copy()
  forecasted['anomaly'] = 0
  forecasted.loc[forecasted['fact'] > forecasted['yhat_upper'], 'anomaly'] = 1
  forecasted.loc[forecasted['fact'] < forecasted['yhat_lower'], 'anomaly'] = -1
  #anomaly importances
  forecasted['importance'] = 0
  forecasted.loc[forecasted['anomaly'] ==1, 'importance'] = (forecasted['fact'] - forecasted['yhat_upper'])/forecast['fact']
  forecasted.loc[forecasted['anomaly'] ==-1, 'importance'] = (forecasted['yhat_lower'] - forecasted['fact'])/forecast['fact']

  return forecasted

pred = detect_anomalies(pred)
print(pred)

anomaly = pred[pred['anomaly'] == 1]

#fig = plt.scatter(x=pred['ds'], y=pred['fact'], color='anomaly', title='Anomaly', cmap=color_discrete_map)
#fig.show()
plt.scatter(x=anomaly['ds'], y=anomaly['fact'], label='Anomaly', color="r")
plt.plot(pred['ds'], pred['fact'], label='Normal')

plt.legend() 
plt.show()