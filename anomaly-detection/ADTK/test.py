

import os
import pandas as pd
from sklearn.cluster import DBSCAN
from influxdb_client import InfluxDBClient
from adtk.detector import MinClusterDetector, ThresholdAD, OutlierDetector
from adtk.data import validate_series
from sklearn.cluster import Birch
from sklearn.neighbors import LocalOutlierFactor
from adtk.visualization import plot
import tkinter
import matplotlib
import matplotlib.pyplot as plt

token = os.environ['INFLUX_TOKEN']
org = "my-org"
client = InfluxDBClient(url="http://localhost:8086", token=token, org=org)

query_test = '''from(bucket: "my-org-bucket")
  |> range(start: 2024-06-03T00:00:00Z ,  stop: 2024-06-03T23:59:00Z)
  |> filter(fn: (r) => r["_measurement"] == "telemetry")
  |> filter(fn: (r) => r["_field"] == "value")
  |> filter(fn: (r) => r["ServiceTag"]  == "9Z38MH3" )
  |> filter(fn: (r) => r["FQDD"] == "SystemUsage")
  |> filter(fn: (r) => r["Label"] == "SystemUsage CPUUsage")
  |> pivot(rowKey:["_time"], columnKey: ["Label"], valueColumn: "_value")
  |> drop(columns:["_start", "_stop", "host", "_field", "_measurement"])'''

query_train = '''from(bucket: "my-org-bucket")
  |> range(start: 2024-06-03T00:00:00Z ,  stop: 2024-06-03T23:59:00Z)
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
df_train = df_train.drop(columns=["result", "table"])
df_train = df_train.set_index("_time")
df_train.head()

#print(df_train.head().to_string())

# Get test dataset
df_test = query_api.query_data_frame(query_train)
#df_test.head()

print(df_test.head().to_string())

df_test["_time"] = pd.to_datetime(df_test["_time"].astype(str), errors='coerce')
df_test = df_test.drop(columns=["result", "table", "FQDD", "HostName", "HostTags", "ServiceTag"])
df_test = df_test.rename(columns={'SystemUsage CPUUsage': 'value'})
df_test = df_test.dropna(subset=['_time'])
df_test = df_test.set_index("_time")
#df_test.head()

print(df_test.info(show_counts=True))
print(df_test.dtypes)

print(df_test.head().to_string())

###
# Anomoly Detection
###
df_test = validate_series(df_test)

print(plt.style.available)

matplotlib.use('TkAgg')
# ThresholdAD
#threshold_ad = ThresholdAD(high=30, low=15)
#anomalies = threshold_ad.detect(df_test)
#plot(df_test, anomaly=anomalies, ts_linewidth=1, ts_markersize=3, anomaly_markersize=5, anomaly_color='red', anomaly_tag="marker")

# OutlierDetector
#outlier_detector = OutlierDetector(LocalOutlierFactor(contamination=0.05))
#anomalies = outlier_detector.fit_detect(df_test)
#plot(df_test, anomaly=anomalies, ts_linewidth=1, ts_markersize=3, anomaly_color='red', anomaly_alpha=0.3, curve_group='all')

# MinClusterDetector
min_cluster_detector = MinClusterDetector(Birch(n_clusters=10))
anomalies = min_cluster_detector.fit_detect(df_test)
plot(df_test, anomaly=anomalies, ts_linewidth=1, ts_markersize=3, anomaly_color='red', anomaly_alpha=0.3, curve_group='all')


plt.show()