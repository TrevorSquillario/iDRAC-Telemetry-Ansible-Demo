import time
import json
import pickle
import re
import logging
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv, DataFrame, concat
from sklearn.preprocessing import MinMaxScaler, MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import LSTM, Dense, Dropout, RepeatVector, TimeDistributed
from influxdb_client import InfluxDBClient

# https://github.com/claimed-framework/component-library/blob/master/component-library/anomaly/anomaly-score-unsupervised.ipynb
# https://github.com/claimed-framework/component-library/blob/master/component-library/anomaly/anomaly-score-unsupervised/Dockerfile

# Number of LSTM layers, default: 3
lstm_layers = int(os.environ.get('lstm_layers', '3'))

# Number of LSTM layers, default: 10
lstm_cells_per_layer = int(os.environ.get('lstm_cells_per_layer', '10'))

# LSTM time steps, default: 10
timesteps = int(os.environ.get('timesteps', '10'))

# Time series dimensionality, default: 3
dim = int(os.environ.get('dim', '3'))

# batch size, default: 32
batch_size = int(os.environ.get('batch_size', '32'))

# epochs, default: 3
epochs = int(os.environ.get('epochs', '3'))

parameters = list(
    map(lambda s: re.sub('$', '"', s),
        map(
            lambda s: s.replace('=', '="'),
            filter(
                lambda s: s.find('=') > -1 and bool(re.match(r'[A-Za-z0-9_]*=[.\/A-Za-z0-9]*', s)),
                sys.argv
            )
    )))

lstm_layers = int(lstm_layers)
lstm_cells_per_layer = int(lstm_cells_per_layer)
timesteps = int(timesteps)
dim = int(dim)
batch_size = int(batch_size)
modelname = "cpuusage.keras"

'''
def lstm_data_transform(data, num_steps=5):
    x = []
    for i in range(data.shape[0]):
        # compute a new (sliding window) index
        end_ix = i + num_steps        # if index is larger than the size of the dataset, we stop
        if end_ix >= data.shape[0]:
            break        # Get a sequence of data for x
        seq = data[i:end_ix]
        x.append(seq)
    return np.array(x)
'''

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)        
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

loss_history = []
loss_history_total = []

class LossHistory(Callback):
    def on_train_begin(self, logs):
        loss_history = [] 

    def on_train_batch_end(self, batch, logs):
        print('Loss on_train_batch_end '+str(logs.get('loss')))
        loss_history.append(logs.get('loss'))
        loss_history_total.append(logs.get('loss'))

# Create LSTM Model
'''
def create_model_orig():
    model = Sequential()
    for _ in range(lstm_layers):
        model.add(LSTM(lstm_cells_per_layer,input_shape=(timesteps,dim),return_sequences=True))
    model.add(Dense(dim))
    model.compile(loss='mae', optimizer='adam')
    return model
'''

def create_model(X_train):
    model = Sequential()
    model.add(LSTM(
        units=64, 
        input_shape=(X_train.shape[1], X_train.shape[2])
    ))
    model.add(Dropout(rate=0.2))
    model.add(RepeatVector(n=X_train.shape[1]))
    model.add(LSTM(units=64, return_sequences=True))
    model.add(Dropout(rate=0.2))
    model.add(TimeDistributed(Dense(units=X_train.shape[2])))
    model.compile(loss='mae', optimizer='adam')
    return model

def train(model, X_train, y_train, batch_size=32):
    #model.fit(data, data, epochs=epochs, batch_size=batch_size, validation_data=(data, data), verbose=0, shuffle=False, callbacks=[LossHistory()])
    model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=batch_size,
        validation_split=0.1,
        shuffle=False,
        verbose=1,
        callbacks=[LossHistory()]
    )
    model.save(modelname)

def test(X_test, df):
    X_test_pred = model.predict(X_test)

    test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)

    THRESHOLD = 0.65

    test_score_df = pd.DataFrame(index=df[TIME_STEPS:].index)
    test_score_df['loss'] = test_mae_loss
    test_score_df['threshold'] = THRESHOLD
    test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
    test_score_df['value'] = df[TIME_STEPS:].value

    anomalies = test_score_df[test_score_df.anomaly == True]
    return anomalies

def test_reading(timestamp, value):
    #value = np.expand_dims(value, 0)
    data = [{ '_time': timestamp, 'value': value}]
    df = pd.DataFrame(data)
    df["_time"] = pd.to_datetime(df["_time"].astype(str), errors='coerce')
    df = df.set_index("_time")
    #df['value'] = scaler.transform(df[['value']])
    print(df)
    print(df.info(show_counts=True))
    print(df.dtypes)
    Xs = []
    Xs.append(df[['value']].iloc[0:1].values)
    X_train_point = np.array(Xs)
    print(X_train_point)
    print(X_train_point.shape)

    X_test_pred = model.predict(x=X_train_point, batch_size=1).reshape(-1)

    test_mae_loss = np.mean(np.abs(X_test_pred - value), axis=0)

    THRESHOLD = 0.65

    #test_score_df = pd.DataFrame(index=df_test[TIME_STEPS:].index)
    test_score_df = {
        '_time': timestamp,
        'loss': test_mae_loss,
        'threshold': THRESHOLD,
        'value': value
    }
    test_score_df['anomaly'] = test_score_df['loss'] > test_score_df['threshold']
    return test_score_df['anomaly'] == True

'''
def doNN(data):
    #data = scaleData(data)
    train(data)
'''

###
# Query & Prepare Dataset
###

token = os.environ['INFLUX_TOKEN']
org = "my-org"
client = InfluxDBClient(url="http://localhost:8086", token=token, org=org)

query_train = '''from(bucket: "my-org-bucket")
  |> range(start: 2024-06-03T00:00:00Z ,  stop: 2024-06-03T23:59:00Z)
  |> filter(fn: (r) => r["_measurement"] == "telemetry")
  |> filter(fn: (r) => r["_field"] == "value")
  |> filter(fn: (r) => r["ServiceTag"]  == "9Z38MH3" )
  |> filter(fn: (r) => r["FQDD"] == "SystemUsage")
  |> filter(fn: (r) => r["Label"] == "SystemUsage CPUUsage")
  |> pivot(rowKey:["_time"], columnKey: ["Label"], valueColumn: "_value")
  |> drop(columns:["_start", "_stop", "host", "_field", "_measurement"])'''

query_test = '''from(bucket: "my-org-bucket")
  |> range(start: 2024-06-04T00:00:00Z ,  stop: 2024-06-04T23:59:00Z)
  |> filter(fn: (r) => r["_measurement"] == "telemetry")
  |> filter(fn: (r) => r["_field"] == "value")
  |> filter(fn: (r) => r["ServiceTag"]  == "9Z38MH3" )
  |> filter(fn: (r) => r["FQDD"] == "SystemUsage")
  |> filter(fn: (r) => r["Label"] == "SystemUsage CPUUsage")
  |> pivot(rowKey:["_time"], columnKey: ["Label"], valueColumn: "_value")
  |> drop(columns:["_start", "_stop", "host", "_field", "_measurement"])'''

query_api = client.query_api()

# Get training dataset
df_train = query_api.query_data_frame(query_train)
df_train.head()

#print(df_train.to_string())

df_train["_time"] = pd.to_datetime(df_train["_time"].astype(str), errors='coerce')
df_train = df_train.drop(columns=["result", "table", "FQDD", "HostName", "HostTags", "ServiceTag"])
df_train = df_train.rename(columns={'SystemUsage CPUUsage': 'value'})
df_train = df_train.dropna(subset=['_time'])
df_train = df_train.set_index("_time")
#df_train.head()

print(df_train.head().to_string())

# Get test dataset
df_test = query_api.query_data_frame(query_train)

print(df_test.head().to_string())

df_test["_time"] = pd.to_datetime(df_test["_time"].astype(str), errors='coerce')
df_test = df_test.drop(columns=["result", "table", "FQDD", "HostName", "HostTags", "ServiceTag"])
df_test = df_test.rename(columns={'SystemUsage CPUUsage': 'value'})
df_test = df_test.dropna(subset=['_time'])
df_test = df_test.set_index("_time")

print(df_test.info(show_counts=True))
print(df_test.dtypes)

print(df_test.head().to_string())   

###
# Anomoly Detection via LSTM AutoEncoder
###


# Train Model
scaler = StandardScaler()
scaler = scaler.fit(df_train[['value']])

df_train['value'] = scaler.transform(df_train[['value']])
df_test['value'] = scaler.transform(df_test[['value']])

TIME_STEPS = 30

# reshape to [samples, time_steps, n_features]
X_train, y_train = create_dataset(df_train[['value']], df_train.value, TIME_STEPS)

if os.path.isfile(modelname):
    model = keras.models.load_model(modelname)
else:
    model = create_model(X_train)
    train(model, X_train, y_train)
    print(loss_history)

# Test Model
X_test, y_test = create_dataset(df_test[['value']], df_test.value, TIME_STEPS)
#print(X_test)
#print(X_test.shape)

anomalies = test(X_test, df_test)
print("---ANOMALIES---\n")
print(anomalies.head().to_string())

# Single Point Test
#weights = model.get_weights()
#single_item_model = create_model(batch_size=1)
#single_item_model.set_weights(weights)
#single_item_model.compile(loss='mae', optimizer='adam')
#timestamp = "2024-06-03 13:08:58.016000+00:00"
#value = 22.822222
#anomaly = test_reading(timestamp, value)
#print(anomaly)

# Backtesting
for index, anomaly in anomalies.iterrows():
    #print(anomaly.index, anomaly['value'])
    timestamp = anomaly.index
    value = anomaly['value']
    anomaly_result = test_reading(timestamp, value)
    print(anomaly_result)