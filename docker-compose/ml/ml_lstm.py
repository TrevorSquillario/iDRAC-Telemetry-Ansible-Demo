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
from flask import request
from flask import Flask
from flask import send_file

# https://github.com/claimed-framework/component-library/blob/master/component-library/anomaly/anomaly-score-unsupervised.ipynb
# https://github.com/claimed-framework/component-library/blob/master/component-library/anomaly/anomaly-score-unsupervised/Dockerfile
# https://github.com/susanli2016/Machine-Learning-with-Python/blob/master/Timeseries%20anomaly%20detection%20using%20LSTM%20Autoencoder%20JNJ.ipynb

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

scaler = StandardScaler()
lstm_layers = int(lstm_layers)
lstm_cells_per_layer = int(lstm_cells_per_layer)
timesteps = int(timesteps)
dim = int(dim)
batch_size = int(batch_size)

loss_history = []
loss_history_total = []

class LossHistory(Callback):
    def on_train_begin(self, logs):
        loss_history = [] 

    def on_train_batch_end(self, batch, logs):
        print('Loss on_train_batch_end '+str(logs.get('loss')))
        loss_history.append(logs.get('loss'))
        loss_history_total.append(logs.get('loss'))

def create_dataset(X, y, time_steps=1):
    '''
        Convert input data into 3-D array combining TIME_STEPS. 
        The shape of the array should be [samples, TIME_STEPS, features], as required for LSTM network.
    '''

    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:(i+time_steps)].values)
        ys.append(y.iloc[i+time_steps])
    return np.array(Xs), np.array(ys)

def create_dataset_from_metric_report(data, num_steps=5):
    df = pd.json_normalize(data)
    print(df.to_string())

    df["Timestamp"] = pd.to_datetime(df["Timestamp"].astype(str), errors='coerce')
    df = df.drop(columns=["ID", "Context", "Label", "HostName", "HostTags", "System"])
    df = df.rename(columns={'Value': 'value', 'Timestamp': '_time'})
    df = df.dropna(subset=['_time'])
    df = df.set_index("_time")
    scaler.fit(df[['value']])
    df['value'] = scaler.transform(df[['value']])

    print("DEBUG: df string\n")
    print(df.to_string())
    print("DEBUG: df shape\n")
    print(df.shape)


    # Test Model
    X_test, y_test = create_dataset(X=df[['value']], y=df['value'], time_steps=timesteps)

    print("DEBUG: X_test dataset\n")
    print(X_test)
    print("DEBUG: X_test shape\n")
    print(X_test.shape)

    return X_test, df


# Create LSTM Model
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

def train(model, X_train, y_train, model_name, model_save_path, batch_size=32):
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
    save_path = "{path}/{name}".format(path=model_save_path, name=model_name)
    print("Saving model to {save}".format(save=save_path))
    model.save(save_path)

def test(X_test, df, model):
    X_test_pred = model.predict(X_test, verbose=0)

    test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)

    print(df)
    print(test_mae_loss)

    THRESHOLD = 0.65

    # Setting index after N timesteps from past in test dataset
    #test_score_df = pd.DataFrame(index=df[timesteps:].index)
    test_score_df = pd.DataFrame(df[timesteps:])
    test_score_df.index = df[timesteps:].index
    test_score_df['loss'] = test_mae_loss
    test_score_df['threshold'] = THRESHOLD
    test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
    test_score_df['value'] = df[timesteps:]["value"]
    # Do an inverse transform on the value column to get back to the actual value
    # test_score_df['value']scaler.inverse_transform(test_score_df['value'])

    #print(X_test_pred)
    #print(X_test)
    #print(test_mae_loss)
    print("DEBUG: anomaly results raw\n")
    print(test_score_df) 
    anomalies = test_score_df[test_score_df.anomaly == True]
    return anomalies

def train_server_metric(bucket, query_api, service_tag, metric_label, model_name, model_save_path):
    query_train = '''from(bucket: "{bucket}")
    |> range(start: -1y ,  stop: now())
    |> filter(fn: (r) => r["_measurement"] == "telemetry")
    |> filter(fn: (r) => r["_field"] == "value")
    |> filter(fn: (r) => r["ServiceTag"]  == "{service_tag}" )
    |> filter(fn: (r) => r["FQDD"] == "SystemUsage")
    |> filter(fn: (r) => r["Label"] == "{metric_label}")
    |> pivot(rowKey:["_time"], columnKey: ["Label"], valueColumn: "_value")
    |> drop(columns:["_start", "_stop", "host", "_field", "_measurement"])'''.format(bucket=bucket, service_tag=service_tag, metric_label=metric_label)

    print(query_train)
    # Get training dataset
    df_train = query_api.query_data_frame(query_train)

    df_train["_time"] = pd.to_datetime(df_train["_time"].astype(str), errors='coerce')
    df_train = df_train.drop(columns=["result", "table", "FQDD", "HostName", "HostTags", "ServiceTag"])
    df_train = df_train.rename(columns={'SystemUsage CPUUsage': 'value'})
    df_train = df_train.dropna(subset=['_time'])
    df_train = df_train.set_index("_time")
    #df_train.head()

    print(df_train.head().to_string())

    # Train Model
    scaler = scaler.fit(df_train[['value']])

    df_train['value'] = scaler.transform(df_train[['value']])

    # reshape to [samples, time_steps, n_features]
    X_train, y_train = create_dataset(X=df_train[['value']], y=df_train.value, time_steps=timesteps)

    model = create_model(X_train)
    train(model=model, X_train=X_train, y_train=y_train, model_name=model_name, model_save_path=model_save_path, batch_size=timesteps)
    print(loss_history)

    return model

def change_case(str):
    label = str.replace(" ", "_").lower()
    return label

###
# Query & Prepare Dataset
###

token = os.environ['INFLUX_TOKEN']
org = os.environ['INFLUX_ORG']
bucket = os.environ['INFLUX_BUCKET']
client = InfluxDBClient(url="http://influx:8086", token=token, org=org)
query_api = client.query_api()

#train_server_metric(bucket=bucket, service_tag="9Z38MH3", metric_label="SystemUsage CPUUsage", query_api=query_api)

###
# Anomoly Detection via LSTM AutoEncoder
###

'''
# Test Model
X_test, y_test = create_dataset(df_test[['value']], df_test.value, timesteps)
#print(X_test)
#print(X_test.shape)

anomalies = test(X_test, df_test)
print("---ANOMALIES---\n")
print(anomalies.head().to_string())
'''
###
# Flask REST API
###

app = Flask(__name__)

@app.route('/send_data', methods=['POST'])
def send_data():
    message = request.get_json()
    #message = message[1:-1] # get rid of encapsulating quotes
    #json_array = json.loads()
    #data = np.asarray(message)
    #print(data)
    X_test, df = create_dataset_from_metric_report(message, num_steps=timesteps)
    service_tag = "9Z38MH3"
    metric_label = "SystemUsage CPUUsage"
    model_save_path = "/data"
    model_name = "{service_tag}.{metric_label}.keras".format(service_tag=service_tag, metric_label= change_case(metric_label))
    model_path = "{path}/{name}".format(path=model_save_path, name=model_name)
    if os.path.isfile(model_path):
        model = keras.models.load_model(model_path)
    else:
        model = train_server_metric(bucket, query_api, service_tag, metric_label, model_name, model_save_path)

    anomalies = test(X_test=X_test, df=df, model=model)
    print("---ANOMALIES---\n")
    print(anomalies.to_string())
    #test(data)
    #return json.dumps(loss_history)
    # json.dumps(anomalies, default=DataFrame.to_dict)
    return json.dumps(message)

@app.route('/reset_model', methods=['GET'])
def reset_model():
    loss_history = []
    loss_history_total = []
    #model = create_model()
    return "done"

@app.route('/get_loss_as_json', methods=['GET'])
def get_loss_as_json():
    return json.dumps(loss_history_total)

app.run(host="0.0.0.0", port=80, debug=True)