import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from pandas.plotting import register_matplotlib_converters
from influxdb_client import InfluxDBClient
from sklearn.preprocessing import StandardScaler

# https://github.com/curiousily/Deep-Learning-For-Hackers/blob/master/14.time-series-anomaly-detection.ipynb

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

###
# Query & Prepare Dataset
###

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

register_matplotlib_converters()
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 22, 10

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

#train_size = int(len(df_train) * 0.95)
#test_size = len(df_test) - train_size
#train, test = df_train.iloc[0:train_size], df.iloc[train_size:len(df)]
print(df_train.shape, df_test.shape)

scaler = StandardScaler()
scaler = scaler.fit(df_train[['value']])

df_train['value'] = scaler.transform(df_train[['value']])
df_test['value'] = scaler.transform(df_test[['value']])

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)        
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

TIME_STEPS = 30

# reshape to [samples, time_steps, n_features]
X_train, y_train = create_dataset(df_train[['value']], df_train.value, TIME_STEPS)
X_test, y_test = create_dataset(df_test[['value']], df_test.value, TIME_STEPS)

print(X_train.shape)

model = keras.Sequential()
model.add(keras.layers.LSTM(
    units=64, 
    input_shape=(X_train.shape[1], X_train.shape[2])
))
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.RepeatVector(n=X_train.shape[1]))
model.add(keras.layers.LSTM(units=64, return_sequences=True))
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=X_train.shape[2])))
model.compile(loss='mae', optimizer='adam')

history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    shuffle=False
)

X_train_pred = model.predict(X_train)

train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)

#sns.displot(train_mae_loss, bins=50, kde=True)
#plt.show()

X_test_pred = model.predict(X_test)

test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)

THRESHOLD = 0.65

test_score_df = pd.DataFrame(index=df_test[TIME_STEPS:].index)
test_score_df['loss'] = test_mae_loss
test_score_df['threshold'] = THRESHOLD
test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
test_score_df['value'] = df_test[TIME_STEPS:].value

#plt.plot(test_score_df.index, test_score_df.loss, label='loss')
#plt.plot(test_score_df.index, test_score_df.threshold, label='threshold')
#plt.xticks(rotation=25)
#plt.legend()
#plt.show()

anomalies = test_score_df[test_score_df.anomaly == True]
print(anomalies.head().to_string())

X = df_test[["value"]]

plt.plot(
  df_test[TIME_STEPS:].index, 
  scaler.inverse_transform(X[TIME_STEPS:]), 
  label='CPUUsage'
)

A = anomalies[["value"]]

anomalies["value"] = scaler.inverse_transform(A)

ax = sns.scatterplot(
    data=anomalies,
    x="_time",
    y="value",
    color=sns.color_palette()[3],
    s=52,
    label='anomaly'
)

plt.xticks(rotation=25)
plt.legend()

plt.show()
