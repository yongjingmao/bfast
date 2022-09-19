# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 12:43:59 2022

@author: uqymao1
"""
import tensorflow as tf
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#%%
df = pd.read_csv(r'data/GroundCover.csv')
df['GC'] = 100-np.array(df['bare50percentile'])

df['Datetime'] = pd.to_datetime(df['date'])
season = (df['Datetime'].dt.month-1)//3+1
year = df['Datetime'].dt.year

df = df.groupby([year,season]).agg(
    {'Datetime':'first', 'bare50percentile':'first', 'rain(mm)':'sum'}
    ).set_index('Datetime')

df.columns = ['GC', 'Rain']
df['GC'] = 100-df['GC']


scaler1 = StandardScaler()
df[['GC', 'Rain']] = scaler1.fit_transform(df[['GC', 'Rain']])
df.dropna(inplace=True)


df.plot()
# zip_path = tf.keras.utils.get_file(
#     origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
#     fname='jena_climate_2009_2016.csv.zip',
#     extract=True)
# csv_path, _ = os.path.splitext(zip_path)
# df = pd.read_csv(csv_path)[['Date Time', 'T (degC)']]
# df['Date Time'] = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')
# df.set_index('Date Time', inplace=True)
#%%
def df_to_X_y(df, window_size):
  df_as_np = df.to_numpy()
  X = []
  y = []
  idx = []
  for i in range(len(df_as_np)-window_size):
    row = [[a] for a in df_as_np[i:i+window_size, :].flatten()]
    X.append(row)
    label = df_as_np[i+window_size, :].flatten()
    y.append(label)
    idx.append(df.index[i+window_size])
  return np.array(X), np.array(y), idx


WINDOW_SIZE = 4
X1, y1, idx = df_to_X_y(df, WINDOW_SIZE)
X_train, X_test_val, y_train, y_test_val, idx_train, idx_test_val = train_test_split(X1, y1, idx, test_size=0.3, 
                                                    shuffle=False)

X_test, X_val, y_test, y_val, idx_test, idx_val = train_test_split(X_test_val, y_test_val, idx_test_val, test_size=0.1, 
                                                    shuffle=False)

#%%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# model = Sequential()
# model.add(InputLayer((WINDOW_SIZE, 1)))
# model.add(LSTM(64))
# model.add(Dropout(0.2))
# #model.add(Dense(8, 'relu'))
# model.add(Dense(1, 'linear')) 
# model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.001), metrics=[RootMeanSquaredError()])
# model.summary()

model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64, activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(y_train.shape[1], 'linear'))

model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.001), metrics=[RootMeanSquaredError()])
model.summary()


history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100)
plt.semilogy(history.history['loss'])

#%% Use oberved value to predict the next step
target_name = 'GC'
target_idx = df.columns.get_loc(target_name)
test_predictions = model.predict(X_test)
test_results = pd.DataFrame(data={'Test Pred':test_predictions[:,target_idx], 'Test Obs':y_test[:,target_idx]}, index=idx_test)
train_predictions = model.predict(X_train)
train_results = pd.DataFrame(data={'Train Pred':train_predictions[:,target_idx], 'Train Obs':y_train[:,target_idx]}, index=idx_train)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
test_results.plot(ax=ax, linestyle='--', color=['Red', 'Green'])
train_results.plot(ax=ax, linestyle='-', color=['Red', 'Green'])
ax.legend(ncol=2)

#%% Use predicted value to predict next step
target_name = 'GC'
target_idx = df.columns.get_loc(target_name)

test_predictions1 = model.predict(X_test)
test_results1 = pd.DataFrame(data={'Test Pred (Obs-based)':test_predictions1[:,target_idx], 'Test Obs':y_test[:,target_idx]}, index=idx_test)
train_predictions = model.predict(X_train)
train_results = pd.DataFrame(data={'Train Pred':train_predictions[:,target_idx], 'Train Obs':y_train[:,target_idx]}, index=idx_train)

X_pred = X_test.copy()
for i in range(len(X_pred)-WINDOW_SIZE):
    y_pred = model.predict(X_test[i].reshape((1, X1.shape[1], 1))).flatten()
    for j in range(1, WINDOW_SIZE+1):
        X_pred[i+j][np.arange(y1.shape[1])+(j-1)*y1.shape[1]] = y_pred.reshape(y1.shape[1],1)

test_predictions2 = model.predict(X_pred)
test_results2 = pd.DataFrame(data={'Test Pred (Pred-based)':test_predictions2[:,target_idx], 'Test Obs':y_test[:,target_idx]}, index=idx_test)


import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8,4))
test_results1.plot(ax=ax, linestyle='--', color=['Red', 'Green'])
test_results2['Test Pred (Pred-based)'].plot(ax=ax, linestyle='--', color=['Blue'])
train_results.plot(ax=ax, linestyle='-', color=['Red', 'Green'])
ax.legend(ncol=2)    