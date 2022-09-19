# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 14:00:41 2022

@author: uqymao1
"""

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

from pmdarima.arima import auto_arima
#%%
# Prepare datasets

df = pd.read_csv(r'data/GroundCover.csv')
df['GC'] = 100-np.array(df['bare50percentile'])

df['Datetime'] = pd.to_datetime(df['date'])
df = df[['Datetime', 'GC']].dropna().set_index('Datetime')

df.plot()

#%%
# Define functions

def adfuller_test(sales):
    result=adfuller(sales)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    if result[1] <= 0.05:
        print("P value is less than 0.05 that means we can reject the null hypothesis(Ho). Therefore we can conclude that data has no unit root and is stationary")
    else:
        print("Weak evidence against null hypothesis that means time series has a unit root which indicates that it is non-stationary ")
adfuller_test(df['GC'])



result = seasonal_decompose(df, model='multiplicative', period=3)
fig = result.plot()


#%% Model set-up
# Train Test Split Index
train_size = 0.95
split_idx = round(len(df)* train_size)

# Split
train = df.iloc[:split_idx]
test = df.iloc[split_idx:]

# Visualize split
fig,ax= plt.subplots(figsize=(12,8))
kws = dict(marker='o')
plt.plot(train, label='Train', **kws)
plt.plot(test, label='Test', **kws)
ax.legend(bbox_to_anchor=[1,1]);

#%%
model = auto_arima(train, start_p=0, start_q=0, start_d=0,
                           max_p=5, max_q=5, max_d=5, m=12,
                           start_P=0, max_P=5, start_Q=0, max_Q=5,
                           seasonal=True,
                           D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
model.summary()
model.plot_diagnostics()

#%%
predictions = []
for i in range(70, len(df)):
    data = df.iloc[0:i]
    prediction = model.fit_predict(data, n_periods=1)
    predictions.append(prediction)



fig, ax = plt.subplots(1, 1, figsize=(15, 5))
ax.plot(df['GC'])
ax.plot(df.index[70:], predictions)


#%%
prediction, confint = model.predict(n_periods=len(test), return_conf_int=True)
cf = pd.DataFrame(confint)
prediction_series = pd.Series(data=np.array(prediction),index=test.index)
fig, ax = plt.subplots(1, 1, figsize=(15, 5))
ax.plot(df['GC'])
ax.plot(prediction_series)
ax.fill_between(prediction_series.index,
                cf[0],
                cf[1],color='grey',alpha=.3)
