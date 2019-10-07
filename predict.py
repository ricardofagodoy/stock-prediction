import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.offline as py
import plotly.graph_objs as go
from keras.models import Sequential
from keras.layers import Dense

features_header = ['Time', 'Open', 'High', 'Low', 'Close', 'Tickvol', 'Spread', 'Growth']
label = 'Close'
data_csv = 'EURUSD_M5_201805292305_201910011910.csv'

# Load data from CSV
ds = pd.read_csv(data_csv, sep='\\t')

# Create Time feature (in seconds)
ds['Date'] = pd.to_datetime(ds['Date'] + ' ' + ds['Time'])
ds['Time'] = pd.to_datetime(ds['Time'])
ds['Time'] = ds['Time'].dt.hour*60 + ds['Time'].dt.minute

# Create new feature to instant growth
ds['Growth'] = ds['Close'].sub(ds['Open'])

# Save original data frame
ds_orig = ds.copy()

# Normalize some fields
ds_num = ds[['Time', 'Tickvol', 'Spread', 'Growth']]
ds_norm = (ds_num - ds_num.mean()) / (ds_num.max() - ds_num.min())
ds[ds_norm.columns] = ds_norm

# Split train and test data
train = ds[features_header].sample(frac=0.8, random_state=200)
test = ds[features_header].drop(train.index)

# Network
model = Sequential([
  Dense(64, activation='relu', input_shape=(len(features_header) - 1,)),
  Dense(64, activation='relu'),
  Dense(1),
])

model.compile(
  optimizer='sgd',
  loss='mean_squared_error',
  metrics=['mean_squared_error']
)

print('Train Features: \n%s' % train.drop(label, axis=1))
print('Train Labels: \n%s' % train[label])

model.fit(
  train.drop(label, axis=1),
  train[label],
  epochs=2,
  batch_size=32
)

# Result of training
print(model.evaluate(test.drop(label, axis=1), test[label]))

# Export features and weights
ds.to_csv('features.csv', index=False)
model.save_weights('model.h5')

# Use it!
print('Predicted: \n%s' % model.predict(test.drop(label, axis=1)[:10]))
print('Labels: \n%s' % test[label][:10])

# Check results on candle graph
'''
data = go.Candlestick(x=ds_orig['Date'][:5],
                      open=ds_orig['Open'][:5],
                      high=ds_orig['High'][:5],
                      low=ds_orig['Low'][:5],
                      close=ds_orig['Close'][:5]
                    )
go.Figure(data=[data], layout={'title': {'text': 'Label'}}).show()
'''