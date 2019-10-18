import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import plotly.offline as py
import plotly.graph_objs as go
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json

features_header = ['Time', 'Open', 'High', 'Low', 'Tickvol', 'Growth']
label = 'Close'
data_csv = 'data/EURUSD_H1_200509010000_201910140000_validation.csv'

# Load model json
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

# Create model and load weights
model = model_from_json(loaded_model_json)
model.load_weights("model.h5")
print("Loaded model from disk")

# Load data from CSV
ds = pd.read_csv(data_csv, sep='\\t')

# Create Time feature (in seconds)
ds['Date'] = pd.to_datetime(ds['Date'] + ' ' + ds['Time'])
ds['Time'] = pd.to_datetime(ds['Time'])
ds['Time'] = ds['Time'].dt.hour*60 + ds['Time'].dt.minute

# Create new feature to instant growth
ds['Growth'] = ds['Close'].sub(ds['Open'])

# Normalize some fields
ds_num = ds[['Time', 'Tickvol', 'Growth']]
ds_norm = (ds_num - ds_num.mean()) / (ds_num.max() - ds_num.min())
ds[ds_norm.columns] = ds_norm

# Data to predict
input_data, input_label = ds[features_header], ds[label]

print('Input Data: \n%s' % input_data)
print('Input Labels: \n%s' % input_label)

model.compile(
  optimizer='adam',
  loss='mean_squared_error'
)

# Predict!
qtde = 20
predictions = model.predict(input_data[:qtde])
#print('%s\n%s' % (input_label[:qtde], predictions))

# Correct
go.Figure(data=[
  go.Candlestick(x=ds['Date'][:qtde],
                      open=ds['Open'][:qtde],
                      high=ds['High'][:qtde],
                      low=ds['Low'][:qtde],
                      close=input_label[:qtde]
  )], layout={'title': {'text': 'Correct  '}}).show()

# Predicted
go.Figure(data=[
  go.Candlestick(x=ds['Date'][:qtde],
                      open=ds['Open'][:qtde],
                      high=ds['High'][:qtde],
                      low=ds['Low'][:qtde],
                      close=list(map(lambda p : p[0], predictions[:qtde]))
  )], layout={'title': {'text': 'Predicted'}}).show()