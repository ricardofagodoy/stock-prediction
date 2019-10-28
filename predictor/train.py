import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

# Properties
features_header = ['Time', 'Open', 'High', 'Low', 'Tickvol', 'Growth']
label = 'Close'
data_csv = 'data/EURUSD_H1_200509010000_201910140000.csv'
epochs=15
batch_size=10

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

# Split train and test data
sample_80 = ds.sample(frac=0.8, random_state=200)
sample_20 = ds.drop(sample_80.index)

# Training data
train_data, train_label = sample_80[features_header], sample_80[label]

# Test data
test_data, test_label = sample_20[features_header], sample_20[label]

# Network
model = Sequential([
  Dense(128, activation='relu', input_shape=(len(features_header),)),
  Dense(32, activation='relu'),
  Dense(8, activation='relu'),
  Dense(1),
])

# Save model to json
model_json = model.to_json()
with open("model/model.json", "w") as json_file:
    json_file.write(model_json)

# Model parameters
model.compile(
  optimizer='adam',
  loss='mean_squared_error'
)

print('Train Features: \n%s' % train_data)
print('Train Labels: \n%s' % train_label)

# Train with data
model.fit(
  train_data,
  train_label,
  epochs=epochs,
  batch_size=batch_size
)

# Result of training
score = model.evaluate(test_data, test_label, verbose=0)
print('%s: %s' % (model.metrics_names[0], score))

# Export trained weights
model.save_weights('model/model.h5')
print('Trained weights exported to model.h5')