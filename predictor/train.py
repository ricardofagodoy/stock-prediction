import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

# Properties
data_csv = 'data/EURUSD_H1_200509010000_201910140000.csv'
epochs=15
batch_size=10

# Load data from CSV
ds = pd.read_csv(data_csv, sep='\\t')

# Feature engineering (creating new features over existing)
df = pd.DataFrame()

df['open'] = ds['Open']
df['high'] = ds['High']
df['low'] = ds['Low']
df['close'] = ds['Open']
df['tickvol'] = ds['Tickvol']
df['time_of_day'] = pd.to_datetime(ds['Time']).dt.hour
df['growth'] = ds['Close'].sub(ds['Open'])
df['is_up'] = df['growth'] > 0

is_up = df['is_up'].values
candles_since_reversion = []

for i in range(len(is_up)):

  candles_since_reversions = np.full(4, None).tolist()
  curr = i-1

  if curr < 0:
    candles_since_reversion_1.append(None)
    continue

  while curr >= 0:
      
    if is_up[i] != is_up[curr]:
      candles_since_reversion_1.append(i-curr)
      break

    curr -= 1

def candles_since_reversion(row):

  curr = row.name-1

  while curr >= 0:
      
    if df.loc[row.name, 'is_up'] != df.loc[curr, 'is_up']:
      return row.name-curr

    curr -= 1

#df['candles_since_reversion_1'] = df.apply(candles_since_reversion, axis=1)
df['candles_since_reversion_1'] = candles_since_reversion_1

'''
df['candles_since_reversion_2']
df['candles_since_reversion_3']
df['candles_since_reversion_4']
df['reversion_1_strength']
df['reversion_2_strength']
df['reversion_3_strength']
df['reversion_4_strength']
df['is_up_candle']
df['is_down_candle']
'''

# Label
df['is_going_up'] = ds['Close'].shift(1).sub(ds['Open'].shift(1)) > 0

df.dropna(inplace=True)

print(df)
exit(0)

# Normalize
ds_num = ds[['Time', 'Tickvol', 'Growth']]
ds_norm = (ds_num - ds_num.mean()) / (ds_num.max() - ds_num.min())
ds[ds_norm.columns] = ds_norm

# z-index

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