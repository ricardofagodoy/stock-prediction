import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import tensorflow as tf
from keras.callbacks.callbacks import EarlyStopping
from features_engineering import create_basic_features
from features_engineering import calculate_candles_since_n_reversions
from features_engineering import normalization
from features_engineering import one_hot_encode

# Properties
data_csv = 'data/EURUSD_H1_200509010000_201910140000.csv'
label = 'is_going_up'
epochs=500
batch_size=10

# Load data from CSV
ds = pd.read_csv(data_csv, sep='\\t')

# Feature engineering (creating new features over existing)
df = create_basic_features(ds)

# Count of candles since last reversions in price
candles_since_reversions = calculate_candles_since_n_reversions(df['is_up'].values, 4)
df['candles_since_reversion_1'] = candles_since_reversions[0]
df['candles_since_reversion_2'] = candles_since_reversions[1]
df['candles_since_reversion_3'] = candles_since_reversions[2]
df['candles_since_reversion_4'] = candles_since_reversions[3]

# Label
df[label] = (ds['Close'].shift(1).sub(ds['Open'].shift(1)) > 0).astype(int)

df.dropna(inplace=True)
print('Columns:\n %s' % df.columns)
print('Dtypes:\n %s' % df.dtypes)

# One-hot encoding for needed fields
df = pd.concat([df, one_hot_encode(df['hour_of_day'])], axis=1)
df.drop('hour_of_day', axis=1, inplace=True)

# Normalize or Standardize
df = normalization(df)

# Export features data and labels to file
df.to_csv('features/training.csv', index=None, header=True)

#exit(0)

# Split train and test data
sample_80 = df.sample(frac=0.8, random_state=200)
sample_20 = df.drop(sample_80.index)

# Training data
train_data, train_label = sample_80.drop(label, axis=1), sample_80[label]

# Test data
test_data, test_label = sample_20.drop(label, axis=1), sample_20[label]

print('train data: \n%s\n%s' % (train_data.columns, train_data.head()))
print('train label: \n%s' % train_label.name)

# Network
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(64, activation='relu', input_shape=(len(train_data.columns),)),
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid'),
])

# Save model to json
model_json = model.to_json()
with open("model/model.json", "w") as json_file:
    json_file.write(model_json)

# Model parameters
model.compile(
  loss='binary_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

# Callbacks to fit
log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
early_stop_callback = EarlyStopping(monitor='loss', mode='min', patience=25, verbose=1)

callbacks = [tensorboard_callback, early_stop_callback]

# Train with data
model.fit(
  train_data,
  train_label,
  epochs=epochs,
  batch_size=batch_size,
  validation_data=(test_data, test_label), 
  callbacks=callbacks
)

# Confusion Matrix
predictions = model.predict(test_data)

confusion = tf.math.confusion_matrix(
  labels=test_label,
  predictions=predictions.round(0), 
  num_classes=2
)

print(confusion)

# Confusion Matrix in percentual
confusion_percentage = tf.Variable(tf.zeros((2,2), tf.float64)) 
confusion_percentage[0,0].assign(confusion[0,0] / (confusion[0,0] + confusion[0,1]))
confusion_percentage[0,1].assign(confusion[0,1] / (confusion[0,0] + confusion[0,1]))
confusion_percentage[1,0].assign(confusion[1,0] / (confusion[1,0] + confusion[1,1]))
confusion_percentage[1,1].assign(confusion[1,1] / (confusion[1,0] + confusion[1,1]))

print(confusion_percentage)

# Accuracy
_, accuracy = model.evaluate(test_data, test_label, batch_size=batch_size)
print('Accuracy: %.2f%%' % (accuracy*100))

# Export trained weights
model.save_weights('model/model.h5')
print('\nTrained weights exported to model/model.h5')