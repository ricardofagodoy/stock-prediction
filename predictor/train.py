import datetime
import tensorflow as tf
from keras.callbacks.callbacks import EarlyStopping
from feature_engineering import obtain_features_split

# Properties
data_csv='data/EURUSD_M5_201805292305_201910011910.csv'
epochs=10
batch_size=12
num_layers=4
initial_neurons=32
neurons_multiplier=2.5
dropout_rate=0.08
activation='relu'
loss_function='binary_crossentropy'
optimizer='adam' # rmsprop
metrics=['accuracy']

# Obtain features
train_data, train_label, test_data, test_label = obtain_features_split(data_csv)

print('Train columns: \n%s\n%s' % (train_data.columns, train_data.dtypes))
print('\nTrain label: \n%s\n' % train_label.name)

# Build network layers
layers = []
neurons=initial_neurons

# Input
layers.append(tf.keras.layers.Dense(neurons, activation=activation, input_dim=len(train_data.columns)))

# Layers
for n in range(num_layers):
  neurons *= neurons_multiplier
  layers.append(tf.keras.layers.Dense(neurons, activation=activation))
  layers.append(tf.keras.layers.Dropout(dropout_rate))

# Output
layers.append(tf.keras.layers.Dense(1, activation='sigmoid'))

# Model
model = tf.keras.models.Sequential(layers)

# Save model to json
with open("model/model.json", "w") as json_file:
    json_file.write(model.to_json())

print('\nModel architecture exported to model/model.json\n')

# Model parameters
model.compile(
  loss=loss_function,
  optimizer=optimizer,
  metrics=metrics
)

# Callbacks
log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
early_stop_callback = EarlyStopping(monitor='loss', mode='min', patience=25, verbose=1)

callbacks = [tensorboard_callback, early_stop_callback]

# Train with data
model.fit(
  train_data.values,
  train_label.values,
  epochs=epochs,
  batch_size=batch_size,
  callbacks=callbacks
)

# Confusion Matrix
predictions = model.predict(test_data.values)

confusion = tf.math.confusion_matrix(
  labels=test_label,
  predictions=predictions.round(0), 
  num_classes=2
)

# Confusion Matrix in percentual
confusion_percentage = tf.Variable(tf.zeros((2,2), tf.float64)) 
confusion_percentage[0,0].assign(confusion[0,0] / (confusion[0,0] + confusion[0,1]))
confusion_percentage[0,1].assign(confusion[0,1] / (confusion[0,0] + confusion[0,1]))
confusion_percentage[1,0].assign(confusion[1,0] / (confusion[1,0] + confusion[1,1]))
confusion_percentage[1,1].assign(confusion[1,1] / (confusion[1,0] + confusion[1,1]))

print('\nConfusion matrixes:\n\n%s\n\n%s' % (confusion.numpy(), confusion_percentage.numpy()))

# Accuracy
_, accuracy = model.evaluate(test_data.values, test_label.values, batch_size=batch_size, verbose=0)
print('\nAccuracy: %.2f%%' % (accuracy*100))

# Export trained weights
model.save_weights('model/model.h5')
print('\nTrained weights exported to model/model.h5')