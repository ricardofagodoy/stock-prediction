import datetime
from keras.callbacks.callbacks import EarlyStopping
from feature_engineering import obtain_features_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

# Properties
data_csv='data/EURUSD_M5_201805292305_201910011910.csv'

def create_model():

  import tensorflow as tf

  epochs=1
  batch_size=1200
  num_layers=4
  initial_neurons=32
  neurons_multiplier=2.5
  dropout_rate=0.08
  activation='relu'
  loss_function='binary_crossentropy'
  optimizer='adam' # rmsprop
  metrics=['accuracy']

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
  return tf.keras.models.Sequential(layers)

# Obtain features
train_data, train_label, test_data, test_label = obtain_features_split(data_csv)

print('Train columns: \n%s\n%s' % (train_data.columns, train_data.dtypes))
print('\nTrain label: \n%s\n' % train_label.name)

# Hyperparameter optimization
activation = ['softmax', 'relu', 'tanh', 'sigmoid', 'linear']
neurons = [5, 10, 15, 25, 35, 50]
optimizer = ['SGD', 'Adam', 'Adamax']
param_grid = dict(activation=activation, neurons = neurons, optimizer = optimizer)

clf = KerasClassifier(build_fn= create_model, epochs= 20, batch_size=40, verbose= 1)

print(clf.get_params().keys())
exit(0)

model = GridSearchCV(estimator=clf, param_grid=param_grid, n_jobs=-1, cv=3)

# Train with data
model.fit(
  train_data.values,
  train_label.values
)

print("Max Accuracy Registred: {} using {}".format(round(model.best_score_,3), model.best_params_))

# Metrics
predictions = model.predict(test_data.values).round(0)

print('\nConfusion Matrix: \n%s\n' % confusion_matrix(predictions, test_label))
print(classification_report(predictions, test_label))
print('\nAcc: %s' % accuracy_score(predictions, test_label))