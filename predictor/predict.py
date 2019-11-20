from tensorflow import keras
from feature_engineering import obtain_features

# Tick to predict if it's UP or DOWN
data_csv = 'data/tick'

# Features
features = obtain_features(data_csv)

# Load model json
json_file = open('model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

# Create model and load weights
model = keras.models.model_from_json(loaded_model_json)
model.load_weights("model/model.h5")
print("\nLoaded model from disk")

# Predict (last row of dataframe)
headers = features.tail(1).columns
feature = features.tail(1).values
print('\nPredicting to features:\n%s\n%s\n' % (headers, feature))

prediction = model.predict(feature)[0, 0]
print('\nPrediction: %s\n' % prediction)

# Save to file
with open("model/prediction", "w") as prediction_file:
    prediction_file.write(str(prediction))

print('Prediction saved to model/prediction')