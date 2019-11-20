import numpy as np
import pandas as pd

def obtain_features_split(data_csv):

  # Obtain features
  df = obtain_features(data_csv, True)

  # The goal
  label = 'is_going_up'

  # Create label values to train
  df[label] = (df['close'].shift(1).sub(df['open'].shift(1)) > 0).astype(int)

  # Drop NaN columns
  df.dropna(inplace=True)

  # Split train and test data
  sample_80 = df.sample(frac=0.8, random_state=200)
  sample_20 = df.drop(sample_80.index)

  # Training data
  train_data, train_label = sample_80.drop(label, axis=1), sample_80[label]

  # Test data
  test_data, test_label = sample_20.drop(label, axis=1), sample_20[label]

  return train_data, train_label, test_data, test_label

def obtain_features(data_csv, write_features_to_disk=False):

  # Load data from CSV
  ds = pd.read_csv(data_csv, sep='\\t', engine='python')

  # Final dataframe
  df = pd.DataFrame()

  # Basic features
  df['open'] = format_tick(ds['Open'])
  df['high'] = format_tick(ds['High'])
  df['low'] = format_tick(ds['Low'])
  df['close'] = format_tick(ds['Close'])
  df['tickvol'] = ds['Tickvol']
  df['growth'] = df['close'].sub(df['open'])
  df['is_up'] = (df['growth'] > 0).astype(int)

  # Count of candles since last reversions in price
  candles_since_reversions = calculate_candles_since_n_reversions(df['is_up'].values, 4)
  df['candles_since_reversion_1'] = candles_since_reversions[0]
  df['candles_since_reversion_2'] = candles_since_reversions[1]
  df['candles_since_reversion_3'] = candles_since_reversions[2]
  df['candles_since_reversion_4'] = candles_since_reversions[3]

  # Drop NaN columns
  df.dropna(inplace=True)

  # One-hot encoding for needed fields
  df['hour_of_day'] = pd.to_datetime(ds['Time']).dt.hour
  df = pd.concat([df, one_hot_encode(df['hour_of_day'])], axis=1)
  df.drop('hour_of_day', axis=1, inplace=True)

  # Normalize or Standardize
  df = normalization(df)

  # Export features to file
  if write_features_to_disk:
    df.to_csv('features/features.csv', index=None, header=True)
    print('\nFeatures samples available at features folder\n')

  return df

''' ************ Helpers ************ '''
def format_tick(tick):
  return (tick * 10e4).round(0).astype(int)

def calculate_candles_since_n_reversions(is_up_arr, n):
  
  lines = len(is_up_arr)
  results = [np.full(lines, None).tolist()] * n

  for i in range(lines):
    reversions = _calculate_candles_since_n_reversions(is_up_arr, n, i)
    
    for r, v in enumerate(reversions):
      results[r][i] = v

  return results

def _calculate_candles_since_n_reversions(is_up_arr, n, i):

  count = 0
  candles_since_reversions = np.full(n, None).tolist()
  curr = i-1

  while curr >= 0 and count < n:
        
    if is_up_arr[i] != is_up_arr[curr]:
      candles_since_reversions[count] = i-curr
      count+=1

    curr -= 1

  return candles_since_reversions

def normalization(dataset):

  dtypes = list(zip(dataset.dtypes.index, map(str, dataset.dtypes)))

  for column, dtype in dtypes:
      if dtype in ['float64', 'int64']:
        dataset[column] = ((dataset[column] - dataset[column].min()) / (dataset[column].max() - dataset[column].min())).round(5)
        
  return dataset

def one_hot_encode(column):
  return pd.get_dummies(column, prefix=column.name)