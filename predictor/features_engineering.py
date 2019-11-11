import numpy as np
import pandas as pd

def create_basic_features(ds):

    df = pd.DataFrame()

    df['open'] = format_tick(ds['Open'])
    df['high'] = format_tick(ds['High'])
    df['low'] = format_tick(ds['Low'])
    df['close'] = format_tick(ds['Close'])
    df['tickvol'] = ds['Tickvol']
    df['hour_of_day'] = pd.to_datetime(ds['Time']).dt.hour
    df['growth'] = df['close'].sub(df['open'])
    df['is_up'] = (df['growth'] > 0).astype(int)

    return df

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