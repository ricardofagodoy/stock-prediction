import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Properties
data_csv = 'data/EURUSD_H1_200509010000_201910140000.csv'

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

