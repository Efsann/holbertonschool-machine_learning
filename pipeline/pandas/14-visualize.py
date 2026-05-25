#!/usr/bin/env python3
"""
Visualizes the Coinbase dataset after applying specific transformations
"""

import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# 1. Weighted_Price sütununu silirik
df = df.drop(columns=['Weighted_Price'])

# 2. Timestamp sütununun adını Date olaraq dəyişirik
df = df.rename(columns={'Timestamp': 'Date'})

# 3. Timestamp dəyərlərini datetime formatına çeviririk
df['Date'] = pd.to_datetime(df['Date'], unit='s')

# 4. Date sütununu indeks təyin edirik
df = df.set_index('Date')

# 5. Boşluqları (NaN) tələb olunan qaydalara əsasən doldururuq
df['Close'] = df['Close'].ffill()
df['High'] = df['High'].fillna(df['Close'])
df['Low'] = df['Low'].fillna(df['Close'])
df['Open'] = df['Open'].fillna(df['Close'])
df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

# 6. Yalnız 2017-ci il və sonrasını seçirik
df = df.loc['2017':]

# 7. Gündəlik olaraq qruplaşdırır (resample) və aqreqasiya edirik
df = df.resample('D').agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
})

# Nəticəni ekrana yazdırırıq
print(df)

# Qrafiki çəkirik
df.plot()
plt.show()
