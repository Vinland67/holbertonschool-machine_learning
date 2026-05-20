#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# 1. Sütunu silirik və adını dəyişirik
df = df.drop(columns=['Weighted_Price'])
df = df.rename(columns={'Timestamp': 'Date'})

# 2. Tarix formatına çeviririk (Yalnız gün səviyyəsində)
df['Date'] = pd.to_datetime(df['Date'], unit='s')

# 3. Boşluqları şərtə uyğun doldururuq
df['Close'] = df['Close'].ffill()
df['High'] = df['High'].fillna(df['Close'])
df['Low'] = df['Low'].fillna(df['Close'])
df['Open'] = df['Open'].fillna(df['Close'])
df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

# 4. İndeks təyin edib 2017-dən sonrasını götürürük
df = df.set_index('Date')
df = df.loc['2017-01-01':]

# 5. Günlük qruplaşdırma (Resample)
df = df.resample('D').agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
})

# 6. Qrafiki çəkirik
df.plot()
plt.show()
