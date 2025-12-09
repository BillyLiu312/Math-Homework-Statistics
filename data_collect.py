import yfinance as yf
import numpy as np
import os
import matplotlib.pyplot as plt

df = yf.download('^GSPC', start='2000-01-01', end='2023-12-31')
df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
df = df.dropna()

# 确保 data 目录存在并保存为 CSV
os.makedirs('data', exist_ok=True)
df[['Close', 'Log_Ret']].to_csv(os.path.join('data', 'SP500_log_returns.csv'), index=True)