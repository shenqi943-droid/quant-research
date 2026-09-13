import pandas as pd

df = pd.read_csv("data/stock_daily_ohlc.csv")

print(f"Total rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"Unique tickers: {df['ticker'].nunique()}")
print(f"\nSample:")
print(df.head())
print(f"\nMissing values:")
print(df.isnull().sum())