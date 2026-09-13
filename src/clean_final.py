import pandas as pd

df = pd.read_csv("data/stock_daily_clean.csv")

exclude = ['MRSH', 'Q', 'FISV', 'SNDK', 'XYZ', 'EXE', 'CPAY', 'EG', 'BNY']
df_clean = df[~df['ticker'].isin(exclude)]

print(f"Remaining tickers: {df_clean['ticker'].nunique()}")
print(f"Remaining rows: {len(df_clean)}")

trading_days = df_clean.groupby('ticker')['date'].count()
print(f"\nTrading days - min: {trading_days.min()} ({trading_days.idxmin()})")
print(f"Trading days - max: {trading_days.max()} ({trading_days.idxmax()})")
print(f"Trading days - mean: {trading_days.mean():.0f}")

df_clean.to_csv("data/stock_daily_final.csv", index=False)

final_tickers = df_clean[['ticker']].drop_duplicates()
universe = pd.read_csv("data/sp500_pairs_universe.csv")
universe_final = universe[universe['Symbol'].isin(final_tickers['ticker'])]
universe_final.to_csv("data/pairs_universe_final.csv", index=False)
print(f"\nSaved {len(universe_final)} tickers to data/pairs_universe_final.csv")