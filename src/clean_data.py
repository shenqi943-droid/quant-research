import pandas as pd

df = pd.read_csv("data/stock_daily_ohlc.csv")

# 1. 转换时间戳（Unix毫秒 → 日期）
df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
df['date'] = df['date'].dt.date

# 2. 删除无用列
df = df.drop(columns=['otc', 'timestamp'])

# 3. 按ticker和日期排序
df = df.sort_values(['ticker', 'date']).reset_index(drop=True)

# 4. 检查交易日一致性
print("=== 数据概况 ===")
print(f"总行数: {len(df)}")
print(f"股票数: {df['ticker'].nunique()}")
print(f"日期范围: {df['date'].min()} to {df['date'].max()}")
print(f"总交易日数: {df['date'].nunique()}")

# 5. 检查每只股票的交易日数量
trading_days = df.groupby('ticker')['date'].count()
print(f"\n=== 交易日数量分布 ===")
print(f"平均: {trading_days.mean():.0f}")
print(f"最少: {trading_days.min()} ({trading_days.idxmin()})")
print(f"最多: {trading_days.max()} ({trading_days.idxmax()})")

# 6. 检查是否有重复
duplicates = df.duplicated(subset=['ticker', 'date']).sum()
print(f"\n重复行: {duplicates}")

# 7. 检查日期缺口（以AAPL为基准）
aapl_dates = set(df[df['ticker'] == 'AAPL']['date'])
missing_dates = {}
for ticker in df['ticker'].unique():
    ticker_dates = set(df[df['ticker'] == ticker]['date'])
    missing = aapl_dates - ticker_dates
    if missing:
        missing_dates[ticker] = len(missing)

if missing_dates:
    print(f"\n=== 有日期缺口的股票 ===")
    for t, n in sorted(missing_dates.items(), key=lambda x: -x[1])[:10]:
        print(f"{t}: {n} days missing")
else:
    print("\n所有股票交易日完全一致")

# 8. 保存清洗后的数据
df.to_csv("data/stock_daily_clean.csv", index=False)
print(f"\nSaved to data/stock_daily_clean.csv")