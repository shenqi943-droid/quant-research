import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("data/stock_daily_final.csv")
pairs = pd.read_csv("data/coint_pairs_significant.csv")

prices = df.pivot(index='date', columns='ticker', values='close')
prices.index = pd.to_datetime(prices.index)

periods = {
    'train': ('2023-01-01', '2025-03-31'),
    'test1': ('2025-05-01', '2025-12-31'),
    'test2': ('2026-01-01', '2026-06-30'),
    'test3': ('2026-07-01', '2026-09-01'),
}

results = []

for _, row in pairs.iterrows():
    t1, t2 = row['ticker_1'], row['ticker_2']
    beta = row['hedge_ratio']

    if t1 not in prices.columns or t2 not in prices.columns:
        continue

    for period, (start, end) in periods.items():
        p1 = prices[t1][start:end].dropna()
        p2 = prices[t2][start:end].dropna()
        common = p1.index.intersection(p2.index)

        if len(common) < 20:
            continue

        p1, p2 = p1[common], p2[common]

        vol_1 = p1.pct_change().std()
        vol_2 = p2.pct_change().std()
        vol_mean = (vol_1 + vol_2) / 2

        log_spread = np.log(p1) - beta * np.log(p2)
        log_spread_std = log_spread.std()
        log_spread_mean = log_spread.mean()

        results.append({
            'ticker_1': t1, 'ticker_2': t2,
            'sub_industry': row['sub_industry'],
            'period': period,
            'vol_mean': vol_mean,
            'log_spread_std': log_spread_std,
            'log_spread_mean': log_spread_mean,
            'n_days': len(common),
        })

results_df = pd.DataFrame(results)
results_df.to_csv("data/sub1_v3_metrics.csv", index=False)

print("=== 各时期汇总 ===")
summary = results_df.groupby('period').agg({
    'vol_mean': ['median', 'mean'],
    'log_spread_std': ['median', 'mean'],
}).round(4)
print(summary)

print("\n=== 逐对指标（前10对）===")
sample = results_df.pivot_table(
    index=['ticker_1', 'ticker_2'],
    columns='period',
    values=['vol_mean', 'log_spread_std'],
)
print(sample.head(10))

print("\n=== log_spread_std按子行业和时期（中位数）===")
pivot = results_df.pivot_table(
    values='log_spread_std', index='sub_industry', columns='period', aggfunc='median'
).round(4)
print(pivot)