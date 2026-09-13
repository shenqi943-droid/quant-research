import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint
import warnings
warnings.filterwarnings('ignore')

# 读取数据
df = pd.read_csv("data/stock_daily_final.csv")
pairs_old = pd.read_csv("data/coint_pairs_significant.csv")
pairs_new = pd.read_csv("data/coint_pairs_updated.csv")

prices = df.pivot(index='date', columns='ticker', values='close')
prices.index = pd.to_datetime(prices.index)

# 极端配对列表（test2残差>0）
extreme_pairs = [('AAPL','HPE'), ('ADBE','TRMB'), ('DDOG','PTC'), ('AMP','ARES'), ('PNC','TFC')]

def is_extreme(t1, t2):
    return (t1, t2) in extreme_pairs or (t2, t1) in extreme_pairs

periods = {
    'train': ('2023-01-01', '2025-06-30'),
    'test1': ('2025-07-01', '2025-12-31'),
    'test2': ('2026-01-01', '2026-06-30'),
    'test3': ('2026-07-01', '2026-09-01'),
}

results = []

# train和test1用旧配对池
for _, row in pairs_old.iterrows():
    t1, t2 = row['ticker_1'], row['ticker_2']
    if t1 not in prices.columns or t2 not in prices.columns:
        continue
    for period in ['train', 'test1']:
        start, end = periods[period]
        p1 = prices[t1][start:end].dropna()
        p2 = prices[t2][start:end].dropna()
        common = p1.index.intersection(p2.index)
        if len(common) < 20:
            continue
        p1, p2 = p1[common], p2[common]
        
        # 协整p值
        _, pval, _ = coint(p1, p2)
        # 该时期的beta
        beta = np.polyfit(p2, p1, 1)[0]
        
        results.append({
            'ticker_1': t1, 'ticker_2': t2,
            'sub_industry': row['sub_industry'],
            'period': period,
            'coint_p': pval,
            'beta': beta,
            'is_extreme': is_extreme(t1, t2)
        })

# test2和test3用新配对池
for _, row in pairs_new.iterrows():
    t1, t2 = row['ticker_1'], row['ticker_2']
    if t1 not in prices.columns or t2 not in prices.columns:
        continue
    for period in ['test2', 'test3']:
        start, end = periods[period]
        p1 = prices[t1][start:end].dropna()
        p2 = prices[t2][start:end].dropna()
        common = p1.index.intersection(p2.index)
        if len(common) < 20:
            continue
        p1, p2 = p1[common], p2[common]
        
        _, pval, _ = coint(p1, p2)
        beta = np.polyfit(p2, p1, 1)[0]
        
        results.append({
            'ticker_1': t1, 'ticker_2': t2,
            'sub_industry': row['sub_industry'],
            'period': period,
            'coint_p': pval,
            'beta': beta,
            'is_extreme': is_extreme(t1, t2)
        })

results_df = pd.DataFrame(results)

# ============ p值汇总 ============
print("=" * 60)
print("Cointegration p-value by Period and Group")
print("=" * 60)

for period in ['train', 'test1', 'test2', 'test3']:
    subset = results_df[results_df['period'] == period]
    if len(subset) == 0:
        continue
    print(f"\n=== {period} ===")
    print(f"All pairs (n={len(subset)}): median p = {subset['coint_p'].median():.4f}")
    
    extreme = subset[subset['is_extreme']]
    normal = subset[~subset['is_extreme']]
    
    if len(extreme) > 0:
        print(f"Extreme (n={len(extreme)}): median p = {extreme['coint_p'].median():.4f}")
    if len(normal) > 0:
        print(f"Normal (n={len(normal)}): median p = {normal['coint_p'].median():.4f}")

# ============ beta汇总 ============
print("\n" + "=" * 60)
print("Beta by Period and Group")
print("=" * 60)

for period in ['train', 'test1', 'test2', 'test3']:
    subset = results_df[results_df['period'] == period]
    if len(subset) == 0:
        continue
    print(f"\n=== {period} ===")
    print(f"All pairs: median beta = {subset['beta'].median():.4f}")
    
    extreme = subset[subset['is_extreme']]
    normal = subset[~subset['is_extreme']]
    
    if len(extreme) > 0:
        print(f"Extreme: median beta = {extreme['beta'].median():.4f}")
    if len(normal) > 0:
        print(f"Normal: median beta = {normal['beta'].median():.4f}")

# ============ 极端配对的详细数据 ============
print("\n" + "=" * 60)
print("Extreme Pairs: p-value and beta by Period")
print("=" * 60)

for t1, t2 in extreme_pairs:
    print(f"\n{t1}-{t2}:")
    for period in ['train', 'test1', 'test2', 'test3']:
        row = results_df[(results_df['ticker_1'] == t1) & 
                         (results_df['ticker_2'] == t2) & 
                         (results_df['period'] == period)]
        if len(row) > 0:
            print(f"  {period}: p={row['coint_p'].values[0]:.4f}, beta={row['beta'].values[0]:.4f}")

results_df.to_csv("data/sub2_pvalue_beta.csv", index=False)
print("\nSaved to data/sub2_pvalue_beta.csv")