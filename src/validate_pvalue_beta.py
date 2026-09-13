import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint, adfuller
import warnings
warnings.filterwarnings('ignore')

# 读取数据
df = pd.read_csv("data/stock_daily_final.csv")
pairs_old = pd.read_csv("data/coint_pairs_significant.csv")
pairs_new = pd.read_csv("data/coint_pairs_updated.csv")

prices = df.pivot(index='date', columns='ticker', values='close')
prices.index = pd.to_datetime(prices.index)

periods = {
    'train': ('2023-01-01', '2025-06-30'),
    'test1': ('2025-07-01', '2025-12-31'),
    'test2': ('2026-01-01', '2026-06-30'),
    'test3': ('2026-07-01', '2026-09-01'),
}

# 极端配对
extreme_pairs = [('AAPL','HPE'), ('ADBE','TRMB'), ('DDOG','PTC'), ('AMP','ARES'), ('PNC','TFC')]
def is_extreme(t1, t2):
    return (t1, t2) in extreme_pairs or (t2, t1) in extreme_pairs

results = []

# 对每个配对，在每个时期：
# 1. 用训练期β算价差 → ADF p值（旧βp值）
# 2. 用该时期自己的β算价差 → ADF p值（新βp值）

def compute_pvalues(p1, p2, beta_fixed):
    """计算用固定β和用该时期β的ADF p值"""
    # 旧β（固定）
    spread_fixed = p1 - beta_fixed * p2
    try:
        adf_p_fixed = adfuller(spread_fixed.dropna())[1]
    except:
        adf_p_fixed = np.nan
    
    # 新β（该时期自己的）
    try:
        beta_period = np.polyfit(p2, p1, 1)[0]
        spread_period = p1 - beta_period * p2
        adf_p_period = adfuller(spread_period.dropna())[1]
    except:
        beta_period = np.nan
        adf_p_period = np.nan
    
    return adf_p_fixed, adf_p_period, beta_period

# 处理旧配对池（train, test1）
for _, row in pairs_old.iterrows():
    t1, t2 = row['ticker_1'], row['ticker_2']
    beta_train = row['hedge_ratio']
    
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
        
        adf_p_fixed, adf_p_period, beta_period = compute_pvalues(p1, p2, beta_train)
        
        results.append({
            'ticker_1': t1, 'ticker_2': t2,
            'sub_industry': row['sub_industry'],
            'period': period,
            'beta_train': beta_train,
            'beta_period': beta_period,
            'adf_p_fixed': adf_p_fixed,
            'adf_p_period': adf_p_period,
            'is_extreme': is_extreme(t1, t2)
        })

# 处理新配对池（test2, test3）
for _, row in pairs_new.iterrows():
    t1, t2 = row['ticker_1'], row['ticker_2']
    beta_train = row['hedge_ratio']  # 这个是新训练期的beta
    
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
        
        adf_p_fixed, adf_p_period, beta_period = compute_pvalues(p1, p2, beta_train)
        
        results.append({
            'ticker_1': t1, 'ticker_2': t2,
            'sub_industry': row['sub_industry'],
            'period': period,
            'beta_train': beta_train,
            'beta_period': beta_period,
            'adf_p_fixed': adf_p_fixed,
            'adf_p_period': adf_p_period,
            'is_extreme': is_extreme(t1, t2)
        })

results_df = pd.DataFrame(results)
results_df.to_csv("data/pvalue_comparison.csv", index=False)

# ============ 汇总 ============
print("=" * 70)
print("ADF p-value: Fixed Beta vs Period Beta")
print("=" * 70)

for period in ['train', 'test1', 'test2', 'test3']:
    subset = results_df[results_df['period'] == period]
    if len(subset) == 0:
        continue
    
    print(f"\n=== {period} (n={len(subset)}) ===")
    print(f"ADF p (fixed beta): median = {subset['adf_p_fixed'].median():.4f}")
    print(f"ADF p (period beta): median = {subset['adf_p_period'].median():.4f}")
    print(f"Pairs with adf_p_period < 0.05: {(subset['adf_p_period'] < 0.05).sum()}/{len(subset)}")
    print(f"Pairs with adf_p_fixed < 0.05: {(subset['adf_p_fixed'] < 0.05).sum()}/{len(subset)}")

# ============ 极端 vs 普通 ============
print("\n" + "=" * 70)
print("By Group: Extreme vs Normal")
print("=" * 70)

for period in ['test1', 'test2', 'test3']:
    subset = results_df[results_df['period'] == period]
    print(f"\n=== {period} ===")
    
    for group_name, group_data in [('Extreme', subset[subset['is_extreme']]), 
                                    ('Normal', subset[~subset['is_extreme']])]:
        if len(group_data) == 0:
            continue
        print(f"{group_name} (n={len(group_data)}):")
        print(f"  ADF p (fixed beta): median = {group_data['adf_p_fixed'].median():.4f}")
        print(f"  ADF p (period beta): median = {group_data['adf_p_period'].median():.4f}")
        print(f"  Pairs with adf_p_period < 0.05: {(group_data['adf_p_period'] < 0.05).sum()}/{len(group_data)}")