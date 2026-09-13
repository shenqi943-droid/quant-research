import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint, adfuller
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# 读取数据
df = pd.read_csv("data/stock_daily_final.csv")
pairs_old = pd.read_csv("data/coint_pairs_significant.csv")   # 44对
pairs_new = pd.read_csv("data/coint_pairs_updated.csv")       # 28对

prices = df.pivot(index='date', columns='ticker', values='close')
prices.index = pd.to_datetime(prices.index)

periods = {
    'train': ('2023-01-01', '2025-06-30'),
    'test1': ('2025-07-01', '2025-12-31'),
    'test2': ('2026-01-01', '2026-06-30'),
    'test3': ('2026-07-01', '2026-09-01'),
}

def compute_lifecycle_metrics(p1, p2, beta):
    """计算一个配对在一个时期内的生命周期指标"""
    # 1. 协整p值
    if len(p1) < 20:
        return None
    try:
        _, coint_p, _ = coint(p1, p2)
    except:
        coint_p = np.nan
    
    # 2. 价差与半衰期
    spread = p1 - beta * p2
    
    spread_lag = spread.shift(1).dropna()
    spread_diff = spread.diff().dropna()
    common = spread_lag.index.intersection(spread_diff.index)
    if len(common) > 10:
        y = spread_diff[common]
        x = spread_lag[common]
        beta_ou = np.polyfit(x, y, 1)[0]
        half_life = -np.log(2) / beta_ou if beta_ou < 0 else np.inf
        # λ值（调整系数）
        lambda_val = beta_ou
    else:
        half_life = np.inf
        lambda_val = np.nan
    
    # 3. 滚动相关性
    ret1 = p1.pct_change()
    ret2 = p2.pct_change()
    rolling_corr = ret1.rolling(min(60, len(p1)//2)).corr(ret2).mean()
    
    # 4. Hurst指数（简化版：用R/S法）
    def hurst(ts):
        ts = np.array(ts)
        if len(ts) < 20:
            return np.nan
        lags = range(2, min(20, len(ts)//2))
        tau = [np.sqrt(np.std(ts[lag:] - ts[:-lag])) for lag in lags]
        try:
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        except:
            return np.nan
    
    hurst_val = hurst(spread.values)
    
    return {
        'coint_p': coint_p,
        'half_life': half_life,
        'lambda': lambda_val,
        'rolling_corr': rolling_corr,
        'hurst': hurst_val
    }

# ============ 对所有配对计算 ============
results = []

# 用旧配对池（44对）计算train和test1
for _, row in pairs_old.iterrows():
    t1, t2 = row['ticker_1'], row['ticker_2']
    beta = row['hedge_ratio']
    
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
        
        metrics = compute_lifecycle_metrics(p1, p2, beta)
        if metrics:
            results.append({
                'ticker_1': t1, 'ticker_2': t2,
                'sub_industry': row['sub_industry'],
                'period': period, 'pair_source': 'old',
                **metrics
            })

# 用新配对池（28对）计算test2和test3
for _, row in pairs_new.iterrows():
    t1, t2 = row['ticker_1'], row['ticker_2']
    beta = row['hedge_ratio']
    
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
        
        metrics = compute_lifecycle_metrics(p1, p2, beta)
        if metrics:
            results.append({
                'ticker_1': t1, 'ticker_2': t2,
                'sub_industry': row['sub_industry'],
                'period': period, 'pair_source': 'new',
                **metrics
            })

results_df = pd.DataFrame(results)
results_df.to_csv("data/sub2_lifecycle_metrics.csv", index=False)

# ============ 汇总 ============
print("=" * 60)
print("Lifecycle Metrics by Period (All Pairs)")
print("=" * 60)

summary = results_df.groupby('period').agg({
    'coint_p': 'median',
    'half_life': 'median',
    'lambda': 'median',
    'rolling_corr': 'median',
    'hurst': 'median',
    'ticker_1': 'count'
}).round(4)
summary.columns = ['coint_p_median', 'half_life_median', 'lambda_median', 
                   'rolling_corr_median', 'hurst_median', 'n_pairs']
print(summary)

# ============ 标记极端配对 ============
# 从之前的结果中读取test2残差
sub1_updated = pd.read_csv("data/sub1_updated_metrics.csv")
test2_data = sub1_updated[sub1_updated['period'] == 'test2'].copy()

# 计算残差
X = sm.add_constant(test2_data[['vol_mean']])
model = sm.OLS(test2_data['log_spread_std'], X).fit()
test2_data['resid'] = model.resid

# 极端配对 = 残差 > 0
extreme_pairs = test2_data[test2_data['resid'] > 0][['ticker_1', 'ticker_2']].values.tolist()
print(f"\nExtreme pairs (resid > 0 in test2): {len(extreme_pairs)}")
for p in extreme_pairs:
    print(f"  {p[0]}-{p[1]}")

# 标记
results_df['is_extreme'] = results_df.apply(
    lambda r: (r['ticker_1'], r['ticker_2']) in [tuple(p) for p in extreme_pairs] or 
              (r['ticker_2'], r['ticker_1']) in [tuple(p) for p in extreme_pairs],
    axis=1
)

# ============ 极端 vs 普通 对比 ============
print("\n" + "=" * 60)
print("Extreme vs Normal Pairs")
print("=" * 60)

for period in ['test1', 'test2', 'test3']:
    subset = results_df[results_df['period'] == period]
    print(f"\n=== {period} ===")
    
    if len(subset[subset['is_extreme']]) > 0:
        print("Extreme pairs:")
        print(subset[subset['is_extreme']][['coint_p', 'half_life', 'rolling_corr', 'hurst']].median().round(4))
    
    print("Normal pairs:")
    print(subset[~subset['is_extreme']][['coint_p', 'half_life', 'rolling_corr', 'hurst']].median().round(4))

results_df.to_csv("data/sub2_lifecycle_metrics_final.csv", index=False)
print("\nSaved to data/sub2_lifecycle_metrics_final.csv")