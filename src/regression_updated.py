import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

# 读取数据
df = pd.read_csv("data/stock_daily_final.csv")
pairs = pd.read_csv("data/coint_pairs_updated.csv")

prices = df.pivot(index='date', columns='ticker', values='close')
prices.index = pd.to_datetime(prices.index)

# 时期划分
periods = {
    'test1': ('2025-07-01', '2025-12-31'),
    'test2': ('2026-01-01', '2026-06-30'),
    'test3': ('2026-07-01', '2026-09-01'),
}

# ============ 计算新配对池的指标 ============
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
        
        results.append({
            'ticker_1': t1, 'ticker_2': t2,
            'sub_industry': row['sub_industry'],
            'period': period,
            'vol_mean': vol_mean,
            'log_spread_std': log_spread_std,
        })

df_new = pd.DataFrame(results)
df_new.to_csv("data/sub1_updated_metrics.csv", index=False)

print(f"New pairs: {len(pairs)}")
print(f"Total observations: {len(df_new)}")

# ============ 分时期回归 ============
print("\n" + "=" * 60)
print("Regression by Period (Updated Pairs): log_spread_std ~ vol_mean")
print("=" * 60)

resid_store = {}

for period in ['test1', 'test2', 'test3']:
    subset = df_new[df_new['period'] == period].copy()
    
    X = sm.add_constant(subset[['vol_mean']])
    y = subset['log_spread_std']
    model = sm.OLS(y, X).fit()
    
    resid = model.resid
    resid_store[period] = resid
    
    print(f"\n=== {period} (n={len(subset)}) ===")
    print(f"R² = {model.rsquared:.4f}")
    print(f"vol_mean coef = {model.params['vol_mean']:.4f} (p={model.pvalues['vol_mean']:.4f})")
    print(f"Residual std = {resid.std():.4f}")
    print(f"Residual median = {resid.median():.4f}")
    print(f"Residual skew = {resid.skew():.4f}")
    print(f"Residual kurtosis = {resid.kurtosis():.4f}")
    print(f"Residual max = {resid.max():.4f}")

# ============ 残差对比 ============
print("\n" + "=" * 60)
print("Summary: Residuals by Period (Updated Pairs)")
print("=" * 60)

for period in ['test1', 'test2', 'test3']:
    resid = resid_store[period]
    print(f"{period}: std={resid.std():.4f}, median={resid.median():.4f}, "
          f"skew={resid.skew():.4f}, kurtosis={resid.kurtosis():.4f}")

# ============ Mann-Whitney U检验 ============
print("\n" + "=" * 60)
print("Mann-Whitney U Test: test1 vs test2")
print("=" * 60)

stat, pval = mannwhitneyu(resid_store['test1'], resid_store['test2'], alternative='two-sided')
print(f"U statistic = {stat:.4f}")
print(f"p-value = {pval:.4f}")

# ============ 识别极端配对 ============
print("\n" + "=" * 60)
print("Top 10 Pairs by Residual in test2 (Updated Pairs)")
print("=" * 60)

test2_data = df_new[df_new['period'] == 'test2'].copy()
X = sm.add_constant(test2_data[['vol_mean']])
model = sm.OLS(test2_data['log_spread_std'], X).fit()
test2_data['resid'] = model.resid

top10 = test2_data.nlargest(10, 'resid')[
    ['ticker_1', 'ticker_2', 'sub_industry', 'log_spread_std', 'vol_mean', 'resid']
]
print(top10.to_string(index=False))

# 集中度
positive_resid = test2_data[test2_data['resid'] > 0]['resid']
top5_resid = test2_data.nlargest(5, 'resid')['resid'].sum()
print(f"\nSum of top 5 positive residuals: {top5_resid:.4f}")
print(f"Sum of all positive residuals: {positive_resid.sum():.4f}")
print(f"Top 5 share: {top5_resid / positive_resid.sum() * 100:.2f}%")