import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

# 读取sub1指标
df = pd.read_csv("data/sub1_v3_metrics.csv")

# 读取VIX
vix = pd.read_csv("data/vix_clean.csv", parse_dates=['date'])

# 各时期VIX均值
periods = {
    'train': ('2023-01-01', '2025-03-31'),
    'test1': ('2025-05-01', '2025-12-31'),
    'test2': ('2026-01-01', '2026-06-30'),
    'test3': ('2026-07-01', '2026-09-01'),
}

vix_means = {}
for period, (start, end) in periods.items():
    subset = vix[(vix['date'] >= start) & (vix['date'] <= end)]
    vix_means[period] = subset['vix'].mean()

df['vix_mean'] = df['period'].map(vix_means)

# 虚拟变量
df['test_period'] = df['period'].apply(lambda x: 0 if x == 'train' else 1)

# ============ 相关性 ============
period_stats = df.groupby('period').agg({
    'log_spread_std': 'median',
    'vix_mean': 'first',
}).reset_index()

corr, pval = stats.pearsonr(period_stats['vix_mean'], period_stats['log_spread_std'])
print(f"=== Correlation: VIX vs log_spread_std ===")
print(f"Pearson r = {corr:.4f}, p = {pval:.4f}")

# ============ 回归1：控制个股波动 ============
print("\n=== Regression 1: log_spread_std ~ test_period + vol_mean ===")
X1 = df[['test_period', 'vol_mean']]
X1 = sm.add_constant(X1)
y = df['log_spread_std']
model1 = sm.OLS(y, X1).fit()
print(model1.summary())

# ============ 回归2：加入VIX ============
print("\n=== Regression 2: log_spread_std ~ test_period + vol_mean + vix_mean ===")
X2 = df[['test_period', 'vol_mean', 'vix_mean']]
X2 = sm.add_constant(X2)
model2 = sm.OLS(y, X2).fit()
print(model2.summary())