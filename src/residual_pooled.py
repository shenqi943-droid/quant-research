import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

# 读取数据
df = pd.read_csv("data/sub1_v3_metrics.csv")

# 只保留测试期
df_test = df[df['period'].isin(['test1', 'test2', 'test3'])].copy()

print("=" * 60)
print("Pooled Regression on All Test Periods: log_spread_std ~ vol_mean")
print("=" * 60)

# 混合回归：所有测试期放在一起
X = sm.add_constant(df_test[['vol_mean']])
y = df_test['log_spread_std']
model = sm.OLS(y, X).fit()

print(model.summary())

# 用统一系数计算每个测试期的残差
df_test['predicted'] = model.predict(X)
df_test['resid'] = df_test['log_spread_std'] - df_test['predicted']

# ============ 各时期残差统计 ============
print("\n" + "=" * 60)
print("Residuals by Period (using pooled coefficients)")
print("=" * 60)

resid_store = {}

for period in ['test1', 'test2', 'test3']:
    subset = df_test[df_test['period'] == period]
    resid = subset['resid']
    resid_store[period] = resid
    
    print(f"\n=== {period} (n={len(subset)}) ===")
    print(f"Residual std = {resid.std():.4f}")
    print(f"Residual median = {resid.median():.4f}")
    print(f"Residual mean = {resid.mean():.4f}")
    print(f"Residual skew = {resid.skew():.4f}")
    print(f"Residual kurtosis = {resid.kurtosis():.4f}")
    print(f"Residual max = {resid.max():.4f}")

# ============ 汇总 ============
print("\n" + "=" * 60)
print("Summary: Residuals by Period (pooled coefficients)")
print("=" * 60)

for period in ['test1', 'test2', 'test3']:
    resid = resid_store[period]
    print(f"{period}: std={resid.std():.4f}, median={resid.median():.4f}, "
          f"skew={resid.skew():.4f}, kurtosis={resid.kurtosis():.4f}")

# ============ Mann-Whitney U检验 ============
print("\n" + "=" * 60)
print("Mann-Whitney U Test: Pairwise Comparison")
print("=" * 60)

for p1, p2 in [('test1', 'test2'), ('test2', 'test3'), ('test1', 'test3')]:
    stat, pval = mannwhitneyu(resid_store[p1], resid_store[p2], alternative='two-sided')
    print(f"{p1} vs {p2}: U={stat:.4f}, p={pval:.4f}")

# ============ 保存 ============
df_test.to_csv("data/resid_pooled.csv", index=False)
print("\nSaved to data/resid_pooled.csv")

# ============ test2的极端配对 ============
print("\n" + "=" * 60)
print("Top 5 Pairs by Residual in test2 (pooled coefficients)")
print("=" * 60)

test2_data = df_test[df_test['period'] == 'test2']
print(test2_data.nlargest(5, 'resid')[
    ['ticker_1', 'ticker_2', 'sub_industry', 'log_spread_std', 'vol_mean', 'predicted', 'resid']
].to_string(index=False))

# 集中度
positive_resid = test2_data[test2_data['resid'] > 0]['resid']
top5_resid = test2_data.nlargest(5, 'resid')['resid'].sum()
print(f"\nSum of top 5 positive residuals: {top5_resid:.4f}")
print(f"Sum of all positive residuals: {positive_resid.sum():.4f}")
print(f"Top 5 share: {top5_resid / positive_resid.sum() * 100:.2f}%")