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
print("Regression by Period: log_spread_std ~ vol_mean")
print("=" * 60)

resid_store = {}

for period in ['test1', 'test2', 'test3']:
    subset = df_test[df_test['period'] == period].copy()
    
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
    print(f"Residual min = {resid.min():.4f}")

# ============ 残差对比 ============
print("\n" + "=" * 60)
print("Summary: Residuals by Period")
print("=" * 60)

for period in ['test1', 'test2', 'test3']:
    resid = resid_store[period]
    print(f"{period}: std={resid.std():.4f}, "
          f"median={resid.median():.4f}, "
          f"skew={resid.skew():.4f}, "
          f"kurtosis={resid.kurtosis():.4f}")

# ============ Mann-Whitney U检验 ============
print("\n" + "=" * 60)
print("Mann-Whitney U Test: test1 vs test2")
print("=" * 60)

stat, pval = mannwhitneyu(
    resid_store['test1'], 
    resid_store['test2'], 
    alternative='two-sided'
)
print(f"U statistic = {stat:.4f}")
print(f"p-value = {pval:.4f}")

if pval < 0.05:
    print("→ Residuals in test2 are significantly different from test1")
else:
    print("→ No significant difference between test1 and test2 residuals")