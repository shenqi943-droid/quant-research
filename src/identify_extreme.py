import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# 读取数据
df = pd.read_csv("data/sub1_v3_metrics.csv")
df_test = df[df['period'].isin(['test1', 'test2', 'test3'])].copy()

# ============ test2的残差 ============
test2_data = df_test[df_test['period'] == 'test2'].copy()
X = sm.add_constant(test2_data[['vol_mean']])
model = sm.OLS(test2_data['log_spread_std'], X).fit()
test2_data['resid'] = model.resid

# ============ 按残差排序，查看最大的10对 ============
print("=" * 80)
print("Top 10 Pairs by Residual in test2 (highest excess spread volatility)")
print("=" * 80)
top10 = test2_data.nlargest(10, 'resid')[
    ['ticker_1', 'ticker_2', 'sub_industry', 'log_spread_std', 'vol_mean', 'resid']
]
print(top10.to_string(index=False))

# ============ 按残差排序，查看最小的5对（负残差） ============
print("\n" + "=" * 80)
print("Bottom 5 Pairs by Residual in test2 (lowest spread volatility vs prediction)")
print("=" * 80)
bottom5 = test2_data.nsmallest(5, 'resid')[
    ['ticker_1', 'ticker_2', 'sub_industry', 'log_spread_std', 'vol_mean', 'resid']
]
print(bottom5.to_string(index=False))

# ============ 按子行业统计残差 ============
print("\n" + "=" * 80)
print("Residual by Sub-Industry in test2")
print("=" * 80)
sub_ind_stats = test2_data.groupby('sub_industry').agg({
    'resid': ['count', 'median', 'mean', 'max'],
    'log_spread_std': 'median',
}).round(4)
print(sub_ind_stats)

# ============ 残差是否集中在少数配对？ ============
print("\n" + "=" * 80)
print("Concentration of Extreme Residuals")
print("=" * 80)
resid_sorted = test2_data['resid'].sort_values(ascending=False)
total_resid_std = test2_data['resid'].std()
print(f"Total residual std: {total_resid_std:.4f}")
print(f"Top 5 pairs contribute to residual variance:")
for i in range(5):
    pair = test2_data.nlargest(5, 'resid').iloc[i]
    print(f"  {pair['ticker_1']}-{pair['ticker_2']}: resid={pair['resid']:.4f}")

# 计算前5对配对的残差占所有正残差的比例
positive_resid = test2_data[test2_data['resid'] > 0]['resid']
top5_resid = test2_data.nlargest(5, 'resid')['resid'].sum()
print(f"\nSum of top 5 positive residuals: {top5_resid:.4f}")
print(f"Sum of all positive residuals: {positive_resid.sum():.4f}")
print(f"Top 5 share: {top5_resid / positive_resid.sum() * 100:.2f}%")

# ============ 对比test1和test2的极端配对 ============
print("\n" + "=" * 80)
print("Comparison: Same Pairs in test1 vs test2")
print("=" * 80)

# 获取test2残差最大的5对
top5_pairs = test2_data.nlargest(5, 'resid')[['ticker_1', 'ticker_2']].values.tolist()

for t1, t2 in top5_pairs:
    test1_row = df_test[(df_test['period'] == 'test1') & 
                        (df_test['ticker_1'] == t1) & 
                        (df_test['ticker_2'] == t2)]
    test2_row = df_test[(df_test['period'] == 'test2') & 
                        (df_test['ticker_1'] == t1) & 
                        (df_test['ticker_2'] == t2)]
    test3_row = df_test[(df_test['period'] == 'test3') & 
                        (df_test['ticker_1'] == t1) & 
                        (df_test['ticker_2'] == t2)]
    
    if len(test1_row) > 0 and len(test2_row) > 0:
        print(f"\n{t1}-{t2}:")
        print(f"  test1: log_spread_std={test1_row['log_spread_std'].values[0]:.4f}, vol_mean={test1_row['vol_mean'].values[0]:.4f}")
        print(f"  test2: log_spread_std={test2_row['log_spread_std'].values[0]:.4f}, vol_mean={test2_row['vol_mean'].values[0]:.4f}")
        if len(test3_row) > 0:
            print(f"  test3: log_spread_std={test3_row['log_spread_std'].values[0]:.4f}, vol_mean={test3_row['vol_mean'].values[0]:.4f}")