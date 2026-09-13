import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 读取完整数据
df = pd.read_csv("data/sub2_pvalue_beta.csv")

# 极端配对列表
extreme_pairs = [('AAPL','HPE'), ('ADBE','TRMB'), ('DDOG','PTC'), ('AMP','ARES'), ('PNC','TFC')]

def is_extreme(t1, t2):
    return (t1, t2) in extreme_pairs or (t2, t1) in extreme_pairs

df['is_extreme'] = df.apply(lambda r: is_extreme(r['ticker_1'], r['ticker_2']), axis=1)

# 对每个配对，计算β在train→test1→test2→test3的变化幅度
# 用最大β与最小β之差 / train β的绝对值，作为漂移幅度

pairs_list = df[['ticker_1', 'ticker_2', 'sub_industry']].drop_duplicates()
drift_results = []

for _, row in pairs_list.iterrows():
    t1, t2 = row['ticker_1'], row['ticker_2']
    pair_data = df[(df['ticker_1'] == t1) & (df['ticker_2'] == t2)]
    
    betas = pair_data[['period', 'beta']].set_index('period')['beta'].to_dict()
    
    # 只保留有train的配对（用于计算漂移）
    if 'train' not in betas:
        continue
    
    train_beta = betas['train']
    all_betas = [betas[p] for p in ['train', 'test1', 'test2', 'test3'] if p in betas]
    
    if len(all_betas) < 3:
        continue
    
    max_beta = max(all_betas)
    min_beta = min(all_betas)
    drift_abs = max_beta - min_beta
    drift_pct = drift_abs / abs(train_beta) if abs(train_beta) > 0.01 else np.nan
    
    # 符号是否翻转
    sign_flip = (max_beta > 0) and (min_beta < 0)
    
    drift_results.append({
        'ticker_1': t1,
        'ticker_2': t2,
        'sub_industry': row['sub_industry'],
        'train_beta': train_beta,
        'max_beta': max_beta,
        'min_beta': min_beta,
        'drift_abs': drift_abs,
        'drift_pct': drift_pct,
        'sign_flip': sign_flip,
        'is_extreme': is_extreme(t1, t2)
    })

drift_df = pd.DataFrame(drift_results)

# ============ 极端配对 ============
print("=" * 80)
print("Beta Drift: Extreme Pairs")
print("=" * 80)
extreme = drift_df[drift_df['is_extreme']].sort_values('drift_pct', ascending=False)
print(extreme[['ticker_1', 'ticker_2', 'train_beta', 'max_beta', 'min_beta', 'drift_abs', 'drift_pct', 'sign_flip']].to_string(index=False))

# ============ 普通配对 ============
print("\n" + "=" * 80)
print("Beta Drift: Normal Pairs (Top 10 by drift_pct)")
print("=" * 80)
normal = drift_df[~drift_df['is_extreme']].sort_values('drift_pct', ascending=False)
print(normal[['ticker_1', 'ticker_2', 'train_beta', 'max_beta', 'min_beta', 'drift_abs', 'drift_pct', 'sign_flip']].head(10).to_string(index=False))

# ============ 汇总对比 ============
print("\n" + "=" * 80)
print("Summary: Beta Drift by Group")
print("=" * 80)

print(f"\nExtreme pairs (n={len(extreme)}):")
print(f"  drift_pct median: {extreme['drift_pct'].median():.2f}")
print(f"  drift_pct mean: {extreme['drift_pct'].mean():.2f}")
print(f"  sign_flip count: {extreme['sign_flip'].sum()}/{len(extreme)}")

print(f"\nNormal pairs (n={len(normal)}):")
print(f"  drift_pct median: {normal['drift_pct'].median():.2f}")
print(f"  drift_pct mean: {normal['drift_pct'].mean():.2f}")
print(f"  sign_flip count: {normal['sign_flip'].sum()}/{len(normal)}")

# ============ 按漂移幅度分类 ============
print("\n" + "=" * 80)
print("Distribution of drift_pct")
print("=" * 80)

for group, name in [(extreme, 'Extreme'), (normal, 'Normal')]:
    print(f"\n{name}:")
    print(f"  drift_pct < 50%: {len(group[group['drift_pct'] < 0.5])}")
    print(f"  50% <= drift_pct < 100%: {len(group[(group['drift_pct'] >= 0.5) & (group['drift_pct'] < 1.0)])}")
    print(f"  drift_pct >= 100%: {len(group[group['drift_pct'] >= 1.0])}")