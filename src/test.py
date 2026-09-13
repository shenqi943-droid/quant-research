import pandas as pd

df = pd.read_csv("data/sub1_metrics.csv")

# 逐时期看vol_mean和spread_vol的绝对水平
print("=== vol_mean ===")
print(df.groupby('period')['vol_mean'].describe())
print("\n=== spread_vol ===")
print(df.groupby('period')['spread_vol'].describe())

# 看训练期和测试期的价差波动分布
print("\n=== spread_vol按时期分布 ===")
for period in ['train', 'test1', 'test2', 'test3']:
    subset = df[df['period'] == period]['spread_vol']
    print(f"{period}: median={subset.median():.4f}, mean={subset.mean():.4f}, max={subset.max():.4f}")