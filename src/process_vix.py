import pandas as pd

# 读取VIX数据（Tab分隔）
vix = pd.read_csv("data/vix.csv", sep='\t')
vix.columns = ['date', 'vix']
vix['date'] = pd.to_datetime(vix['date'], format='%Y/%m/%d')
vix = vix.dropna()
vix = vix.sort_values('date').reset_index(drop=True)

periods = {
    'train': ('2023-01-01', '2025-03-31'),
    'test1': ('2025-05-01', '2025-12-31'),
    'test2': ('2026-01-01', '2026-06-30'),
    'test3': ('2026-07-01', '2026-09-01'),
}

print("=== VIX均值 ===")
for period, (start, end) in periods.items():
    subset = vix[(vix['date'] >= start) & (vix['date'] <= end)]
    print(f"{period}: {subset['vix'].mean():.2f} (n={len(subset)})")

vix.to_csv("data/vix_clean.csv", index=False)
print("\nSaved to data/vix_clean.csv")