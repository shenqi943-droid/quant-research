import pandas as pd

df = pd.read_csv("data/sp500_constituents.csv")

target_sectors = ['Information Technology', 'Financials', 'Energy']
df_target = df[df['GICS Sector'].isin(target_sectors)]

df_target = df_target[['Symbol', 'Security', 'GICS Sector', 'GICS Sub-Industry']]

df_target.to_csv("data/sp500_pairs_universe.csv", index=False)

print(f"Saved {len(df_target)} stocks")
print(df_target.head())