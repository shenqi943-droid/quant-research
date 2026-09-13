import pandas as pd
df = pd.read_csv("data/sp500_constituents.csv")
print(df.columns.tolist())
print(df.head())
print(df.shape)
print(df['GICS Sector'].value_counts())

df = pd.read_csv("data/sp500_constituents.csv")
target_sectors = ['Information Technology', 'Financials', 'Energy']
df_target = df[df['GICS Sector'].isin(target_sectors)]

print(df_target.groupby(['GICS Sector', 'GICS Sub-Industry']).size().sort_values(ascending=False))