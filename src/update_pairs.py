import pandas as pd
import numpy as np
from itertools import combinations
from statsmodels.tsa.stattools import coint
import warnings
warnings.filterwarnings('ignore')

# 读取数据
df = pd.read_csv("data/stock_daily_final.csv")
universe = pd.read_csv("data/pairs_universe_final.csv")

prices = df.pivot(index='date', columns='ticker', values='close')
prices.index = pd.to_datetime(prices.index)

# ============ 用2023.7-2025.12重新筛选配对 ============
train_start = '2023-07-01'
train_end = '2025-12-31'
prices_train = prices[(prices.index >= train_start) & (prices.index <= train_end)]

print(f"Training period: {prices_train.index[0]} to {prices_train.index[-1]}")
print(f"Training days: {len(prices_train)}")

# 按子行业分组
sub_industries = universe.groupby('GICS Sub-Industry')['Symbol'].apply(list).to_dict()
min_stocks = 5
valid_subs = {k: v for k, v in sub_industries.items() if len(v) >= min_stocks}

results = []

for sub_ind, tickers in valid_subs.items():
    tickers = [t for t in tickers if t in prices.columns]
    if len(tickers) < 2:
        continue
    
    for t1, t2 in combinations(tickers, 2):
        try:
            s1 = prices_train[t1].dropna()
            s2 = prices_train[t2].dropna()
            common = s1.index.intersection(s2.index)
            if len(common) < 200:
                continue
            s1, s2 = s1[common], s2[common]
            
            score, pvalue, _ = coint(s1, s2)
            beta = np.polyfit(s2, s1, 1)[0]
            spread = s1 - beta * s2
            
            spread_lag = spread.shift(1).dropna()
            spread_diff = spread.diff().dropna()
            common_idx = spread_lag.index.intersection(spread_diff.index)
            y = spread_diff[common_idx]
            x = spread_lag[common_idx]
            beta_ou = np.polyfit(x, y, 1)[0]
            half_life = -np.log(2) / beta_ou if beta_ou < 0 else np.inf
            
            results.append({
                'sub_industry': sub_ind,
                'ticker_1': t1,
                'ticker_2': t2,
                'coint_pvalue': pvalue,
                'hedge_ratio': beta,
                'half_life': half_life
            })
        except:
            continue

results_df = pd.DataFrame(results)

# 筛选显著配对
significant = results_df[
    (results_df['coint_pvalue'] < 0.05) & 
    (results_df['half_life'] < 60) & 
    (results_df['half_life'] > 0)
].copy()

print(f"\nTotal pairs tested: {len(results_df)}")
print(f"Significant pairs (p<0.05, 0<half_life<60): {len(significant)}")

# 检查AAPL-HPE是否还在
aapl_hpe = significant[(significant['ticker_1'] == 'AAPL') & (significant['ticker_2'] == 'HPE')]
print(f"\nAAPL-HPE in new pool: {'YES' if len(aapl_hpe) > 0 else 'NO'}")
if len(aapl_hpe) > 0:
    print(aapl_hpe.to_string(index=False))

# 保存新配对池
significant.to_csv("data/coint_pairs_updated.csv", index=False)
print(f"\nSaved to data/coint_pairs_updated.csv")