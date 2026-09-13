import pandas as pd
import numpy as np
from itertools import combinations
from statsmodels.tsa.stattools import coint, adfuller
import warnings
warnings.filterwarnings('ignore')

# 读取数据
df = pd.read_csv("data/stock_daily_final.csv")
universe = pd.read_csv("data/pairs_universe_final.csv")

# 转成宽表
prices = df.pivot(index='date', columns='ticker', values='close')
print(f"Price matrix shape: {prices.shape}")

# 按子行业分组
sub_industries = universe.groupby('GICS Sub-Industry')['Symbol'].apply(list).to_dict()

# 筛选股票数>=5的子行业
min_stocks = 5
valid_subs = {k: v for k, v in sub_industries.items() if len(v) >= min_stocks}
print(f"\nSub-industries with >= {min_stocks} stocks: {len(valid_subs)}")

# ============ 协整检验 ============
# 训练期：2023.1 - 2025.6
train_end = '2025-06-30'
prices_train = prices[prices.index <= train_end]

print(f"\nTraining period: {prices_train.index[0]} to {prices_train.index[-1]}")
print(f"Training days: {len(prices_train)}")

results = []

for sub_ind, tickers in valid_subs.items():
    # 过滤：只保留在价格矩阵中的ticker
    tickers = [t for t in tickers if t in prices.columns]
    if len(tickers) < 2:
        continue
    
    # 对该子行业内的所有配对进行协整检验
    for t1, t2 in combinations(tickers, 2):
        try:
            s1 = prices_train[t1].dropna()
            s2 = prices_train[t2].dropna()
            
            # 对齐
            common_idx = s1.index.intersection(s2.index)
            if len(common_idx) < 200:
                continue
            
            s1 = s1[common_idx]
            s2 = s2[common_idx]
            
            # 协整检验（Engle-Granger）
            score, pvalue, _ = coint(s1, s2)
            
            # 计算对冲比率（OLS）
            beta = np.polyfit(s2, s1, 1)[0]
            spread = s1 - beta * s2
            
            # 半衰期
            spread_lag = spread.shift(1).dropna()
            spread_diff = spread.diff().dropna()
            common = spread_lag.index.intersection(spread_diff.index)
            
            if len(common) > 10:
                y = spread_diff[common]
                x = spread_lag[common]
                beta_ou = np.polyfit(x, y, 1)[0]
                if beta_ou < 0:
                    half_life = -np.log(2) / beta_ou
                else:
                    half_life = np.inf
            else:
                half_life = np.inf
            
            results.append({
                'sub_industry': sub_ind,
                'ticker_1': t1,
                'ticker_2': t2,
                'coint_pvalue': pvalue,
                'hedge_ratio': beta,
                'half_life': half_life
            })
        except Exception as e:
            continue

# 保存所有结果
results_df = pd.DataFrame(results)
results_df.to_csv("data/coint_results_all.csv", index=False)

print(f"\nTotal pairs tested: {len(results_df)}")
print(f"Pairs with p < 0.05: {len(results_df[results_df['coint_pvalue'] < 0.05])}")
print(f"Pairs with p < 0.01: {len(results_df[results_df['coint_pvalue'] < 0.01])}")

# 筛选显著配对
significant = results_df[
    (results_df['coint_pvalue'] < 0.05) & 
    (results_df['half_life'] < 60) & 
    (results_df['half_life'] > 0)
].copy()

print(f"\nSignificant pairs (p<0.05, 0<half_life<60): {len(significant)}")
print(f"\nBy sub-industry:")
print(significant.groupby('sub_industry').size().sort_values(ascending=False))

# 保存显著配对
significant.to_csv("data/coint_pairs_significant.csv", index=False)
print(f"\nSaved to data/coint_pairs_significant.csv")