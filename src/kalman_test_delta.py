import pandas as pd
import numpy as np
from pykalman import KalmanFilter
import warnings
warnings.filterwarnings('ignore')

# 读取数据
df = pd.read_csv("data/stock_daily_final.csv")
pairs_new = pd.read_csv("data/coint_pairs_updated.csv")

prices = df.pivot(index='date', columns='ticker', values='close')
prices.index = pd.to_datetime(prices.index)

periods = {
    'test1': ('2025-07-01', '2025-12-31'),
    'test2': ('2026-01-01', '2026-06-30'),
    'test3': ('2026-07-01', '2026-09-01'),
}

def kalman_beta(y, x, delta=1e-5):
    """用卡尔曼滤波估计时变beta"""
    n = len(y)
    # 状态: [beta, alpha]
    # 观测: y = beta * x + alpha
    trans_cov = delta / (1 - delta) * np.eye(2)
    
    kf = KalmanFilter(
        n_dim_obs=1,
        n_dim_state=2,
        initial_state_mean=[0, 0],
        initial_state_covariance=np.ones((2, 2)),
        transition_matrices=np.eye(2),
        observation_matrices=np.array([[x[0], 1]]),
        transition_covariance=trans_cov,
        observation_covariance=1.0
    )
    
    # 递归估计
    state_means = np.zeros((n, 2))
    state_covs = np.zeros((n, 2, 2))
    state_mean = kf.initial_state_mean
    state_cov = kf.initial_state_covariance
    
    for t in range(n):
        # 更新观测矩阵
        obs_mat = np.array([[x[t], 1]])
        
        # 预测
        pred_mean = state_mean
        pred_cov = state_cov + trans_cov
        
        # 更新
        obs_cov = 1.0
        kalman_gain = pred_cov @ obs_mat.T @ np.linalg.inv(obs_mat @ pred_cov @ obs_mat.T + obs_cov)
        state_mean = pred_mean + kalman_gain.flatten() * (y[t] - obs_mat @ pred_mean)
        state_cov = (np.eye(2) - kalman_gain @ obs_mat) @ pred_cov
        
        state_means[t] = state_mean
        state_covs[t] = state_cov
    
    return state_means[:, 0], state_means[:, 1]  # beta, alpha

def backtest_pair(p1, p2, delta, entry_z=2.0, exit_z=0.5, stop_z=4.0, max_hold=30):
    """对单个配对回测"""
    log_p1 = np.log(p1.values)
    log_p2 = np.log(p2.values)
    
    # 卡尔曼滤波估计beta
    beta_t, alpha_t = kalman_beta(log_p1, log_p2, delta)
    
    # 计算价差
    spread = log_p1 - beta_t * log_p2 - alpha_t
    
    # Z-score（滚动60天）
    spread_series = pd.Series(spread, index=p1.index)
    mean = spread_series.rolling(60).mean()
    std = spread_series.rolling(60).std()
    z = (spread_series - mean) / std
    
    # 回测
    position = 0
    entry_price = 0
    entry_day = 0
    pnl = []
    
    for i in range(60, len(z)):
        if np.isnan(z.iloc[i]):
            continue
        
        current_z = z.iloc[i]
        current_spread = spread[i]
        
        if position == 0:
            if current_z > entry_z:
                position = -1  # 做空价差
                entry_price = current_spread
                entry_day = i
            elif current_z < -entry_z:
                position = 1  # 做多价差
                entry_price = current_spread
                entry_day = i
        else:
            # 检查平仓条件
            hold_days = i - entry_day
            exit_signal = False
            
            if abs(current_z) < exit_z:
                exit_signal = True
            elif abs(current_z) > stop_z:
                exit_signal = True
            elif hold_days > max_hold:
                exit_signal = True
            
            if exit_signal:
                if position == 1:
                    pnl.append(current_spread - entry_price)
                else:
                    pnl.append(entry_price - current_spread)
                position = 0
    
    return pnl, z

# 对每个delta值测试
deltas = [1e-6, 1e-5, 1e-4, 1e-3]

print("=" * 70)
print("Kalman Filter Delta Comparison")
print("=" * 70)

for delta in deltas:
    print(f"\n=== delta = {delta} ===")
    
    for period in ['test1', 'test2', 'test3']:
        start, end = periods[period]
        all_pnl = []
        n_pairs = 0
        
        for _, row in pairs_new.iterrows():
            t1, t2 = row['ticker_1'], row['ticker_2']
            if t1 not in prices.columns or t2 not in prices.columns:
                continue
            
            p1 = prices[t1][start:end].dropna()
            p2 = prices[t2][start:end].dropna()
            common = p1.index.intersection(p2.index)
            if len(common) < 60:
                continue
            p1, p2 = p1[common], p2[common]
            
            pnl, _ = backtest_pair(p1, p2, delta)
            if len(pnl) > 0:
                all_pnl.extend(pnl)
                n_pairs += 1
        
        if len(all_pnl) > 0:
            total_pnl = sum(all_pnl)
            win_rate = sum(1 for x in all_pnl if x > 0) / len(all_pnl)
            avg_pnl = np.mean(all_pnl)
            print(f"  {period}: n_pairs={n_pairs}, total_pnl={total_pnl:.4f}, "
                  f"avg={avg_pnl:.4f}, win_rate={win_rate:.2%}, n_trades={len(all_pnl)}")