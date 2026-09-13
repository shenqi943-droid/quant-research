# Quant Research: Pairs Trading under Accelerated Capital Rotation

## 1. Project Overview

This repository contains the code and analysis for a research project on pairs trading in the U.S. equity market. The study investigates whether accelerated capital rotation affects the effectiveness of cointegration-based pairs trading strategies.

**Research Question:**
In the current U.S. equity market, broad indices remain stable while individual stocks diverge, with capital rotating rapidly between technology and other sectors. When capital rotation accelerates, does the spread volatility between cointegrated stock pairs exhibit excess expansion relative to individual stock volatility? If excess expansion occurs, does it indicate the pair relationship is changing or even breaking down? And how do these changes affect the effectiveness of pairs trading strategies?

**Hypothesis:**
During periods of accelerated capital rotation, cointegrated stock pairs exhibit excess spread volatility relative to individual stock volatility. Compared to normal pairs, extreme pairs with excess spread volatility break down faster in their pair relationships. With appropriate stop-loss mechanisms, excess spread volatility can generate higher returns.

**Three Sub-Questions:**
1. Does spread volatility exhibit excess expansion relative to individual stock volatility during accelerated capital rotation?
2. If excess expansion occurs, does it accompany structural changes in pair relationships?
3. How do these changes affect the profitability of pairs trading strategies?

## 2. Repository Structure

```
quant-research/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── src/
│   ├── download_sp500.py
│   ├── filter_sp500.py
│   ├── fetch_data.py
│   ├── process_vix.py
│   ├── clean_data.py
│   ├── clean_final.py
│   ├── find_pairs.py
│   ├── update_pairs.py
│   ├── compute_sub1.py
│   ├── regression_by_period.py
│   ├── residual_pooled.py
│   ├── sub2_pvalue_beta.py
│   ├── compute_sub2.py
│   ├── validate_pvalue_beta.py
│   ├── kalman_test_delta.py
│   ├── kalman_final.py
│   └── simulate_100.py
├── data/
└── outputs/
```

## 3. Setup Instructions

### 3.1 Create Virtual Environment

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 3.2 Install Required Packages

```bash
pip install -r requirements.txt
```

### 3.3 Configure API Credentials

Create a `.env` file in the project root:

```
MASSIVE_API_KEY=your_api_key_here
```

A template is provided in `.env.example`. 

## 4. Required Packages

See `requirements.txt`. Main dependencies: pandas, numpy, statsmodels, pykalman, massive, python-dotenv, scipy, matplotlib.

## 5. Data Download

### 5.1 S&P 500 Constituent List

```bash
python src/download_sp500.py
```

Downloads the S&P 500 constituent list from a public GitHub repository (`datasets/s-and-p-500-companies`), including ticker, GICS sector, and GICS sub-industry.

### 5.2 Filter Target Sectors

```bash
python src/filter_sp500.py
```

Filters for three target sectors (Information Technology, Financials, Energy), yielding 170 stocks across 15 sub-industries.

### 5.3 Fetch OHLC Data from Massive API

```bash
python src/fetch_data.py
```

Retrieves daily OHLCV data for each ticker from 2023-01-01 to 2026-09-01 using the Massive REST API Python client.

### 5.4 Process VIX Data

Download VIX data from FRED (series VIXCLS) and save as `data/vix_raw.csv`, then run:

```bash
python src/process_vix.py
```

## 6. Data Processing

```bash
python src/clean_data.py
python src/clean_final.py
```

These scripts: convert Unix millisecond timestamps to date format; remove stocks with insufficient trading days; align trading days across all stocks; output 161 stocks with 919 trading days each (147,959 rows).

## 7. Main Analysis

### 7.1 Sub-Question 1: Excess Spread Volatility

```bash
python src/find_pairs.py
python src/compute_sub1.py
python src/regression_by_period.py
python src/residual_pooled.py
python src/update_pairs.py
python src/regression_updated.py
```

### 7.2 Sub-Question 2: Pair Relationship Changes

```bash
python src/sub2_pvalue_beta.py
python src/compute_sub2.py
python src/validate_pvalue_beta.py
```

### 7.3 Sub-Question 3: Strategy Backtest

```bash
python src/kalman_test_delta.py
python src/kalman_final.py
python src/simulate_100.py
```

## 8. Backtest

The backtest is implemented in `src/kalman_final.py`. Strategy rules: Entry |Z| > 2; Exit |Z| < 0.5; Hard stop-loss |Z| > 4; Time stop-loss holding > 30 days; Transaction cost 0.2% round-trip.

To run:

```bash
python src/kalman_final.py
```

## 9. Outputs

The code generates: `data/coint_pairs_significant.csv` (44 pairs from training period); `data/coint_pairs_updated.csv` (28 pairs from updated period); `data/sub1_v3_metrics.csv` (Sub-question 1 metrics); `data/sub2_lifecycle_metrics_final.csv` (Sub-question 2 metrics); `data/pvalue_comparison.csv` (P-value comparison).

## 10. Data Quality Checks

The following checks are performed: duplicate rows (none found); trading day alignment (161 stocks with 919 days each); missing values (no missing values in OHLC data); stocks with insufficient data (9 stocks excluded: MRSH, Q, FISV, SNDK, XYZ, EXE, CPAY, EG, BNY).

## 11. Assistance and Source Disclosure

**Academic Papers Consulted:** Engle, R.F. and Granger, C.W.J. (1987). Co-integration and error correction. Pandini, D. (2026). Stationary, But Not Profitable? A Critical Look at Pairs Trading. Kalman, R.E. (1960). A new approach to linear filtering and prediction problems.

**External Datasets Used:** S&P 500 constituent list from GitHub repository `datasets/s-and-p-500-companies`; VIX data from FRED (series VIXCLS).

**AI Tools Used:** DeepSeek, used for code debugging, research design discussion, and report writing assistance.

**Assistance Received:** No assistance from other individuals.
