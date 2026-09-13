import os
import time
import pandas as pd
from massive import RESTClient
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("MASSIVE_API_KEY")
client = RESTClient(api_key=api_key)

universe = pd.read_csv("data/sp500_pairs_universe.csv")
tickers = universe['Symbol'].tolist()

print(f"Total tickers to fetch: {len(tickers)}")

all_data = []
failed = []

for i, ticker in enumerate(tickers):
    try:
        aggs = []
        for a in client.list_aggs(
            ticker=ticker,
            multiplier=1,
            timespan="day",
            from_="2023-01-01",
            to="2026-09-01",
            limit=50000
        ):
            aggs.append(a)
        
        if aggs:
            df = pd.DataFrame(aggs)
            df['ticker'] = ticker
            all_data.append(df)
        
        if (i + 1) % 20 == 0:
            print(f"Progress: {i+1}/{len(tickers)}")
        
        time.sleep(0.1)  
        
    except Exception as e:
        print(f"Failed: {ticker} - {e}")
        failed.append(ticker)

if all_data:
    combined = pd.concat(all_data, ignore_index=True)
    combined.to_csv("data/stock_daily_ohlc.csv", index=False)
    print(f"\nSaved {len(combined)} rows for {len(all_data)} tickers")
    print(f"Failed: {failed}")