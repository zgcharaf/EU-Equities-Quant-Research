
import yfinance as yf
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
# Define end = today, start = six years ago
end = datetime.today().date()
start = end - relativedelta(years=3)

print(f"Fetching data from {start} to {end}")
def fetch_six_years(tickers, start, end, interval="1d"):
    """
    Returns a dict mapping each ticker to its DataFrame of
    daily OHLCV (adjusted) from start to end.
    """
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end + relativedelta(days=1),  # include end date
        interval=interval,
        group_by='ticker',
        auto_adjust=True,
        threads=True,
    )

    data = {}
    for t in tickers:
        # handle single vs multi‐ticker return shape
        df = raw[t].copy() if len(tickers) > 1 else raw.copy()
        df.index.name = "Date"
        data[t] = df
    return data
# Our universe
df_tickers = pd.read_csv('constituents.csv')
tickers = {}

for ticker, name  in df_tickers[['ticker', 'name']].values: 
    tickers[name]=ticker
    
tickers
# Fetch data
hist_data = fetch_six_years(
    list(tickers.values()),
    start=start,
    end=end,
    interval="1d"
)

# Display first 5 rows of each
for name, sym in tickers.items():
    print(f"### {name} ({sym})")
for ticker in tickers.values(): 
    hist_data[ticker]= hist_data[ticker].reset_index()
    valid_hist_data = {}
skipped = []
full_data_set = pd.DataFrame()

for ticker in tickers.values():
    df = hist_data.get(ticker)

    # Check: DataFrame exists, not empty, and "Close" column has at least one non-NaN value
    if isinstance(df, pd.DataFrame) and not df.empty and df["Close"].notna().any():
        valid_hist_data[ticker] = df
        df['ticker'] = ticker
        full_data_set = pd.concat([full_data_set, df])
    else:
        skipped.append(ticker)
# Replace with filtered dict
hist_data = valid_hist_data
indicies_table = pd.read_csv('market_indices.csv')
indices_tickers = {}
for country,ticker   in indicies_table[[ 'Country', 'ticker']].values: 
    indices_tickers[country]=ticker
index_data = fetch_six_years(
    list(indices_tickers.values()),
    start=start,
    end=end,
    interval="1d"
)    
for ticker in indices_tickers.values(): 
    index_data[ticker]= index_data[ticker].reset_index()
    valid_index_data = {}
skipped = []
full_index_data = pd.DataFrame()

for k, v in indices_tickers.items():
    df = index_data.get(v)

    if isinstance(df, pd.DataFrame) and not df.empty and "Close" in df.columns:
        # % of missing values in Close
        missing_ratio = df["Close"].isna().mean()

        if missing_ratio < 0.5 and df["Close"].notna().any():
            valid_index_data[v] = df
            df = df.copy()  # avoid SettingWithCopyWarning
            df['ticker'] = v
            df['Country'] = k
            full_index_data = pd.concat([full_index_data, df])
        else:
            skipped.append(v)
    else:
        skipped.append(v)

# Replace with filtered dict
index_data = valid_index_data
full_index_data.to_csv('index_data.csv')
full_data_set.to_csv('Prices.csv')