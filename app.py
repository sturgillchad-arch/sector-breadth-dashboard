import io
import numpy as np
import pandas as pd
import requests
import yfinance as yf


def get_sp500_sectors():
    """Fetch current S&P 500 constituents and their GICS sectors with a browser header."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/115.0.0.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    # Wrap raw HTML in StringIO to avoid future parser deprecation warnings
    table = pd.read_html(io.StringIO(response.text))[0]
    table["Symbol"] = table["Symbol"].str.replace(".", "-", regex=False)
    return table[["Symbol", "GICS Sector", "Security"]]


def calculate_stock_indicators(df):
    """Calculate moving averages, highs/lows, returns, and RSI."""
    results = {}

    for ticker in df.columns.levels[1]:
        sub = df.xs(ticker, axis=1, level=1).dropna()
        if len(sub) < 252:
            continue

        close = sub["Close"]
        high = sub["High"]
        low = sub["Low"]
        volume = sub["Volume"]
        last_price = close.iloc[-1]

        # Moving Averages
        sma10 = close.rolling(10).mean().iloc[-1]
        sma20 = close.rolling(20).mean().iloc[-1]
        sma50 = close.rolling(50).mean().iloc[-1]
        sma100 = close.rolling(100).mean().iloc[-1]
        sma200 = close.rolling(200).mean().iloc[-1]

        # Highs / Lows (4-week ~ 20 days, 52-week ~ 252 days)
        hi_20 = high.iloc[-20:].max()
        lo_20 = low.iloc[-20:].min()
        hi_252 = high.iloc[-252:].max()
        lo_252 = low.iloc[-252:].min()

        # 14-period RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(14).mean().iloc[-1]
        avg_loss = loss.rolling(14).mean().iloc[-1]
        rs = avg_gain / avg_loss if avg_loss != 0 else np.nan
        rsi = 100 - (100 / (1 + rs)) if pd.notnull(rs) else 50.0

        # Performance & RVOL
        ret_1d = (close.iloc[-1] / close.iloc[-2] - 1) * 100
        ret_5d = (
            (close.iloc[-1] / close.iloc[-6] - 1) * 100
            if len(close) > 6
            else 0.0
        )
        avg_vol20 = volume.rolling(20).mean().iloc[-1]
        rvol = volume.iloc[-1] / avg_vol20 if avg_vol20 > 0 else 1.0

        results[ticker] = {
            "last_price": last_price,
            "above_10d": last_price > sma10,
            "above_20d": last_price > sma20,
            "above_50d": last_price > sma50,
            "above_100d": last_price > sma100,
            "above_200d": last_price > sma200,
            "new_4w_high": last_price >= hi_20,
            "new_4w_low": last_price <= lo_20,
            "new_52w_high": last_price >= hi_252,
            "new_52w_low": last_price <= lo_252,
            "rsi": rsi,
            "ret_1d": ret_1d,
            "ret_5d": ret_5d,
            "rvol": rvol,
        }

    return pd.DataFrame.from_dict(results, orient="index")


def generate_sector_breadth_report():
    print("1. Fetching S&P 500 sector mapping...")
    sp_table = get_sp500_sectors()
    tickers = sp_table["Symbol"].tolist()

    print(f"2. Downloading historical data for {len(tickers)} symbols...")
    data = yf.download(
        tickers, period="1y", group_by="ticker", threads=True, progress=False
    )

    print("3. Computing breadth indicators...")
    metrics_df = calculate_stock_indicators(data)
    merged = sp_table.merge(
        metrics_df, left_on="Symbol", right_index=True, how="inner"
    )

    # Sector Breadth Table
    breadth = (
        merged.groupby("GICS Sector")
        .agg(
            Total_Count=("Symbol", "count"),
            Pct_Above_10D=("above_10d", lambda x: f"{x.mean() * 100:.1f}%"),
            Pct_Above_20D=("above_20d", lambda x: f"{x.mean() * 100:.1f}%"),
            Pct_Above_50D=("above_50d", lambda x: f"{x.mean() * 100:.1f}%"),
            Pct_Above_100D=("above_100d", lambda x: f"{x.mean() * 100:.1f}%"),
            Pct_Above_200D=("above_200d", lambda x: f"{x.mean() * 100:.1f}%"),
            High_4W=("new_4w_high", lambda x: f"{x.mean() * 100:.1f}%"),
            High_52W=("new_52w_high", lambda x: f"{x.mean() * 100:.1f}%"),
            Low_4W=("new_4w_low", lambda x: f"{x.mean() * 100:.1f}%"),
            Low_52W=("new_52w_low", lambda x: f"{x.mean() * 100:.1f}%"),
            RSI_Over_70=("rsi", lambda x: f"{(x > 70).mean() * 100:.1f}%"),
            RSI_Under_30=("rsi", lambda x: f"{(x < 30).mean() * 100:.1f}%"),
        )
        .reset_index()
    )

    print("\n" + "=" * 80)
    print("SECTOR BREADTH DASHBOARD")
    print("=" * 80)
    print(breadth.to_string(index=False))

    # Top Performers per Sector
    print("\n" + "=" * 80)
    print("TOP PERFORMERS BY SECTOR (5-Day Return)")
    print("=" * 80)
    top_performers = merged.sort_values(
        ["GICS Sector", "ret_5d"], ascending=[True, False]
    )
    for sector, group in top_performers.groupby("GICS Sector"):
        print(f"\n[{sector}]")
        top3 = group.head(3)[
            ["Symbol", "Security", "ret_1d", "ret_5d", "rvol", "rsi"]
        ]
        top3.columns = ["Ticker", "Company", "1D %", "5D %", "RVOL", "RSI"]
        print(
            top3.to_string(
                index=False,
                formatters={
                    "1D %": "{:+.2f}%".format,
                    "5D %": "{:+.2f}%".format,
                    "RVOL": "{:.2f}x".format,
                    "RSI": "{:.1f}".format,
                },
            )
        )


if __name__ == "__main__":
    generate_sector_breadth_report()