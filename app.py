import io
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

# Configure page settings
st.set_page_config(
    page_title="S&P 500 Sector Breadth Dashboard",
    page_icon="📈",
    layout="wide",
)


@st.cache_data(ttl=3600)
def get_sp500_sectors():
    """Fetch current S&P 500 constituents and their GICS sectors."""
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

    table = pd.read_html(io.StringIO(response.text))[0]
    table["Symbol"] = table["Symbol"].str.replace(".", "-", regex=False)
    return table[["Symbol", "GICS Sector", "Security"]]


@st.cache_data(ttl=3600)
def fetch_market_data(tickers):
    """Download 1 year of daily historical data for all constituents in bulk."""
    data = yf.download(
        tickers,
        period="1y",
        group_by="ticker",
        auto_adjust=True,
        progress=False,
    )
    return data


def calculate_stock_indicators(df, tickers):
    """Calculate moving averages, high/low breakouts, RSI, and RVOL safely."""
    results = {}

    for ticker in tickers:
        try:
            # Handle multi-index formats safely across yfinance versions
            if isinstance(df.columns, pd.MultiIndex):
                if ticker in df.columns.levels[0]:
                    sub = df[ticker].dropna(how="all")
                elif ticker in df.columns.levels[1]:
                    sub = df.xs(ticker, axis=1, level=1).dropna(how="all")
                else:
                    continue
            else:
                sub = df.dropna(how="all")

            if len(sub) < 30:
                continue

            close = sub["Close"].dropna()
            if len(close) < 30:
                continue

            high = sub["High"].dropna() if "High" in sub.columns else close
            low = sub["Low"].dropna() if "Low" in sub.columns else close
            volume = (
                sub["Volume"].dropna()
                if "Volume" in sub.columns
                else pd.Series(1, index=close.index)
            )

            last_price = float(close.iloc[-1])

            # Moving Averages
            sma10 = (
                float(close.rolling(10).mean().iloc[-1])
                if len(close) >= 10
                else last_price
            )
            sma20 = (
                float(close.rolling(20).mean().iloc[-1])
                if len(close) >= 20
                else last_price
            )
            sma50 = (
                float(close.rolling(50).mean().iloc[-1])
                if len(close) >= 50
                else last_price
            )
            sma100 = (
                float(close.rolling(100).mean().iloc[-1])
                if len(close) >= 100
                else last_price
            )
            sma200 = (
                float(close.rolling(200).mean().iloc[-1])
                if len(close) >= 200
                else last_price
            )

            # Highs / Lows (4-week ~ 20 trading days, 52-week ~ 252 trading days)
            lookback_4w = min(len(high), 21)
            lookback_52w = min(len(high), 253)

            prior_hi_20 = (
                float(high.iloc[-lookback_4w:-1].max())
                if lookback_4w > 1
                else last_price
            )
            prior_lo_20 = (
                float(low.iloc[-lookback_4w:-1].min())
                if lookback_4w > 1
                else last_price
            )
            prior_hi_252 = (
                float(high.iloc[-lookback_52w:-1].max())
                if lookback_52w > 1
                else last_price
            )
            prior_lo_252 = (
                float(low.iloc[-lookback_52w:-1].min())
                if lookback_52w > 1
                else last_price
            )

            today_high = float(high.iloc[-1])
            today_low = float(low.iloc[-1])

            new_4w_high = bool(today_high >= prior_hi_20)
            new_4w_low = bool(today_low <= prior_lo_20)
            new_52w_high = bool(today_high >= prior_hi_252)
            new_52w_low = bool(today_low <= prior_lo_252)

            # 14-period RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0.0)
            loss = -delta.where(delta < 0, 0.0)
            avg_gain = gain.rolling(14).mean().iloc[-1]
            avg_loss = loss.rolling(14).mean().iloc[-1]
            rs = avg_gain / avg_loss if avg_loss != 0 else np.nan
            rsi = (
                float(100 - (100 / (1 + rs)))
                if (pd.notnull(rs) and not np.isnan(rs))
                else 50.0
            )

            # Returns & Relative Volume (RVOL)
            ret_1d = (
                float((close.iloc[-1] / close.iloc[-2] - 1) * 100)
                if len(close) >= 2
                else 0.0
            )
            ret_5d = (
                float((close.iloc[-1] / close.iloc[-6] - 1) * 100)
                if len(close) >= 6
                else 0.0
            )

            avg_vol20 = (
                float(volume.rolling(20).mean().iloc[-1])
                if len(volume) >= 20
                else float(volume.iloc[-1])
            )
            rvol = (
                float(volume.iloc[-1] / avg_vol20)
                if (avg_vol20 > 0 and not np.isnan(avg_vol20))
                else 1.0
            )

            results[ticker] = {
                "last_price": last_price,
                "above_10d": last_price > sma10,
                "above_20d": last_price > sma20,
                "above_50d": last_price > sma50,
                "above_100d": last_price > sma100,
                "above_200d": last_price > sma200,
                "new_4w_high": new_4w_high,
                "new_4w_low": new_4w_low,
                "new_52w_high": new_52w_high,
                "new_52w_low": new_52w_low,
                "rsi": rsi,
                "ret_1d": ret_1d,
                "ret_5d": ret_5d,
                "rvol": rvol,
            }
        except Exception:
            continue

    return pd.DataFrame.from_dict(results, orient="index")


# --- UI Presentation ---

st.title("📊 S&P 500 Sector Breadth & Performance Dashboard")
st.caption(
    "Daily market participation, momentum extremes, and sector leaders."
)

with st.spinner("Fetching latest market data across S&P 500 constituents..."):
    sp_table = get_sp500_sectors()
    tickers = sp_table["Symbol"].tolist()
    data = fetch_market_data(tickers)
    metrics_df = calculate_stock_indicators(data, tickers)

    # Merge individual stock metrics with GICS sector mapping
    merged = sp_table.merge(
        metrics_df, left_on="Symbol", right_index=True, how="inner"
    )

# Compute Sector-Level Breadth Aggregations
breadth = (
    merged.groupby("GICS Sector")
    .agg(
        Count=("Symbol", "count"),
        Above_10D=("above_10d", lambda x: round(float(x.mean() * 100), 1)),
        Above_20D=("above_20d", lambda x: round(float(x.mean() * 100), 1)),
        Above_50D=("above_50d", lambda x: round(float(x.mean() * 100), 1)),
        Above_100D=("above_100d", lambda x: round(float(x.mean() * 100), 1)),
        Above_200D=("above_200d", lambda x: round(float(x.mean() * 100), 1)),
        High_4W=("new_4w_high", lambda x: round(float(x.mean() * 100), 1)),
        High_52W=("new_52w_high", lambda x: round(float(x.mean() * 100), 1)),
        Low_4W=("new_4w_low", lambda x: round(float(x.mean() * 100), 1)),
        Low_52W=("new_52w_low", lambda x: round(float(x.mean() * 100), 1)),
        RSI_Under_30=(
            "rsi",
            lambda x: round(float((x < 30).mean() * 100), 1),
        ),
        RSI_Over_70=(
            "rsi",
            lambda x: round(float((x > 70).mean() * 100), 1),
        ),
    )
    .reset_index()
)

# Sort sectors by 20-Day Moving Average participation (Rank 1 to 11)
breadth = breadth.sort_values(by="Above_20D", ascending=False).reset_index(
    drop=True
)
breadth.insert(0, "Rank", range(1, len(breadth) + 1))

# Rename columns to clean dashboard labels
column_mapping = {
    "GICS Sector": "Sector",
    "Count": "Stocks",
    "Above_10D": "> 10D SMA",
    "Above_20D": "> 20D SMA",
    "Above_50D": "> 50D SMA",
    "Above_100D": "> 100D SMA",
    "Above_200D": "> 200D SMA",
    "High_4W": "4W High",
    "High_52W": "52W High",
    "Low_4W": "4W Low",
    "Low_52W": "52W Low",
    "RSI_Under_30": "RSI < 30",
    "RSI_Over_70": "RSI > 70",
}
breadth = breadth.rename(columns=column_mapping)

# Percentage columns for formatting
pct_cols = [
    "> 10D SMA",
    "> 20D SMA",
    "> 50D SMA",
    "> 100D SMA",
    "> 200D SMA",
    "4W High",
    "52W High",
    "4W Low",
    "52W Low",
    "RSI < 30",
    "RSI > 70",
]

st.subheader("Sector Breadth Dashboard")
st.dataframe(
    breadth.style.background_gradient(
        subset=[
            "> 10D SMA",
            "> 20D SMA",
            "> 50D SMA",
            "> 100D SMA",
            "> 200D SMA",
        ],
        cmap="Blues",
        vmin=0.0,
        vmax=100.0,
    )
    .background_gradient(
        subset=["4W High", "52W High", "RSI > 70"],
        cmap="BuGn",
        vmin=0.0,
        vmax=30.0,
    )
    .background_gradient(
        subset=["4W Low", "52W Low", "RSI < 30"],
        cmap="Reds",
        vmin=0.0,
        vmax=30.0,
    )
    .format({col: "{:.1f}%" for col in pct_cols}),
    use_container_width=True,
    height=450,
    hide_index=True,
)

st.divider()

# Sector Drill-Down Section
st.subheader("Top Sector Performers & Volume Leaders")
selected_sector = st.selectbox(
    "Select Sector to Inspect:", sorted(merged["GICS Sector"].unique())
)

sector_stocks = merged[merged["GICS Sector"] == selected_sector]

col1, col2 = st.columns(2)

with col1:
    st.markdown("##### 🚀 Top 5 Performers (5-Day Return)")
    top_5d = sector_stocks.sort_values(by="ret_5d", ascending=False).head(5)[
        ["Symbol", "Security", "last_price", "ret_1d", "ret_5d", "rvol", "rsi"]
    ]
    top_5d.columns = [
        "Ticker",
        "Company",
        "Price ($)",
        "1D %",
        "5D %",
        "RVOL",
        "RSI",
    ]
    st.dataframe(
        top_5d.style.format(
            {
                "Price ($)": "${:.2f}",
                "1D %": "{:+.2f}%",
                "5D %": "{:+.2f}%",
                "RVOL": "{:.2f}x",
                "RSI": "{:.1f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

with col2:
    st.markdown("##### 📊 Top 5 Relative Volume Leaders (RVOL)")
    top_rvol = sector_stocks.sort_values(by="rvol", ascending=False).head(5)[
        ["Symbol", "Security", "last_price", "ret_1d", "ret_5d", "rvol", "rsi"]
    ]
    top_rvol.columns = [
        "Ticker",
        "Company",
        "Price ($)",
        "1D %",
        "5D %",
        "RVOL",
        "RSI",
    ]
    st.dataframe(
        top_rvol.style.format(
            {
                "Price ($)": "${:.2f}",
                "1D %": "{:+.2f}%",
                "5D %": "{:+.2f}%",
                "RVOL": "{:.2f}x",
                "RSI": "{:.1f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )