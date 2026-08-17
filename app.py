import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

# Configure page layout
st.set_page_config(
    page_title="Institutional Sector Breadth & Alpha Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sector ETF Benchmark Mapping
SECTOR_ETFS = {
    "Information Technology": "XLK",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Consumer Discretionary": "XLY",
    "Industrials": "XLI",
    "Communication Services": "XLC",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Materials": "XLB",
}


@st.cache_data(ttl=3600)
def get_sp500_constituents():
    """Scrape S&P 500 constituents and GICS sectors."""
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
def fetch_all_market_data(tickers):
    """Download 1 year of daily historical data for all stocks + SPY + Sector ETFs."""
    all_symbols = list(
        set(tickers + list(SECTOR_ETFS.values()) + ["SPY", "^VIX"])
    )
    data = yf.download(
        all_symbols,
        period="1y",
        group_by="ticker",
        auto_adjust=True,
        progress=False,
    )
    return data


def extract_price_series(df, ticker):
    """Safely extract clean Close, High, Low, Volume series for any ticker."""
    try:
        if isinstance(df.columns, pd.MultiIndex):
            if ticker in df.columns.levels[0]:
                sub = df[ticker].dropna(how="all")
            elif ticker in df.columns.levels[1]:
                sub = df.xs(ticker, axis=1, level=1).dropna(how="all")
            else:
                return None
        else:
            sub = df.dropna(how="all")

        if len(sub) < 30:
            return None

        close = sub["Close"].dropna()
        high = sub["High"].dropna() if "High" in sub.columns else close
        low = sub["Low"].dropna() if "Low" in sub.columns else close
        volume = (
            sub["Volume"].dropna()
            if "Volume" in sub.columns
            else pd.Series(1, index=close.index)
        )
        return {"close": close, "high": high, "low": low, "volume": volume}
    except Exception:
        return None


def calculate_stock_metrics(df, tickers, spy_close):
    """Calculate breadth, momentum, high/low breakouts, and Relative Strength vs SPY."""
    results = {}

    spy_ret_1m = (
        (spy_close.iloc[-1] / spy_close.iloc[-22] - 1) * 100
        if len(spy_close) >= 22
        else 0.0
    )
    spy_ret_3m = (
        (spy_close.iloc[-1] / spy_close.iloc[-63] - 1) * 100
        if len(spy_close) >= 63
        else 0.0
    )

    for ticker in tickers:
        series = extract_price_series(df, ticker)
        if not series:
            continue

        close = series["close"]
        high = series["high"]
        low = series["low"]
        volume = series["volume"]

        if len(close) < 50:
            continue

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

        # Highs / Lows
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

        # RSI(14)
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

        # Performance Returns
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
        ret_1m = (
            float((close.iloc[-1] / close.iloc[-22] - 1) * 100)
            if len(close) >= 22
            else 0.0
        )
        ret_3m = (
            float((close.iloc[-1] / close.iloc[-63] - 1) * 100)
            if len(close) >= 63
            else 0.0
        )

        # Alpha vs SPY (Excess Return)
        alpha_1m = ret_1m - spy_ret_1m
        alpha_3m = ret_3m - spy_ret_3m

        # RVOL (20-Day Average Volume Ratio)
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

        # Trend Template Score
        trend_score = sum(
            [
                last_price > sma20,
                last_price > sma50,
                last_price > sma200,
                sma20 > sma50,
                sma50 > sma200,
            ]
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
            "ret_1m": ret_1m,
            "ret_3m": ret_3m,
            "alpha_1m": alpha_1m,
            "alpha_3m": alpha_3m,
            "rvol": rvol,
            "trend_score": trend_score,
            "sma20": sma20,
            "sma50": sma50,
            "sma200": sma200,
        }

    return pd.DataFrame.from_dict(results, orient="index")


def compute_sector_etf_performance(df, spy_close):
    """Compute performance and RS metrics for all 11 GICS Sector ETFs vs SPY."""
    sector_data = []
    spy_ret_1w = (spy_close.iloc[-1] / spy_close.iloc[-6] - 1) * 100
    spy_ret_1m = (spy_close.iloc[-1] / spy_close.iloc[-22] - 1) * 100
    spy_ret_3m = (spy_close.iloc[-1] / spy_close.iloc[-63] - 1) * 100

    for sector, etf in SECTOR_ETFS.items():
        series = extract_price_series(df, etf)
        if not series:
            continue
        c = series["close"]
        r_1w = (c.iloc[-1] / c.iloc[-6] - 1) * 100
        r_1m = (c.iloc[-1] / c.iloc[-22] - 1) * 100
        r_3m = (c.iloc[-1] / c.iloc[-63] - 1) * 100

        # Relative Strength Ratio (ETF / SPY)
        rs_ratio = (c / spy_close).dropna()
        rs_10d_ma = rs_ratio.rolling(10).mean().iloc[-1]
        rs_current = rs_ratio.iloc[-1]
        rs_momentum = (rs_current / rs_10d_ma - 1) * 100

        # Determine Quadrant
        if r_1m >= spy_ret_1m and rs_momentum >= 0:
            quadrant = "🟢 Leading"
        elif r_1m >= spy_ret_1m and rs_momentum < 0:
            quadrant = "🟡 Weakening"
        elif r_1m < spy_ret_1m and rs_momentum >= 0:
            quadrant = "🔵 Improving"
        else:
            quadrant = "🔴 Lagging"

        sector_data.append(
            {
                "Sector": sector,
                "ETF": etf,
                "Price": c.iloc[-1],
                "1W %": r_1w,
                "1M %": r_1m,
                "3M %": r_3m,
                "Alpha 1M": r_1m - spy_ret_1m,
                "Alpha 3M": r_3m - spy_ret_3m,
                "RS Momentum": rs_momentum,
                "Rotation Phase": quadrant,
            }
        )

    return pd.DataFrame(sector_data)


# --- DATA PIPELINE INGESTION ---

with st.spinner("Downloading live institutional market feeds & calculating alpha..."):
    sp_table = get_sp500_constituents()
    all_tickers = sp_table["Symbol"].tolist()
    raw_data = fetch_all_market_data(all_tickers)

    spy_series = extract_price_series(raw_data, "SPY")
    spy_close = spy_series["close"] if spy_series else None

    if spy_close is not None:
        stock_metrics_df = calculate_stock_metrics(
            raw_data, all_tickers, spy_close
        )
        merged_df = sp_table.merge(
            stock_metrics_df, left_on="Symbol", right_index=True, how="inner"
        )
        sector_perf_df = compute_sector_etf_performance(raw_data, spy_close)
    else:
        st.error(
            "Error downloading SPY benchmark data. Please refresh the page."
        )
        st.stop()


# --- SIDEBAR ---

st.sidebar.title("⚡ Portfolio Action Center")

spy_last = spy_close.iloc[-1]
spy_chg_1d = (spy_close.iloc[-1] / spy_close.iloc[-2] - 1) * 100
st.sidebar.metric("S&P 500 (SPY)", f"${spy_last:.2f}", f"{spy_chg_1d:+.2f}%")

vix_series = extract_price_series(raw_data, "^VIX")
if vix_series:
    vix_last = vix_series["close"].iloc[-1]
    vix_chg = (
        vix_series["close"].iloc[-1] / vix_series["close"].iloc[-2] - 1
    ) * 100
    st.sidebar.metric("Volatility (VIX)", f"{vix_last:.2f}", f"{vix_chg:+.2f}%")

st.sidebar.divider()

pct_stocks_above_50d = (merged_df["above_50d"].mean()) * 100
pct_stocks_above_200d = (merged_df["above_200d"].mean()) * 100

st.sidebar.markdown("### 🧭 Market Breadth Regime")
if pct_stocks_above_50d > 60 and pct_stocks_above_200d > 60:
    regime = "🟢 Bullish Expansion"
    advice = "Favor aggressive trend continuation, LEAPS, & covered calls in Leading sectors."
elif pct_stocks_above_50d < 40 and pct_stocks_above_200d < 50:
    regime = "🔴 Bearish Contraction"
    advice = "Raise cash, trim lagging sectors, sell defensive cash-secured puts on dips."
else:
    regime = "🟡 Neutral / Choppy Rotation"
    advice = "Focus on relative strength leaders with strong RVOL; avoid broad index beta."

st.sidebar.info(f"**Regime:** {regime}\n\n**Action:** {advice}")
st.sidebar.markdown(f"* S&P 500 > 50D SMA: **{pct_stocks_above_50d:.1f}%**")
st.sidebar.markdown(f"* S&P 500 > 200D SMA: **{pct_stocks_above_200d:.1f}%**")


# --- MAIN WORKSPACE ---

st.title("🏛️ Institutional S&P 500 Sector & Alpha Dashboard")
st.caption(
    "Real-time money flow rotation, relative strength alpha vs. SPY, and stock breakout radar."
)

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Sector Breadth Matrix",
        "🔄 Relative Strength & Rotation",
        "🎯 Alpha Outperformer Radar",
        "🔍 Single Stock & Sector Deep Dive",
    ]
)

# --- TAB 1: BREADTH MATRIX ---
with tab1:
    st.subheader("S&P 500 Large Cap Breadth Matrix")
    st.caption(
        "Constituent participation, momentum extremes, and multi-timeframe moving averages."
    )

    breadth = (
        merged_df.groupby("GICS Sector")
        .agg(
            Stocks=("Symbol", "count"),
            Above_10D=("above_10d", lambda x: round(float(x.mean() * 100), 1)),
            Above_20D=("above_20d", lambda x: round(float(x.mean() * 100), 1)),
            Above_50D=("above_50d", lambda x: round(float(x.mean() * 100), 1)),
            Above_100D=(
                "above_100d",
                lambda x: round(float(x.mean() * 100), 1),
            ),
            Above_200D=(
                "above_200d",
                lambda x: round(float(x.mean() * 100), 1),
            ),
            High_4W=("new_4w_high", lambda x: round(float(x.mean() * 100), 1)),
            High_52W=(
                "new_52w_high",
                lambda x: round(float(x.mean() * 100), 1),
            ),
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

    breadth = breadth.sort_values(by="Above_20D", ascending=False).reset_index(
        drop=True
    )
    breadth.insert(0, "Rank", range(1, len(breadth) + 1))

    column_mapping = {
        "GICS Sector": "Sector",
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

# --- TAB 2: RELATIVE STRENGTH & ROTATION ---
with tab2:
    st.subheader("Sector Relative Strength vs. S&P 500 (SPY)")
    st.caption(
        "Identifies where institutional money is actively flowing in or out over multiple time horizons."
    )

    c1, c2 = st.columns([1.2, 1])

    with c1:
        st.markdown("##### 🏆 Sector Performance & Alpha Matrix")
        styled_sector_df = sector_perf_df.sort_values(
            by="Alpha 1M", ascending=False
        )
        st.dataframe(
            styled_sector_df.style.background_gradient(
                subset=["Alpha 1M", "Alpha 3M", "RS Momentum"],
                cmap="RdYlGn",
                vmin=-5.0,
                vmax=5.0,
            ).format(
                {
                    "Price": "${:.2f}",
                    "1W %": "{:+.2f}%",
                    "1M %": "{:+.2f}%",
                    "3M %": "{:+.2f}%",
                    "Alpha 1M": "{:+.2f}%",
                    "Alpha 3M": "{:+.2f}%",
                    "RS Momentum": "{:+.2f}%",
                }
            ),
            use_container_width=True,
            hide_index=True,
            height=430,
        )

    with c2:
        st.markdown("##### 🧭 Sector Rotation Quadrants (RRG Style)")
        fig = px.scatter(
            sector_perf_df,
            x="Alpha 1M",
            y="RS Momentum",
            text="ETF",
            color="Rotation Phase",
            color_discrete_map={
                "🟢 Leading": "#00C853",
                "🟡 Weakening": "#FFD600",
                "🔵 Improving": "#2979FF",
                "🔴 Lagging": "#D50000",
            },
            size=[14] * len(sector_perf_df),
            hover_name="Sector",
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.update_traces(textposition="top center")
        fig.update_layout(
            height=430,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="1-Month Alpha vs SPY (%)",
            yaxis_title="RS Momentum / Trend (%)",
        )
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 3: ALPHA OUTPERFORMER RADAR ---
with tab3:
    st.subheader("🎯 Alpha Radar: Top Stocks Outperforming the S&P 500")
    st.caption(
        "Screen high-momentum institutional leaders trading in uptrends with active volume expansion."
    )

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        sector_filter = st.selectbox(
            "Filter by Sector:",
            ["All Sectors"] + sorted(merged_df["GICS Sector"].unique()),
        )
    with f2:
        min_alpha_1m = st.slider("Min 1M Alpha vs SPY (%):", -5.0, 20.0, 3.0)
    with f3:
        min_rvol = st.slider("Min Relative Volume (RVOL):", 0.5, 3.0, 1.0)
    with f4:
        trend_template_only = st.checkbox(
            "Uptrend Only (> 20D, 50D, 200D)", value=True
        )

    filtered = merged_df.copy()
    if sector_filter != "All Sectors":
        filtered = filtered[filtered["GICS Sector"] == sector_filter]

    filtered = filtered[
        (filtered["alpha_1m"] >= min_alpha_1m) & (filtered["rvol"] >= min_rvol)
    ]

    if trend_template_only:
        filtered = filtered[
            (filtered["above_20d"])
            & (filtered["above_50d"])
            & (filtered["above_200d"])
        ]

    def classify_setup(row):
        if row["new_4w_high"] and row["rvol"] >= 1.3:
            return "🔥 Volume Breakout"
        elif (
            abs(row["last_price"] - row["sma20"]) / row["sma20"] < 0.02
            and row["rsi"] < 60
        ):
            return "🎯 20D SMA Pullback"
        elif row["trend_score"] == 5 and row["alpha_3m"] > 10:
            return "💎 Stage 2 Leader"
        elif row["rsi"] > 70:
            return "⚡ Momentum Stretch"
        else:
            return "📈 Steady Uptrend"

    if not filtered.empty:
        filtered["Trade Setup"] = filtered.apply(classify_setup, axis=1)

        display_cols = [
            "Symbol",
            "Security",
            "GICS Sector",
            "Trade Setup",
            "last_price",
            "ret_1d",
            "ret_5d",
            "alpha_1m",
            "alpha_3m",
            "rvol",
            "rsi",
            "trend_score",
        ]
        out_df = filtered[display_cols].sort_values(
            by="alpha_1m", ascending=False
        )
        out_df.columns = [
            "Ticker",
            "Company",
            "Sector",
            "Setup Type",
            "Price ($)",
            "1D %",
            "5D %",
            "1M Alpha",
            "3M Alpha",
            "RVOL",
            "RSI",
            "Trend Score (0-5)",
        ]

        st.markdown(f"**Found {len(out_df)} Outperforming Leaders:**")
        st.dataframe(
            out_df.style.background_gradient(
                subset=["1M Alpha", "3M Alpha"], cmap="Greens", vmin=0, vmax=25
            )
            .background_gradient(subset=["RVOL"], cmap="Blues", vmin=1.0, vmax=3.0)
            .format(
                {
                    "Price ($)": "${:.2f}",
                    "1D %": "{:+.2f}%",
                    "5D %": "{:+.2f}%",
                    "1M Alpha": "{:+.2f}%",
                    "3M Alpha": "{:+.2f}%",
                    "RVOL": "{:.2f}x",
                    "RSI": "{:.1f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
            height=480,
        )

        csv_data = out_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Outperforming Stocks to CSV",
            data=csv_data,
            file_name="SP500_Alpha_Outperformers.csv",
            mime="text/csv",
        )
    else:
        st.warning(
            "No stocks match the selected filter criteria. Try lowering the alpha or RVOL threshold."
        )

# --- TAB 4: DEEP DIVE ---
with tab4:
    st.subheader("🔍 Deep Dive Technical Checklist")
    selected_ticker = st.selectbox(
        "Select Ticker to Inspect:",
        sorted(merged_df["Symbol"].unique()),
        index=0,
    )

    stock_row = merged_df[merged_df["Symbol"] == selected_ticker].iloc[0]
    stock_series = extract_price_series(raw_data, selected_ticker)

    if stock_series:
        c_series = stock_series["close"]

        d1, d2, d3, d4, d5 = st.columns(5)
        d1.metric("Current Price", f"${stock_row['last_price']:.2f}")
        d2.metric("1M Alpha vs SPY", f"{stock_row['alpha_1m']:+.2f}%")
        d3.metric("3M Alpha vs SPY", f"{stock_row['alpha_3m']:+.2f}%")
        d4.metric("RVOL (20D)", f"{stock_row['rvol']:.2f}x")
        d5.metric("RSI(14)", f"{stock_row['rsi']:.1f}")

        chart_fig = go.Figure()
        chart_fig.add_trace(
            go.Scatter(
                x=c_series.index[-120:],
                y=c_series.iloc[-120:],
                name="Close Price",
                line=dict(color="#2962FF", width=2),
            )
        )
        chart_fig.add_trace(
            go.Scatter(
                x=c_series.index[-120:],
                y=c_series.rolling(20).mean().iloc[-120:],
                name="20D SMA",
                line=dict(color="#FF6D00", width=1.5),
            )
        )
        chart_fig.add_trace(
            go.Scatter(
                x=c_series.index[-120:],
                y=c_series.rolling(50).mean().iloc[-120:],
                name="50D SMA",
                line=dict(color="#00C853", width=1.5),
            )
        )
        chart_fig.add_trace(
            go.Scatter(
                x=c_series.index[-120:],
                y=c_series.rolling(200).mean().iloc[-120:],
                name="200D SMA",
                line=dict(color="#D50000", width=1.5),
            )
        )

        chart_fig.update_layout(
            title=f"{selected_ticker} ({stock_row['Security']}) - Price vs Key Moving Averages (Past 6 Months)",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=420,
            margin=dict(l=20, r=20, t=40, b=20),
            hovermode="x unified",
        )
        st.plotly_chart(chart_fig, use_container_width=True)

        st.markdown("##### 📋 Technical Health Checklist")
        chk1 = "✅" if stock_row["above_20d"] else "❌"
        chk2 = "✅" if stock_row["above_50d"] else "❌"
        chk3 = "✅" if stock_row["above_200d"] else "❌"
        chk4 = "✅" if stock_row["alpha_1m"] > 0 else "❌"
        chk5 = "✅" if 40 <= stock_row["rsi"] <= 70 else "⚠️"

        ch_col1, ch_col2 = st.columns(2)
        ch_col1.markdown(f"- {chk1} **Above 20D SMA:** Short-term trend support")
        ch_col1.markdown(
            f"- {chk2} **Above 50D SMA:** Intermediate institutional trend"
        )
        ch_col1.markdown(
            f"- {chk3} **Above 200D SMA:** Long-term bull market filter"
        )
        ch_col2.markdown(
            f"- {chk4} **1-Month Alpha Positive:** Generating excess return vs SPY"
        )
        ch_col2.markdown(
            f"- {chk5} **RSI In Health Zone (40-70):** Not severely overbought or broken down"
        )