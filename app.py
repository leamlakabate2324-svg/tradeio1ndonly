import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="TradeAI", page_icon="📈", layout="wide")

FUTURES = {
    "E-mini Nasdaq-100 (NQ)": {"ticker": "NQ=F", "point_value": 20.0, "tick_size": 0.25},
    "Micro E-mini Nasdaq-100 (MNQ)": {"ticker": "MNQ=F", "point_value": 2.0, "tick_size": 0.25},
    "E-mini S&P 500 (ES)": {"ticker": "ES=F", "point_value": 50.0, "tick_size": 0.25},
    "Micro E-mini S&P 500 (MES)": {"ticker": "MES=F", "point_value": 5.0, "tick_size": 0.25},
    "E-mini Dow (YM)": {"ticker": "YM=F", "point_value": 5.0, "tick_size": 1.0},
    "Micro E-mini Dow (MYM)": {"ticker": "MYM=F", "point_value": 0.5, "tick_size": 1.0},
    "E-mini Russell 2000 (RTY)": {"ticker": "RTY=F", "point_value": 50.0, "tick_size": 0.10},
    "Crude Oil (CL)": {"ticker": "CL=F", "point_value": 1000.0, "tick_size": 0.01},
    "Micro Crude Oil (MCL)": {"ticker": "MCL=F", "point_value": 100.0, "tick_size": 0.01},
    "Gold (GC)": {"ticker": "GC=F", "point_value": 100.0, "tick_size": 0.10},
    "Micro Gold (MGC)": {"ticker": "MGC=F", "point_value": 10.0, "tick_size": 0.10},
    "Silver (SI)": {"ticker": "SI=F", "point_value": 5000.0, "tick_size": 0.005},
    "10-Year Treasury Note (ZN)": {"ticker": "ZN=F", "point_value": 1000.0, "tick_size": 0.015625},
    "Euro FX (6E)": {"ticker": "6E=F", "point_value": 125000.0, "tick_size": 0.00005},
    "Corn (ZC)": {"ticker": "ZC=F", "point_value": 50.0, "tick_size": 0.25},
    "Soybeans (ZS)": {"ticker": "ZS=F", "point_value": 50.0, "tick_size": 0.25},
    "Wheat (ZW)": {"ticker": "ZW=F", "point_value": 50.0, "tick_size": 0.25},
}

def demo_data(n=1800, seed=7):
    rng = np.random.default_rng(seed)
    close = 18000 * np.exp(np.cumsum(rng.normal(0, 0.0015, n)))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0, 0.001, n)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close}, index=idx)

def download_data(ticker, interval, period):
    import yfinance as yf
    data = yf.download(ticker, interval=interval, period=period, auto_adjust=False,
                       progress=False, threads=False)
    if data is None or data.empty:
        raise ValueError("The data provider returned no rows for this symbol/interval/period.")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data[["Open", "High", "Low", "Close"]].dropna()
    data.index = pd.to_datetime(data.index)
    return data

def normalize_csv(upload):
    df = pd.read_csv(upload)
    # Accept common names such as timestamp/date/datetime and lowercase OHLC.
    lower = {c.lower().strip(): c for c in df.columns}
    time_col = next((lower[k] for k in ("datetime", "timestamp", "date", "time") if k in lower), None)
    required = {}
    for col in ("open", "high", "low", "close"):
        if col not in lower:
            raise ValueError("CSV must contain Open, High, Low and Close columns (case-insensitive).")
        required[col.title()] = lower[col]
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.set_index(time_col)
    else:
        df.index = pd.RangeIndex(len(df), name="Bar")
    out = df[[required["Open"], required["High"], required["Low"], required["Close"]]].copy()
    out.columns = ["Open", "High", "Low", "Close"]
    out = out.apply(pd.to_numeric, errors="coerce").dropna()
    if len(out) < 60:
        raise ValueError("Please provide at least 60 valid OHLC bars.")
    return out

def add_indicators(df, fast=20, slow=50, rsi_len=14, atr_len=14):
    x = df.copy()
    x["EMA fast"] = x["Close"].ewm(span=fast, adjust=False).mean()
    x["EMA slow"] = x["Close"].ewm(span=slow, adjust=False).mean()
    delta = x["Close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/rsi_len, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/rsi_len, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    x["RSI"] = 100 - 100/(1 + rs)
    prev = x["Close"].shift()
    tr = pd.concat([(x["High"]-x["Low"]), (x["High"]-prev).abs(),
                    (x["Low"]-prev).abs()], axis=1).max(axis=1)
    x["ATR"] = tr.ewm(alpha=1/atr_len, adjust=False).mean()
    up = x["EMA fast"] > x["EMA slow"]
    down = x["EMA fast"] < x["EMA slow"]
    long_sig = up & (x["RSI"].shift() < 45) & (x["RSI"] >= 45)
    short_sig = down & (x["RSI"].shift() > 55) & (x["RSI"] <= 55)
    x["Signal"] = np.select([long_sig, short_sig], [1, -1], default=0)
    return x

def run_backtest(df, point_value, fee_per_contract, risk_pct, atr_mult, rr,
                 starting_equity, max_contracts, max_bars):
    equity = float(starting_equity)
    peak = equity
    max_dd = 0.0
    trades = []
    i = 0
    while i < len(df) - 1:
        row = df.iloc[i]
        side = int(row["Signal"]) if pd.notna(row["Signal"]) else 0
        if side == 0 or pd.isna(row["ATR"]) or row["ATR"] <= 0:
            i += 1
            continue
        entry_i = i + 1
        entry = float(df["Open"].iloc[entry_i])
        risk_points = float(atr_mult * row["ATR"])
        stop = entry - side * risk_points
        target = entry + side * risk_points * rr
        risk_cash_one = risk_points * point_value
        risk_budget = equity * risk_pct
        contracts = min(max_contracts, int(risk_budget // risk_cash_one)) if risk_cash_one > 0 else 0
        if contracts < 1:
            i += 1
            continue
        exit_price = None
        exit_i = None
        reason = "Time exit"
        end = min(entry_i + max_bars, len(df))
        for j in range(entry_i, end):
            hi, lo = float(df["High"].iloc[j]), float(df["Low"].iloc[j])
            hit_stop = lo <= stop if side == 1 else hi >= stop
            hit_target = hi >= target if side == 1 else lo <= target
            if hit_stop:  # conservative if both touched in the same bar
                exit_price, exit_i, reason = stop, j, "Stop"
                break
            if hit_target:
                exit_price, exit_i, reason = target, j, "Target"
                break
        if exit_price is None:
            exit_i = min(entry_i + max_bars - 1, len(df)-1)
            exit_price = float(df["Close"].iloc[exit_i])
        gross = (exit_price-entry) * side * point_value * contracts
        fees = 2 * fee_per_contract * contracts
        pnl = gross - fees
        before = equity
        equity += pnl
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak-equity)/peak if peak > 0 else 0)
        trades.append({"Entry time": str(df.index[entry_i]), "Side": "Long" if side == 1 else "Short",
                       "Entry": entry, "Exit": exit_price, "Contracts": contracts,
                       "Exit reason": reason, "P&L ($)": pnl,
                       "R": pnl / (risk_cash_one * contracts) if risk_cash_one else 0,
                       "Equity ($)": equity})
        i = exit_i + 1
    if not trades:
        return {"Trades": 0, "Win rate": 0, "Profit factor": 0, "Average R": 0,
                "Return": 0, "Max drawdown": 0}, pd.DataFrame()
    t = pd.DataFrame(trades)
    winners = t[t["P&L ($)"] > 0]["P&L ($)"].sum()
    losers = abs(t[t["P&L ($)"] <= 0]["P&L ($)"].sum())
    stats = {"Trades": len(t), "Win rate": (t["P&L ($)"] > 0).mean()*100,
             "Profit factor": winners/losers if losers else float("inf"),
             "Average R": t["R"].mean(),
             "Return": (equity/starting_equity-1)*100, "Max drawdown": max_dd*100}
    return stats, t

st.title("📈 TradeAI")
st.caption("Futures research dashboard · Educational simulator · No trade execution")
with st.sidebar:
    st.header("Market & data")
    source = st.radio("Data source", ["Demo data", "Yahoo Finance", "Upload OHLC CSV"])
    market_name = st.selectbox("Futures contract", list(FUTURES.keys()))
    spec = FUTURES[market_name]
    ticker = st.text_input("Yahoo symbol", spec["ticker"])
    interval = st.selectbox("Interval", ["1h", "1d", "15m", "5m"], index=0)
    period_options = ["1mo", "3mo", "6mo", "1y", "2y", "5y"]
    period = st.selectbox("History", period_options, index=3)
    upload = st.file_uploader("OHLC CSV", type=["csv"]) if source == "Upload OHLC CSV" else None
    st.divider()
    st.header("Strategy settings")
    fast = st.number_input("Fast EMA", 2, 200, 20)
    slow = st.number_input("Slow EMA", 3, 300, 50)
    atr_mult = st.slider("Stop distance (ATR)", 0.5, 5.0, 1.5, 0.1)
    rr = st.slider("Reward : risk", 0.5, 5.0, 2.0, 0.25)
    risk_pct = st.slider("Risk budget per simulated trade (%)", 0.1, 2.0, 1.0, 0.1)/100
    starting_equity = st.number_input("Starting simulated equity ($)", 1000, 1000000, 25000, 1000)
    fee = st.number_input("Fee per contract, per side ($)", 0.0, 100.0, 2.5, 0.25)
    max_contracts = st.number_input("Maximum simulated contracts", 1, 100, 5)
    max_bars = st.number_input("Maximum bars in a trade", 1, 200, 20)

try:
    if source == "Demo data":
        raw = demo_data()
        data_label = "Synthetic demo data (not a real market)"
    elif source == "Yahoo Finance":
        raw = download_data(ticker, interval, period)
        data_label = f"{ticker} · {interval} · {period} · Yahoo Finance"
    else:
        if upload is None:
            st.info("Upload a CSV with Open, High, Low and Close columns to begin.")
            st.stop()
        raw = normalize_csv(upload)
        data_label = f"Uploaded file · {len(raw):,} bars"
    data = add_indicators(raw, int(fast), int(slow))
    st.caption(data_label)
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

if len(data) < 60:
    st.error("At least 60 bars are needed for this strategy.")
    st.stop()

latest = data.iloc[-1]
trend = "UPTREND" if latest["EMA fast"] > latest["EMA slow"] else "DOWNTREND" if latest["EMA fast"] < latest["EMA slow"] else "MIXED"
signal = "LONG SETUP" if latest["Signal"] == 1 else "SHORT SETUP" if latest["Signal"] == -1 else "NO SETUP"

m1, m2, m3, m4 = st.columns(4)
m1.metric("Market trend", trend)
m2.metric("Latest close", f'{latest["Close"]:,.2f}')
m3.metric("RSI (14)", f'{latest["RSI"]:.1f}' if pd.notna(latest["RSI"]) else "—")
m4.metric("Latest signal", signal)

tab1, tab2, tab3 = st.tabs(["📊 Chart analyzer", "🧪 Backtester", "📋 Data & methodology"])
with tab1:
    chart = go.Figure()
    chart.add_trace(go.Candlestick(x=data.index, open=data["Open"], high=data["High"],
                                   low=data["Low"], close=data["Close"], name="Price"))
    chart.add_trace(go.Scatter(x=data.index, y=data["EMA fast"], name=f"EMA {fast}", line=dict(width=1.4)))
    chart.add_trace(go.Scatter(x=data.index, y=data["EMA slow"], name=f"EMA {slow}", line=dict(width=1.4)))
    longs = data[data["Signal"] == 1]
    shorts = data[data["Signal"] == -1]
    chart.add_trace(go.Scatter(x=longs.index, y=longs["Low"], mode="markers", name="Long signal",
                               marker=dict(symbol="triangle-up", size=10)))
    chart.add_trace(go.Scatter(x=shorts.index, y=shorts["High"], mode="markers", name="Short signal",
                               marker=dict(symbol="triangle-down", size=10)))
    chart.update_layout(height=620, xaxis_rangeslider_visible=False, template="plotly_dark",
                        margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
    st.plotly_chart(chart, use_container_width=True)
    rsi_chart = go.Figure()
    rsi_chart.add_trace(go.Scatter(x=data.index, y=data["RSI"], name="RSI 14"))
    rsi_chart.add_hline(y=45, line_dash="dash")
    rsi_chart.add_hline(y=55, line_dash="dash")
    rsi_chart.update_layout(height=220, template="plotly_dark", yaxis_range=[0,100], margin=dict(l=10,r=10,t=20,b=10))
    st.plotly_chart(rsi_chart, use_container_width=True)
    st.info("Signals are rule-based examples, not AI predictions. A signal is calculated at bar close; simulated entry is at the next bar open.")

with tab2:
    st.subheader("Historical strategy simulation")
    st.write(f"Contract point value used: **${spec['point_value']:,.4g} per point** · Tick size reference: **{spec['tick_size']}**")
    st.caption("For uploaded data, verify the selected contract's point value yourself. Yahoo symbols and continuous futures data may not match every broker's contract history.")
    stats, trades = run_backtest(data, spec["point_value"], fee, risk_pct, atr_mult, rr,
                                 starting_equity, int(max_contracts), int(max_bars))
    if stats["Trades"] == 0:
        st.warning("No simulated trades met the rules and sizing constraints in this dataset.")
    else:
        a,b,c,d,e,f = st.columns(6)
        a.metric("Trades", f'{stats["Trades"]}')
        b.metric("Win rate", f'{stats["Win rate"]:.1f}%')
        c.metric("Profit factor", f'{stats["Profit factor"]:.2f}')
        d.metric("Average R", f'{stats["Average R"]:.2f}')
        e.metric("Total return", f'{stats["Return"]:.1f}%')
        f.metric("Max drawdown", f'{stats["Max drawdown"]:.1f}%')
        eq = go.Figure()
        eq.add_trace(go.Scatter(y=trades["Equity ($)"], mode="lines", name="Simulated equity"))
        eq.update_layout(height=300, template="plotly_dark", xaxis_title="Trade number", yaxis_title="Equity ($)")
        st.plotly_chart(eq, use_container_width=True)
        st.dataframe(trades, use_container_width=True, hide_index=True)
        st.download_button("Download trade history CSV", trades.to_csv(index=False), "tradeai_backtest.csv", "text/csv")
    st.warning("Backtests are hypothetical. They can be distorted by data quality, contract rolls, slippage, fees, session settings and assumptions about intrabar price order. This tool does not connect to a broker or place trades.")

with tab3:
    st.subheader("How this version works")
    st.markdown("""
    - **Trend:** fast EMA above slow EMA = upward trend; below = downward trend.
    - **Long setup:** upward trend and RSI crosses back above 45.
    - **Short setup:** downward trend and RSI crosses back below 55.
    - **Stop:** ATR multiplied by the selected stop distance.
    - **Target:** stop distance multiplied by the reward:risk setting.
    - **Entry assumption:** next bar open after the signal bar closes.
    - **Sizing:** whole contracts, based on risk budget and contract point value, capped by the maximum contracts setting.
    - **Costs:** a flat fee per contract per side. Slippage and exchange fees are not separately modeled.
    """)
    st.subheader("Data format for CSV upload")
    st.code("Date,Open,High,Low,Close\\n2025-01-02 09:30,21000,21010,20995,21005\\n...")
    st.caption("The app is a research prototype. Validate symbols, contract specifications, data adjustments and costs before interpreting results.")
