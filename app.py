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
