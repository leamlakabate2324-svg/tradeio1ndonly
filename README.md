# TradeAI starter

A local Streamlit futures research dashboard based on the supplied EMA/RSI/ATR analyzer.

## Run locally

1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Create and activate a virtual environment (optional but recommended).
4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Start the website:

   ```bash
   streamlit run app.py
   ```

Streamlit will print a local URL, usually http://localhost:8501.

## Data options

- Demo data: synthetic prices for checking the interface only.
- Yahoo Finance: uses continuous futures symbols where available. Yahoo data may be delayed, incomplete, or unavailable for some contracts and intervals.
- CSV upload: supply OHLC data with Open, High, Low, Close columns. A Date, Datetime, Timestamp, or Time column is optional.

## Important limitations

- The strategy is rule-based, not machine-learning AI.
- Backtests are hypothetical and do not place orders.
- Futures sizing uses a point-value setting and whole contracts. Verify the contract specification for the exact instrument and data feed.
- The prototype does not model margin, exchange fees, slippage, bid/ask spread, funding, session templates, or realistic contract-roll adjustments.
- It is for learning and software development, not a recommendation to trade.
