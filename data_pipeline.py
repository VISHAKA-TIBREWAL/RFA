"""
data_pipeline.py
Fetches NIFTY Bank index data, individual bank stocks, VIX, USD/INR
Uses yfinance for real market data
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


# ─── CONFIG ────────────────────────────────────────────────────────────────────

BANK_STOCKS = {
    "HDFCBANK":  "HDFCBANK.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "SBIN":      "SBIN.NS",
    "AXISBANK":  "AXISBANK.NS",
    "KOTAKBANK": "KOTAKBANK.NS",
    "INDUSINDBK":"INDUSINDBK.NS",
    "BANKBARODA": "BANKBARODA.NS",
    "PNB":       "PNB.NS",
    "IDFCFIRSTB":"IDFCFIRSTB.NS",
    "FEDERALBNK":"FEDERALBNK.NS",
}

INDICES = {
    "NIFTY_BANK": "^NSEBANK",
    "NIFTY_50":   "^NSEI",
    "VIX":        "^INDIAVIX",
    "USD_INR":    "USDINR=X",
}

DEFAULT_START = "2019-01-01"
DEFAULT_END   = datetime.today().strftime("%Y-%m-%d")


# ─── FETCH FUNCTIONS ────────────────────────────────────────────────────────────

def fetch_ticker(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV for a single ticker."""
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            print(f"  [WARN] No data for {ticker}")
            return pd.DataFrame()
        # Handle MultiIndex columns (yfinance issue)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [str(c).lower() for c in df.columns]
        return df
    except Exception as e:
        print(f"  [ERROR] {ticker}: {e}")
        return pd.DataFrame()


def fetch_all_data(start: str = DEFAULT_START, end: str = DEFAULT_END) -> dict:
    """
    Returns a dict with keys:
      'indices'  → DataFrame (Date index, columns: nifty_bank, nifty_50, vix, usd_inr)
      'stocks'   → dict of {name: DataFrame}
    """
    print("=" * 60)
    print("📥 Fetching market data...")
    print("=" * 60)

    # ── Indices ──
    idx_frames = {}
    for name, ticker in INDICES.items():
        print(f"  Fetching {name} ({ticker})...")
        df = fetch_ticker(ticker, start, end)
        if not df.empty:
            idx_frames[name.lower()] = df["close"].rename(name.lower())

    indices_df = pd.DataFrame(idx_frames)
    indices_df.index = pd.to_datetime(indices_df.index)
    indices_df.dropna(how="all", inplace=True)

    # ── Individual Stocks ──
    stocks = {}
    for name, ticker in BANK_STOCKS.items():
        print(f"  Fetching {name} ({ticker})...")
        df = fetch_ticker(ticker, start, end)
        if not df.empty:
            df.index = pd.to_datetime(df.index)
            stocks[name] = df

    print(f"\n✅ Fetched indices shape: {indices_df.shape}")
    print(f"✅ Fetched {len(stocks)} bank stocks")
    return {"indices": indices_df, "stocks": stocks}


def build_returns(indices_df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log-returns for all index columns."""
    rets = np.log(indices_df / indices_df.shift(1)).dropna()
    rets.columns = [c + "_ret" for c in rets.columns]
    return rets


def build_stock_returns(stocks: dict) -> pd.DataFrame:
    """Compute daily close-price returns for all bank stocks."""
    closes = {}
    for name, df in stocks.items():
        if "close" in df.columns:
            closes[name] = df["close"]
    close_df = pd.DataFrame(closes)
    close_df.index = pd.to_datetime(close_df.index)
    rets = np.log(close_df / close_df.shift(1)).dropna()
    return rets


def align_data(indices_df: pd.DataFrame, stock_rets: pd.DataFrame) -> pd.DataFrame:
    """Merge index data and stock returns on common trading dates."""
    combined = indices_df.join(stock_rets, how="inner")
    combined.sort_index(inplace=True)
    return combined


# ─── MAIN (DEMO) ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data = fetch_all_data()
    idx  = data["indices"]
    stk  = data["stocks"]

    print("\n📊 Indices sample:")
    print(idx.tail(5))

    rets = build_returns(idx)
    stock_rets = build_stock_returns(stk)

    combined = align_data(idx, stock_rets)
    print(f"\n📊 Combined shape: {combined.shape}")
    print(combined.tail(3))

    # Save for next steps
    idx.to_csv("indices_raw.csv")
    combined.to_csv("combined_data.csv")
    print("\n💾 Saved: indices_raw.csv, combined_data.csv")
