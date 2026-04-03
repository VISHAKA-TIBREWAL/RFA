"""
feature_engineering.py
Computes 30+ features from raw market data for NIFTY Banking Risk System.

Feature Groups:
  1. Volatility          (rolling std, EWMA vol, vol-of-vol)
  2. Drawdown            (rolling max drawdown, underwater duration)
  3. Volume Stress       (z-score, spike detection)
  4. Correlation         (bank vs market, cross-bank, rolling)
  5. Momentum / Trend    (RSI, rate-of-change, risk velocity)
  6. External Macro      (VIX level + change, USD/INR + change)
  7. Tail Risk           (CVaR, skewness, kurtosis)
  8. Relative Risk       (bank vs broad market spread)
"""

import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════════════
# 1 ── VOLATILITY FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

def rolling_volatility(returns: pd.Series, windows=(5, 10, 20)) -> pd.DataFrame:
    """Annualised rolling standard deviation for multiple windows."""
    frames = {}
    for w in windows:
        frames[f"vol_{w}d"] = returns.rolling(w).std() * np.sqrt(252)
    return pd.DataFrame(frames, index=returns.index)


def ewma_volatility(returns: pd.Series, span=20) -> pd.Series:
    """Exponentially weighted moving average volatility."""
    return (returns.ewm(span=span).std() * np.sqrt(252)).rename("ewma_vol")


def vol_of_vol(returns: pd.Series, window=20, inner=5) -> pd.Series:
    """Volatility of short-window volatility — measures instability."""
    short_vol = returns.rolling(inner).std()
    return short_vol.rolling(window).std().rename("vol_of_vol")


# ═══════════════════════════════════════════════════════════════════════════════
# 2 ── DRAWDOWN FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

def rolling_drawdown(price: pd.Series, window=60) -> pd.DataFrame:
    """
    Computes:
      max_drawdown   : worst peak-to-trough over window
      underwater_dur : consecutive days below rolling max
    """
    roll_max = price.rolling(window, min_periods=1).max()
    dd       = (price - roll_max) / roll_max               # always ≤ 0

    # Underwater duration (streak counter)
    is_under = (dd < 0).astype(int)
    streak   = is_under * (is_under.groupby((is_under != is_under.shift()).cumsum()).cumcount() + 1)

    return pd.DataFrame({
        "max_drawdown":    dd,
        "underwater_dur":  streak,
    }, index=price.index)


def current_drawdown(price: pd.Series) -> pd.Series:
    """Current distance from all-time high."""
    peak = price.cummax()
    return ((price - peak) / peak).rename("current_drawdown")


# ═══════════════════════════════════════════════════════════════════════════════
# 3 ── VOLUME STRESS
# ═══════════════════════════════════════════════════════════════════════════════

def volume_stress(volume: pd.Series, window=20) -> pd.DataFrame:
    """
    vol_zscore  : z-score of volume vs rolling window
    vol_spike   : binary flag (z > 2)
    """
    mu    = volume.rolling(window).mean()
    sigma = volume.rolling(window).std()
    z     = (volume - mu) / (sigma + 1e-9)

    return pd.DataFrame({
        "vol_zscore": z,
        "vol_spike":  (z > 2.0).astype(int),
    }, index=volume.index)


# ═══════════════════════════════════════════════════════════════════════════════
# 4 ── CORRELATION FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

def bank_market_corr(bank_rets: pd.Series, market_rets: pd.Series, window=30) -> pd.Series:
    """Rolling correlation of bank index with broader NIFTY 50."""
    return bank_rets.rolling(window).corr(market_rets).rename("bank_market_corr")


def cross_bank_corr(stock_rets: pd.DataFrame, window=20) -> pd.Series:
    """
    Average pairwise rolling correlation across all bank stocks.
    High value → contagion / systemic risk.
    """
    n = len(stock_rets.columns)
    corr_sum = pd.Series(0.0, index=stock_rets.index)
    count    = 0
    cols     = stock_rets.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            corr_sum += stock_rets[cols[i]].rolling(window).corr(stock_rets[cols[j]])
            count     += 1
    avg_corr = corr_sum / max(count, 1)
    return avg_corr.rename("cross_bank_corr")


def correlation_trend(corr_series: pd.Series, window=10) -> pd.Series:
    """Rate of change in correlation — rising corr = warning."""
    return corr_series.diff(window).rename("corr_trend")


# ═══════════════════════════════════════════════════════════════════════════════
# 5 ── MOMENTUM / TREND
# ═══════════════════════════════════════════════════════════════════════════════

def rsi(price: pd.Series, window=14) -> pd.Series:
    """Relative Strength Index."""
    delta  = price.diff()
    gain   = delta.clip(lower=0).rolling(window).mean()
    loss   = (-delta.clip(upper=0)).rolling(window).mean()
    rs     = gain / (loss + 1e-9)
    return (100 - 100 / (1 + rs)).rename("rsi")


def rate_of_change(price: pd.Series, window=5) -> pd.Series:
    """% change over window — momentum signal."""
    return ((price / price.shift(window)) - 1).rename(f"roc_{window}d")


def moving_avg_spread(price: pd.Series, short=10, long=50) -> pd.Series:
    """
    (short MA - long MA) / long MA
    Negative → downtrend → stress
    """
    ma_s = price.rolling(short).mean()
    ma_l = price.rolling(long).mean()
    return ((ma_s - ma_l) / (ma_l + 1e-9)).rename("ma_spread")


# ═══════════════════════════════════════════════════════════════════════════════
# 6 ── EXTERNAL MACRO (VIX + USD/INR)
# ═══════════════════════════════════════════════════════════════════════════════

def vix_features(vix: pd.Series, window=20) -> pd.DataFrame:
    """
    vix_level   : raw VIX
    vix_change  : 1-day change
    vix_zscore  : z-score vs rolling window
    vix_spike   : binary flag
    """
    z = (vix - vix.rolling(window).mean()) / (vix.rolling(window).std() + 1e-9)
    return pd.DataFrame({
        "vix_level":  vix,
        "vix_change": vix.diff(),
        "vix_zscore": z,
        "vix_spike":  (z > 1.5).astype(int),
    }, index=vix.index)


def fx_features(usd_inr: pd.Series, window=20) -> pd.DataFrame:
    """
    fx_level    : USD/INR rate
    fx_change   : 1-day change
    fx_zscore   : z-score (large rise = capital outflow risk)
    """
    z = (usd_inr - usd_inr.rolling(window).mean()) / (usd_inr.rolling(window).std() + 1e-9)
    return pd.DataFrame({
        "fx_level":   usd_inr,
        "fx_change":  usd_inr.diff(),
        "fx_zscore":  z,
    }, index=usd_inr.index)


# ═══════════════════════════════════════════════════════════════════════════════
# 7 ── TAIL RISK
# ═══════════════════════════════════════════════════════════════════════════════

def cvar(returns: pd.Series, window=60, alpha=0.05) -> pd.Series:
    """
    Conditional Value at Risk (Expected Shortfall).
    Average return in worst alpha% of days.
    More negative = higher tail risk.
    """
    def _cvar(x):
        q = np.percentile(x, alpha * 100)
        tail = x[x <= q]
        return tail.mean() if len(tail) > 0 else np.nan

    return returns.rolling(window).apply(_cvar, raw=True).rename("cvar_5pct")


def rolling_skew_kurt(returns: pd.Series, window=30) -> pd.DataFrame:
    """
    skewness  : negative = left tail (crash risk)
    kurtosis  : fat tails = extreme events
    """
    sk = returns.rolling(window).apply(lambda x: skew(x), raw=True)
    kt = returns.rolling(window).apply(lambda x: kurtosis(x), raw=True)
    return pd.DataFrame({
        "return_skew": sk,
        "return_kurt": kt,
    }, index=returns.index)


# ═══════════════════════════════════════════════════════════════════════════════
# 8 ── RELATIVE RISK (BANK vs MARKET)
# ═══════════════════════════════════════════════════════════════════════════════

def relative_risk(bank_rets: pd.Series, market_rets: pd.Series, window=20) -> pd.DataFrame:
    """
    rel_return  : bank return - market return (outperformance)
    beta        : market sensitivity
    rel_vol     : bank vol - market vol (excess risk)
    """
    bank_vol   = bank_rets.rolling(window).std()
    market_vol = market_rets.rolling(window).std()

    cov_bm = bank_rets.rolling(window).cov(market_rets)
    var_m  = market_rets.rolling(window).var()
    beta   = cov_bm / (var_m + 1e-9)

    rel_ret = (bank_rets.rolling(window).mean() - market_rets.rolling(window).mean()) * 252

    return pd.DataFrame({
        "bank_rel_return": rel_ret,
        "bank_beta":       beta,
        "bank_rel_vol":    bank_vol - market_vol,
    }, index=bank_rets.index)


# ═══════════════════════════════════════════════════════════════════════════════
# 9 ── STOCK-LEVEL FEATURES (per individual bank)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_stock_features(stock_rets: pd.Series, index_rets: pd.Series,
                            window=20) -> pd.DataFrame:
    """
    Per-stock feature set used for BUY / HOLD / AVOID recommendation.
    """
    mu      = stock_rets.rolling(window).mean() * 252
    sig     = stock_rets.rolling(window).std() * np.sqrt(252)
    sharpe  = mu / (sig + 1e-9)

    cov_si  = stock_rets.rolling(window).cov(index_rets)
    var_i   = index_rets.rolling(window).var()
    beta    = cov_si / (var_i + 1e-9)

    corr_si = stock_rets.rolling(window).corr(index_rets)
    roc     = (stock_rets.rolling(window).sum())            # cumulative return over window

    return pd.DataFrame({
        "annualised_return": mu,
        "volatility":        sig,
        "sharpe":            sharpe,
        "beta":              beta,
        "corr_with_index":   corr_si,
        "momentum":          roc,
    }, index=stock_rets.index)


# ═══════════════════════════════════════════════════════════════════════════════
# MASTER FUNCTION — builds ALL features at once
# ═══════════════════════════════════════════════════════════════════════════════

def build_all_features(indices_df: pd.DataFrame,
                       stock_rets: pd.DataFrame) -> pd.DataFrame:
    """
    indices_df columns expected:
      nifty_bank, nifty_50, vix, usd_inr
      (close prices for indices)

    stock_rets:
      log-returns for each bank stock
    """
    bank_price  = indices_df["nifty_bank"].dropna()
    market_price= indices_df["nifty_50"].dropna()

    # Compute returns from prices
    bank_rets   = np.log(bank_price   / bank_price.shift(1)).dropna()
    market_rets = np.log(market_price / market_price.shift(1)).dropna()

    common_idx  = bank_rets.index.intersection(market_rets.index)
    bank_rets   = bank_rets.loc[common_idx]
    market_rets = market_rets.loc[common_idx]

    frames = []

    # ── Volatility ──
    frames.append(rolling_volatility(bank_rets, windows=(5, 10, 20)))
    frames.append(ewma_volatility(bank_rets))
    frames.append(vol_of_vol(bank_rets))

    # ── Drawdown ──
    aligned_price = bank_price.reindex(common_idx).ffill()
    frames.append(rolling_drawdown(aligned_price))
    frames.append(current_drawdown(aligned_price))

    # ── Correlation ──
    bm_corr = bank_market_corr(bank_rets, market_rets)
    frames.append(bm_corr.to_frame())

    if stock_rets is not None and len(stock_rets.columns) >= 2:
        aligned_sr = stock_rets.reindex(common_idx).dropna(axis=1, how="all")
        cb_corr    = cross_bank_corr(aligned_sr)
        frames.append(cb_corr.to_frame())
        frames.append(correlation_trend(cb_corr).to_frame())

    frames.append(correlation_trend(bm_corr).to_frame())

    # ── Momentum ──
    frames.append(rsi(aligned_price).to_frame())
    frames.append(rate_of_change(aligned_price).to_frame())
    frames.append(moving_avg_spread(aligned_price).to_frame())

    # ── VIX ──
    if "vix" in indices_df.columns:
        vix = indices_df["vix"].reindex(common_idx).ffill()
        frames.append(vix_features(vix))

    # ── FX ──
    if "usd_inr" in indices_df.columns:
        fx = indices_df["usd_inr"].reindex(common_idx).ffill()
        frames.append(fx_features(fx))

    # ── Tail Risk ──
    frames.append(cvar(bank_rets).to_frame())
    frames.append(rolling_skew_kurt(bank_rets))

    # ── Relative Risk ──
    frames.append(relative_risk(bank_rets, market_rets))

    # ── Combine ──
    feature_df = pd.concat(frames, axis=1)
    feature_df = feature_df.loc[common_idx]
    feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    feature_df.fillna(method="ffill", limit=5, inplace=True)
    feature_df.dropna(inplace=True)

    print(f"✅ Feature matrix: {feature_df.shape[0]} rows × {feature_df.shape[1]} features")
    print(f"   Features: {list(feature_df.columns)}")
    return feature_df


# ─── DEMO ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    print("Testing feature engineering with synthetic data...")
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    np.random.seed(42)

    # Synthetic prices
    bank_price   = pd.Series(10000 * np.cumprod(1 + np.random.randn(500) * 0.01), index=dates)
    market_price = pd.Series(15000 * np.cumprod(1 + np.random.randn(500) * 0.008), index=dates)
    vix          = pd.Series(15 + np.random.randn(500) * 5, index=dates).clip(lower=10)
    usd_inr      = pd.Series(70 + np.cumsum(np.random.randn(500) * 0.1), index=dates)

    indices_df = pd.DataFrame({
        "nifty_bank": bank_price,
        "nifty_50":   market_price,
        "vix":        vix,
        "usd_inr":    usd_inr,
    })

    # Synthetic stock returns
    stocks = {}
    for name in ["HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK"]:
        stocks[name] = pd.Series(np.random.randn(500) * 0.012, index=dates)
    stock_rets = pd.DataFrame(stocks)

    features = build_all_features(indices_df, stock_rets)
    print(features.tail(3))
    features.to_csv("features.csv")
    print("💾 Saved: features.csv")
