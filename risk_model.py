"""
risk_model.py
Core risk scoring engine for NIFTY Banking Risk System.

Pipeline:
  1. Normalise each sub-component to 0-100 (higher = safer)
  2. Weighted composite → Risk Score (0-100)
  3. Rule-based state classification
  4. Per-stock risk scoring + BUY / HOLD / AVOID recommendation
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

STATE_MAP = {
    "Stable":   (80, 100),
    "Low Risk": (65,  80),
    "Stress":   (40,  65),
    "Crisis":   ( 0,  40),
    # Recovery: special — score rising from Crisis zone
}

COMPONENT_WEIGHTS = {
    "stability":  0.40,   # volatility + drawdown
    "volume":     0.20,   # volume stress
    "network":    0.20,   # correlations (systemic)
    "external":   0.20,   # VIX + FX
}


# ═══════════════════════════════════════════════════════════════════════════════
# SUB-COMPONENT SCORES (each → 0-100, higher = safer)
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_scale(series: pd.Series, invert: bool = True) -> pd.Series:
    """
    MinMax scale to 0-1 then ×100.
    invert=True  → high raw value = low safety score (e.g. volatility)
    invert=False → high raw value = high safety score
    """
    arr     = series.values.reshape(-1, 1)
    scaler  = MinMaxScaler()
    scaled  = scaler.fit_transform(arr).flatten()
    if invert:
        scaled = 1 - scaled
    return pd.Series(scaled * 100, index=series.index)


def stability_score(features: pd.DataFrame) -> pd.Series:
    """
    40% weight — combines volatility and drawdown signals.
    Higher score → lower volatility, shallower drawdown → safer.
    """
    sub_scores = []

    # Volatility (invert — high vol = low score)
    if "vol_20d" in features.columns:
        sub_scores.append(_safe_scale(features["vol_20d"].fillna(method="ffill"), invert=True))
    if "ewma_vol" in features.columns:
        sub_scores.append(_safe_scale(features["ewma_vol"].fillna(method="ffill"), invert=True))
    if "vol_of_vol" in features.columns:
        sub_scores.append(_safe_scale(features["vol_of_vol"].fillna(method="ffill"), invert=True))

    # Drawdown (invert — deeper DD = lower score)
    if "max_drawdown" in features.columns:
        sub_scores.append(_safe_scale(features["max_drawdown"].abs().fillna(0), invert=True))
    if "current_drawdown" in features.columns:
        sub_scores.append(_safe_scale(features["current_drawdown"].abs().fillna(0), invert=True))
    if "underwater_dur" in features.columns:
        sub_scores.append(_safe_scale(features["underwater_dur"].fillna(0), invert=True))

    if not sub_scores:
        return pd.Series(50.0, index=features.index)
    return pd.concat(sub_scores, axis=1).mean(axis=1)


def volume_score(features: pd.DataFrame) -> pd.Series:
    """
    20% weight — volume stress (spike = panic).
    High volume z-score = danger.
    """
    sub_scores = []
    if "vol_zscore" in features.columns:
        sub_scores.append(_safe_scale(features["vol_zscore"].abs().fillna(0), invert=True))
    if "vol_spike" in features.columns:
        # spike present: -30 points
        sub_scores.append(100 - features["vol_spike"].fillna(0) * 30)

    if not sub_scores:
        return pd.Series(50.0, index=features.index)
    return pd.concat(sub_scores, axis=1).mean(axis=1).clip(0, 100)


def network_score(features: pd.DataFrame) -> pd.Series:
    """
    20% weight — systemic correlation risk.
    Higher cross-bank correlation = contagion = danger.
    """
    sub_scores = []
    if "bank_market_corr" in features.columns:
        sub_scores.append(_safe_scale(features["bank_market_corr"].fillna(0).abs(), invert=True))
    if "cross_bank_corr" in features.columns:
        sub_scores.append(_safe_scale(features["cross_bank_corr"].fillna(0).abs(), invert=True))
    if "corr_trend" in features.columns:
        # Rapidly rising corr = danger
        sub_scores.append(_safe_scale(features["corr_trend"].fillna(0), invert=True))

    if not sub_scores:
        return pd.Series(50.0, index=features.index)
    return pd.concat(sub_scores, axis=1).mean(axis=1)


def external_score(features: pd.DataFrame) -> pd.Series:
    """
    20% weight — VIX + USD/INR macro signals.
    High VIX = fear. Sharp USD/INR rise = capital flight.
    """
    sub_scores = []
    if "vix_level" in features.columns:
        sub_scores.append(_safe_scale(features["vix_level"].fillna(method="ffill"), invert=True))
    if "vix_zscore" in features.columns:
        sub_scores.append(_safe_scale(features["vix_zscore"].fillna(0), invert=True))
    if "vix_spike" in features.columns:
        sub_scores.append(100 - features["vix_spike"].fillna(0) * 30)
    if "fx_zscore" in features.columns:
        # Rising USD/INR = bad
        sub_scores.append(_safe_scale(features["fx_zscore"].fillna(0), invert=True))

    if not sub_scores:
        return pd.Series(50.0, index=features.index)
    return pd.concat(sub_scores, axis=1).mean(axis=1).clip(0, 100)


# ═══════════════════════════════════════════════════════════════════════════════
# COMPOSITE RISK SCORE
# ═══════════════════════════════════════════════════════════════════════════════

def compute_risk_score(features: pd.DataFrame) -> pd.DataFrame:
    """
    Returns DataFrame with:
      stability_score, volume_score, network_score, external_score,
      risk_score        (0-100, higher = SAFER)
      risk_velocity     (day-over-day change)
      risk_momentum     (5-day trend)
    """
    s = stability_score(features)
    v = volume_score(features)
    n = network_score(features)
    e = external_score(features)

    w = COMPONENT_WEIGHTS
    composite = (
        w["stability"] * s +
        w["volume"]    * v +
        w["network"]   * n +
        w["external"]  * e
    )

    # Smooth with 3-day EMA to reduce noise
    composite_smooth = composite.ewm(span=3).mean()

    result = pd.DataFrame({
        "stability_score": s,
        "volume_score":    v,
        "network_score":   n,
        "external_score":  e,
        "risk_score":      composite_smooth,
        "risk_velocity":   composite_smooth.diff(1),
        "risk_momentum":   composite_smooth.diff(5),
    }, index=features.index)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# STATE CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

def classify_state(risk_scores: pd.Series, window_recovery: int = 5) -> pd.Series:
    """
    Maps risk score → market state.
    Special case: Recovery = score rising for ≥ window_recovery days from Crisis.

    States: Stable | Low Risk | Stress | Crisis | Recovery
    """
    states = []
    prev_crisis = False
    recovery_count = 0

    for i, (date, score) in enumerate(risk_scores.items()):
        velocity = risk_scores.iloc[i] - risk_scores.iloc[i - 1] if i > 0 else 0

        if score >= 80:
            state = "Stable"
            prev_crisis     = False
            recovery_count  = 0
        elif score >= 65:
            state = "Low Risk"
            prev_crisis    = False
            recovery_count = 0
        elif score >= 40:
            state = "Stress"
            if prev_crisis and velocity > 0:
                recovery_count += 1
                if recovery_count >= window_recovery:
                    state = "Recovery"
            else:
                recovery_count = 0
                prev_crisis = False
        else:
            state = "Crisis"
            prev_crisis    = True
            recovery_count = 0

        states.append(state)

    return pd.Series(states, index=risk_scores.index, name="market_state")


def state_to_color(state: str) -> str:
    """Returns hex color for UI rendering."""
    colors = {
        "Stable":   "#00c853",
        "Low Risk": "#76ff03",
        "Stress":   "#ffab00",
        "Crisis":   "#d50000",
        "Recovery": "#00b8d4",
    }
    return colors.get(state, "#9e9e9e")


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSITION PROBABILITY (RULE-BASED BASELINE)
# ═══════════════════════════════════════════════════════════════════════════════

# Historical approximate transition matrix (can be refined with HMM)
TRANSITION_MATRIX = {
    "Stable":   {"Stable": 0.90, "Low Risk": 0.08, "Stress": 0.02, "Crisis": 0.00, "Recovery": 0.00},
    "Low Risk": {"Stable": 0.15, "Low Risk": 0.70, "Stress": 0.13, "Crisis": 0.02, "Recovery": 0.00},
    "Stress":   {"Stable": 0.05, "Low Risk": 0.15, "Stress": 0.55, "Crisis": 0.20, "Recovery": 0.05},
    "Crisis":   {"Stable": 0.00, "Low Risk": 0.05, "Stress": 0.30, "Crisis": 0.45, "Recovery": 0.20},
    "Recovery": {"Stable": 0.10, "Low Risk": 0.35, "Stress": 0.35, "Crisis": 0.05, "Recovery": 0.15},
}


def get_next_state_probs(current_state: str) -> dict:
    """Returns probability distribution over next states."""
    return TRANSITION_MATRIX.get(current_state, {})


# ═══════════════════════════════════════════════════════════════════════════════
# PER-STOCK RISK SCORING + RECOMMENDATION
# ═══════════════════════════════════════════════════════════════════════════════

STOCK_SCORE_THRESHOLDS = {
    "BUY":   70,   # stock score ≥ 70
    "HOLD":  45,   # 45 ≤ score < 70
    # AVOID: score < 45
}


def score_individual_stock(stock_rets: pd.Series,
                            index_rets: pd.Series,
                            market_risk_score: float,
                            window: int = 20) -> dict:
    """
    Produces a single summary dict for the latest date.

    Inputs:
      stock_rets       : log-return series for the stock
      index_rets       : log-return series for NIFTY Bank
      market_risk_score: composite risk score on latest date (0-100)
    """
    if len(stock_rets) < window + 5:
        return {"error": "insufficient data"}

    r  = stock_rets.dropna()
    ir = index_rets.reindex(r.index).dropna()
    r  = r.reindex(ir.index)

    # Annualised stats (last window days)
    recent_r  = r.iloc[-window:]
    recent_ir = ir.iloc[-window:]

    ann_ret = recent_r.mean() * 252
    ann_vol = recent_r.std() * np.sqrt(252)
    sharpe  = ann_ret / (ann_vol + 1e-9)

    cov  = recent_r.cov(recent_ir)
    var  = recent_ir.var()
    beta = cov / (var + 1e-9)

    corr = recent_r.corr(recent_ir)
    mom  = recent_r.sum()                  # cumulative return over window

    dd   = r.iloc[-window:]
    cum  = dd.cumsum()
    roll = cum.cummax()
    max_dd = (cum - roll).min()

    # ── Stock Score ──
    # Penalise: high vol, high beta, deep DD, high corr during stress
    vol_pen  = np.clip(max(0, ann_vol - 0.25) * 50, 0, 20)      # penalty for vol > 25%, capped at 20
    beta_pen = max(0, beta - 1.1) * 10                          # penalty for beta > 1.1
    dd_pen   = np.clip(abs(max_dd) * 50, 0, 20)                 # penalty for drawdown, capped at 20
    corr_pen = max(0, corr - 0.7) * 10              

    base_score = market_risk_score                              # inherit market health
    stock_score = base_score - vol_pen - beta_pen - dd_pen - corr_pen
    
    # Scale down rewards/penalties from sharpe & momentum so they don't break bounds
    stock_score += np.clip(sharpe * 2, -10, 10)
    stock_score += np.clip(mom * 50, -10, 10)
    stock_score = float(np.clip(stock_score, 0, 100))

    # ── Recommendation ──
    if stock_score >= STOCK_SCORE_THRESHOLDS["BUY"] and market_risk_score >= 50:
        recommendation = "BUY"
        confidence     = "HIGH" if stock_score >= 80 else "MODERATE"
    elif stock_score >= STOCK_SCORE_THRESHOLDS["HOLD"]:
        recommendation = "HOLD"
        confidence     = "MODERATE"
    else:
        recommendation = "AVOID"
        confidence     = "HIGH" if stock_score < 25 else "MODERATE"

    # ── Explanation ──
    reasons = []
    if ann_vol > 0.25:
        reasons.append(f"High volatility ({ann_vol:.1%} annualised)")
    if beta > 1.2:
        reasons.append(f"High market sensitivity (β = {beta:.2f})")
    if max_dd < -0.10:
        reasons.append(f"Deep drawdown ({max_dd:.1%} over {window}d)")
    if corr > 0.85:
        reasons.append("Strongly correlated with market stress")
    if mom < 0:
        reasons.append("Negative momentum — falling trend")
    if not reasons:
        reasons.append("Low volatility, solid momentum, healthy beta")

    return {
        "stock_score":      round(stock_score, 1),
        "recommendation":   recommendation,
        "confidence":       confidence,
        "ann_return":       round(ann_ret * 100, 2),
        "ann_volatility":   round(ann_vol * 100, 2),
        "sharpe":           round(sharpe, 2),
        "beta":             round(beta, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "momentum":         round(mom * 100, 2),
        "corr_with_index":  round(corr, 3),
        "reasons":          reasons,
    }


def score_all_stocks(stock_rets: pd.DataFrame,
                     index_rets: pd.Series,
                     market_risk_score: float,
                     window: int = 20) -> pd.DataFrame:
    """
    Returns a sorted DataFrame of stock scores + recommendations.
    """
    results = []
    for col in stock_rets.columns:
        s = stock_rets[col]
        r = score_individual_stock(s, index_rets, market_risk_score, window)
        r["stock"] = col
        results.append(r)

    df = pd.DataFrame(results).set_index("stock")
    df.sort_values("stock_score", ascending=False, inplace=True)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# FULL RISK PIPELINE (convenience function)
# ═══════════════════════════════════════════════════════════════════════════════

def run_risk_pipeline(features: pd.DataFrame,
                      stock_rets: pd.DataFrame,
                      index_rets: pd.Series) -> dict:
    """
    Runs the full pipeline and returns a results dict ready for the UI.
    """
    # 1. Risk scores
    scores = compute_risk_score(features)

    # 2. State classification
    states = classify_state(scores["risk_score"])

    # 3. Combine
    output = scores.join(states)

    # 4. Latest snapshot
    latest_score = float(scores["risk_score"].iloc[-1])
    latest_state = states.iloc[-1]
    latest_vel   = float(scores["risk_velocity"].iloc[-1])
    latest_mom   = float(scores["risk_momentum"].iloc[-1])

    # 5. Stock recommendations
    stock_analysis = score_all_stocks(stock_rets, index_rets, latest_score)

    # 6. Transition probs
    next_probs = get_next_state_probs(latest_state)

    return {
        "timeseries":       output,
        "latest_score":     latest_score,
        "latest_state":     latest_state,
        "risk_velocity":    latest_vel,
        "risk_momentum":    latest_mom,
        "state_color":      state_to_color(latest_state),
        "stock_analysis":   stock_analysis,
        "next_state_probs": next_probs,
    }


# ─── DEMO ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from feature_engineering import build_all_features

    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=500, freq="B")

    # Inject a crisis (COVID-like) around day 300
    bank_ret = np.random.randn(500) * 0.01
    bank_ret[295:315] = np.random.randn(20) * 0.04   # crisis spike
    bank_price = pd.Series(10000 * np.cumprod(1 + bank_ret), index=dates)

    mkt_ret   = np.random.randn(500) * 0.008
    mkt_price = pd.Series(15000 * np.cumprod(1 + mkt_ret), index=dates)

    vix = pd.Series(15 + np.random.randn(500) * 3, index=dates)
    vix.iloc[295:315] += 20                          # VIX spikes in crisis

    usd_inr = pd.Series(75 + np.cumsum(np.random.randn(500) * 0.05), index=dates)

    indices_df = pd.DataFrame({
        "nifty_bank": bank_price,
        "nifty_50":   mkt_price,
        "vix":        vix,
        "usd_inr":    usd_inr,
    })

    stock_names = ["HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK"]
    stock_rets  = pd.DataFrame(
        {n: np.random.randn(500) * 0.012 for n in stock_names}, index=dates
    )
    # Make SBI risky during crisis
    stock_rets["SBIN"].iloc[295:315] += 0.04

    features   = build_all_features(indices_df, stock_rets)
    bank_idx_r = np.log(bank_price / bank_price.shift(1)).dropna()

    results = run_risk_pipeline(features, stock_rets.reindex(features.index), bank_idx_r.reindex(features.index))

    print(f"\n📊 Latest Risk Score : {results['latest_score']:.1f}")
    print(f"📊 Market State      : {results['latest_state']}")
    print(f"📊 Risk Velocity     : {results['risk_velocity']:.2f}")
    print(f"\n🔮 Next State Probs  : {results['next_state_probs']}")
    print(f"\n📈 Stock Analysis:")
    print(results["stock_analysis"][["stock_score", "recommendation", "confidence", "beta", "ann_volatility"]])
