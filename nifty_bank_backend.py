import enum
from datetime import datetime, timedelta
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ===========================
#  Config / constants
# ===========================

# Yahoo Finance symbol for NIFTY Bank index
NIFTY_BANK_SYMBOL = "^NSEBANK"  # fallback to ^BANKNIFTY if ever needed

# Default history window for fetching data (in years)
DEFAULT_YEARS_HISTORY = 10

SHORT_VOL_WINDOW = 20
LONG_VOL_WINDOW = 60
VOLUME_WINDOW = 20


# ===========================
#  Risk state definitions
# ===========================


class RiskRegime(str, enum.Enum):
    stable = "Stable"
    early_stress = "Early Stress"
    crisis = "Crisis"
    recovery = "Recovery"


class RiskStatePoint(BaseModel):
    date: datetime
    close: float
    short_vol: float
    long_vol: float
    drawdown: float
    volume_stress: float
    instability: float
    recovery_signal: float
    state: RiskRegime


# ===========================
#  Data fetching (Yahoo)
# ===========================


class NiftyBankDataFetcher:
    """
    Thin wrapper around yfinance with simple in-memory cache.
    Handles the MultiIndex columns Yahoo sometimes returns.
    """

    def __init__(self, symbol: str = NIFTY_BANK_SYMBOL):
        self.symbol = symbol
        self._cache: Optional[pd.DataFrame] = None

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        # If Yahoo returns MultiIndex (Price, Ticker), drop ticker level
        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel(1, axis=1)
        # Standardize column names
        df = df.rename(
            columns={
                "Open": "Open",
                "High": "High",
                "Low": "Low",
                "Close": "Close",
                "Adj Close": "AdjClose",
                "Volume": "Volume",
            }
        )
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df

    def _download_history(self, start: datetime, end: datetime) -> pd.DataFrame:
        df = yf.download(
            self.symbol,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
        if df.empty:
            raise RuntimeError(
                "No data returned from Yahoo Finance. "
                "Check symbol, internet connection, or date range."
            )
        df = self._normalize_columns(df)
        return df

    def get_history(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Return OHLCV history, using cached data if available and sufficient.
        """
        if end is None:
            end = datetime.utcnow()
        if start is None:
            start = end - timedelta(days=DEFAULT_YEARS_HISTORY * 365)

        if self._cache is not None:
            cached_start, cached_end = self._cache.index[0], self._cache.index[-1]
            if cached_start <= start and cached_end >= end:
                return self._cache.loc[start:end].copy()

        df = self._download_history(start, end)
        if self._cache is None:
            self._cache = df
        else:
            self._cache = (
                pd.concat([self._cache, df]).sort_index().drop_duplicates()
            )

        return df.loc[start:end].copy()


# ===========================
#  Risk model core
# ===========================


class DynamicRiskModel:
    """
    Constructs risk-state variables and classifies regimes.
    Uses quantile-based thresholds so that regimes adapt
    to the NIFTY Bank index's own distribution.
    """

    def __init__(
        self,
        short_vol_window: int = SHORT_VOL_WINDOW,
        long_vol_window: int = LONG_VOL_WINDOW,
        volume_window: int = VOLUME_WINDOW,
    ):
        self.short_vol_window = short_vol_window
        self.long_vol_window = long_vol_window
        self.volume_window = volume_window

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Log returns
        df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

        # Realized volatility (short + long)
        df["short_vol"] = df["log_return"].rolling(self.short_vol_window).std()
        df["long_vol"] = df["log_return"].rolling(self.long_vol_window).std()

        # Drawdown from running max
        rolling_max = df["Close"].cummax()
        df["drawdown"] = (df["Close"] - rolling_max) / rolling_max

        # Volume stress: rolling z-score
        vol_mean = df["Volume"].rolling(self.volume_window).mean()
        vol_std = df["Volume"].rolling(self.volume_window).std()
        df["volume_stress"] = (df["Volume"] - vol_mean) / vol_std

        # Instability: change in short volatility
        df["instability"] = df["short_vol"].diff()

        # Recovery signal: volatility falling while price improves from recent min
        roll_min = df["Close"].rolling(self.short_vol_window).min()
        price_from_min = (df["Close"] - roll_min) / roll_min
        df["recovery_signal"] = -df["short_vol"].diff() * np.sign(
            price_from_min.diff()
        )

        df = df.dropna(
            subset=[
                "short_vol",
                "long_vol",
                "drawdown",
                "volume_stress",
                "instability",
                "recovery_signal",
            ]
        )

        return df

    def _compute_thresholds(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Use empirical quantiles to adapt thresholds to the regime distribution.
        """
        thresholds = {
            "vol_q50": df["short_vol"].quantile(0.50),
            "vol_q75": df["short_vol"].quantile(0.75),
            "vol_q90": df["short_vol"].quantile(0.90),
            "dd_q25": df["drawdown"].quantile(0.25),
            "dd_q10": df["drawdown"].quantile(0.10),
            "inst_q50": df["instability"].quantile(0.50),
            "rec_q50": df["recovery_signal"].quantile(0.50),
        }
        return thresholds

    def classify_regime(self, row, th: Dict[str, float]) -> RiskRegime:
        vol = row["short_vol"]
        dd = row["drawdown"]
        inst = row["instability"]
        rec = row["recovery_signal"]

        # Crisis: very high volatility + deep drawdown
        if (vol >= th["vol_q90"]) and (dd <= th["dd_q10"]):
            return RiskRegime.crisis

        # Early stress: volatility above median, drawdown worsening / instability positive
        if (vol >= th["vol_q50"]) and (inst > th["inst_q50"]) and (
            dd <= th["dd_q25"]
        ):
            return RiskRegime.early_stress

        # Stable: low volatility, mild drawdown, no acceleration
        if (vol < th["vol_q50"]) and (dd > th["dd_q25"]) and (
            inst <= th["inst_q50"]
        ):
            return RiskRegime.stable

        # Recovery: volatility easing and recovery signal positive
        return RiskRegime.recovery

    def attach_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        thresholds = self._compute_thresholds(df)
        df["Risk_State"] = df.apply(
            lambda r: self.classify_regime(r, thresholds), axis=1
        )
        return df

    def compute_transition_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Empirical first-order Markov transition matrix between regimes.
        """
        # Use the enum values (Stable, Early Stress, etc.) as state labels
        codes = df["Risk_State"].map(lambda r: r.value)
        prev = codes.shift(1).dropna()
        curr = codes.dropna()
        tm = pd.crosstab(prev, curr, normalize="index").fillna(0.0)
        tm.index.name = "From"
        tm.columns.name = "To"
        return tm

    def simulate_state_paths(
        self,
        transition_matrix: pd.DataFrame,
        start_state: RiskRegime,
        n_steps: int = 200,
        n_paths: int = 20,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        """
        Simulate sequences of risk regimes using the estimated Markov chain.
        """
        rng = np.random.default_rng(random_state)

        # Ensure all regimes appear as both rows and columns
        for regime in [s.value for s in RiskRegime]:
            if regime not in transition_matrix.columns:
                transition_matrix[regime] = 0.0
            if regime not in transition_matrix.index:
                transition_matrix.loc[regime] = 0.0

        transition_matrix = transition_matrix.sort_index().sort_index(axis=1)
        states = list(transition_matrix.columns)
        probs = transition_matrix.values
        state_to_idx = {s: i for i, s in enumerate(states)}

        paths = np.empty((n_paths, n_steps), dtype=object)
        paths[:, 0] = start_state.value

        for t in range(1, n_steps):
            for p in range(n_paths):
                curr_state = paths[p, t - 1]
                row_idx = state_to_idx[curr_state]
                paths[p, t] = rng.choice(states, p=probs[row_idx])

        return paths


# ===========================
#  FastAPI backend
# ===========================


app = FastAPI(
    title="NIFTY Bank Dynamic Risk Backend",
    description="Dynamic risk-state and recovery modeling for the NIFTY Bank Index.",
    version="1.0.0",
)

# Allow the Vite frontend (localhost:5173) to call this API from the browser
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_fetcher = NiftyBankDataFetcher()
_model = DynamicRiskModel()


class SimulationRequest(BaseModel):
    start_state: RiskRegime
    n_steps: int = 200
    n_paths: int = 20
    random_seed: Optional[int] = 42


@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


@app.get("/risk-state/latest", response_model=RiskStatePoint)
def get_latest_risk_state():
    end = datetime.utcnow()
    start = end - timedelta(days=DEFAULT_YEARS_HISTORY * 365)

    raw = _fetcher.get_history(start, end)
    feats = _model.build_features(raw)
    feats = _model.attach_regimes(feats)

    last = feats.iloc[-1]

    return RiskStatePoint(
        date=last.name.to_pydatetime(),
        close=float(last["Close"]),
        short_vol=float(last["short_vol"]),
        long_vol=float(last["long_vol"]),
        drawdown=float(last["drawdown"]),
        volume_stress=float(last["volume_stress"]),
        instability=float(last["instability"]),
        recovery_signal=float(last["recovery_signal"]),
        state=RiskRegime(last["Risk_State"]),
    )


@app.get("/risk-state/history", response_model=List[RiskStatePoint])
def get_risk_state_history(
    start: Optional[datetime] = Query(
        None,
        description="Start date (inclusive). If omitted, uses ~10 years ago.",
    ),
    end: Optional[datetime] = Query(
        None,
        description="End date (exclusive). If omitted, uses now.",
    ),
    limit: int = Query(
        1000,
        ge=1,
        le=5000,
        description="Max number of points to return (downsampled if needed).",
    ),
):
    if end is None:
        end = datetime.utcnow()
    if start is None:
        start = end - timedelta(days=DEFAULT_YEARS_HISTORY * 365)

    raw = _fetcher.get_history(start, end)
    feats = _model.build_features(raw)
    feats = _model.attach_regimes(feats)

    if len(feats) > limit:
        idx = np.linspace(0, len(feats) - 1, num=limit, dtype=int)
        feats = feats.iloc[idx]

    result: List[RiskStatePoint] = []
    for ts, row in feats.iterrows():
        result.append(
            RiskStatePoint(
                date=ts.to_pydatetime(),
                close=float(row["Close"]),
                short_vol=float(row["short_vol"]),
                long_vol=float(row["long_vol"]),
                drawdown=float(row["drawdown"]),
                volume_stress=float(row["volume_stress"]),
                instability=float(row["instability"]),
                recovery_signal=float(row["recovery_signal"]),
                state=RiskRegime(row["Risk_State"]),
            )
        )
    return result


@app.get("/risk-state/transition-matrix")
def get_transition_matrix():
    end = datetime.utcnow()
    start = end - timedelta(days=DEFAULT_YEARS_HISTORY * 365)

    raw = _fetcher.get_history(start, end)
    feats = _model.build_features(raw)
    feats = _model.attach_regimes(feats)

    tm = _model.compute_transition_matrix(feats)
    return tm.to_dict(orient="index")


@app.post("/risk-state/simulate")
def simulate_risk_states(request: SimulationRequest):
    end = datetime.utcnow()
    start = end - timedelta(days=DEFAULT_YEARS_HISTORY * 365)

    raw = _fetcher.get_history(start, end)
    feats = _model.build_features(raw)
    feats = _model.attach_regimes(feats)
    tm = _model.compute_transition_matrix(feats)

    paths = _model.simulate_state_paths(
        transition_matrix=tm,
        start_state=request.start_state,
        n_steps=request.n_steps,
        n_paths=request.n_paths,
        random_state=request.random_seed,
    )

    return {"paths": paths.tolist(), "states": [s.value for s in RiskRegime]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "nifty_bank_backend:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )

