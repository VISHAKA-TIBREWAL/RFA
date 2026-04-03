"""
models.py
ML models for NIFTY Banking Risk System.

Models:
  1. HiddenMarkovRiskModel     – learns latent market regimes (GaussianHMM)
  2. LSTMTransitionPredictor   – temporal sequence model (PyTorch)
  3. RandomForestCrisisPredictor – crisis 3-day-ahead prediction (ensemble)
  4. LogisticRegressionBaseline  – linear baseline
  5. IsolationForestAnomaly    – unsupervised anomaly baseline
  6. StaticThresholdBaseline   – simplest possible baseline

Each exposes: fit(X, y) / predict(X) / predict_proba(X) / save(path) / load(path)
"""

import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble           import RandomForestClassifier, IsolationForest
from sklearn.linear_model       import LogisticRegression
from sklearn.preprocessing      import StandardScaler
from sklearn.pipeline           import Pipeline

# Optional: hmmlearn
try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("[WARN] hmmlearn not installed. HMM model disabled. pip install hmmlearn")

# Optional: PyTorch
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARN] PyTorch not installed. LSTM model disabled. pip install torch")


# ═══════════════════════════════════════════════════════════════════════════════
# LABEL GENERATION  (shared utility)
# ═══════════════════════════════════════════════════════════════════════════════

def make_crisis_label(risk_scores: pd.Series, threshold: float = 40.0,
                       horizon: int = 3) -> pd.Series:
    """
    Binary label: 1 if risk score drops below threshold within next `horizon` days.
    Used for supervised crisis prediction.
    """
    in_crisis = (risk_scores < threshold).astype(int)
    label     = in_crisis.rolling(horizon).max().shift(-horizon)
    return label.dropna().astype(int)


def make_state_label(states: pd.Series) -> pd.Series:
    """Maps state strings → integer codes."""
    mapping = {"Stable": 0, "Low Risk": 1, "Stress": 2, "Crisis": 3, "Recovery": 4}
    return states.map(mapping)


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  HIDDEN MARKOV MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class HiddenMarkovRiskModel:
    """
    GaussianHMM learns latent hidden states from risk-score features.
    States are numbered 0..n_components-1 — we map them to meaningful labels.
    """

    def __init__(self, n_states: int = 5, n_iter: int = 200, covariance_type="full"):
        self.n_states        = n_states
        self.n_iter          = n_iter
        self.covariance_type = covariance_type
        self.model           = None
        self.state_map       = {}    # hidden state id → label
        self.scaler          = StandardScaler()

    # ── internal ──────────────────────────────────────────────────────────────

    def _prepare(self, X: pd.DataFrame) -> np.ndarray:
        """Select & scale HMM input features."""
        cols = [c for c in ["risk_score", "risk_velocity", "risk_momentum",
                             "stability_score", "external_score"] if c in X.columns]
        if not cols:
            cols = X.columns.tolist()[:5]
        return self.scaler.fit_transform(X[cols].fillna(0).values)

    def _map_states_to_labels(self, X_scaled: np.ndarray, risk_scores: pd.Series):
        """
        Map HMM hidden states to readable labels by comparing mean risk score.
        Lower mean risk score → 'Crisis'; higher → 'Stable'
        """
        preds      = self.model.predict(X_scaled)
        state_means = {}
        for s in range(self.n_states):
            idx = (preds == s)
            if idx.sum() > 0:
                state_means[s] = risk_scores.values[idx].mean()
            else:
                state_means[s] = 50.0

        sorted_states = sorted(state_means, key=state_means.get)  # ascending risk
        labels        = ["Crisis", "Stress", "Low Risk", "Stable", "Recovery"]
        for i, s in enumerate(sorted_states):
            self.state_map[s] = labels[min(i, len(labels) - 1)]

    # ── public API ────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, risk_scores: pd.Series):
        if not HMM_AVAILABLE:
            print("[HMM] hmmlearn not available. Skipping.")
            return self
        X_scaled = self._prepare(X)
        self.model = GaussianHMM(
            n_components    = self.n_states,
            covariance_type = self.covariance_type,
            n_iter          = self.n_iter,
            random_state    = 42,
        )
        self.model.fit(X_scaled)
        self._map_states_to_labels(X_scaled, risk_scores)
        print(f"[HMM] Converged: {self.model.monitor_.converged}")
        return self

    def predict_states(self, X: pd.DataFrame) -> pd.Series:
        if not HMM_AVAILABLE or self.model is None:
            return pd.Series("Unknown", index=X.index)
        X_scaled = self.scaler.transform(
            X[[c for c in ["risk_score", "risk_velocity", "risk_momentum",
                            "stability_score", "external_score"] if c in X.columns]
              ].fillna(0).values
        )
        raw = self.model.predict(X_scaled)
        return pd.Series([self.state_map.get(s, "Unknown") for s in raw], index=X.index)

    def predict_state_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Returns posterior state probabilities for each timestep."""
        if not HMM_AVAILABLE or self.model is None:
            return pd.DataFrame()
        X_scaled = self.scaler.transform(
            X[[c for c in ["risk_score", "risk_velocity", "risk_momentum",
                            "stability_score", "external_score"] if c in X.columns]
              ].fillna(0).values
        )
        proba = self.model.predict_proba(X_scaled)
        cols  = [self.state_map.get(i, str(i)) for i in range(self.n_states)]
        return pd.DataFrame(proba, columns=cols, index=X.index)

    def save(self, path: str = "hmm_model.pkl"):
        joblib.dump({"model": self.model, "map": self.state_map, "scaler": self.scaler}, path)

    def load(self, path: str = "hmm_model.pkl"):
        data = joblib.load(path)
        self.model      = data["model"]
        self.state_map  = data["map"]
        self.scaler     = data["scaler"]


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  LSTM TRANSITION PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════

if TORCH_AVAILABLE:
    class _LSTMNet(nn.Module):
        def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                batch_first=True, dropout=dropout)
            self.fc   = nn.Sequential(
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :]).squeeze(-1)


class LSTMTransitionPredictor:
    """
    LSTM that takes a sequence of risk features → predicts P(crisis in next 3 days).
    Input shape: (batch, seq_len, n_features)
    """

    def __init__(self, seq_len: int = 10, hidden_size: int = 64,
                 num_layers: int = 2, lr: float = 1e-3, epochs: int = 50):
        self.seq_len     = seq_len
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.lr          = lr
        self.epochs      = epochs
        self.model       = None
        self.scaler      = StandardScaler()
        self.feature_cols= None

    def _select_features(self, X: pd.DataFrame) -> list:
        preferred = ["risk_score", "risk_velocity", "risk_momentum",
                     "stability_score", "volume_score", "network_score",
                     "external_score", "vol_20d", "max_drawdown",
                     "vix_level", "bank_market_corr"]
        return [c for c in preferred if c in X.columns] or X.columns.tolist()[:8]

    def _make_sequences(self, X_scaled: np.ndarray, y: np.ndarray = None):
        Xs, ys = [], []
        for i in range(self.seq_len, len(X_scaled)):
            Xs.append(X_scaled[i - self.seq_len:i])
            if y is not None:
                ys.append(y[i])
        X_seq = np.array(Xs, dtype=np.float32)
        if y is not None:
            return X_seq, np.array(ys, dtype=np.float32)
        return X_seq

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if not TORCH_AVAILABLE:
            print("[LSTM] PyTorch not available. Skipping.")
            return self

        self.feature_cols = self._select_features(X)
        X_np = self.scaler.fit_transform(X[self.feature_cols].fillna(0).values)
        y_np = y.values

        X_seq, y_seq = self._make_sequences(X_np, y_np)

        input_size = X_seq.shape[2]
        self.model = _LSTMNet(input_size, self.hidden_size, self.num_layers)

        # Class weights for imbalanced crisis labels
        pos_weight = torch.tensor(
            (len(y) - y.sum()) / (y.sum() + 1e-6),
            dtype=torch.float32
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer  = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler  = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        X_t = torch.from_numpy(X_seq)
        y_t = torch.from_numpy(y_seq)

        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            # forward through raw network to use BCEWithLogitsLoss
            out, _ = self.model.lstm(X_t)
            logits  = self.model.fc[:-1](out[:, -1, :]).squeeze(-1)  # pre-sigmoid
            loss = criterion(logits, y_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            if (epoch + 1) % 10 == 0:
                print(f"  [LSTM] Epoch {epoch+1}/{self.epochs} | Loss: {loss.item():.4f}")

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns P(crisis) for each time step (NaN for first seq_len rows)."""
        if not TORCH_AVAILABLE or self.model is None:
            return np.full(len(X), np.nan)

        X_np  = self.scaler.transform(X[self.feature_cols].fillna(0).values)
        X_seq = self._make_sequences(X_np)

        self.model.eval()
        with torch.no_grad():
            X_t   = torch.from_numpy(X_seq)
            proba = self.model(X_t).numpy()

        # Pad beginning with NaN
        pad    = np.full(self.seq_len, np.nan)
        return np.concatenate([pad, proba])

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, path: str = "lstm_model.pt"):
        if self.model:
            torch.save({"state_dict":    self.model.state_dict(),
                        "feature_cols":  self.feature_cols,
                        "scaler":        self.scaler,
                        "seq_len":       self.seq_len,
                        "hidden_size":   self.hidden_size,
                        "num_layers":    self.num_layers}, path)

    def load(self, path: str = "lstm_model.pt"):
        if not TORCH_AVAILABLE:
            return
        ck = torch.load(path)
        self.feature_cols = ck["feature_cols"]
        self.scaler       = ck["scaler"]
        self.seq_len      = ck["seq_len"]
        self.hidden_size  = ck["hidden_size"]
        self.num_layers   = ck["num_layers"]
        self.model        = _LSTMNet(len(self.feature_cols), self.hidden_size, self.num_layers)
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  RANDOM FOREST CRISIS PREDICTOR  (main model)
# ═══════════════════════════════════════════════════════════════════════════════

class RandomForestCrisisPredictor:
    """Best performing single model — uses rolling train window."""

    FEATURE_COLS = [
        "risk_score", "risk_velocity", "risk_momentum",
        "stability_score", "volume_score", "network_score", "external_score",
        "vol_5d", "vol_10d", "vol_20d", "ewma_vol", "vol_of_vol",
        "max_drawdown", "current_drawdown", "underwater_dur",
        "vol_zscore", "bank_market_corr",
        "vix_level", "vix_zscore", "fx_zscore",
        "rsi", "roc_5d", "ma_spread",
        "cvar_5pct", "return_skew", "bank_beta",
    ]

    def __init__(self, n_estimators: int = 200, max_depth: int = 8,
                 train_window: int = 500, min_train: int = 252):
        self.n_estimators  = n_estimators
        self.max_depth     = max_depth
        self.train_window  = train_window
        self.min_train     = min_train
        self.pipeline      = None
        self.feature_cols_ = None

    def _get_cols(self, X: pd.DataFrame) -> list:
        return [c for c in self.FEATURE_COLS if c in X.columns]

    def fit(self, X: pd.DataFrame, y: pd.Series):
        cols = self._get_cols(X)
        self.feature_cols_ = cols
        Xf   = X[cols].fillna(0)
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("rf",     RandomForestClassifier(
                n_estimators = self.n_estimators,
                max_depth    = self.max_depth,
                class_weight = "balanced",
                n_jobs       = -1,
                random_state = 42,
            ))
        ])
        self.pipeline.fit(Xf, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Xf = X[self.feature_cols_].fillna(0)
        return self.pipeline.predict(Xf)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        Xf = X[self.feature_cols_].fillna(0)
        return self.pipeline.predict_proba(Xf)[:, 1]

    def feature_importance(self) -> pd.Series:
        rf  = self.pipeline.named_steps["rf"]
        imp = pd.Series(rf.feature_importances_, index=self.feature_cols_)
        return imp.sort_values(ascending=False)

    def walk_forward_predict(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """
        Walk-forward (expanding window) out-of-sample predictions.
        No data leakage — only past data used for each prediction.
        """
        cols     = self._get_cols(X)
        preds    = np.full(len(X), np.nan)
        preds_p  = np.full(len(X), np.nan)

        for i in range(self.min_train, len(X)):
            X_train = X[cols].iloc[max(0, i - self.train_window):i].fillna(0)
            y_train = y.iloc[max(0, i - self.train_window):i]

            if y_train.sum() == 0:   # no crisis in training window
                continue

            tmp_pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("rf",     RandomForestClassifier(
                    n_estimators = self.n_estimators,
                    max_depth    = self.max_depth,
                    class_weight = "balanced",
                    n_jobs       = -1,
                    random_state = 42,
                ))
            ])
            tmp_pipe.fit(X_train, y_train)

            X_test     = X[cols].iloc[[i]].fillna(0)
            preds[i]   = tmp_pipe.predict(X_test)[0]
            preds_p[i] = tmp_pipe.predict_proba(X_test)[0, 1]

        return preds, preds_p

    def save(self, path: str = "rf_model.pkl"):
        joblib.dump({"pipeline": self.pipeline, "cols": self.feature_cols_}, path)

    def load(self, path: str = "rf_model.pkl"):
        d = joblib.load(path)
        self.pipeline      = d["pipeline"]
        self.feature_cols_ = d["cols"]


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  LOGISTIC REGRESSION BASELINE
# ═══════════════════════════════════════════════════════════════════════════════

class LogisticRegressionBaseline:
    FEATURE_COLS = ["risk_score", "risk_velocity", "vol_20d",
                    "max_drawdown", "vix_level", "bank_market_corr"]

    def __init__(self):
        self.pipeline      = Pipeline([
            ("scaler", StandardScaler()),
            ("lr",     LogisticRegression(class_weight="balanced", max_iter=500,
                                          random_state=42))
        ])
        self.feature_cols_ = None

    def _get_cols(self, X):
        return [c for c in self.FEATURE_COLS if c in X.columns]

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.feature_cols_ = self._get_cols(X)
        self.pipeline.fit(X[self.feature_cols_].fillna(0), y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(X[self.feature_cols_].fillna(0))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X[self.feature_cols_].fillna(0))[:, 1]

    def save(self, path="lr_model.pkl"): joblib.dump(self.pipeline, path)
    def load(self, path="lr_model.pkl"): self.pipeline = joblib.load(path)


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  ISOLATION FOREST BASELINE
# ═══════════════════════════════════════════════════════════════════════════════

class IsolationForestBaseline:
    FEATURE_COLS = ["vol_20d", "max_drawdown", "vol_zscore", "vix_level", "bank_market_corr"]

    def __init__(self, contamination: float = 0.08):
        self.contamination = contamination
        self.model         = IsolationForest(contamination=contamination,
                                             n_estimators=200,
                                             random_state=42, n_jobs=-1)
        self.scaler        = StandardScaler()
        self.feature_cols_ = None

    def _get_cols(self, X):
        return [c for c in self.FEATURE_COLS if c in X.columns]

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_cols_ = self._get_cols(X)
        Xs = self.scaler.fit_transform(X[self.feature_cols_].fillna(0))
        self.model.fit(Xs)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Xs = self.scaler.transform(X[self.feature_cols_].fillna(0))
        raw = self.model.predict(Xs)
        return (raw == -1).astype(int)   # 1 = anomaly (crisis)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Uses negative anomaly score as proxy for crisis probability."""
        Xs     = self.scaler.transform(X[self.feature_cols_].fillna(0))
        scores = -self.model.decision_function(Xs)   # higher = more anomalous
        mn, mx = scores.min(), scores.max()
        return (scores - mn) / (mx - mn + 1e-9)     # normalise to 0-1

    def save(self, path="if_model.pkl"): joblib.dump({"m": self.model, "s": self.scaler, "c": self.feature_cols_}, path)
    def load(self, path="if_model.pkl"):
        d = joblib.load(path); self.model = d["m"]; self.scaler = d["s"]; self.feature_cols_ = d["c"]


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  STATIC THRESHOLD BASELINE
# ═══════════════════════════════════════════════════════════════════════════════

class StaticThresholdBaseline:
    """
    If risk_score < threshold → predict crisis.
    No training — pure rule.
    """
    def __init__(self, threshold: float = 40.0):
        self.threshold = threshold

    def fit(self, X, y=None): return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (X["risk_score"] < self.threshold).astype(int).values

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Inverted normalised risk score as proxy probability."""
        scores = X["risk_score"].fillna(50).values
        return np.clip((self.threshold - scores) / self.threshold, 0, 1)

    def save(self, path=None): pass
    def load(self, path=None): pass


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    np.random.seed(42)
    n = 600
    dates = pd.date_range("2020-01-01", periods=n, freq="B")

    # Synthetic features
    risk_score   = pd.Series(50 + np.cumsum(np.random.randn(n) * 2), index=dates).clip(0, 100)
    # Inject crisis
    risk_score.iloc[300:320] = 25
    risk_score.iloc[320:340] = 35

    features = pd.DataFrame({
        "risk_score":      risk_score,
        "risk_velocity":   risk_score.diff().fillna(0),
        "risk_momentum":   risk_score.diff(5).fillna(0),
        "stability_score": risk_score + np.random.randn(n) * 5,
        "volume_score":    50 + np.random.randn(n) * 10,
        "network_score":   50 + np.random.randn(n) * 8,
        "external_score":  50 + np.random.randn(n) * 12,
        "vol_20d":         0.15 + np.random.randn(n) * 0.05,
        "max_drawdown":    -np.abs(np.random.randn(n) * 0.05),
        "vix_level":       15 + np.random.randn(n) * 5,
        "bank_market_corr":0.6 + np.random.randn(n) * 0.1,
        "vol_zscore":      np.random.randn(n),
        "cvar_5pct":       -0.03 + np.random.randn(n) * 0.01,
    })
    features.fillna(0, inplace=True)

    from risk_model import make_crisis_label   # keep import local
    label = make_crisis_label(risk_score, threshold=40, horizon=3)
    common = features.index.intersection(label.index)
    Xc = features.loc[common]
    yc = label.loc[common]

    print(f"Crisis rate: {yc.mean():.2%} ({yc.sum()} / {len(yc)} days)")

    # ── RF ──
    rf = RandomForestCrisisPredictor(n_estimators=50, train_window=300, min_train=100)
    rf.fit(Xc, yc)
    rf_pred = rf.predict(Xc)
    print(f"\nRF accuracy (in-sample): {(rf_pred == yc.values).mean():.3f}")
    print("Top features:")
    print(rf.feature_importance().head(5))

    # ── HMM ──
    if HMM_AVAILABLE:
        hmm = HiddenMarkovRiskModel(n_states=5)
        hmm.fit(Xc, risk_score.loc[common])
        hmm_states = hmm.predict_states(Xc)
        print(f"\nHMM state distribution:\n{hmm_states.value_counts()}")

    # ── Baselines ──
    lr = LogisticRegressionBaseline()
    lr.fit(Xc, yc)
    lr_pred = lr.predict(Xc)
    print(f"\nLR accuracy (in-sample): {(lr_pred == yc.values).mean():.3f}")

    st = StaticThresholdBaseline(threshold=40.0)
    st_pred = st.predict(Xc)
    print(f"Threshold accuracy: {(st_pred == yc.values).mean():.3f}")
