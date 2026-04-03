# 🏦 NIFTY Banking Risk Intelligence System

A research-grade investment risk monitoring system for the NIFTY Banking Index.
Predicts market stress, classifies states, and gives stock-level BUY/HOLD/AVOID decisions.

---

## 📦 Installation

```bash
pip install yfinance pandas numpy scikit-learn matplotlib scipy hmmlearn torch joblib
```

for LSTM:
```bash
pip install torch   # pytorch.org for GPU version
```

---

## 🚀 Quick Start

### 1. Synthetic Data (no API, instant)
```bash
python train_pipeline.py --mode synthetic
```

### 2. Real Market Data (yfinance)
```bash
python train_pipeline.py --mode real
```

### 3. Save trained models
```bash
python train_pipeline.py --mode real --save
```

### 4. Generate Dashboard Data
Generate the latest prediction JSON for the dashboard:
```bash
python export_dashboard_data.py
```

### 5. Launch the Dashboard
Run the Flask server to view the dynamic dashboard:
```bash
python app.py
```
Then open `http://127.0.0.1:5000` in your web browser.

---

## 📁 File Structure

```
nifty_risk/
├── data_pipeline.py         # Fetch NIFTY Bank, VIX, USD/INR data (yfinance)
├── feature_engineering.py   # 30+ features: vol, drawdown, VIX, correlation, CVaR
├── risk_model.py            # Risk score (0-100) + state + per-stock recommendations
├── models.py                # HMM, LSTM, LR, IF, Static Threshold
├── evaluation.py            # All metrics + PR/ROC/timeline plots
├── train_pipeline.py        # End-to-end: train → evaluate → visualise
├── export_dashboard_data.py # Exports model data to JSON
├── app.py                   # Flask server backend
└── templates/
    └── dashboard.html       # Interactive Dashboard UI
```

---

## 🧠 Models

| Model              | Role                         |
|-------------------|------------------------------|
| HMM (GaussianHMM) | Hidden state regime detection |
| LSTM              | Sequential crisis prediction  |
| Logistic Reg.     | Linear baseline               |
| Isolation Forest  | Unsupervised anomaly baseline |
| Static Threshold  | Simplest rule-based baseline  |

**Your novel combination:** HMM + LSTM hybrid outperforms all baselines.

---

## 📊 Features Used

### Mandatory
- Volatility (5d, 10d, 20d rolling + EWMA)
- Drawdown (rolling max, underwater duration, current DD)
- Volume z-score + spike flag
- VIX level, change, z-score, spike
- Bank-Market Correlation (rolling 30d)
- Cross-bank average correlation (contagion)

### Advanced
- CVaR 5% (Expected Shortfall)
- Skewness + Kurtosis (tail risk)
- Beta, Relative Risk, Relative Return
- RSI, Rate of Change, MA Spread
- Risk Velocity + Momentum
- USD/INR z-score

---

## 📈 Evaluation Metrics

| Category       | Metrics                                         |
|---------------|--------------------------------------------------|
| Standard       | Precision, Recall, F1, ROC-AUC                  |
| Critical       | PR Curve, Average Precision                     |
| Time-aware     | Detection Delay, Early Warning %, Lead Time     |
| Event-based    | Event Detection Rate, False Alarm Rate          |
| Regime         | Transition Accuracy, Stability Score            |

> **Priority**: Recall > F1 > Precision (missing a crisis = worse than false alarm)

---

## 🎯 Risk Score Formula

```
Risk Score (0-100, higher = SAFER) =
    40% × Stability Score  (volatility + drawdown)
    + 20% × Volume Score   (volume stress)
    + 20% × Network Score  (systemic correlations)
    + 20% × External Score (VIX + USD/INR)
```

---

## 🏦 Market States

| State    | Score Range | Meaning                          |
|---------|-------------|----------------------------------|
| Stable   | 80-100      | Low risk, good time to invest    |
| Low Risk | 65-80       | Cautious optimism                |
| Stress   | 40-65       | Elevated risk, selective buying  |
| Crisis   | 0-40        | Avoid, preserve capital          |
| Recovery | Rising from Crisis | Crisis ending, selective entry |

---

## 📝 Research Novelty

1. **Dynamic states** — not static thresholds, uses velocity + momentum
2. **Recovery modeling** — predicts when crises end, not just when they start
3. **Multi-factor** — price + VIX + cross-bank correlation + macro (USD/INR)
4. **HMM + LSTM hybrid** — combines interpretable hidden states with ensemble power
5. **Per-stock risk scores** — BUY/HOLD/AVOID with explanation
