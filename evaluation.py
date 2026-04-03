"""
evaluation.py
Full evaluation framework for NIFTY Banking Risk System.

Metrics:
  Standard:     Precision, Recall, F1, ROC-AUC
  Time-aware:   Detection Delay, Lead Time, Early Warning Accuracy
  Event-based:  Event Detection Rate, False Alarm Rate
  Regime:       Transition Accuracy, Stability Score
  Plots:        PR Curve, ROC Curve, Detection Timeline, Model Comparison Table
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, roc_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay,
)


# ═══════════════════════════════════════════════════════════════════════════════
# CORE CLASSIFICATION METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                             y_proba: np.ndarray = None) -> dict:
    """Standard classification metrics."""
    mask = ~np.isnan(y_pred.astype(float))
    yt   = y_true[mask].astype(int)
    yp   = y_pred[mask].astype(int)

    metrics = {
        "precision":       round(precision_score(yt, yp, zero_division=0), 4),
        "recall":          round(recall_score(yt, yp, zero_division=0),    4),
        "f1":              round(f1_score(yt, yp, zero_division=0),        4),
        "n_test":          int(mask.sum()),
        "crisis_rate_pct": round(yt.mean() * 100, 2),
    }

    if y_proba is not None:
        mask_p = ~np.isnan(y_proba.astype(float))
        common = mask & mask_p
        if common.sum() > 10:
            try:
                metrics["roc_auc"] = round(
                    roc_auc_score(y_true[common].astype(int), y_proba[common]), 4)
                metrics["avg_precision"] = round(
                    average_precision_score(y_true[common].astype(int), y_proba[common]), 4)
            except ValueError:
                pass

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# TIME-AWARE METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def _find_crisis_episodes(y_true: np.ndarray, min_gap: int = 5) -> list:
    """Return list of (start_idx, end_idx) for each crisis episode."""
    episodes = []
    in_crisis = False
    start     = 0
    for i, v in enumerate(y_true):
        if v == 1 and not in_crisis:
            in_crisis = True
            start     = i
        elif v == 0 and in_crisis:
            in_crisis = False
            episodes.append((start, i - 1))
    if in_crisis:
        episodes.append((start, len(y_true) - 1))

    # merge close episodes
    merged = []
    for ep in episodes:
        if merged and ep[0] - merged[-1][1] < min_gap:
            merged[-1] = (merged[-1][0], ep[1])
        else:
            merged.append(list(ep))
    return merged


def detection_delay(y_true: np.ndarray, y_pred: np.ndarray,
                    dates: pd.DatetimeIndex = None) -> dict:
    """
    For each actual crisis episode, measure how many days after onset
    the model first raised an alert. Negative = early warning.
    """
    episodes = _find_crisis_episodes(y_true)
    delays   = []

    for start, end in episodes:
        # Find first prediction = 1 in window [start-10 : end+5]
        window_start = max(0, start - 10)
        window_end   = min(len(y_pred), end + 5)
        window_pred  = y_pred[window_start:window_end]

        first_alert = np.where(window_pred == 1)[0]
        if len(first_alert) > 0:
            # delay relative to crisis start
            alert_idx = window_start + first_alert[0]
            delays.append(alert_idx - start)   # negative = early
        else:
            delays.append(np.nan)              # missed crisis

    detected  = [d for d in delays if not np.isnan(d)]
    missed    = sum(1 for d in delays if np.isnan(d))

    return {
        "n_episodes":          len(episodes),
        "detected":            len(detected),
        "missed":              missed,
        "event_detection_rate":round(len(detected) / max(len(episodes), 1), 4),
        "avg_delay_days":      round(float(np.nanmean(delays)), 2) if delays else np.nan,
        "early_warning_pct":   round(sum(1 for d in delays if isinstance(d, float) and d < 0) / max(len(episodes), 1), 4),
        "delays_per_episode":  delays,
    }


def false_alarm_rate(y_true: np.ndarray, y_pred: np.ndarray,
                     window: int = 5) -> dict:
    """
    False alarm: prediction=1 with no actual crisis in next `window` days.
    """
    false_alarms = 0
    true_alarms  = 0

    for i, p in enumerate(y_pred):
        if p == 1:
            # check if real crisis in next window
            horizon = y_true[i:min(i + window, len(y_true))]
            if horizon.sum() > 0:
                true_alarms  += 1
            else:
                false_alarms += 1

    total_alerts = true_alarms + false_alarms
    return {
        "total_alerts":    total_alerts,
        "true_alarms":     true_alarms,
        "false_alarms":    false_alarms,
        "false_alarm_rate":round(false_alarms / max(total_alerts, 1), 4),
        "precision@window": round(true_alarms / max(total_alerts, 1), 4),
    }


def regime_transition_accuracy(true_states: pd.Series,
                                pred_states: pd.Series) -> dict:
    """
    Checks: when a state transition actually occurs, did the model predict it?
    Transition = consecutive states differ.
    """
    true_trans = (true_states != true_states.shift(1)).astype(int).iloc[1:]
    pred_trans = (pred_states != pred_states.shift(1)).astype(int).iloc[1:]

    common = true_trans.index.intersection(pred_trans.index)
    if len(common) == 0:
        return {"error": "No common index for transition comparison"}

    tt = true_trans.loc[common].values
    pt = pred_trans.loc[common].values

    tp = ((tt == 1) & (pt == 1)).sum()
    fp = ((tt == 0) & (pt == 1)).sum()
    fn = ((tt == 1) & (pt == 0)).sum()

    prec  = tp / max(tp + fp, 1)
    rec   = tp / max(tp + fn, 1)
    f1    = 2 * prec * rec / max(prec + rec, 1e-9)

    return {
        "transition_precision": round(prec, 4),
        "transition_recall":    round(rec,  4),
        "transition_f1":        round(f1,   4),
        "n_true_transitions":   int(tt.sum()),
        "n_pred_transitions":   int(pt.sum()),
    }


def model_stability_score(y_pred: np.ndarray) -> float:
    """Measures how much the model flickers (changes between 0/1 rapidly)."""
    if len(y_pred) < 2:
        return 1.0
    flips = (y_pred[1:] != y_pred[:-1]).mean()
    return round(1 - flips, 4)   # 1 = perfectly stable


# ═══════════════════════════════════════════════════════════════════════════════
# FULL EVALUATION REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_model(name: str,
                   y_true: np.ndarray,
                   y_pred: np.ndarray,
                   y_proba: np.ndarray,
                   dates: pd.DatetimeIndex = None) -> dict:
    """One-shot comprehensive evaluation for a single model."""
    clf      = classification_metrics(y_true, y_pred, y_proba)
    dd       = detection_delay(y_true, y_pred, dates)
    far      = false_alarm_rate(y_true, y_pred)
    stab     = model_stability_score(y_pred[~np.isnan(y_pred.astype(float))])

    return {
        "model":               name,
        **clf,
        **dd,
        **far,
        "stability_score":     stab,
    }


def compare_models(results: list) -> pd.DataFrame:
    """
    Takes list of evaluate_model dicts, returns comparison DataFrame.
    """
    keep_cols = [
        "model", "precision", "recall", "f1", "roc_auc",
        "event_detection_rate", "avg_delay_days",
        "false_alarm_rate", "stability_score",
    ]
    rows = []
    for r in results:
        row = {k: r.get(k, np.nan) for k in keep_cols}
        rows.append(row)

    df = pd.DataFrame(rows).set_index("model")
    return df.round(4)


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

COLORS = {
    "Random Forest": "#e63946",
    "LSTM":          "#457b9d",
    "HMM":           "#2a9d8f",
    "Logistic":      "#e9c46a",
    "Isolation Forest": "#f4a261",
    "Static Threshold": "#a8dadc",
}


def plot_pr_curves(models_dict: dict, y_true: np.ndarray,
                   save_path: str = "pr_curves.png"):
    """
    models_dict = {"Model Name": y_proba_array, ...}
    Plots Precision-Recall curves for all models.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    for name, proba in models_dict.items():
        mask = ~np.isnan(proba.astype(float))
        yt   = y_true[mask].astype(int)
        yp   = proba[mask]
        if len(np.unique(yt)) < 2:
            continue
        prec, rec, _ = precision_recall_curve(yt, yp)
        ap = average_precision_score(yt, yp)
        color = COLORS.get(name, "#ffffff")
        ax.plot(rec, prec, color=color, lw=2,
                label=f"{name} (AP={ap:.3f})")

    ax.axhline(y_true.mean(), color="#555", linestyle="--", alpha=0.6,
               label=f"Baseline (AP={y_true.mean():.3f})")

    ax.set_xlabel("Recall",    color="#e6edf3", fontsize=12)
    ax.set_ylabel("Precision", color="#e6edf3", fontsize=12)
    ax.set_title("Precision-Recall Curves — Crisis Detection",
                 color="#e6edf3", fontsize=14, fontweight="bold")
    ax.tick_params(colors="#8b949e")
    ax.legend(facecolor="#21262d", labelcolor="#e6edf3", fontsize=9)
    ax.grid(alpha=0.15, color="#30363d")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"💾 Saved: {save_path}")


def plot_roc_curves(models_dict: dict, y_true: np.ndarray,
                    save_path: str = "roc_curves.png"):
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    for name, proba in models_dict.items():
        mask = ~np.isnan(proba.astype(float))
        yt   = y_true[mask].astype(int)
        yp   = proba[mask]
        if len(np.unique(yt)) < 2:
            continue
        fpr, tpr, _ = roc_curve(yt, yp)
        auc          = roc_auc_score(yt, yp)
        color = COLORS.get(name, "#ffffff")
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate", color="#e6edf3", fontsize=12)
    ax.set_ylabel("True Positive Rate",  color="#e6edf3", fontsize=12)
    ax.set_title("ROC Curves — Crisis Detection",
                 color="#e6edf3", fontsize=14, fontweight="bold")
    ax.tick_params(colors="#8b949e")
    ax.legend(facecolor="#21262d", labelcolor="#e6edf3", fontsize=9)
    ax.grid(alpha=0.15, color="#30363d")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"💾 Saved: {save_path}")


def plot_detection_timeline(dates: pd.DatetimeIndex,
                             risk_scores: np.ndarray,
                             y_true: np.ndarray,
                             predictions_dict: dict,
                             save_path: str = "detection_timeline.png"):
    """
    Timeline showing risk score with actual crises and model alerts overlaid.
    """
    n_models = len(predictions_dict)
    fig      = plt.figure(figsize=(16, 4 + n_models * 1.5))
    fig.patch.set_facecolor("#0d1117")

    gs = gridspec.GridSpec(1 + n_models, 1, hspace=0.08)

    # ── Risk score panel ──
    ax0 = fig.add_subplot(gs[0])
    ax0.set_facecolor("#161b22")

    ax0.plot(dates, risk_scores, color="#58a6ff", lw=1.2, label="Risk Score")
    ax0.axhline(40, color="#d50000", ls="--", lw=0.8, alpha=0.6, label="Crisis threshold")
    ax0.axhline(65, color="#ffab00", ls="--", lw=0.8, alpha=0.6, label="Stress threshold")

    # Shade actual crisis periods
    crisis_idx = np.where(y_true == 1)[0]
    for ci in crisis_idx:
        ax0.axvspan(dates[ci], dates[min(ci + 1, len(dates) - 1)],
                    alpha=0.18, color="#d50000")

    ax0.set_ylabel("Risk Score", color="#e6edf3", fontsize=9)
    ax0.set_ylim(0, 105)
    ax0.tick_params(colors="#8b949e", labelbottom=False)
    ax0.legend(loc="upper right", facecolor="#21262d",
               labelcolor="#e6edf3", fontsize=7)
    ax0.grid(alpha=0.1, color="#30363d")
    ax0.set_title("Crisis Detection Timeline — All Models",
                  color="#e6edf3", fontsize=13, fontweight="bold", pad=8)
    for spine in ax0.spines.values():
        spine.set_edgecolor("#30363d")

    # ── Per-model alert panels ──
    for i, (name, preds) in enumerate(predictions_dict.items()):
        ax = fig.add_subplot(gs[i + 1])
        ax.set_facecolor("#161b22")
        color = COLORS.get(name, "#ffffff")

        # Shade actual crises
        for ci in crisis_idx:
            ax.axvspan(dates[ci], dates[min(ci + 1, len(dates) - 1)],
                       alpha=0.12, color="#d50000")

        # Model alerts
        mask    = ~np.isnan(preds.astype(float)) & (preds == 1)
        alert_d = dates[mask]
        ax.vlines(alert_d, 0, 1, color=color, lw=1.2, alpha=0.8)

        ax.set_ylabel(name, color=color, fontsize=8, rotation=0,
                      ha="right", va="center", labelpad=70)
        ax.set_yticks([])
        ax.tick_params(colors="#8b949e",
                       labelbottom=(i == n_models - 1))
        ax.grid(alpha=0.05, color="#30363d")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"💾 Saved: {save_path}")


def plot_model_comparison_table(comparison_df: pd.DataFrame,
                                 save_path: str = "model_comparison.png"):
    """
    Renders comparison DataFrame as a styled table image.
    """
    display_cols = [c for c in [
        "precision", "recall", "f1", "roc_auc",
        "event_detection_rate", "avg_delay_days",
        "false_alarm_rate", "stability_score",
    ] if c in comparison_df.columns]

    df_show = comparison_df[display_cols].copy()
    df_show = df_show.rename(columns={
        "precision":            "Precision",
        "recall":               "Recall",
        "f1":                   "F1",
        "roc_auc":              "ROC-AUC",
        "event_detection_rate": "Event Detect",
        "avg_delay_days":       "Delay (days)",
        "false_alarm_rate":     "False Alarm",
        "stability_score":      "Stability",
    })

    fig, ax = plt.subplots(figsize=(14, 0.6 * (len(df_show) + 2)))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    ax.axis("off")

    col_labels = ["Model"] + df_show.columns.tolist()
    cell_text  = [[idx] + [f"{v:.3f}" if isinstance(v, float) else str(v)
                            for v in row]
                   for idx, row in df_show.iterrows()]

    tbl = ax.table(
        cellText    = cell_text,
        colLabels   = col_labels,
        loc         = "center",
        cellLoc     = "center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.8)

    # Style header
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#1f6feb")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Style rows
    row_colors = ["#161b22", "#21262d"]
    for i in range(1, len(cell_text) + 1):
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor(row_colors[i % 2])
            tbl[i, j].set_text_props(color="#e6edf3")
            tbl[i, j].set_edgecolor("#30363d")

    # Highlight best F1 row
    f1_col_idx = col_labels.index("F1") if "F1" in col_labels else None
    if f1_col_idx is not None:
        f1_vals = df_show["F1"].dropna()
        if len(f1_vals):
            best_row = f1_vals.idxmax()
            best_idx = list(df_show.index).index(best_row) + 1
            for j in range(len(col_labels)):
                tbl[best_idx, j].set_facecolor("#0d4429")

    ax.set_title("Model Comparison — NIFTY Banking Risk System",
                 color="#e6edf3", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"💾 Saved: {save_path}")


def plot_feature_importance(importance: pd.Series,
                             save_path: str = "feature_importance.png",
                             top_n: int = 15):
    """Bar chart of feature importances (for RF model)."""
    top = importance.head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    bars = ax.barh(top.index[::-1], top.values[::-1], color="#1f6feb", alpha=0.85)
    ax.set_xlabel("Importance", color="#e6edf3", fontsize=11)
    ax.set_title(f"Top {top_n} Feature Importances (Random Forest)",
                 color="#e6edf3", fontsize=13, fontweight="bold")
    ax.tick_params(colors="#8b949e")
    ax.grid(axis="x", alpha=0.15, color="#30363d")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"💾 Saved: {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    np.random.seed(42)
    n     = 500
    dates = pd.date_range("2020-01-01", periods=n, freq="B")

    # Synthetic true labels (two crisis episodes)
    y_true = np.zeros(n, dtype=int)
    y_true[150:170] = 1
    y_true[320:345] = 1

    # Synthetic risk score
    rs = 70 + np.random.randn(n) * 5
    rs[145:175] = 30
    rs[315:350] = 25

    # Synthetic model predictions
    def noisy_pred(y, noise=0.15, delay=2):
        p = np.roll(y, delay).astype(float)
        mask = np.random.rand(n) < noise
        p[mask] = 1 - p[mask]
        return p

    models_proba = {
        "Random Forest":    np.clip(noisy_pred(y_true, noise=0.05, delay=1) + np.random.randn(n)*0.05, 0, 1),
        "LSTM":             np.clip(noisy_pred(y_true, noise=0.08, delay=2) + np.random.randn(n)*0.08, 0, 1),
        "Logistic":         np.clip(noisy_pred(y_true, noise=0.12, delay=3) + np.random.randn(n)*0.10, 0, 1),
        "Isolation Forest": np.clip(noisy_pred(y_true, noise=0.18, delay=4) + np.random.randn(n)*0.12, 0, 1),
        "Static Threshold": np.clip((40 - rs) / 40, 0, 1),
    }
    models_pred = {k: (v >= 0.5).astype(int) for k, v in models_proba.items()}

    # ── Evaluate all ──
    all_results = []
    for name in models_proba:
        r = evaluate_model(name, y_true,
                           models_pred[name].astype(float),
                           models_proba[name], dates)
        all_results.append(r)
        print(f"\n{name}:")
        for k, v in r.items():
            if k not in ("model", "delays_per_episode"):
                print(f"  {k:<25} {v}")

    cmp = compare_models(all_results)
    print("\n📊 Model Comparison Table:")
    print(cmp.to_string())

    # ── Save plots ──
    plot_pr_curves(models_proba, y_true,         "pr_curves.png")
    plot_roc_curves(models_proba, y_true,        "roc_curves.png")
    plot_detection_timeline(dates, rs, y_true,
                            models_pred,         "detection_timeline.png")
    plot_model_comparison_table(cmp,             "model_comparison.png")
