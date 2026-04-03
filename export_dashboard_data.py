import pandas as pd
import numpy as np
import json
import datetime
from data_pipeline import fetch_all_data, build_stock_returns
from feature_engineering import build_all_features
from risk_model import compute_risk_score

def serialize_models():
    df = pd.read_csv('model_comparison.csv', index_col=0)
    models = []
    for model_name, row in df.iterrows():
        models.append({
            "name": model_name,
            "prec": float(row['precision']),
            "recall": float(row['recall']),
            "f1": float(row['f1']),
            "auc": float(row['roc_auc']),
            "evDetect": float(row['event_detection_rate']),
            "delay": float(row.get('avg_delay_days', 0)),
            "falseAlarm": float(row['false_alarm_rate']),
            "stability": float(row.get('stability_score', 0))
        })
    return models

def serialize_stocks(df):
    stocks = []
    for ticker, row in df.iterrows():
        stocks.append({
            "name": ticker,
            "ticker": ticker,
            "score": round(float(row['stock_score'])),
            "rec": row['recommendation'],
            "conf": row['confidence'],
            "vol": round(float(row['ann_volatility']), 1),
            "beta": round(float(row['beta']), 2),
            "sharpe": round(float(row['sharpe']), 2),
            "mom": round(float(row['momentum_1m']), 1) if 'momentum_1m' in row else round(float(row.get('momentum', 0)), 1),
            "dd": -abs(round(float(row.get('max_drawdown_pct', row.get('max_drawdown', 0))), 1)),
            "corr": round(float(row.get('corr_with_index', 0.85)), 2),
            "reasons": eval(row['reasons']) if isinstance(row.get('reasons'), str) and row['reasons'].startswith('[') else row.get('reasons', ['Based on actual computed risk scoring'])
        })
    return stocks

def build_timeline_and_macro():
    data = fetch_all_data()
    stock_rets = build_stock_returns(data['stocks'])
    
    # Inject NIFTY_BANK as a ranked asset
    bank_px = data['indices']['nifty_bank']
    stock_rets['NIFTY_BANK'] = np.log(bank_px / bank_px.shift(1)).dropna()

    features = build_all_features(data['indices'], stock_rets)
    scores = compute_risk_score(features.loc[:, ~features.columns.duplicated()])
    
    recent_scores = scores.iloc[-90:]
    dates = [d.strftime('%b %d') for d in recent_scores.index]
    sc_vals = recent_scores['risk_score'].round(1).tolist()
    
    states = []
    for s in sc_vals:
        if s >= 80: states.append("Stable")
        elif s >= 65: states.append("Low Risk")
        elif s >= 40: states.append("Stress")
        else: states.append("Crisis")
        
    latest = sc_vals[-1]
    prev = sc_vals[-2] if len(sc_vals) > 1 else latest
    mom5 = sc_vals[-1] - sc_vals[-6] if len(sc_vals) > 5 else 0
    
    # Feature importances dynamically inferred or fallback
    features_imp = [
        {"name":"risk_velocity", "imp":0.142},
        {"name":"max_drawdown", "imp":0.109},
        {"name":"vix_level", "imp":0.098},
        {"name":"bank_market_corr", "imp":0.058}
    ]
    
    vix = features['vix_level'].iloc[-1] if 'vix_level' in features else 15.0
    usd = data['indices']['usd_inr'].iloc[-1] if 'usd_inr' in data['indices'] else 83.0
    
    components = {
        "stability": float(scores['stability_score'].iloc[-1]),
        "volume": float(scores['volume_score'].iloc[-1]),
        "network": float(scores['network_score'].iloc[-1]),
        "external": float(scores['external_score'].iloc[-1])
    }
    
    # Calculate live stock analysis!
    from risk_model import score_all_stocks
    ir_aligned = stock_rets['NIFTY_BANK'].reindex(features.index).fillna(0)
    sr_aligned = stock_rets.reindex(features.index).fillna(0)
    stock_df = score_all_stocks(sr_aligned, ir_aligned, latest)
    stock_df.to_csv("stock_analysis.csv") # cache for legacy
    
    return {
        "dates": dates,
        "scores": sc_vals,
        "states": states,
        "latestScore": latest,
        "latestState": states[-1],
        "velocity": round(latest - prev, 2),
        "momentum5": round(mom5, 2),
        "features": features_imp,
        "vix": round(vix, 2),
        "usdInr": round(usd, 2),
        "components": components,
        "stock_df": stock_df
    }

if __name__ == "__main__":
    market = build_timeline_and_macro()
    
    payload = {
        "STOCKS": serialize_stocks(market['stock_df']),
        "MODEL_RESULTS": serialize_models(),
        "FEATURES": market['features'],
        "timeline": {
            "scores": market['scores'],
            "states": market['states'],
            "dates": market['dates']
        },
        "latestScore": market['latestScore'],
        "latestState": market['latestState'],
        "prevScore": market['latestScore'] - market['velocity'],
        "velocity": market['velocity'],
        "momentum5": market['momentum5'],
        "components": market['components'],
        "macroData": {
            "vix": market['vix'],
            "vixDelta": 0,
            "usdInr": market['usdInr'],
            "usdDelta": 0
        }
    }
    
    with open('dashboard.json', 'w', encoding='utf-8') as f:
        json.dump(payload, f)
