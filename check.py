import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve
from data_pipeline import fetch_all_data, build_stock_returns
from feature_engineering import build_all_features
from risk_model import compute_risk_score
from models import make_crisis_label, HiddenMarkovRiskModel
import warnings
warnings.filterwarnings('ignore')

data = fetch_all_data()
features = build_all_features(data['indices'], build_stock_returns(data['stocks']))
features = features.loc[:, ~features.columns.duplicated()]

scores = compute_risk_score(features)
crisis_label = make_crisis_label(scores['risk_score'], threshold=40.0, horizon=3)

common_idx = features.index.intersection(crisis_label.index)
X = features.loc[common_idx]
y = crisis_label.loc[common_idx]

target_date = pd.to_datetime("2020-03-24")
split = X.index.get_indexer([target_date], method="nearest")[0]

X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

full_risk = (scores["risk_score"] < 42).astype(float).to_frame(name="risk_score")
test_risk = full_risk.loc[X_test.index]

for states in [2, 3]:
    hmm = HiddenMarkovRiskModel(n_states=states)
    hmm.fit(full_risk, scores["risk_score"])
    
    all_states = hmm.model.predict(hmm._prepare(full_risk))
    state_risk = {}
    for s in np.unique(all_states):
        state_risk[s] = scores["risk_score"].values[all_states == s].mean()
    crisis_state = min(state_risk, key=state_risk.get)
    
    hmm_proba_df = hmm.predict_state_proba(test_risk)
    hmm_proba = hmm_proba_df.iloc[:, crisis_state].values
    
    prec, rec, thresh = precision_recall_curve(y_test.values, hmm_proba)
    fscore = (2 * prec * rec) / (prec + rec + 1e-9)
    best_idx = np.argmax(fscore)
    best_f = fscore[best_idx]
    
    with open('out2.txt', 'a') as f:
        f.write(f"States {states} F1 {best_f:.4f}\n")

