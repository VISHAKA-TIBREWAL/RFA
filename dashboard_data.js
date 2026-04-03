
const IS_REAL_DATA = true;

const STOCKS = [{"name": "HDFCBANK", "ticker": "HDFCBANK", "score": 0, "rec": "AVOID", "conf": "HIGH", "vol": 38.6, "beta": 1.03, "sharpe": -5.16, "mom": 0, "dd": -0.0, "corr": 0.85, "reasons": ["['High volatility (38.6% annualised)', 'Deep drawdown (-18.2% over 20d)', 'Negative momentum \u2014 falling trend']"]}, {"name": "ICICIBANK", "ticker": "ICICIBANK", "score": 0, "rec": "AVOID", "conf": "HIGH", "vol": 27.7, "beta": 0.74, "sharpe": -5.56, "mom": 0, "dd": -0.0, "corr": 0.85, "reasons": ["['High volatility (27.7% annualised)', 'Deep drawdown (-12.4% over 20d)', 'Negative momentum \u2014 falling trend']"]}, {"name": "SBIN", "ticker": "SBIN", "score": 0, "rec": "AVOID", "conf": "HIGH", "vol": 36.1, "beta": 0.84, "sharpe": -5.43, "mom": 0, "dd": -0.0, "corr": 0.85, "reasons": ["['High volatility (36.1% annualised)', 'Deep drawdown (-18.2% over 20d)', 'Negative momentum \u2014 falling trend']"]}, {"name": "AXISBANK", "ticker": "AXISBANK", "score": 0, "rec": "AVOID", "conf": "HIGH", "vol": 37.0, "beta": 1.04, "sharpe": -4.63, "mom": 0, "dd": -0.0, "corr": 0.85, "reasons": ["['High volatility (37.0% annualised)', 'Deep drawdown (-15.2% over 20d)', 'Strongly correlated with market stress', 'Negative momentum \u2014 falling trend']"]}, {"name": "KOTAKBANK", "ticker": "KOTAKBANK", "score": 0, "rec": "AVOID", "conf": "HIGH", "vol": 29.8, "beta": 0.85, "sharpe": -6.06, "mom": 0, "dd": -0.0, "corr": 0.85, "reasons": ["['High volatility (29.8% annualised)', 'Deep drawdown (-14.2% over 20d)', 'Strongly correlated with market stress', 'Negative momentum \u2014 falling trend']"]}, {"name": "INDUSINDBK", "ticker": "INDUSINDBK", "score": 0, "rec": "AVOID", "conf": "HIGH", "vol": 46.4, "beta": 1.19, "sharpe": -5.19, "mom": 0, "dd": -0.0, "corr": 0.85, "reasons": ["['High volatility (46.4% annualised)', 'Deep drawdown (-22.0% over 20d)', 'Negative momentum \u2014 falling trend']"]}, {"name": "BANKBARODA", "ticker": "BANKBARODA", "score": 0, "rec": "AVOID", "conf": "HIGH", "vol": 41.7, "beta": 0.94, "sharpe": -7.05, "mom": 0, "dd": -0.0, "corr": 0.85, "reasons": ["['High volatility (41.7% annualised)', 'Deep drawdown (-19.8% over 20d)', 'Negative momentum \u2014 falling trend']"]}, {"name": "PNB", "ticker": "PNB", "score": 0, "rec": "AVOID", "conf": "HIGH", "vol": 44.6, "beta": 1.07, "sharpe": -5.3, "mom": 0, "dd": -0.0, "corr": 0.85, "reasons": ["['High volatility (44.6% annualised)', 'Deep drawdown (-19.5% over 20d)', 'Negative momentum \u2014 falling trend']"]}, {"name": "IDFCFIRSTB", "ticker": "IDFCFIRSTB", "score": 0, "rec": "AVOID", "conf": "HIGH", "vol": 40.8, "beta": 1.14, "sharpe": -5.43, "mom": 0, "dd": -0.0, "corr": 0.85, "reasons": ["['High volatility (40.8% annualised)', 'Deep drawdown (-17.9% over 20d)', 'Strongly correlated with market stress', 'Negative momentum \u2014 falling trend']"]}, {"name": "FEDERALBNK", "ticker": "FEDERALBNK", "score": 0, "rec": "AVOID", "conf": "HIGH", "vol": 40.6, "beta": 1.07, "sharpe": -3.26, "mom": 0, "dd": -0.0, "corr": 0.85, "reasons": ["['High volatility (40.6% annualised)', 'Deep drawdown (-13.0% over 20d)', 'Negative momentum \u2014 falling trend']"]}];

const MODEL_RESULTS = [{"name": "HMM", "prec": 0.8333, "recall": 0.9375, "f1": 0.8824, "auc": 0.9677, "evDetect": 1.0, "delay": 1.0, "falseAlarm": 0.1111, "stability": 0.9987}, {"name": "LSTM", "prec": 0.9375, "recall": 0.9375, "f1": 0.9375, "auc": 0.9997, "evDetect": 1.0, "delay": 0.0, "falseAlarm": 0.0, "stability": 0.9993}, {"name": "Logistic", "prec": 0.381, "recall": 1.0, "f1": 0.5517, "auc": 0.9997, "evDetect": 1.0, "delay": 0.0, "falseAlarm": 0.5952, "stability": 0.998}, {"name": "Isolation Forest", "prec": 0.0349, "recall": 1.0, "f1": 0.0674, "auc": 0.9997, "evDetect": 1.0, "delay": 0.0, "falseAlarm": 0.963, "stability": 0.9407}, {"name": "Static Threshold", "prec": 0.8667, "recall": 0.8125, "f1": 0.8387, "auc": 0.9061, "evDetect": 1.0, "delay": 0.0, "falseAlarm": 0.0667, "stability": 0.998}];

const FEATURES = [{"name": "risk_velocity", "imp": 0.142}, {"name": "max_drawdown", "imp": 0.109}, {"name": "vix_level", "imp": 0.098}, {"name": "bank_market_corr", "imp": 0.058}];

const timeline = {
    scores: [75.7, 73.4, 72.1, 73.6, 74.3, 74.9, 75.4, 75.6, 75.7, 75.8, 76.1, 76.7, 75.6, 75.2, 74.9, 75.1, 75.0, 75.1, 74.7, 74.5, 74.6, 74.8, 75.4, 75.7, 76.2, 75.9, 75.7, 75.5, 75.5, 75.1, 75.0, 74.7, 75.0, 75.2, 74.3, 73.4, 72.7, 72.9, 73.3, 73.6, 73.4, 72.5, 71.7, 71.3, 70.3, 70.1, 71.0, 71.5, 71.8, 71.6, 71.5, 71.9, 72.1, 72.4, 72.6, 72.9, 73.4, 73.4, 72.5, 72.1, 72.2, 72.8, 71.8, 70.2, 71.1, 71.3, 71.8, 72.2, 71.7, 69.2, 67.0, 66.9, 66.2, 64.2, 64.4, 62.5, 62.1, 60.5, 60.7, 61.3, 61.8, 60.5, 60.2, 58.2, 57.8, 57.3, 55.8, 54.1, 54.8, 55.5],
    states: ["Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk", "Stress", "Stress", "Stress", "Stress", "Stress", "Stress", "Stress", "Stress", "Stress", "Stress", "Stress", "Stress", "Stress", "Stress", "Stress", "Stress", "Stress"],
    dates: ["Nov 20", "Nov 21", "Nov 24", "Nov 25", "Nov 26", "Nov 27", "Nov 28", "Dec 01", "Dec 02", "Dec 03", "Dec 04", "Dec 05", "Dec 08", "Dec 09", "Dec 10", "Dec 11", "Dec 12", "Dec 15", "Dec 16", "Dec 17", "Dec 18", "Dec 19", "Dec 22", "Dec 23", "Dec 24", "Dec 26", "Dec 29", "Dec 30", "Dec 31", "Jan 01", "Jan 02", "Jan 05", "Jan 06", "Jan 07", "Jan 08", "Jan 09", "Jan 12", "Jan 13", "Jan 14", "Jan 16", "Jan 19", "Jan 20", "Jan 21", "Jan 22", "Jan 23", "Jan 27", "Jan 28", "Jan 29", "Jan 30", "Feb 02", "Feb 03", "Feb 04", "Feb 05", "Feb 06", "Feb 09", "Feb 10", "Feb 11", "Feb 12", "Feb 13", "Feb 16", "Feb 17", "Feb 18", "Feb 19", "Feb 20", "Feb 23", "Feb 24", "Feb 25", "Feb 26", "Feb 27", "Mar 02", "Mar 04", "Mar 05", "Mar 06", "Mar 09", "Mar 10", "Mar 11", "Mar 12", "Mar 13", "Mar 16", "Mar 17", "Mar 18", "Mar 19", "Mar 20", "Mar 23", "Mar 24", "Mar 25", "Mar 27", "Mar 30", "Apr 01", "Apr 02"]
};

const latestScore = 55.5;
const latestState = "Stress";
const prevScore = 54.8;
const velocity = 0.7;
const momentum5 = -2.3;

const components = {"stability": 73.34486245931839, "volume": 50.0, "network": 14.61052057557289, "external": 69.8322516987858};

const macroData = {
    vix: 25.52,
    vixDelta: 0,
    usdInr: 92.64,
    usdDelta: 0
};

const crisisProb = Math.round(Math.max(0, Math.min(98, 100 - latestScore*1.05)));

const TRANSITION = {
  "Stable":   {Stable:0.90,LowRisk:0.08,Stress:0.02,Crisis:0.00,Recovery:0.00},
  "Low Risk": {Stable:0.15,LowRisk:0.70,Stress:0.13,Crisis:0.02,Recovery:0.00},
  "Stress":   {Stable:0.05,LowRisk:0.15,Stress:0.55,Crisis:0.20,Recovery:0.05},
  "Crisis":   {Stable:0.00,LowRisk:0.05,Stress:0.30,Crisis:0.45,Recovery:0.20},
  "Recovery": {Stable:0.10,LowRisk:0.35,Stress:0.35,Crisis:0.05,Recovery:0.15},
};
const nextProbs = TRANSITION[latestState] || TRANSITION["Stress"];
const recoverChance = Math.round(
  ((nextProbs.Stable||0)+(nextProbs.LowRisk||0)+(nextProbs.Recovery||0))*100
);
