"""Purged chronological training; save model identity with feature schema."""
import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from lightgbm import LGBMClassifier
import config
from technical_indicators import FEATURE_COLUMNS

MODEL_SAVE_PATH = './ml_model/'

def train_and_evaluate_models_for_streamlit(prepared_df, tickers, prediction_horizon_days,
                                            target_rise_percent, model_dir=MODEL_SAVE_PATH):
    if prediction_horizon_days < 1 or target_rise_percent < 0:
        raise ValueError('Invalid prediction settings')
    if prepared_df.attrs.get('prediction_horizon_days') not in (None, prediction_horizon_days):
        raise ValueError('Prepared labels have a different horizon; prepare data again')
    if prepared_df.attrs.get('target_rise_percent') not in (None, target_rise_percent):
        raise ValueError('Prepared labels have a different rise threshold; prepare data again')
    folder = Path(model_dir)
    folder.mkdir(parents=True, exist_ok=True)
    metrics = {}
    for ticker in tickers:
        frame = prepared_df[prepared_df.Ticker == ticker].sort_index().dropna(subset=['target','label_end']+FEATURE_COLUMNS)
        frame = frame[frame.label_end <= frame.index.max()]
        if len(frame) < 30:
            raise ValueError(f'{ticker}: at least 30 fully labeled feature rows required')
        boundary = frame.index[int(len(frame)*.8)]
        train = frame[(frame.index < boundary) & (frame.label_end < boundary)]
        test = frame[frame.index >= boundary]
        if len(train) < 10 or test.empty or train.target.nunique() < 2:
            raise ValueError(f'{ticker}: insufficient purged data or only one training class')
        model = VotingClassifier([
            ('rf',RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)),
            ('lgbm',LGBMClassifier(n_estimators=100, random_state=42, n_jobs=1, verbosity=-1))], voting='hard')
        model.fit(train[FEATURE_COLUMNS], train.target.astype(int))
        predictions = model.predict(test[FEATURE_COLUMNS])
        metrics[ticker] = {'Accuracy':float(accuracy_score(test.target,predictions)),
                          'Precision (Class 1)':float(precision_score(test.target,predictions,zero_division=0)),
                          'Recall (Class 1)':float(recall_score(test.target,predictions,zero_division=0)),
                          'F1 Score (Class 1)':float(f1_score(test.target,predictions,zero_division=0)),
                          'train_count':len(train),'test_count':len(test),
                          'first_test_signal':boundary.isoformat(),
                          'last_train_label':pd.Timestamp(train.label_end.max()).isoformat()}
        name = f'ensemble_voting_classifier_hard_{ticker}.pkl'
        joblib.dump(model, folder/name)
        metadata = dict(symbol=ticker,features=FEATURE_COLUMNS,
                        trained_through_label=pd.Timestamp(train.label_end.max()).isoformat(),
                        validation_through=pd.Timestamp(test.label_end.max()).isoformat(),
                        prediction_horizon_days=int(prediction_horizon_days),target_rise_percent=float(target_rise_percent))
        (folder/(name+'.json')).write_text(json.dumps(metadata,indent=2))
    return metrics, FEATURE_COLUMNS.copy()

def main():
    from data_handler import get_prepared_data
    frame = get_prepared_data()
    cutoff = frame.index.unique().sort_values()[int(frame.index.nunique()*.8)]
    training = frame.loc[frame.index < cutoff].copy()
    metrics,_ = train_and_evaluate_models_for_streamlit(training, config.STOCK_UNIVERSE,
                   config.PREDICTION_HORIZON_DAYS, config.TARGET_RISE_PERCENT)
    print(json.dumps(metrics,indent=2))

if __name__ == '__main__':
    main()
