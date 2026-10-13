import json
from pathlib import Path
import joblib
import pandas as pd
from technical_indicators import FEATURE_COLUMNS

ML_MODEL_FEATURE_COLUMNS = FEATURE_COLUMNS

def generate_signals_from_ml(df_with_indicators, model_dir, ml_model_feature_columns=None):
    columns = FEATURE_COLUMNS if ml_model_feature_columns is None else ml_model_feature_columns
    if columns != FEATURE_COLUMNS:
        raise ValueError('Feature schema differs; retrain models')
    frames = []
    for ticker, frame in df_with_indicators.groupby('Ticker', sort=True):
        path = Path(model_dir)/f'ensemble_voting_classifier_hard_{ticker}.pkl'
        metadata_path = Path(str(path)+'.json')
        if not path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f'{ticker}: model and metadata required; train first')
        metadata = json.loads(metadata_path.read_text())
        if metadata.get('symbol') != ticker or metadata.get('features') != columns:
            raise ValueError(f'{ticker}: model identity/feature mismatch')
        # Reserve every date touched by training OR reported model validation.
        cutoff = pd.Timestamp(metadata['validation_through'])
        frame = frame.loc[frame.index > cutoff].sort_index()
        if frame.empty:
            raise ValueError(f'{ticker}: no untouched dates after {cutoff.date()}; download a later period')
        if frame[columns].isna().any().any():
            raise ValueError(f'{ticker}: missing prediction features')
        model = joblib.load(path)  # trusted locally trained artifacts only
        frames.append(pd.DataFrame({'signal':model.predict(frame[columns]).astype(int),'Ticker':ticker},index=frame.index))
    return pd.concat(frames).sort_index(kind='stable') if frames else pd.DataFrame(columns=['signal','Ticker'])
