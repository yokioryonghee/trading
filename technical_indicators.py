"""Causal OHLCV features with explicit future-label boundaries."""
import numpy as np
import pandas as pd

FEATURE_COLUMNS = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'EMA_short',
                   'EMA_long', 'RSI', 'MACD', 'MACD_hist', 'ATR_14', 'return_1d',
                   'high_low_diff_pct', 'open_close_diff_pct', 'volume_change_pct',
                   'close_div_ema_long']

def normalize_ohlcv(raw):
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        fields = {'open', 'high', 'low', 'close', 'volume', 'adj close'}
        levels = [i for i in range(df.columns.nlevels)
                  if fields & {str(v).lower() for v in df.columns.get_level_values(i)}]
        if len(levels) != 1:
            raise ValueError('Cannot identify OHLCV field level')
        field_level = levels[0]
        other = [i for i in range(df.columns.nlevels) if i != field_level]
        if any(len(df.columns.get_level_values(i).unique()) != 1 for i in other):
            raise ValueError('Expected one ticker per OHLCV frame')
        df.columns = df.columns.get_level_values(field_level)
    df.columns = [str(c).upper() for c in df.columns]
    required = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']
    if not set(required) <= set(df.columns) or df.columns.duplicated().any():
        raise ValueError('Expected unique OPEN,HIGH,LOW,CLOSE,VOLUME columns')
    df.index = pd.to_datetime(df.index, errors='coerce')
    df = df.loc[df.index.notna(), required].sort_index()
    if df.index.duplicated().any():
        raise ValueError('Duplicate trading dates')
    df = df.apply(pd.to_numeric, errors='coerce')
    if not np.isfinite(df.to_numpy()).all() or (df.iloc[:, :4] <= 0).any().any() or (df.VOLUME < 0).any():
        raise ValueError('OHLCV must be finite, prices positive and volume nonnegative')
    if ((df.HIGH < df[['OPEN', 'CLOSE']].max(axis=1)) |
        (df.LOW > df[['OPEN', 'CLOSE']].min(axis=1)) | (df.LOW > df.HIGH)).any():
        raise ValueError('Inconsistent OHLC range')
    df.index.name = 'Date'
    return df

def prepare_features(raw, ema_short=20, ema_long=60, rsi_period=21,
                     macd_fast=12, macd_slow=26, macd_signal=9,
                     horizon=20, rise=.1):
    periods = [ema_short, ema_long, rsi_period, macd_fast, macd_slow, macd_signal, horizon]
    if any(not isinstance(v, (int, np.integer)) or v < 1 for v in periods):
        raise ValueError('Indicator periods and horizon must be positive integers')
    if not np.isfinite(rise) or rise < 0 or ema_short >= ema_long or macd_fast >= macd_slow:
        raise ValueError('Invalid feature parameters')
    df = normalize_ohlcv(raw)
    close = df.CLOSE
    df['EMA_short'] = close.ewm(span=ema_short, adjust=False, min_periods=ema_short).mean()
    df['EMA_long'] = close.ewm(span=ema_long, adjust=False, min_periods=ema_long).mean()
    delta = close.diff()
    gains = delta.clip(lower=0).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    losses = (-delta.clip(upper=0)).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    rsi = 100 - 100 / (1 + gains / losses.replace(0, np.nan))
    rsi = rsi.mask((losses == 0) & (gains > 0), 100).mask((losses == 0) & (gains == 0), 50)
    df['RSI'] = rsi
    df['MACD'] = (close.ewm(span=macd_fast, adjust=False, min_periods=macd_fast).mean() -
                  close.ewm(span=macd_slow, adjust=False, min_periods=macd_slow).mean())
    df['MACD_hist'] = df.MACD - df.MACD.ewm(span=macd_signal, adjust=False, min_periods=macd_signal).mean()
    true_range = pd.concat([df.HIGH-df.LOW, (df.HIGH-close.shift()).abs(),
                            (df.LOW-close.shift()).abs()], axis=1).max(axis=1)
    df['ATR_14'] = true_range.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    df['return_1d'] = close.pct_change(fill_method=None)
    df['high_low_diff_pct'] = (df.HIGH-df.LOW)/close
    df['open_close_diff_pct'] = (close-df.OPEN)/df.OPEN
    df['volume_change_pct'] = df.VOLUME.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0)
    df['close_div_ema_long'] = close/df.EMA_long
    # At t use HIGH[t+1:t+h+1]; require the entire horizon.
    df['future_high'] = df.HIGH.shift(-1).iloc[::-1].rolling(horizon, min_periods=horizon).max().iloc[::-1]
    df['target'] = ((df.future_high/close-1) >= rise).astype(float).where(df.future_high.notna())
    df['label_end'] = pd.Series(df.index, index=df.index).shift(-horizon)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLUMNS)
    df.attrs['prediction_horizon_days'] = int(horizon)
    df.attrs['target_rise_percent'] = float(rise)
    return df
