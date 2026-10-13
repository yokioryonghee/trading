"""Download adjusted daily bars and keep unknown future labels missing."""
from pathlib import Path
import pandas as pd
import yfinance as yf
import config
from technical_indicators import normalize_ohlcv, prepare_features

def fetch_and_save_data_for_ticker(ticker, start_date=None, end_date=None,
                                  data_path=None, csv_filename_template=None):
    start_date = start_date or config.START_DATE
    end_date = end_date or config.END_DATE
    if pd.Timestamp(start_date) >= pd.Timestamp(end_date):
        raise ValueError('Start date must precede end date')
    raw = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True,
                      progress=False, multi_level_index=False, timeout=20)
    if raw is None or raw.empty:
        raise ValueError(f'No market data for {ticker}')
    df = normalize_ohlcv(raw)
    folder = Path(data_path or config.DATA_PATH)
    folder.mkdir(parents=True, exist_ok=True)
    name = (csv_filename_template or config.CSV_FILENAME).format(TICKER=ticker, START_DATE=start_date, END_DATE=end_date)
    df.to_csv(folder/name)
    return df

def calculate_indicators_and_target(df_raw, ema_short_period=None, ema_long_period=None,
                                    rsi_period=None, macd_fast_period=None, macd_slow_period=None,
                                    macd_signal_period=None, prediction_horizon_days=None, target_rise_percent=None):
    return prepare_features(df_raw,
        ema_short=config.EMA_SHORT_PERIOD if ema_short_period is None else ema_short_period,
        ema_long=config.EMA_LONG_PERIOD if ema_long_period is None else ema_long_period,
        rsi_period=config.RSI_PERIOD if rsi_period is None else rsi_period,
        macd_fast=config.MACD_FAST_PERIOD if macd_fast_period is None else macd_fast_period,
        macd_slow=config.MACD_SLOW_PERIOD if macd_slow_period is None else macd_slow_period,
        macd_signal=config.MACD_SIGNAL_PERIOD if macd_signal_period is None else macd_signal_period,
        horizon=config.PREDICTION_HORIZON_DAYS if prediction_horizon_days is None else prediction_horizon_days,
        rise=config.TARGET_RISE_PERCENT if target_rise_percent is None else target_rise_percent)

def get_prepared_data(tickers=None, start_date_str=None, end_date_str=None,
                      ema_short_period=None, ema_long_period=None, rsi_period=None,
                      macd_fast_period=None, macd_slow_period=None, macd_signal_period=None,
                      prediction_horizon_days=None, target_rise_percent=None):
    tickers = config.STOCK_UNIVERSE if tickers is None else tickers
    if not tickers or len(set(tickers)) != len(tickers):
        raise ValueError('Nonempty unique tickers required')
    frames, failures = [], []
    for ticker in tickers:
        try:
            raw = fetch_and_save_data_for_ticker(ticker, start_date_str, end_date_str)
            frame = calculate_indicators_and_target(raw, ema_short_period, ema_long_period, rsi_period,
                         macd_fast_period, macd_slow_period, macd_signal_period,
                         prediction_horizon_days, target_rise_percent)
            if frame.empty:
                raise ValueError('Insufficient feature warmup history')
            frame['Ticker'] = ticker
            frames.append(frame)
        except Exception as exc:
            failures.append(f'{ticker}: {exc}')
    # An incomplete universe changes the experiment; do not silently continue.
    if failures:
        raise ValueError('Market-data preparation failed: ' + '; '.join(failures))
    result = pd.concat(frames).sort_index(kind='stable')
    result.attrs['prediction_horizon_days'] = prediction_horizon_days or config.PREDICTION_HORIZON_DAYS
    result.attrs['target_rise_percent'] = config.TARGET_RISE_PERCENT if target_rise_percent is None else target_rise_percent
    return result

get_prepared_data_for_multiple_tickers = get_prepared_data
