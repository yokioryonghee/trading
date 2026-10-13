"""Per-symbol close signals, executed by the backtester on later opens."""
import pandas as pd
import config

def generate_signals(df_with_indicators, ema_short_period=None, ema_long_period=None,
                     rsi_period=None, rsi_buy_level=None, rsi_sell_level=None):
    buy = config.RSI_TREND_CONFIRM_LEVEL_BUY if rsi_buy_level is None else rsi_buy_level
    sell = config.RSI_TREND_CONFIRM_LEVEL_SELL if rsi_sell_level is None else rsi_sell_level
    if not 0 <= buy <= 100 or not 0 <= sell <= 100:
        raise ValueError('RSI thresholds must be in [0,100]')
    required = {'Ticker','EMA_short','EMA_long','RSI'}
    if not required <= set(df_with_indicators.columns):
        raise ValueError('Ticker, EMA_short, EMA_long, RSI required')
    result = []
    for ticker, frame in df_with_indicators.groupby('Ticker', sort=True):
        frame = frame.sort_index()
        if frame.index.duplicated().any():
            raise ValueError('Duplicate symbol dates')
        above = frame.EMA_short > frame.EMA_long
        previous = above.shift(1)
        signal = pd.Series(0, index=frame.index, dtype=int)
        signal.loc[above & previous.eq(False) & frame.RSI.lt(buy)] = 1
        signal.loc[~above & previous.eq(True) & frame.RSI.gt(sell)] = -1
        result.append(pd.DataFrame({'signal':signal, 'Ticker':ticker}))
    return pd.concat(result).sort_index(kind='stable') if result else pd.DataFrame(columns=['signal','Ticker'])
