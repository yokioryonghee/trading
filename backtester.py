"""Daily-bar research accounting: close signal, next observed open execution."""
import numpy as np
import pandas as pd
from config import (INITIAL_CAPITAL, TRANSACTION_COST_PERCENT,
                    MAX_ALLOCATION_PCT_OF_INITIAL_CAPITAL_PER_TRADE,
                    TAKE_PROFIT_PERCENT, STOP_LOSS_PERCENT, MAX_HOLD_DAYS,
                    MAX_CONCURRENT_POSITIONS)

def run_backtest(data_with_signals, initial_capital=INITIAL_CAPITAL,
                 commission_rate=TRANSACTION_COST_PERCENT,
                 max_allocation_pct_per_trade=MAX_ALLOCATION_PCT_OF_INITIAL_CAPITAL_PER_TRADE,
                 take_profit_pct=TAKE_PROFIT_PERCENT, stop_loss_pct=STOP_LOSS_PERCENT,
                 max_hold_days=MAX_HOLD_DAYS, max_concurrent_positions=MAX_CONCURRENT_POSITIONS):
    if not np.isfinite([initial_capital, commission_rate, max_allocation_pct_per_trade,
                       take_profit_pct, stop_loss_pct]).all():
        raise ValueError('Backtest parameters must be finite')
    if initial_capital <= 0 or not 0 <= commission_rate < 1 or not 0 < max_allocation_pct_per_trade <= 1:
        raise ValueError('Invalid capital, commission or allocation')
    if not 0 <= stop_loss_pct < 1 or take_profit_pct < 0 or max_hold_days < 1 or max_concurrent_positions < 1:
        raise ValueError('Invalid exit/position limits')
    required = {'OPEN', 'CLOSE', 'Ticker', 'signal'}
    if not required <= set(data_with_signals.columns) or data_with_signals.empty:
        raise ValueError('Nonempty dated OPEN,CLOSE,Ticker,signal data required')
    data = data_with_signals.copy().sort_index()
    if not isinstance(data.index, pd.DatetimeIndex) or data.index.hasnans:
        raise ValueError('Valid DatetimeIndex required')
    if data.reset_index().duplicated([data.index.name or 'index', 'Ticker']).any():
        raise ValueError('Duplicate date/ticker rows')
    if not np.isfinite(data[['OPEN','CLOSE','signal']].to_numpy()).all() or (data[['OPEN','CLOSE']] <= 0).any().any():
        raise ValueError('Prices and signals must be finite; prices positive')
    if not data.signal.isin([-1,0,1]).all():
        raise ValueError('Signals must be -1,0,1')
    cash = float(initial_capital)
    positions, pending, marks, trades, rows = {}, {}, {}, [], []
    def sell(ticker, price, date, reason):
        nonlocal cash
        pos = positions.pop(ticker)
        proceeds = pos['shares'] * price * (1-commission_rate)
        cost = pos['shares'] * pos['buy_price'] * (1+commission_rate)
        cash += proceeds
        trades.append(dict(Date=date, Ticker=ticker, Type='SELL', Price=price,
                           Shares=pos['shares'], Proceeds=proceeds, Entry_Price=pos['buy_price'],
                           PnL_Pct=price/pos['buy_price']-1, PnL_Amount=proceeds-cost, Reason=reason))
    dates = data.index.unique().sort_values()
    last_dates=data.reset_index().groupby('Ticker')[data.index.name or 'index'].max().to_dict()
    for date in dates:
        daily = data.loc[[date]].set_index('Ticker')
        # Execute only signals from an earlier completed bar, exits before entries.
        for ticker in list(positions):
            if ticker in daily.index and pending.get(ticker) == -1:
                sell(ticker, float(daily.loc[ticker, 'OPEN']), date, 'Prior close exit')
        candidates = [ticker for ticker in daily.index if pending.get(ticker) == 1 and ticker not in positions]
        # Today's volume/return is unknown at open; ordering is deterministic by ticker.
        for ticker in sorted(candidates):
            if len(positions) >= max_concurrent_positions:
                break
            price = float(daily.loc[ticker, 'OPEN'])
            budget = min(initial_capital*max_allocation_pct_per_trade, cash)
            shares = int(np.floor(budget/(price*(1+commission_rate))))
            if shares:
                cost = shares*price*(1+commission_rate)
                cash -= cost
                positions[ticker] = dict(shares=shares, buy_price=price, sessions=0)
                trades.append(dict(Date=date, Ticker=ticker, Type='BUY', Price=price, Shares=shares, Cost=cost))
        for ticker, bar in daily.iterrows():
            marks[ticker] = float(bar.CLOSE)
            pending[ticker] = int(bar.signal)
            if ticker in positions:
                pos = positions[ticker]
                pos['sessions'] += 1
                change = float(bar.CLOSE)/pos['buy_price']-1
                if (take_profit_pct > 0 and change >= take_profit_pct) or (stop_loss_pct > 0 and change <= -stop_loss_pct) or pos['sessions'] >= max_hold_days:
                    # A close-observed exit is submitted for the next observed open.
                    pending[ticker] = -1
        for ticker in list(positions):
            if last_dates[ticker] == date:
                sell(ticker, float(daily.loc[ticker,'CLOSE']), date, 'End of Backtest')
        value = cash + sum(pos['shares']*marks[ticker] for ticker,pos in positions.items())
        rows.append(dict(Date=date, total_value=value))
    portfolio = pd.DataFrame(rows).set_index('Date')
    portfolio['returns'] = portfolio.total_value.pct_change(fill_method=None).fillna(0.)
    portfolio.iloc[0, portfolio.columns.get_loc('returns')] = portfolio.total_value.iloc[0]/initial_capital-1
    return portfolio, pd.DataFrame(trades, columns=['Date','Ticker','Type','Price','Shares','Cost',
                                                  'Proceeds','Entry_Price','PnL_Pct','PnL_Amount','Reason'])
