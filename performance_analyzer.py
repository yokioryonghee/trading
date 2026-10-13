import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_performance_metrics(portfolio_df,initial_capital):
    if portfolio_df.empty or initial_capital<=0: raise ValueError('Nonempty portfolio and positive capital required')
    values=portfolio_df.total_value.astype(float)
    if not np.isfinite(values).all() or (values<=0).any(): raise ValueError('Invalid portfolio values')
    elapsed=(values.index[-1]-values.index[0]).total_seconds()/86400
    peak=values.cummax().clip(lower=initial_capital)
    std=portfolio_df.returns.std()
    return {'Total Return (%)':float((values.iloc[-1]/initial_capital-1)*100),
            'CAGR (%)':float(((values.iloc[-1]/initial_capital)**(365.25/elapsed)-1)*100) if elapsed>0 else None,
            'Max Drawdown (%)':float(((values/peak-1)*100).min()),
            'Sharpe Ratio':float(portfolio_df.returns.mean()/std*np.sqrt(252)) if np.isfinite(std) and std>0 else None}

def plot_performance(portfolio_df,stock_data_for_benchmark,ticker_name):
    fig,ax=plt.subplots(figsize=(12,7))
    ax.plot(portfolio_df.index,portfolio_df.total_value,label='Portfolio')
    benchmark=stock_data_for_benchmark.CLOSE.reindex(portfolio_df.index).ffill().dropna()
    if not benchmark.empty:
        ax.plot(benchmark.index,benchmark/benchmark.iloc[0]*portfolio_df.total_value.loc[benchmark.index[0]],label=f'{ticker_name} buy and hold')
    ax.set(xlabel='Date',ylabel='Value',title='Daily-bar research backtest'); ax.legend(); ax.grid(True); fig.tight_layout()
    return fig
