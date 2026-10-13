import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from technical_indicators import prepare_features,FEATURE_COLUMNS
from strategy import generate_signals
from backtester import run_backtest

def bars(n=500):
    rng=np.random.default_rng(7); cl=100*np.exp(np.cumsum(rng.normal(0,.015,n))); op=cl*.999
    return pd.DataFrame(dict(OPEN=op,HIGH=np.maximum(op,cl)*1.01,LOW=np.minimum(op,cl)*.99,CLOSE=cl,VOLUME=rng.integers(100,1000,n)),index=pd.bdate_range('2020-01-01',periods=n,name='Date'))

def test_future_window_and_unknown_tail():
    raw=bars(); data=prepare_features(raw,horizon=3)
    day=data.index[10]; pos=raw.index.get_loc(day)
    assert data.loc[day,'future_high']==raw.HIGH.iloc[pos+1:pos+4].max()
    assert data.target.tail(3).isna().all()
    assert data.label_end.tail(3).isna().all()

def test_features_have_no_future_fill():
    raw=bars(); first=prepare_features(raw)
    altered=raw.copy(); altered.iloc[300:,:4]*=10
    second=prepare_features(altered)
    pd.testing.assert_frame_equal(first.loc[first.index<raw.index[300],FEATURE_COLUMNS],second.loc[second.index<raw.index[300],FEATURE_COLUMNS])

def test_single_and_multiindex_same_features():
    raw=bars(); expected=prepare_features(raw)
    for field_first in [True,False]:
        framed=raw.copy()
        framed.columns=pd.MultiIndex.from_tuples([(c,'A') if field_first else ('A',c) for c in raw])
        pd.testing.assert_frame_equal(expected,prepare_features(framed))

def test_symbol_signal_isolation():
    a=prepare_features(bars()); a['Ticker']='A'
    b=a.copy(); b['Ticker']='B'; b['EMA_short']*=2
    expected=generate_signals(a)
    mixed=generate_signals(pd.concat([a,b]).sort_index())
    pd.testing.assert_frame_equal(expected,mixed[mixed.Ticker=='A'])

def test_next_open_execution_and_final_fees():
    dates=pd.bdate_range('2024-01-01',periods=3,name='Date')
    frame=pd.DataFrame({'OPEN':[100,110,120],'CLOSE':[105,115,125],'Ticker':'A','signal':[1,0,0]},index=dates)
    portfolio,trades=run_backtest(frame,initial_capital=1000,commission_rate=.01,max_allocation_pct_per_trade=1,take_profit_pct=0,stop_loss_pct=0,max_hold_days=100)
    shares=int(1000//(110*1.01)); cash=1000-shares*110*1.01
    assert trades.iloc[0].Date==dates[1]
    assert trades.iloc[0].Price==110
    assert portfolio.total_value.iloc[0]==1000
    assert portfolio.total_value.iloc[-1]==pytest.approx(cash+shares*125*.99)

def test_closed_market_preserves_valuation():
    dates=pd.bdate_range('2024-01-01',periods=4,name='Date')
    a=pd.DataFrame(dict(OPEN=[100,100,100],CLOSE=[100,100,100],Ticker='A',signal=[1,0,0]),index=dates[[0,1,3]])
    b=pd.DataFrame(dict(OPEN=100,CLOSE=100,Ticker='B',signal=0),index=dates)
    p,_=run_backtest(pd.concat([a,b]),initial_capital=1000,commission_rate=0,max_allocation_pct_per_trade=1,take_profit_pct=0,stop_loss_pct=0,max_hold_days=100)
    assert p.total_value.eq(1000).all()

def test_purged_training_and_untouched_prediction(tmp_path):
    from train_evaluate_ml import train_and_evaluate_models_for_streamlit
    from ml_strategy import generate_signals_from_ml
    raw=bars(650); data=prepare_features(raw,horizon=5,rise=.01); data['Ticker']='A'
    training=data.iloc[:350].copy()
    metrics,columns=train_and_evaluate_models_for_streamlit(training,['A'],5,.01,tmp_path)
    assert pd.Timestamp(metrics['A']['last_train_label'])<pd.Timestamp(metrics['A']['first_test_signal'])
    output=generate_signals_from_ml(data,tmp_path,columns)
    metadata=json.loads((tmp_path/'ensemble_voting_classifier_hard_A.pkl.json').read_text())
    assert output.index.min()>pd.Timestamp(metadata['validation_through'])
    with pytest.raises(ValueError,match='no untouched'):
        generate_signals_from_ml(training,tmp_path)

def test_missing_model_is_not_successful_cash(tmp_path):
    from ml_strategy import generate_signals_from_ml
    data=prepare_features(bars()); data['Ticker']='A'
    with pytest.raises(FileNotFoundError): generate_signals_from_ml(data,tmp_path)

def test_download_failure_visible(monkeypatch):
    import data_handler
    monkeypatch.setattr(data_handler.yf,'download',lambda *a,**k:pd.DataFrame())
    with pytest.raises(ValueError,match='A: No market data'): data_handler.get_prepared_data(['A'])
