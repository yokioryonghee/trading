# backtester.py (디버깅 print() 문 추가 버전)

import pandas as pd
import numpy as np
from config import (
    INITIAL_CAPITAL, TRANSACTION_COST_PERCENT,
    MAX_ALLOCATION_PCT_OF_INITIAL_CAPITAL_PER_TRADE,
    TAKE_PROFIT_PERCENT, STOP_LOSS_PERCENT, MAX_HOLD_DAYS,
    MAX_CONCURRENT_POSITIONS
)

def run_backtest(data_with_signals, 
                 initial_capital=INITIAL_CAPITAL, 
                 commission_rate=TRANSACTION_COST_PERCENT,
                 max_allocation_pct_per_trade=MAX_ALLOCATION_PCT_OF_INITIAL_CAPITAL_PER_TRADE,
                 take_profit_pct=TAKE_PROFIT_PERCENT,
                 stop_loss_pct=STOP_LOSS_PERCENT,
                 max_hold_days=MAX_HOLD_DAYS,
                 max_concurrent_positions=MAX_CONCURRENT_POSITIONS):
    """
    매매 신호에 따라 백테스팅을 실행하고 포트폴리오 가치 변화 및 거래 내역을 반환합니다.
    이 버전은 다중 종목, 고급 청산 로직, 그리고 거래량 급등 기반 매수 우선순위를 포함합니다.

    data_with_signals: 'CLOSE' 가격, 'signal', 'Ticker', 'VOLUME', 'volume_change_pct' 컬럼이 포함된 DataFrame.
    """
    # 포트폴리오 상태 초기화
    cash = initial_capital
    
    # 각 종목별 현재 보유 포지션을 추적
    # {ticker: {'shares': N, 'buy_price': P, 'entry_date': Date, 'entry_idx': int}}
    current_positions = {} 
    
    # 결과를 저장할 DataFrame (날짜별 총 자산 가치)
    # 데이터의 유니크한 날짜를 인덱스로 사용하고 정렬
    unique_dates = data_with_signals.index.unique().sort_values()
    portfolio = pd.DataFrame(index=unique_dates) 
    
    # 포트폴리오의 'total_value' 컬럼을 초기 자본금으로 채우고, 이후 매일 업데이트
    portfolio['total_value'] = initial_capital

    # 거래 내역을 저장할 리스트
    trades = []

    # 날짜별로 데이터를 그룹화하여 순회
    for i, current_date in enumerate(unique_dates):
        # current_idx를 루프의 가장 상단에서 항상 정의합니다.
        current_idx = unique_dates.get_loc(current_date)
        
        # 해당 날짜의 모든 종목 데이터
        daily_data = data_with_signals.loc[[current_date]].copy() 
        
        # --- 1. 기존 포지션 청산 조건 확인 (손절매, 익절, 시간 초과) ---
        for ticker, pos_info in list(current_positions.items()): 
            ticker_daily_data = daily_data[daily_data['Ticker'] == ticker]
            if ticker_daily_data.empty:
                continue 

            current_price = ticker_daily_data['CLOSE'].iloc[0]
            buy_price = pos_info['buy_price']
            entry_date = pos_info['entry_date']
            shares = pos_info['shares']
            
            current_profit_loss_pct = (current_price / buy_price) - 1

            entry_idx = pos_info['entry_idx']
            days_held_trading = current_idx - entry_idx 
            
            should_close = False
            reason = ""

            # --- DEBUG: 손절매 조건 확인 전 출력 ---
            # 모든 매도 조건(TP, SL, MAX HOLD)을 확인하기 직전에 현재 포지션의 PnL을 출력합니다.
            # 이 print 문을 통해 어떤 조건이 먼저 만족되는지 추적할 수 있습니다.
            # 특히 손실이 발생했는데 손절매가 동작하지 않는 경우를 파악하는 데 유용합니다.
            print(f"DEBUG CHECK [TP/SL/MAX_HOLD]: Date={current_date.strftime('%Y-%m-%d')}, Ticker={ticker}, "
                  f"Price={current_price:.2f}, EntryPrice={buy_price:.2f}, "
                  f"PnL_Pct={current_profit_loss_pct:.4f} (Target SL={-stop_loss_pct:.4f}, Target TP={take_profit_pct:.4f}), "
                  f"DaysHeld={days_held_trading} (MaxHold={max_hold_days}), Shares={shares:.0f}, Cash={cash:.2f}")


            if current_profit_loss_pct >= take_profit_pct:
                should_close = True
                reason = f"TP: {current_profit_loss_pct:.2%}"
            elif current_profit_loss_pct <= -stop_loss_pct:
                should_close = True
                reason = f"SL: {current_profit_loss_pct:.2%}"
            elif days_held_trading >= max_hold_days and shares > 0: 
                should_close = True
                reason = f"MAX HOLD: {days_held_trading} days"

            if should_close:
                proceeds = shares * current_price * (1 - commission_rate)
                total_buy_cost = shares * buy_price * (1 + commission_rate) 
                profit_loss_amount = proceeds - total_buy_cost 

                trades.append({
                    'Date': current_date, 'Ticker': ticker, 'Type': 'SELL', 
                    'Price': current_price, 'Shares': shares, 'Proceeds': proceeds,
                    'Entry_Price': buy_price, 'PnL_Pct': current_profit_loss_pct, 
                    'PnL_Amount': profit_loss_amount, 'Reason': reason
                })
                cash += proceeds
                del current_positions[ticker] 
                print(f"DEBUG SELL [TP/SL/MAX_HOLD]: {current_date.strftime('%Y-%m-%d')} {ticker} SOLD for {reason}. PnL_Pct: {current_profit_loss_pct:.2%}, Shares: {shares:.0f}, Cash: {cash:.2f}")


        # --- 2. 매매 신호 및 포트폴리오 관리 기반 매수/매도 실행 ---
        signals_today = daily_data[daily_data['signal'] != 0].copy() 

        # 매도 신호 처리 (모델 신호에 의한 매도)
        for idx, row in list(signals_today[signals_today['signal'] == -1.0].iterrows()):
            ticker = row['Ticker']
            current_price = row['CLOSE']
            if ticker in current_positions: 
                pos_info = current_positions[ticker]
                shares_to_sell = pos_info['shares']
                buy_price = pos_info['buy_price']
                
                proceeds = shares_to_sell * current_price * (1 - commission_rate)
                profit_loss_pct = (current_price / buy_price) - 1
                total_buy_cost = shares_to_sell * buy_price * (1 + commission_rate)
                profit_loss_amount = proceeds - total_buy_cost
                
                trades.append({
                    'Date': current_date, 'Ticker': ticker, 'Type': 'SELL', 
                    'Price': current_price, 'Shares': shares_to_sell, 'Proceeds': proceeds,
                    'Entry_Price': buy_price, 'PnL_Pct': profit_loss_pct, 
                    'PnL_Amount': profit_loss_amount, 'Reason': 'Signal'
                })
                cash += proceeds
                del current_positions[ticker]
                print(f"DEBUG SELL [SIGNAL]: {current_date.strftime('%Y-%m-%d')} {ticker} SOLD by Signal. PnL_Pct: {profit_loss_pct:.2%}, Shares: {shares:.0f}, Cash: {cash:.2f}")


        # 매수 신호 처리 (모델 신호에 의한 매수 및 거래량 급등 우선순위)
        buy_signals_today = signals_today[signals_today['signal'] == 1.0].copy()

        if not buy_signals_today.empty:
            buy_candidates = buy_signals_today[~buy_signals_today['Ticker'].isin(current_positions.keys())].copy()
            
            if not buy_candidates.empty:
                if 'volume_change_pct' in buy_candidates.columns:
                    buy_candidates['volume_change_pct_filled'] = buy_candidates['volume_change_pct'].fillna(-np.inf)
                    buy_candidates = buy_candidates.sort_values(by='volume_change_pct_filled', ascending=False)
                else:
                    print(f"경고: {current_date}에 'volume_change_pct' 컬럼이 없어 거래량 우선순위를 적용할 수 없습니다. 기본 순서 사용.")
                
                num_can_buy = max_concurrent_positions - len(current_positions)
                
                if num_can_buy > 0:
                    for idx, row in buy_candidates.head(num_can_buy).iterrows(): 
                        ticker = row['Ticker']
                        current_price = row['CLOSE']

                        if ticker in current_positions or cash <= 0:
                            continue
                        
                        investment_amount_limit_based_on_initial_capital = initial_capital * max_allocation_pct_per_trade
                        amount_to_invest = min(investment_amount_limit_based_on_initial_capital, cash) 
                        
                        if current_price > 0:
                            shares_to_buy = np.floor(amount_to_invest / (current_price * (1 + commission_rate))) 
                        else:
                            shares_to_buy = 0

                        cost = shares_to_buy * current_price * (1 + commission_rate)

                        if shares_to_buy > 0 and cost <= cash: 
                            cash -= cost
                            current_positions[ticker] = {
                                'shares': shares_to_buy,
                                'buy_price': current_price,
                                'entry_date': current_date,
                                'entry_idx': current_idx 
                            }
                            trades.append({'Date': current_date, 'Ticker': ticker, 'Type': 'BUY', 
                                           'Price': current_price, 'Shares': shares_to_buy, 'Cost': cost})
                            print(f"DEBUG BUY: {current_date.strftime('%Y-%m-%d')} {ticker} BOUGHT. Price: {current_price:.2f}, Shares: {shares_to_buy:.0f}, Cash left: {cash:.2f}")

        # --- 3. 일별 포트폴리오 가치 업데이트 ---
        current_holdings_value = 0
        for ticker, pos_info in current_positions.items():
            ticker_daily_data = daily_data[daily_data['Ticker'] == ticker]
            if not ticker_daily_data.empty:
                current_holdings_value += pos_info['shares'] * ticker_daily_data['CLOSE'].iloc[0]

        current_total_value = cash + current_holdings_value
        portfolio.loc[current_date, 'total_value'] = current_total_value

    # 모든 거래가 끝난 후 남아있는 포지션 청산 (백테스팅 종료 시점)
    last_available_date = data_with_signals.index.max()
    last_daily_data = data_with_signals.loc[[last_available_date]].copy() 

    for ticker, pos_info in list(current_positions.items()): 
        ticker_last_price_data = last_daily_data[last_daily_data['Ticker'] == ticker]
        if ticker_last_price_data.empty:
            print(f"경고: 백테스팅 종료 시점에 {ticker}의 마지막 날 가격 데이터를 찾을 수 없습니다. 이 포지션은 청산하지 못합니다.")
            continue 

        last_price = ticker_last_price_data['CLOSE'].iloc[0]
        shares_to_sell = pos_info['shares']
        buy_price = pos_info['buy_price']
        
        proceeds = shares_to_sell * last_price * (1 - commission_rate)
        profit_loss_pct = (last_price / buy_price) - 1
        total_buy_cost = shares_to_sell * buy_price * (1 + commission_rate)
        profit_loss_amount = proceeds - total_buy_cost
        
        trades.append({
            'Date': last_available_date, 'Ticker': ticker, 'Type': 'SELL', 
            'Price': last_price, 'Shares': shares_to_sell, 'Proceeds': proceeds,
            'Entry_Price': buy_price, 'PnL_Pct': profit_loss_pct, 
            'PnL_Amount': profit_loss_amount, 'Reason': 'End of Backtest'
        })
        cash += proceeds
        del current_positions[ticker] 
        print(f"DEBUG SELL [END]: {last_available_date.strftime('%Y-%m-%d')} {ticker} SOLD at END. PnL_Pct: {profit_loss_pct:.2%}, Shares: {shares_to_sell:.0f}, Cash: {cash:.2f}")

    portfolio['total_value'] = portfolio['total_value'].ffill()
    portfolio['total_value'] = portfolio['total_value'].bfill()

    portfolio['returns'] = portfolio['total_value'].pct_change().fillna(0)
    
    return portfolio, pd.DataFrame(trades)

if __name__ == '__main__':
    # 이 파일을 직접 실행하면 backtester 로직 테스트
    from data_handler import get_prepared_data
    from strategy import generate_signals
    from config import STOCK_UNIVERSE, INITIAL_CAPITAL 

    print("--- 백테스터 개별 실행 테스트 (다중 종목, 고급 전략) ---")
    data = get_prepared_data(tickers=STOCK_UNIVERSE)
    if data is not None and not data.empty:
        signals_df = generate_signals(data)
        
        if 'volume_change_pct' not in data.columns:
            print("오류: 'volume_change_pct' 컬럼이 데이터에 없습니다. data_handler.py를 확인하세요.")
            exit()

        data_to_merge = data.reset_index() 
        signals_to_merge = signals_df.reset_index() 

        data_for_backtest_full = pd.merge(
            data_to_merge, 
            signals_to_merge[['Date', 'Ticker', 'signal']], 
            on=['Date', 'Ticker'], 
            how='inner'
        )
        data_for_backtest_full.set_index('Date', inplace=True)
        data_for_backtest_full.sort_index(inplace=True)

        required_cols_for_backtest = ['CLOSE', 'VOLUME', 'volume_change_pct', 'Ticker', 'signal']
        missing_cols = [col for col in required_cols_for_backtest if col not in data_for_backtest_full.columns]
        if missing_cols:
            print(f"오류: 백테스팅에 필요한 컬럼이 누락되었습니다: {missing_cols}. data_handler.py 또는 데이터 통합 로직을 확인하세요.")
            # return

        data_for_backtest_final = data_for_backtest_full[required_cols_for_backtest]

        if data_for_backtest_final.empty or 'signal' not in data_for_backtest_final.columns:
            print("백테스팅용 최종 데이터 생성에 실패했거나 신호가 없습니다. 백테스팅을 종료합니다.")
        else:
            portfolio_df, trades_df = run_backtest(data_for_backtest_final, INITIAL_CAPITAL) 

            if not portfolio_df.empty:
                print("\n--- 포트폴리오 가치 변화 (마지막 10일) ---")
                print(portfolio_df.tail(10))
                
                print("\n--- 거래 내역 (처음 5건) ---")
                if not trades_df.empty:
                    print(trades_df.head())
                else:
                    print("거래 내역이 없습니다.")
                
                print("\n--- 거래 내역 (마지막 5건) ---")
                if not trades_df.empty:
                    print(trades_df.tail())
                else:
                    print("거래 내역이 없습니다.")
            else:
                print("백테스팅 결과 포트폴리오 데이터가 비어있습니다.")
    else:
        print("데이터 준비 실패 (백테스터 테스트).")