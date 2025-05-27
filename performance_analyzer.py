# performance_analyzer.py (최종 수정 버전 - 그래프 X축 및 Y축 스케일링 강화)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import TICKER # 시각화 제목 등에 사용

def calculate_performance_metrics(portfolio_df, initial_capital):
    """포트폴리오 DataFrame으로부터 주요 성과 지표를 계산합니다."""
    if portfolio_df.empty:
        return {
            "Total Return (%)": 0, "CAGR (%)": 0,
            "Max Drawdown (%)": 0, "Sharpe Ratio": 0,
            "Number of Trades": 0 
        }

    # 인덱스가 DatetimeIndex가 아니면 변환
    if not isinstance(portfolio_df.index, pd.DatetimeIndex):
        try:
            portfolio_df.index = pd.to_datetime(portfolio_df.index)
        except Exception as e:
            print(f"Error converting portfolio index to DatetimeIndex: {e}")
            # 변환 실패 시 기본값 반환 또는 오류 처리
            return {
                "Total Return (%)": 0, "CAGR (%)": 0,
                "Max Drawdown (%)": 0, "Sharpe Ratio": 0,
                "Number of Trades": 0
            }

    final_value = portfolio_df['total_value'].iloc[-1]
    total_return_pct = (final_value / initial_capital - 1) * 100
    
    # CAGR 계산
    if len(portfolio_df.index) > 1:
        num_days = (portfolio_df.index[-1] - portfolio_df.index[0]).days
        num_years = num_days / 365.25
    else:
        num_years = 0
        
    cagr = 0
    if num_years > 0 and initial_capital > 0 and final_value > 0:
         cagr = ((final_value / initial_capital) ** (1 / num_years) - 1) * 100
    
    # Max Drawdown (MDD) 계산
    portfolio_df['peak'] = portfolio_df['total_value'].cummax()
    portfolio_df['drawdown'] = portfolio_df['total_value'] - portfolio_df['peak']
    portfolio_df['drawdown_pct'] = (portfolio_df['drawdown'] / portfolio_df['peak']) * 100
    max_drawdown_pct = portfolio_df['drawdown_pct'].min() 
    
    # Sharpe Ratio 계산
    daily_returns = portfolio_df['returns']
    sharpe_ratio = 0
    if not daily_returns.empty and daily_returns.std() != 0: 
         sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    
    metrics = {
        "Total Return (%)": round(total_return_pct, 2),
        "CAGR (%)": round(cagr, 2),
        "Max Drawdown (%)": round(max_drawdown_pct, 2),
        "Sharpe Ratio": round(sharpe_ratio, 2)
    }
    return metrics

def plot_performance(portfolio_df, stock_data_for_benchmark, ticker):
    """
    포트폴리오 가치 변화와 벤치마크(원시 주가)를 함께 시각화합니다.
    portfolio_df: run_backtest에서 반환된 포트폴리오 DataFrame
    stock_data_for_benchmark: 원본 주가 데이터 (벤치마크용, run_backtest에서 이미 필터링되어 넘어옴)
    ticker: 종목 티커 (벤치마크로 사용된 종목명)
    """
    plt.figure(figsize=(12, 7))

    # 포트폴리오 가치 플로팅
    plt.plot(portfolio_df.index, portfolio_df['total_value'], label='Portfolio Value', color='blue')

    # 벤치마크 (원시 주가) 플로팅
    if not portfolio_df.empty:
        plot_start_date = portfolio_df.index.min()
        # stock_data_for_benchmark는 run_backtest.py에서 이미 plot_start_date 기준으로 필터링되어 넘어옴
        # 여기서 한번 더 명확하게 자르거나, 넘어온 데이터가 이미 유효하다고 가정합니다.
        # benchmark_aligned = stock_data_for_benchmark['CLOSE'].loc[stock_data_for_benchmark.index >= plot_start_date].copy()
        benchmark_aligned = stock_data_for_benchmark['CLOSE'].copy() # 이미 잘려 넘어왔다고 가정
    else:
        print("경고: 포트폴리오 데이터가 비어있어 벤치마크를 그릴 수 없습니다.")
        plt.close()
        return

    if benchmark_aligned.empty:
        print("경고: 벤치마크 데이터가 필터링된 후 비어있습니다. 벤치마크는 그려지지 않습니다.")
    else:
        initial_benchmark_price = benchmark_aligned.iloc[0]
        # portfolio_df의 첫 total_value를 기준으로 스케일링
        scaled_benchmark = portfolio_df['total_value'].iloc[0] * (benchmark_aligned / initial_benchmark_price)
        
        plt.plot(scaled_benchmark.index, scaled_benchmark, label=f'{ticker} Buy & Hold (Scaled)', color='orange', linestyle='--')

    plt.title(f'Portfolio Performance vs. {ticker} Buy & Hold')
    plt.xlabel('Date')
    plt.ylabel('Value ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # X축 범위를 백테스팅 데이터 기간에 맞춥니다.
    if not portfolio_df.empty:
        plt.xlim(portfolio_df.index.min(), portfolio_df.index.max())

    # Y축 스케일을 자동으로 최적화하도록 합니다.
    # 만약 여전히 문제가 있다면, plt.ylim(min_value, max_value)를 수동으로 설정해야 할 수 있습니다.
    # 예를 들어, plt.ylim(portfolio_df['total_value'].min() * 0.9, portfolio_df['total_value'].max() * 1.1)
    
    plt.show()


if __name__ == '__main__':
    # 이 파일을 직접 실행하면 performance_analyzer 로직 테스트
    from data_handler import get_prepared_data
    from strategy import generate_signals
    from backtester import run_backtest
    from config import INITIAL_CAPITAL, STOCK_UNIVERSE, TICKER 

    print("--- Performance Analyzer 개별 실행 테스트 ---")
    data_all_tickers = get_prepared_data(tickers=STOCK_UNIVERSE)
    if data_all_tickers is None or data_all_tickers.empty:
        print("데이터 준비 실패 (Performance Analyzer 테스트).")
        exit()

    signals_df_all_tickers = generate_signals(data_all_tickers)
    if signals_df_all_tickers.empty:
        print("매매 신호 생성 실패 (Performance Analyzer 테스트).")
        exit()

    data_to_merge = data_all_tickers.reset_index()
    signals_to_merge = signals_df_all_tickers.reset_index()

    data_for_backtest_full = pd.merge(
        data_to_merge, 
        signals_to_merge[['Date', 'Ticker', 'signal']], 
        on=['Date', 'Ticker'], 
        how='inner'
    )
    data_for_backtest_full.set_index('Date', inplace=True)
    data_for_backtest_full.sort_index(inplace=True)

    required_cols = ['CLOSE', 'VOLUME', 'volume_change_pct', 'Ticker', 'signal']
    missing_cols = [col for col in required_cols if col not in data_for_backtest_full.columns]
    if missing_cols:
        print(f"오류: 백테스팅에 필요한 컬럼이 누락되었습니다: {missing_cols}. 데이터 통합 확인.")
        exit()
    data_for_backtest_final = data_for_backtest_full[required_cols]


    portfolio_df, trades_df = run_backtest(data_for_backtest_final, INITIAL_CAPITAL)
    
    if not portfolio_df.empty:
        performance_metrics = calculate_performance_metrics(portfolio_df, INITIAL_CAPITAL)
        buy_trades = len(trades_df[trades_df['Type'] == 'BUY']) if not trades_df.empty else 0
        sell_trades = len(trades_df[trades_df['Type'] == 'SELL']) if not trades_df.empty else 0
        performance_metrics["Number of Buy Trades"] = buy_trades
        performance_metrics["Number of Sell Trades"] = sell_trades
        
        print("\n--- 주요 성과 지표 ---")
        for key, value in performance_metrics.items():
            print(f"{key}: {value}")
        
        # 벤치마크 데이터를 위한 첫 번째 종목 필터링
        benchmark_data_for_plot = data_all_tickers[data_all_tickers['Ticker'] == TICKER].copy()
        
        # plot_performance에 전달하기 전에 벤치마크 데이터도 포트폴리오 시작 날짜에 맞춰 자릅니다.
        if not portfolio_df.empty:
            plot_start_date = portfolio_df.index.min()
            benchmark_data_for_plot = benchmark_data_for_plot.loc[benchmark_data_for_plot.index >= plot_start_date].copy()

        plot_performance(portfolio_df, benchmark_data_for_plot, TICKER)
    else:
        print("포트폴리오 데이터가 비어있어 성과 분석을 수행할 수 없습니다.")