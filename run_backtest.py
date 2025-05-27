# run_backtest.py (최종 수정 버전)

from config import INITIAL_CAPITAL, STOCK_UNIVERSE # TICKER 대신 STOCK_UNIVERSE 사용
from data_handler import get_prepared_data
from strategy import generate_signals
from backtester import run_backtest
from performance_analyzer import calculate_performance_metrics, plot_performance
import pandas as pd # pd.merge를 위해 추가

def main():
    # 시각화 제목 등에 사용될 대표 티커 (STOCK_UNIVERSE의 첫 번째 종목)
    display_ticker = STOCK_UNIVERSE[0] if STOCK_UNIVERSE else "N/A"

    print(f"--- {display_ticker} 외 {len(STOCK_UNIVERSE)-1}개 주식에 대한 백테스팅 시작 ---")
    print(f"초기 자본금: {INITIAL_CAPITAL:,.0f} $")

    # 1. 데이터 준비 (다중 종목 데이터 가져오기)
    print("\n1. 데이터 준비 중...")
    # get_prepared_data는 이제 다중 종목 데이터를 반환합니다.
    data_with_indicators_all_tickers = get_prepared_data(tickers=STOCK_UNIVERSE)
    if data_with_indicators_all_tickers is None or data_with_indicators_all_tickers.empty:
        print("데이터 준비에 실패했습니다. 백테스팅을 종료합니다.")
        return
    print(f"데이터 기간: {data_with_indicators_all_tickers.index.min()} ~ {data_with_indicators_all_tickers.index.max()}")
    print(f"포함된 종목: {data_with_indicators_all_tickers['Ticker'].unique().tolist()}")
    print("데이터 준비 완료.")

    # 2. 매매 신호 생성 (다중 종목 신호 생성)
    print("\n2. 매매 신호 생성 중...")
    # generate_signals는 이제 다중 종목 데이터와 'Ticker' 컬럼을 기대합니다.
    signals_df_all_tickers = generate_signals(data_with_indicators_all_tickers)
    if signals_df_all_tickers.empty:
        print("매매 신호 생성에 실패했거나 신호가 없습니다. 백테스팅을 종료합니다.")
        return
    print("매매 신호 생성 완료.")

    # 3. 백테스팅을 위한 최종 데이터 준비
    print("\n3. 백테스팅 데이터 통합 중...")
    
    # 병합을 위해 Date 인덱스를 일반 컬럼으로 변환
    data_to_merge = data_with_indicators_all_tickers.reset_index() # Date를 컬럼으로 만듦
    signals_to_merge = signals_df_all_tickers.reset_index()       # Date를 컬럼으로 만듦

    # 이제 'Date'와 'Ticker' 컬럼을 기준으로 병합합니다.
    data_for_backtest_full = pd.merge(
        data_to_merge, 
        signals_to_merge[['Date', 'Ticker', 'signal']], # Date, Ticker, signal 컬럼만 사용
        on=['Date', 'Ticker'], # 'Date'와 'Ticker' 두 컬럼을 기준으로 병합
        how='inner' # 양쪽에 모두 있는 데이터만 사용
    )
    
    # 다시 Date를 인덱스로 설정
    data_for_backtest_full.set_index('Date', inplace=True)
    # 인덱스 기준으로 정렬 (선택 사항이지만 좋은 습관)
    data_for_backtest_full.sort_index(inplace=True)

    # 백테스터에 필요한 최종 컬럼들: CLOSE, VOLUME, volume_change_pct, Ticker, signal
    required_cols = ['CLOSE', 'VOLUME', 'volume_change_pct', 'Ticker', 'signal']
    missing_cols = [col for col in required_cols if col not in data_for_backtest_full.columns]
    if missing_cols:
        print(f"오류: 백테스팅에 필요한 컬럼이 누락되었습니다: {missing_cols}. data_handler.py 확인.")
        return

    data_for_backtest_final = data_for_backtest_full[required_cols]
    
    if data_for_backtest_final.empty or 'signal' not in data_for_backtest_final.columns:
        print("백테스팅용 최종 데이터 생성에 실패했거나 신호가 없습니다. 백테스팅을 종료합니다.")
        return
    print("백테스팅 데이터 통합 완료.")

    # 4. 백테스팅 실행
    print("\n4. 백테스팅 실행 중...")
    # run_backtest 함수는 이제 portfolio_df와 trades_df만 반환합니다. (positions_df 제거)
    portfolio_df, trades_df = run_backtest(data_for_backtest_final, INITIAL_CAPITAL)
    print("백테스팅 실행 완료.")

    if portfolio_df.empty:
        print("백테스팅 결과 포트폴리오 데이터가 비어있습니다. 분석을 종료합니다.")
        return

    # 5. 성과 분석
    print("\n5. 성과 분석 중...")
    performance_metrics = calculate_performance_metrics(portfolio_df, INITIAL_CAPITAL)

    # 거래 횟수 추가
    buy_trades = len(trades_df[trades_df['Type'] == 'BUY']) if not trades_df.empty else 0
    sell_trades = len(trades_df[trades_df['Type'] == 'SELL']) if not trades_df.empty else 0
    performance_metrics["Number of Buy Trades"] = buy_trades
    performance_metrics["Number of Sell Trades"] = sell_trades

    print("\n--- 최종 성과 보고서 ---")
    for key, value in performance_metrics.items():
        print(f"{key}: {value}")

    if not trades_df.empty:
        print("\n--- 최근 거래 내역 (5건) ---")
        print(trades_df.tail())
    else:
        print("\n거래 내역이 없습니다.")

    # 6. 시각화
    print("\n6. 결과 시각화 중...")
    # plot_performance 함수는 현재 단일 종목 벤치마크만 지원합니다.
    # 따라서, STOCK_UNIVERSE의 첫 번째 종목 (display_ticker)에 대한 데이터를 벤치마크로 사용합니다.
    
    # data_with_indicators_all_tickers에서 display_ticker에 해당하는 데이터만 필터링
    benchmark_data = data_with_indicators_all_tickers[data_with_indicators_all_tickers['Ticker'] == display_ticker].copy()
    
    plot_performance(portfolio_df, benchmark_data, display_ticker)
    print(f"--- {display_ticker} 외 {len(STOCK_UNIVERSE)-1}개 주식에 대한 백테스팅 종료 ---")

if __name__ == '__main__':
    main()