# run_ml_backtest.py

from config import INITIAL_CAPITAL, STOCK_UNIVERSE #
from data_handler import get_prepared_data #
# from strategy import generate_signals # 기존 규칙 기반 신호 생성 함수 (여기서는 사용 안 함)
from ml_strategy import generate_signals_from_ml # 새로 만든 ML 기반 신호 생성 함수
from backtester import run_backtest #
from performance_analyzer import calculate_performance_metrics, plot_performance #
import pandas as pd

# train_evaluate_ml.py 에서 모델이 저장된 경로
MODEL_SAVE_DIR = './ml_model/' #

def main():
    display_ticker = STOCK_UNIVERSE[0] if STOCK_UNIVERSE else "N/A" #

    print(f"--- ML 모델 기반 백테스팅 시작 ---")
    print(f"대상 종목: {STOCK_UNIVERSE}") #
    print(f"초기 자본금: {INITIAL_CAPITAL:,.0f} $") #
    print(f"사용 모델 경로: {MODEL_SAVE_DIR}")

    # 1. 데이터 준비 (data_handler.py 사용은 동일)
    print("\n1. 데이터 준비 중...")
    data_with_indicators_all_tickers = get_prepared_data(tickers=STOCK_UNIVERSE) #
    if data_with_indicators_all_tickers is None or data_with_indicators_all_tickers.empty:
        print("데이터 준비에 실패했습니다. ML 백테스팅을 종료합니다.")
        return
    print(f"데이터 기간: {data_with_indicators_all_tickers.index.min()} ~ {data_with_indicators_all_tickers.index.max()}")
    print(f"포함된 종목: {data_with_indicators_all_tickers['Ticker'].unique().tolist()}")
    print("데이터 준비 완료.")

    # 2. ML 기반 매매 신호 생성
    print("\n2. ML 기반 매매 신호 생성 중...")
    # generate_signals_from_ml 함수 호출 시, 피처 목록을 전달해야 하지만,
    # ml_strategy.py 내부에 ML_MODEL_FEATURE_COLUMNS로 정의했으므로 여기서는 model_dir만 전달
    signals_df_all_tickers = generate_signals_from_ml(data_with_indicators_all_tickers, MODEL_SAVE_DIR)
    if signals_df_all_tickers.empty:
        print("ML 기반 매매 신호 생성에 실패했거나 신호가 없습니다. 백테스팅을 종료합니다.")
        return
    print(f"생성된 ML 신호 수: {len(signals_df_all_tickers)}")
    if not signals_df_all_tickers.empty:
        print(f"매수 신호 (1.0) 개수: {signals_df_all_tickers[signals_df_all_tickers['signal'] == 1.0].shape[0]}")
        print(f"관망 신호 (0.0) 개수: {signals_df_all_tickers[signals_df_all_tickers['signal'] == 0.0].shape[0]}")
    print("ML 기반 매매 신호 생성 완료.")

    # 3. 백테스팅을 위한 최종 데이터 준비 (기존 run_backtest.py 와 유사)
    print("\n3. 백테스팅용 데이터 통합 중...")
    data_to_merge = data_with_indicators_all_tickers.reset_index()
    signals_to_merge = signals_df_all_tickers.reset_index()

    # Date와 Ticker를 기준으로 병합
    data_for_backtest_full = pd.merge(
        data_to_merge,
        signals_to_merge[['Date', 'Ticker', 'signal']],
        on=['Date', 'Ticker'],
        how='inner' # 신호가 있는 날짜/티커에 대해서만 데이터 사용
    )
    data_for_backtest_full.set_index('Date', inplace=True)
    data_for_backtest_full.sort_index(inplace=True)

    required_cols = ['CLOSE', 'VOLUME', 'volume_change_pct', 'Ticker', 'signal'] #
    missing_cols = [col for col in required_cols if col not in data_for_backtest_full.columns]
    if missing_cols:
        print(f"오류: ML 백테스팅에 필요한 컬럼이 누락되었습니다: {missing_cols}. data_handler.py 또는 데이터 통합 로직을 확인하세요.")
        return
    
    # 'volume_change_pct'가 없으면 backtester.py에서 경고가 나올 수 있으므로 확인.
    # data_handler.py에서 생성되므로 보통은 존재함.
    if 'volume_change_pct' not in data_for_backtest_full.columns:
        print("경고: 'volume_change_pct' 컬럼이 백테스팅 데이터에 없습니다. 거래량 우선순위가 적용되지 않을 수 있습니다.")
        # 필요시 NaN으로 채우거나, 해당 로직을 사용하지 않도록 backtester 수정 고려
        # data_for_backtest_full['volume_change_pct'] = 0.0 # 임시 처리 예시

    data_for_backtest_final = data_for_backtest_full[required_cols]

    if data_for_backtest_final.empty or 'signal' not in data_for_backtest_final.columns:
        print("ML 백테스팅용 최종 데이터 생성에 실패했거나 신호가 없습니다. 백테스팅을 종료합니다.")
        return
    print("백테스팅용 데이터 통합 완료.")

    # 4. 백테스팅 실행 (backtester.py 사용은 동일)
    print("\n4. ML 기반 백테스팅 실행 중...")
    portfolio_df, trades_df = run_backtest(data_for_backtest_final, INITIAL_CAPITAL) #
    print("ML 기반 백테스팅 실행 완료.")

    if portfolio_df.empty:
        print("ML 백테스팅 결과 포트폴리오 데이터가 비어있습니다. 분석을 종료합니다.")
        return

    # 5. 성과 분석 (performance_analyzer.py 사용은 동일)
    print("\n5. 성과 분석 중...")
    performance_metrics = calculate_performance_metrics(portfolio_df, INITIAL_CAPITAL) #

    buy_trades = len(trades_df[trades_df['Type'] == 'BUY']) if not trades_df.empty else 0 #
    sell_trades = len(trades_df[trades_df['Type'] == 'SELL']) if not trades_df.empty else 0 #
    performance_metrics["Number of Buy Trades"] = buy_trades
    performance_metrics["Number of Sell Trades"] = sell_trades

    print("\n--- ML 전략 최종 성과 보고서 ---")
    for key, value in performance_metrics.items():
        print(f"{key}: {value}")

    if not trades_df.empty:
        print("\n--- ML 전략 최근 거래 내역 (5건) ---")
        print(trades_df.tail())
    else:
        print("\n거래 내역이 없습니다.")

    # 6. 시각화 (performance_analyzer.py 사용은 동일)
    print("\n6. 결과 시각화 중...")
    benchmark_data = data_with_indicators_all_tickers[data_with_indicators_all_tickers['Ticker'] == display_ticker].copy() #
    plot_performance(portfolio_df, benchmark_data, f"{display_ticker} (ML Strategy)") #
    print(f"--- ML 모델 기반 백테스팅 종료 ---")

if __name__ == '__main__':
    main()