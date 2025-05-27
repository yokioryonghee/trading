# strategy.py (수정 버전)

import pandas as pd
# config.py에서 필요한 설정값들을 import 합니다.
from config import RSI_TREND_CONFIRM_LEVEL_BUY, RSI_TREND_CONFIRM_LEVEL_SELL
from config import STOCK_UNIVERSE # 다중 종목 처리를 위해 추가

def generate_signals(df_with_indicators):
    """
    주어진 다중 종목 DataFrame으로부터 각 종목별로 이동평균선 골든/데드 크로스 및 RSI를 이용한 매매 신호 생성.
    신호: 1 (매수), -1 (매도), 0 (보유/관망)
    
    df_with_indicators: 'Ticker' 컬럼을 포함하며, 각 종목의 지표가 계산된 DataFrame.
    """
    all_signals = []

    # 각 종목별로 그룹화하여 신호 생성
    for ticker in df_with_indicators['Ticker'].unique():
        ticker_df = df_with_indicators[df_with_indicators['Ticker'] == ticker].copy()
        
        # 신호를 저장할 임시 DataFrame
        signals_for_ticker = pd.DataFrame(index=ticker_df.index)
        signals_for_ticker['signal'] = 0.0 # 기본값은 보유
        signals_for_ticker['Ticker'] = ticker # Ticker 컬럼 다시 추가 (merge 후 사용될 수 있도록)

        # 골든 크로스: 단기 EMA가 장기 EMA를 상향 돌파
        # EMA_short_above_long 컬럼이 이미 있다면 재계산 불필요
        if 'EMA_short' in ticker_df.columns and 'EMA_long' in ticker_df.columns:
            signals_for_ticker['EMA_short_above_long'] = (ticker_df['EMA_short'] > ticker_df['EMA_long']).astype(int)
            signals_for_ticker['golden_cross'] = ((signals_for_ticker['EMA_short_above_long'] == 1) & \
                                                  (signals_for_ticker['EMA_short_above_long'].shift(1) == 0)).astype(int)

            # 데드 크로스: 단기 EMA가 장기 EMA를 하향 돌파
            signals_for_ticker['dead_cross'] = ((signals_for_ticker['EMA_short_above_long'] == 0) & \
                                                (signals_for_ticker['EMA_short_above_long'].shift(1) == 1)).astype(int)
        else:
            # EMA 컬럼이 없으면 신호 생성 불가
            print(f"경고: {ticker}의 데이터에 EMA 컬럼이 없습니다. 신호 생성을 건너뜝니다.")
            all_signals.append(signals_for_ticker[['signal', 'Ticker']]) # 빈 신호라도 추가하여 데이터프레임 구조 유지
            continue


        # RSI 조건 (config.py에서 설정한 값 사용)
        if 'RSI' in ticker_df.columns:
            rsi_condition_buy = ticker_df['RSI'] > RSI_TREND_CONFIRM_LEVEL_BUY
            rsi_condition_sell = ticker_df['RSI'] < RSI_TREND_CONFIRM_LEVEL_SELL
        else:
            print(f"경고: {ticker}의 데이터에 RSI 컬럼이 없습니다. RSI 조건을 사용하지 않습니다.")
            rsi_condition_buy = True # 조건 없이 항상 참 (RSI 조건 무시)
            rsi_condition_sell = True # 조건 없이 항상 참 (RSI 조건 무시)

        # 최종 신호 생성
        # 골든 크로스 발생 및 RSI 조건 만족 시 매수
        signals_for_ticker.loc[(signals_for_ticker['golden_cross'] == 1) & rsi_condition_buy, 'signal'] = 1.0
        # 데드 크로스 발생 및 RSI 조건 만족 시 매도
        signals_for_ticker.loc[(signals_for_ticker['dead_cross'] == 1) & rsi_condition_sell, 'signal'] = -1.0
        
        all_signals.append(signals_for_ticker[['signal', 'Ticker']]) # 최종 신호 및 Ticker 컬럼 추가

    if not all_signals:
        print("모든 종목에 대해 신호 생성에 실패했습니다.")
        return pd.DataFrame(columns=['signal', 'Ticker']) # 빈 DataFrame 반환

    # 모든 종목의 신호를 하나의 DataFrame으로 합치기
    final_signals_df = pd.concat(all_signals).sort_index()
    
    return final_signals_df


if __name__ == '__main__':
    # 이 파일을 직접 실행하면 strategy 로직 테스트
    # data_handler에서 다중 종목 데이터를 가져옵니다.
    from data_handler import get_prepared_data 
    from config import STOCK_UNIVERSE # 테스트를 위해 필요

    print("--- 전략 생성 테스트 (다중 종목) ---")
    data_all_tickers = get_prepared_data(tickers=STOCK_UNIVERSE)
    
    if data_all_tickers is not None and not data_all_tickers.empty:
        signals_df_all_tickers = generate_signals(data_all_tickers)
        
        print("\n--- 생성된 신호 (각 종목별 최근 5개) ---")
        for ticker in STOCK_UNIVERSE:
            print(f"\n--- {ticker} 신호 ---")
            print(signals_df_all_tickers[signals_df_all_tickers['Ticker'] == ticker].tail(5))
        
        print("\n--- 전체 매수 신호 발생일 (처음 5건) ---")
        print(signals_df_all_tickers[signals_df_all_tickers['signal'] == 1.0].head())
        
        print("\n--- 전체 매도 신호 발생일 (처음 5건) ---")
        print(signals_df_all_tickers[signals_df_all_tickers['signal'] == -1.0].head())
        
        print(f"\n총 신호 데이터 행 수: {len(signals_df_all_tickers)}")
        print(f"포함된 종목: {signals_df_all_tickers['Ticker'].unique().tolist()}")
        print(f"매수 신호 총 개수: {signals_df_all_tickers[signals_df_all_tickers['signal'] == 1.0].shape[0]}")
        print(f"매도 신호 총 개수: {signals_df_all_tickers[signals_df_all_tickers['signal'] == -1.0].shape[0]}")

    else:
        print("데이터 준비 실패 (전략 테스트).")