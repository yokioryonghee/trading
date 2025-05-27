# data_handler.py (수정 버전)

import yfinance as yf
import pandas as pd
import numpy as np
import os
import pandas_ta as ta

# config.py에서 필요한 설정값들을 import 합니다.
# TICKER 대신 STOCK_UNIVERSE를 임포트하고, CSV_FILENAME도 CSV_FILENAME_TEMPLATE 역할로 사용합니다.
from config import STOCK_UNIVERSE, START_DATE, END_DATE, DATA_PATH, CSV_FILENAME
from config import EMA_SHORT_PERIOD, EMA_LONG_PERIOD, RSI_PERIOD
from config import MACD_FAST_PERIOD, MACD_SLOW_PERIOD, MACD_SIGNAL_PERIOD
from config import PREDICTION_HORIZON_DAYS, TARGET_RISE_PERCENT

def fetch_and_save_data_for_ticker(ticker, start_date=START_DATE, end_date=END_DATE,
                                    data_path=DATA_PATH, csv_filename_template=CSV_FILENAME):
    """
    Yahoo Finance에서 특정 주식 데이터를 가져오거나, 이미 존재하는 파일에서 로드합니다.
    """
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # 종목별 파일명 생성: template.format을 사용하여 {TICKER} 부분을 현재 티커로 대체
    current_csv_filename = csv_filename_template.format(TICKER=ticker, START_DATE=start_date, END_DATE=end_date)
    file_path = os.path.join(data_path, current_csv_filename)
    
    df = None
    should_download = True

    # 기존 파일이 있다면 삭제 후 새로 다운로드
    if os.path.exists(file_path):
        print(f"기존 데이터 파일 '{file_path}'이(가) 존재합니다. 삭제 후 새로 다운로드합니다.")
        os.remove(file_path)
        should_download = True # 강제로 새로 다운로드

    if should_download:
        print(f"Yahoo Finance에서 데이터를 다운로드합니다: {ticker} ({start_date} ~ {end_date})...")
        try:
            # yfinance.download에 auto_adjust=False 명시적으로 전달하여 MultiIndex 또는 Adj Close 문제 방지
            df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
            
            if not df.empty:
                df.index.name = 'Date'
                df.to_csv(file_path)
                print(f"데이터를 '{file_path}'에 성공적으로 저장했습니다.")
            else:
                print(f"{ticker}에 대한 데이터를 {start_date}부터 {end_date}까지 다운로드할 수 없습니다.")
                return None
        except Exception as e:
            print(f"Yahoo Finance 데이터 다운로드 중 오류 발생: {e}")
            return None
    
    # 다운로드가 완료된 후, 또는 기존 파일이 사용될 경우, 다시 로드하여 데이터프레임 확인
    # 이 부분은 should_download와 상관없이, df가 None일 경우에만 재로드를 시도
    if df is None or df.empty: # 위에서 다운로드 실패 시 df가 None일 수 있음
        if os.path.exists(file_path): # 파일은 있는데 df가 None이면 다시 읽어봐야 함 (예외 처리 후)
            print(f"재시도: 데이터 파일 '{file_path}'을(를) 로드합니다.")
            try:
                df = pd.read_csv(file_path, index_col=0, parse_dates=True, dayfirst=False)
                df.index = pd.to_datetime(df.index, errors='coerce')
                df.dropna(subset=[df.index.name], inplace=True)
                if df.empty:
                    raise ValueError("재로드된 CSV 파일에서 유효한 날짜 데이터를 찾을 수 없습니다.")
                if df.index.name != 'Date':
                    df.index.name = 'Date'
            except Exception as e:
                print(f"재로드 중 오류 발생 ({file_path}): {e}. 데이터 처리에 실패했습니다.")
                return None
        else:
            print("데이터 파일이 존재하지 않아 로드할 수 없습니다.")
            return None

    return df


def calculate_indicators_and_target(df_raw):
    """
    주어진 DataFrame에 기술적 지표와 타겟 변수를 계산하여 추가합니다.
    이 함수는 단일 종목의 데이터프레임을 처리합니다.
    """
    df = df_raw.copy()

    # 인덱스 확인 및 DatetimeIndex로 변환
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index, errors='coerce')
            df.dropna(subset=[df.index.name], inplace=True) 
            if df.empty:
                print("Error: DataFrame is empty after converting index to DatetimeIndex and dropping NaT.")
                return None
        except Exception as e:
            print(f"Error converting index to DatetimeIndex in calculate_indicators_and_target: {e}")
            return None

    # 중복된 인덱스 제거
    df = df[~df.index.duplicated(keep='first')]
    # print(f"지표 계산 전 (중복 제거 후) 데이터 크기: {len(df)} 행") # DEBUG성 출력 주석 처리

    # 컬럼 이름 정규화 및 MultiIndex 처리
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        # print(f"DEBUG: MultiIndex 컬럼을 단일 레벨로 평탄화. 새 컬럼: {df.columns.tolist()}") # DEBUG성 출력 주석 처리

    # 모든 컬럼 이름을 대문자로 통일 (YFinance 기본값)
    df.columns = [col.upper() for col in df.columns]
    # print(f"DEBUG: 컬럼 이름 모두 대문자로 통일. 현재 컬럼: {df.columns.tolist()}") # DEBUG성 출력 주석 처리
    
    # 필수 컬럼이 모두 있는지 최종 확인
    expected_cols = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME'] # 대문자로 변경
    if not all(col in df.columns for col in expected_cols):
        print(f"오류: 필수 컬럼이 누락되었습니다. 필요한 컬럼: {expected_cols}, 현재 컬럼: {df.columns.tolist()}")
        return None

    # 가격 및 거래량 컬럼 데이터 타입을 숫자형으로 강제 변환
    for col in expected_cols: # 정규화된 컬럼 이름을 사용
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 변환 후 NaN이 생긴 행 제거 (필수 가격 데이터가 없는 경우)
    rows_before_price_dropna = len(df)
    df.dropna(subset=expected_cols, inplace=True) # 필수 컬럼에 NaN이 있는 행 제거
    if len(df) < rows_before_price_dropna:
        print(f"경고: 가격/거래량 데이터 NaN 제거 후 {rows_before_price_dropna - len(df)} 행이 제거되었습니다. 현재 {len(df)} 행.")
    if df.empty:
        print("오류: 가격/거래량 NaN 제거 후 데이터가 비어 있습니다.")
        return None

    # print(f"컬럼 이름 정규화 및 숫자형 변환 후 컬럼: {df.columns.tolist()}") # DEBUG성 출력 주석 처리
    # print(f"데이터 타입 확인: \n{df[expected_cols].dtypes}") # DEBUG성 출력 주석 처리


    # 기술적 지표 계산 및 NaN 개수 확인 (이제 대문자 컬럼 이름 사용)
    df['EMA_short'] = ta.ema(df['CLOSE'], length=EMA_SHORT_PERIOD)
    # print(f"DEBUG: EMA_short 계산 후 NaN 개수: {df['EMA_short'].isnull().sum()}") # DEBUG성 출력 주석 처리

    df['EMA_long'] = ta.ema(df['CLOSE'], length=EMA_LONG_PERIOD)
    # print(f"DEBUG: EMA_long 계산 후 NaN 개수: {df['EMA_long'].isnull().sum()}") # DEBUG성 출력 주석 처리

    df['RSI'] = ta.rsi(df['CLOSE'], length=RSI_PERIOD)
    # print(f"DEBUG: RSI 계산 후 NaN 개수: {df['RSI'].isnull().sum()}") # DEBUG성 출력 주석 처리

    # --- MACD 계산 로직 수정: pandas_ta 버전 호환성 강화 ---
    macd_results = ta.macd(df['CLOSE'], fast=MACD_FAST_PERIOD, slow=MACD_SLOW_PERIOD, signal=MACD_SIGNAL_PERIOD)
    
    # print(f"DEBUG: MACD 계산 결과 컬럼: {macd_results.columns.tolist()}") # DEBUG성 출력 주석 처리
    
    # MACD 컬럼 이름 확인 및 할당 (유연하게 처리)
    if len(macd_results.columns) >= 3:
        df['MACD'] = macd_results.iloc[:, 0] # 첫 번째 컬럼 (MACD 라인)
        df['MACD_hist'] = macd_results.iloc[:, 1] # 두 번째 컬럼 (MACD 히스토그램)
        # print(f"DEBUG: MACD/MACD_hist 컬럼 순서대로 할당 완료.") # DEBUG성 출력 주석 처리
    else:
        macd_col_name = f'MACD_{MACD_FAST_PERIOD}_{MACD_SLOW_PERIOD}_{MACD_SIGNAL_PERIOD}'
        macdh_col_name = f'MACDH_{MACD_FAST_PERIOD}_{MACD_SLOW_PERIOD}_{MACD_SIGNAL_PERIOD}'
        
        if macd_col_name in macd_results.columns:
            df['MACD'] = macd_results[macd_col_name]
        else:
            print(f"경고: MACD 라인 컬럼 '{macd_col_name}'을(를) 찾을 수 없습니다. MACD 결과 컬럼: {macd_results.columns.tolist()}")
            if not macd_results.empty and len(macd_results.columns) > 0:
                df['MACD'] = macd_results.iloc[:, 0]
                print("MACD 라인으로 MACD 결과의 첫 번째 컬럼을 사용합니다.")
            else:
                print("MACD 라인 컬럼을 찾을 수 없거나 MACD 결과가 비어있습니다. MACD 계산 실패.")
                return None # MACD 계산 실패 시 None 반환

        if macdh_col_name in macd_results.columns:
            df['MACD_hist'] = macd_results[macdh_col_name]
        else:
            print(f"경고: MACD Histogram 컬럼 '{macdh_col_name}'을(를) 찾을 수 없습니다. MACD 결과 컬럼: {macd_results.columns.tolist()}")
            if not macd_results.empty and len(macd_results.columns) > 1:
                df['MACD_hist'] = macd_results.iloc[:, 1]
                print("MACD Histogram으로 MACD 결과의 두 번째 컬럼을 사용합니다.")
            else:
                print("MACD Histogram 컬럼을 찾을 수 없거나 MACD 결과에 충분한 컬럼이 없습니다. MACD Histogram 계산 실패.")
                df['MACD_hist'] = np.nan # 컬럼이 없다면 NaN으로 채움
                
    # print(f"DEBUG: MACD/MACD_hist 계산 후 (NaN 개수: {df['MACD'].isnull().sum()}, {df['MACD_hist'].isnull().sum()})") # DEBUG성 출력 주석 처리


    atr_col_name = f'ATR_{14}'
    df[atr_col_name] = ta.atr(df['HIGH'], df['LOW'], df['CLOSE'], length=14)
    # print(f"DEBUG: ATR 계산 후 NaN 개수: {df[atr_col_name].isnull().sum()}") # DEBUG성 출력 주석 처리


    # 기타 유용한 피처
    df['return_1d'] = df['CLOSE'].pct_change()
    # print(f"DEBUG: return_1d 계산 후 NaN 개수: {df['return_1d'].isnull().sum()}") # DEBUG성 출력 주석 처리

    df['high_low_diff_pct'] = (df['HIGH'] - df['LOW']) / df['CLOSE']
    df['open_close_diff_pct'] = (df['CLOSE'] - df['OPEN']) / df['OPEN']
    df['volume_change_pct'] = df['VOLUME'].pct_change()
    
    # 'close_div_ema_long'는 EMA_long이 NaN이면 NaN이 되므로, EMA_long이 계산된 후 추가
    df['close_div_ema_long'] = df['CLOSE'] / df['EMA_long'] 
    # print(f"DEBUG: 기타 피처 계산 완료.") # DEBUG성 출력 주석 처리

    # print(f"모든 기술적 지표 계산 후 (NaN 포함): {len(df)} 행") # DEBUG성 출력 주석 처리


    # --- 타겟 변수 'target' 계산 로직 ---
    # future_high 계산 전, High 컬럼에 NaN이 있는지 확인 (혹시 모를 상황 대비)
    if df['HIGH'].isnull().any(): # 대문자 HIGH 사용
        print(f"DEBUG: HIGH 컬럼에 NaN이 있습니다. 개수: {df['HIGH'].isnull().sum()}")

    # 미래 고가(future_high)는 현재 종가 대비 PREDICTION_HORIZON_DAYS 기간 동안의 최고가
    df['future_high'] = df['HIGH'].shift(-1).rolling(window=PREDICTION_HORIZON_DAYS, min_periods=1).max()
    # print(f"DEBUG: future_high 계산 후 NaN 개수: {df['future_high'].isnull().sum()}") # DEBUG성 출력 주석 처리

    # 타겟: 미래 최고가가 현재 종가 대비 TARGET_RISE_PERCENT 이상 상승하는 경우 (1) 아니면 (0)
    df['target'] = ((df['future_high'] / df['CLOSE']) - 1 >= TARGET_RISE_PERCENT).astype(int)
    # print(f"DEBUG: target 계산 후 NaN 개수: {df['target'].isnull().sum()}") # DEBUG성 출력 주석 처리


    # 타겟을 정의할 수 없는 마지막 PREDICTION_HORIZON_DAYS 기간의 데이터는 제거
    initial_rows_before_target_drop = len(df)
    df = df.iloc[:-PREDICTION_HORIZON_DAYS] # 마지막 PREDICTION_HORIZON_DAYS 만큼의 행 제거
    # print(f"타겟 계산 후 마지막 {PREDICTION_HORIZON_DAYS} 행 제거: {len(df)} 행 (이전 {initial_rows_before_target_drop} 행)") # DEBUG성 출력 주석 처리
    
    # 이 시점에서 target 컬럼에 NaN이 얼마나 있는지 확인
    # print(f"DEBUG: target 컬럼의 NaN 개수 (행 제거 후): {df['target'].isnull().sum()}") # DEBUG성 출력 주석 처리


    # 모든 NaN 값 제거 (기술적 지표 계산 초기, 타겟 계산 등으로 인해 발생)
    rows_before_dropna = len(df)
    df.dropna(inplace=True)
    # print(f"dropna() 후 최종 데이터 크기: {len(df)} 행 (제거된 NaN 행: {rows_before_dropna - len(df)})") # DEBUG성 출력 주석 처리

    if df.empty:
        print("경고: 지표/타겟 계산 및 NaN 제거 후 데이터가 비어 있습니다. 데이터 기간 또는 파라미터(예: EMA 기간, PREDICTION_HORIZON_DAYS)를 확인하십시오.")
        return None

    return df

# get_prepared_data 함수를 다중 종목 처리용으로 변경
def get_prepared_data_for_multiple_tickers(tickers=STOCK_UNIVERSE):
    """
    여러 종목에 대해 데이터를 가져와 기술적 지표와 타겟 변수를 계산하고,
    모든 데이터를 하나의 DataFrame으로 합쳐 반환하는 메인 함수.
    """
    all_data = []
    print(f"데이터 준비 중: {len(tickers)}개 종목 ({START_DATE} ~ {END_DATE})")

    for ticker in tickers:
        print(f"\n--- {ticker} 데이터 처리 시작 ---")
        # 각 종목별 데이터 가져오기 및 저장
        df_raw = fetch_and_save_data_for_ticker(ticker)
        if df_raw is None or df_raw.empty:
            print(f"{ticker} 원시 데이터 로드 실패. 다음 종목으로 넘어갑니다.")
            continue
        
        # 각 종목별 지표 및 타겟 계산
        df_with_features_and_target = calculate_indicators_and_target(df_raw)
        
        if df_with_features_and_target is None or df_with_features_and_target.empty:
            print(f"{ticker} 지표/타겟 계산 후 데이터 없음. 다음 종목으로 넘어갑니다.")
            continue
        
        # 'Ticker' 컬럼 추가하여 어떤 종목 데이터인지 구분
        df_with_features_and_target['Ticker'] = ticker
        all_data.append(df_with_features_and_target)
        print(f"--- {ticker} 데이터 처리 완료. 최종 유효 데이터 크기: {len(df_with_features_and_target)} 행 ---")

    if not all_data:
        print("모든 종목에 대해 데이터 준비에 실패했습니다.")
        return None
    
    # 모든 종목의 데이터를 하나의 DataFrame으로 합치기
    # 합칠 때 Date 인덱스를 기준으로 정렬하고, Ticker 컬럼으로 구분
    final_df = pd.concat(all_data).sort_index()
    
    print(f"\n모든 종목 지표 및 타겟 계산 완료. 최종 유효 데이터 크기: {len(final_df)} 행")
    return final_df

# 외부에서 'get_prepared_data()'를 호출할 때 다중 종목 함수가 실행되도록 별칭(alias) 설정
get_prepared_data = get_prepared_data_for_multiple_tickers


if __name__ == '__main__':
    print("--- data_handler.py 단독 실행 테스트 시작 ---")
    # 단독 실행 시, config.py의 STOCK_UNIVERSE에 있는 모든 종목을 처리합니다.
    prepared_df_all_tickers = get_prepared_data(tickers=STOCK_UNIVERSE) 

    if prepared_df_all_tickers is not None and not prepared_df_all_tickers.empty:
        print("\n--- 준비된 전체 데이터 (처음 5행) ---")
        # Ticker 컬럼을 포함하여 출력
        print(prepared_df_all_tickers.head())
        
        print(f"\n--- 준비된 전체 데이터 기간: {prepared_df_all_tickers.index.min()} ~ {prepared_df_all_tickers.index.max()} ---")
        print(f"총 데이터 행 수: {len(prepared_df_all_tickers)}")
        print(f"총 컬럼 수: {len(prepared_df_all_tickers.columns)}")
        print(f"포함된 종목: {prepared_df_all_tickers['Ticker'].unique().tolist()}")
        
        # 각 종목별 타겟 분포 확인
        print("\n--- 각 종목별 타겟 분포 ---")
        # Ticker 별로 그룹화하여 타겟 분포 확인
        print(prepared_df_all_tickers.groupby('Ticker')['target'].value_counts())
        
    else:
        print("데이터 준비에 실패했습니다 (메인 함수).")