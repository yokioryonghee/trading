import pandas as pd
import joblib
import os

# 모델 학습에 사용된 최종 피처 컬럼 리스트
# 이 리스트는 train_evaluate_ml.py에서 모델 학습 시 사용된
# feature_columns_for_model (또는 valid_feature_columns)과 정확히 일치해야 합니다.
# data_handler.py와 train_evaluate_ml.py 분석을 통해 도출된 리스트:
# ml_strategy.py 파일 상단

# 모델 학습에 사용된 최종 피처 컬럼 리스트
# train_evaluate_ml.py 에서 DEBUG 프린트로 확인한 목록으로 교체합니다.
ML_MODEL_FEATURE_COLUMNS = [
    'CLOSE', 'HIGH', 'LOW', 'OPEN', 'VOLUME', 
    'EMA_short', 'EMA_long', 'RSI', 'MACD', 'MACD_hist', 
    'ATR_14', 'return_1d', 'high_low_diff_pct', 'open_close_diff_pct', 
    'volume_change_pct', 'close_div_ema_long'
]

# ... 이하 generate_signals_from_ml 함수 정의 ...

def generate_signals_from_ml(df_with_indicators, model_dir):
    """
    학습된 ML 모델을 사용하여 각 종목별 매매 신호를 생성합니다.
    신호: 1 (매수), 0 (보유/관망).

    Args:
        df_with_indicators (pd.DataFrame): 'Ticker' 컬럼 및 모델 학습에 사용된
                                          모든 피처 컬럼을 포함하는 DataFrame.
                                          data_handler.get_prepared_data()의 반환값.
        model_dir (str): 학습된 .pkl 모델 파일들이 저장된 디렉토리 경로.

    Returns:
        pd.DataFrame: 'Date'를 인덱스로 하고 'signal'과 'Ticker' 컬럼을 가진 DataFrame.
    """
    all_signals = []
    print(f"ML 신호 생성을 시작합니다. 사용할 모델 디렉토리: {model_dir}")
    print(f"예측에 사용할 피처 수: {len(ML_MODEL_FEATURE_COLUMNS)}")
    print(f"사용될 피처 예시 (처음 5개): {ML_MODEL_FEATURE_COLUMNS[:5]}")

    # df_with_indicators에 있는 모든 티커에 대해 반복
    for ticker in df_with_indicators['Ticker'].unique():
        ticker_df = df_with_indicators[df_with_indicators['Ticker'] == ticker].copy()

        if ticker_df.empty:
            print(f"정보: {ticker}에 대한 데이터가 없어 신호 생성을 건너뜁니다.")
            continue

        # 예측에 사용할 피처만 선택
        # ML_MODEL_FEATURE_COLUMNS에 정의된 모든 피처가 ticker_df에 있는지 확인
        missing_features = [col for col in ML_MODEL_FEATURE_COLUMNS if col not in ticker_df.columns]
        if missing_features:
            print(f"경고: {ticker} 데이터에 다음 필수 피처가 없습니다: {missing_features}. 이 종목은 건너뜁니다.")
            # 날짜 인덱스 유지를 위해 빈 신호 추가
            signals_for_ticker_empty = pd.DataFrame(index=ticker_df.index)
            signals_for_ticker_empty['signal'] = 0.0
            signals_for_ticker_empty['Ticker'] = ticker
            all_signals.append(signals_for_ticker_empty[['signal', 'Ticker']])
            continue

        X_live = ticker_df[ML_MODEL_FEATURE_COLUMNS]

        # 모델 학습 시와 동일하게 NaN 처리
        # data_handler.py의 get_prepared_data()는 이미 dropna()를 수행하므로,
        # X_live는 일반적으로 NaN이 없을 것으로 예상됩니다.
        # 하지만, 만약의 경우를 대비해 확인 및 간단한 처리를 추가할 수 있습니다.
        if X_live.isnull().values.any():
            nan_count = X_live.isnull().sum().sum()
            print(f"경고: {ticker}의 예측 입력 데이터에 {nan_count}개의 NaN 값이 있습니다. 해당 행들을 예측에서 제외하거나 0으로 채웁니다.")
            # 예시: NaN이 있는 행을 예측에서 제외 (또는 fillna(0) 등 다른 전략 사용)
            # 여기서는 해당 행에 대해 신호를 0으로 유지하기 위해 X_live에서 dropna는 하지 않고,
            # 예측 후 신호 할당 시점에 NaN이 있는 예측은 0으로 처리될 수 있도록 합니다.
            # 또는, train_evaluate_ml.py 처럼 여기서도 숫자형이 아닌 컬럼을 사전에 제거하는 로직이 필요할 수 있으나,
            # ML_MODEL_FEATURE_COLUMNS 자체가 이미 숫자형으로만 구성되었다고 가정합니다.
            X_live_processed = X_live.fillna(0) # 간단히 0으로 채우는 예시
            if X_live_processed.empty and not X_live.empty : # 모든 값이 NaN이어서 비어버린 경우
                 print(f"경고: {ticker}의 모든 예측용 데이터가 NaN 처리 후 비었습니다. 건너뜁니다.")
                 # (코드 이어짐)
            X_live = X_live_processed


        if X_live.empty:
            print(f"경고: {ticker}에 대한 예측용 유효 피처 데이터가 없습니다. 건너뜁니다.")
            signals_for_ticker_empty = pd.DataFrame(index=ticker_df.index)
            signals_for_ticker_empty['signal'] = 0.0
            signals_for_ticker_empty['Ticker'] = ticker
            all_signals.append(signals_for_ticker_empty[['signal', 'Ticker']])
            continue

        # 신호를 저장할 임시 DataFrame
        signals_for_ticker = pd.DataFrame(index=ticker_df.index)
        signals_for_ticker['signal'] = 0.0 # 기본값은 보유/관망
        signals_for_ticker['Ticker'] = ticker

        # train_evaluate_ml.py에서 저장한 앙상블 모델 사용 가정
        model_filename = f'ensemble_voting_classifier_hard_{ticker}.pkl' #
        model_path = os.path.join(model_dir, model_filename)

        if not os.path.exists(model_path):
            print(f"경고: {ticker}에 대한 모델 파일({model_path})이 없습니다. 이 종목은 0 신호를 사용합니다.")
            all_signals.append(signals_for_ticker[['signal', 'Ticker']])
            continue

        try:
            loaded_model = joblib.load(model_path)
            predictions = loaded_model.predict(X_live) # 모델 예측 (0 또는 1)
            signals_for_ticker['signal'] = predictions.astype(float) # 1.0 또는 0.0
        except ValueError as ve:
            print(f"에러: {ticker} 모델 예측 중 ValueError 발생 (피처 불일치 등): {ve}")
            # 오류 발생 시 해당 티커는 0 신호를 유지
        except Exception as e:
            print(f"에러: {ticker} 모델 로드 또는 예측 중 기타 오류 발생: {e}")
            # 오류 발생 시 해당 티커는 0 신호를 유지

        all_signals.append(signals_for_ticker[['signal', 'Ticker']])

    if not all_signals:
        print("경고: 모든 종목에 대해 ML 신호 생성에 실패했습니다.")
        # Date 컬럼이 없으므로 인덱스 이름으로 설정
        empty_df = pd.DataFrame(columns=['signal', 'Ticker'])
        empty_df.index.name = 'Date'
        return empty_df

    final_signals_df = pd.concat(all_signals).sort_index()
    print(f"ML 신호 생성 완료. 총 신호 수: {len(final_signals_df)}")
    return final_signals_df