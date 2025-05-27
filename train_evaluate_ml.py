# train_evaluate_ml.py (수정 버전 - IndexError 해결)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import lightgbm as lgb
import joblib
import os
#import sys



# 프로젝트 루트 디렉토리를 Python 경로에 추가
#script_dir = os.path.dirname(__file__)
#project_root = os.path.abspath(os.path.join(script_dir, '..'))
#sys.path.append(project_root)

# data_handler와 config 임포트
from data_handler import get_prepared_data
from config import STOCK_UNIVERSE # 학습할 종목 리스트 가져오기

# --- 1. 데이터 로드 및 전처리 ---
try:
    # get_prepared_data 함수를 사용하여 전처리된 다중 종목 데이터(지표 및 타겟 포함)를 가져옵니다.
    # 이제 반환되는 prepared_df는 'Ticker' 컬럼을 포함합니다.
    prepared_df = get_prepared_data()
    if prepared_df is None or prepared_df.empty:
        print("데이터 준비에 실패했습니다. 모델 학습을 종료합니다.")
        exit()
    print(f"준비된 전체 데이터 기간: {prepared_df.index.min()} ~ {prepared_df.index.max()}")
    print(f"준비된 데이터 컬럼: {prepared_df.columns.tolist()}")
    print(f"포함된 종목: {prepared_df['Ticker'].unique().tolist()}")

except Exception as e:
    print(f"데이터 준비 중 오류 발생: {e}")
    exit()

# (임시로 주석처리해놓음. 다시 체크할것)타겟 컬럼 명
#target_column = 'target'
# 피처 컬럼 명 (타겟 컬럼과 Ticker 컬럼을 제외한 모든 컬럼)
#feature_columns = [col for col in prepared_df.columns if col not in [target_column, 'Ticker']]


target_column = 'target'
# 'future_high'와 'ADJ CLOSE'를 학습 피처에서 명시적으로 제외
excluded_cols_for_training = [target_column, 'Ticker', 'future_high', 'ADJ CLOSE']
feature_columns = [col for col in prepared_df.columns if col not in excluded_cols_for_training]
print(f"초기 feature_columns (제외 후): {feature_columns[:5]}... 등 {len(feature_columns)}개") # 디버깅용


# 모델 학습 및 저장 경로 설정
model_save_path = './ml_model/'
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)


# 각 종목별로 모델 학습 및 평가를 진행합니다.
for ticker in STOCK_UNIVERSE:
    print(f"\n--- {ticker} 에 대한 모델 학습 시작 ---")
    
    # 특정 종목의 데이터만 필터링
    ticker_df = prepared_df[prepared_df['Ticker'] == ticker].copy()

    if ticker_df.empty:
        print(f"경고: {ticker} 에 대한 유효한 데이터가 없습니다. 이 종목은 건너뜜니다.")
        continue

    # 시계열 데이터를 위한 학습/테스트 분할
    # 데이터의 80%를 학습, 20%를 테스트 (시간 순서대로)
    train_size = int(len(ticker_df) * 0.8)
    train_df = ticker_df.iloc[:train_size]
    test_df = ticker_df.iloc[train_size:]

    if train_df.empty or test_df.empty:
        print(f"경고: {ticker} 에 대한 학습 또는 테스트 데이터가 충분하지 않습니다. 이 종목은 건너뜁니다.")
        continue

    X_train = train_df[feature_columns]
    y_train = train_df[target_column]
    X_test = test_df[feature_columns]
    y_test = test_df[target_column]
    
    # 모든 피처가 숫자형인지 확인 (모델 학습 전에 최종 점검)
    # NaN이 있는 컬럼은 LightGBM에서 Warning이 나올 수 있으나, RandomForest는 오류 발생 가능
    # 여기서는 비숫자형 컬럼만 제거하고, NaN은 데이터 준비 단계에서 이미 제거됨
    initial_feature_count = len(feature_columns)
    valid_feature_columns = []
    for col in feature_columns:
        if pd.api.types.is_numeric_dtype(X_train[col]):
            valid_feature_columns.append(col)
        else:
            print(f"경고: {ticker}의 학습 데이터에 비숫자형 피처 '{col}'이(가) 있습니다. 해당 컬럼을 제외합니다.")
            
    X_train = X_train[valid_feature_columns]
    X_test = X_test[valid_feature_columns]
    feature_columns_for_model = valid_feature_columns # 실제로 모델에 들어갈 피처 컬럼
    
    if ticker_df.empty:
        pass
    else:
        print(f"DEBUG [{ticker}]: 학습에 사용된 최종 피처 목록 (feature_columns_for_model): {feature_columns_for_model}")

    if X_train.empty or X_test.empty:
        print(f"경고: {ticker} 학습/테스트 데이터에서 유효한 숫자형 피처가 없어 모델 학습을 건너뜁니다.")
        continue

    # 학습/테스트 데이터셋 크기 및 타겟 분포 출력
    print(f"  {ticker} 학습 데이터셋 크기: {len(X_train)}")
    print(f"  {ticker} 테스트 데이터셋 크기: {len(X_test)}")
    print(f"  {ticker} 학습 데이터셋 타겟 분포 (0: {y_train.value_counts().get(0, 0)}, 1: {y_train.value_counts().get(1, 0)})")
    print(f"  {ticker} 테스트 데이터셋 타겟 분포 (0: {y_test.value_counts().get(0, 0)}, 1: {y_test.value_counts().get(1, 0)})")


    # --- 2. 모델 정의 및 학습 ---
    print(f"  --- {ticker} 모델 학습 중 ---")

    # RandomForestClassifier
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    print(f"  {ticker} RandomForestClassifier 학습 완료.")

    # LightGBM Classifier
    lgbm_model = lgb.LGBMClassifier(random_state=42)
    lgbm_model.fit(X_train, y_train)
    print(f"  {ticker} LightGBM Classifier 학습 완료.")

    # 앙상블 모델 (Hard Voting)
    voting_clf_hard = VotingClassifier(estimators=[('rf', rf_model), ('lgbm', lgbm_model)], voting='hard')
    voting_clf_hard.fit(X_train, y_train)
    print("앙상블 (Hard Voting) 모델 학습 완료.")

    # --- 3. 모델 평가 (테스트 세트) ---
    print(f"  --- {ticker} 모델 평가 시작 ---")

    y_pred_rf = rf_model.predict(X_test)
    y_pred_lgbm = lgbm_model.predict(X_test)
    y_pred_ensemble = voting_clf_hard.predict(X_test)
    
    # 평가 지표 출력
    print(f"\n  --- {ticker} RandomForestClassifier 성능 ---")
    # y_test에 단일 클래스만 있을 경우 (0 또는 1) precision_score, recall_score, f1_score는 NaN을 반환하거나 오류를 낼 수 있음.
    # 이를 방지하기 위해 zero_division 파라미터 추가
    print(f"  Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
    print(f"  Precision (Class 1): {precision_score(y_test, y_pred_rf, zero_division=0):.4f}")
    print(f"  Recall (Class 1): {recall_score(y_test, y_pred_rf, zero_division=0):.4f}")
    print(f"  F1 Score (Class 1): {f1_score(y_test, y_pred_rf, zero_division=0):.4f}")

    print(f"\n  --- {ticker} LightGBM Classifier 성능 ---")
    print(f"  Accuracy: {accuracy_score(y_test, y_pred_lgbm):.4f}")
    print(f"  Precision (Class 1): {precision_score(y_test, y_pred_lgbm, zero_division=0):.4f}")
    print(f"  Recall (Class 1): {recall_score(y_test, y_pred_lgbm, zero_division=0):.4f}")
    print(f"  F1 Score (Class 1): {f1_score(y_test, y_pred_lgbm, zero_division=0):.4f}")

    print(f"\n  --- {ticker} 앙상블 (Hard Voting) 모델 성능 ---")
    print(f"  Accuracy: {accuracy_score(y_test, y_pred_ensemble):.4f}")
    print(f"  Precision (Class 1): {precision_score(y_test, y_pred_ensemble, zero_division=0):.4f}")
    print(f"  Recall (Class 1): {recall_score(y_test, y_pred_ensemble, zero_division=0):.4f}")
    print(f"  F1 Score (Class 1): {f1_score(y_test, y_pred_ensemble, zero_division=0):.4f}")

    try:
        # Soft Voting Classifier를 사용하여 ROC AUC 계산
        # LightGBM은 predict_proba를 제공하므로, 'soft' voting 가능
        voting_clf_soft = VotingClassifier(estimators=[('rf', rf_model), ('lgbm', lgbm_model)], voting='soft')
        voting_clf_soft.fit(X_train, y_train)
        y_pred_proba_ensemble = voting_clf_soft.predict_proba(X_test)[:, 1]
        
        # y_test에 단일 클래스만 있을 경우 ROC AUC는 정의되지 않으므로 NaN 반환
        if len(y_test.unique()) > 1:
            print(f"  ROC AUC Score (Soft Voting): {roc_auc_score(y_test, y_pred_proba_ensemble):.4f}")
        else:
            print(f"  ROC AUC Score (Soft Voting): nan (테스트 세트에 단일 클래스만 존재)")

    except AttributeError:
        print("  Warning: Soft Voting ROC AUC could not be calculated. Estimators do not support predict_proba or insufficient data.") 
    except ValueError as ve:
        # ROC AUC 계산 중 ValueError (예: 모든 레이블이 동일할 때) 처리
        print(f"  ROC AUC Score (Soft Voting): nan (ValueError: {ve})")

    print(f"\n  Confusion Matrix (테스트 세트 - {ticker} 앙상블 모델):")
    cm_ensemble = confusion_matrix(y_test, y_pred_ensemble)
    print(cm_ensemble)

    # *** 수정: Confusion Matrix의 크기를 확인하여 인덱싱 오류를 방지합니다. ***
    if cm_ensemble.shape == (2, 2):
        print(f"  TN: {cm_ensemble[0, 0]}, FP: {cm_ensemble[0, 1]}, FN: {cm_ensemble[1, 0]}, TP: {cm_ensemble[1, 1]}")
    elif cm_ensemble.shape == (1, 1):
        # 단일 클래스만 존재하는 경우
        # y_test의 유니크한 값이 0 또는 1 중 하나일 때 발생
        if y_test.value_counts().get(0, 0) == len(y_test): # 실제값이 모두 0 (미상승)인 경우
            print(f"  TN: {cm_ensemble[0, 0]}, FP: 0, FN: 0, TP: 0 (테스트 세트에 급등이 없습니다.)")
        elif y_test.value_counts().get(1, 0) == len(y_test): # 실제값이 모두 1 (급등)인 경우
            print(f"  TN: 0, FP: 0, FN: 0, TP: {cm_ensemble[0, 0]} (테스트 세트에 미상승이 없습니다.)")
        else: # 그 외 (예: y_test는 다중 클래스인데 y_pred가 단일 클래스일 때)
            print(f"  경고: Confusion Matrix가 1x1이지만 y_test는 단일 클래스가 아님. cm: {cm_ensemble}")
            # 이 경우 TP/FP/TN/FN을 정확히 할당하기 어려우므로, 추가적인 분석 필요
            print(f"  Accuracy: {accuracy_score(y_test, y_pred_ensemble):.4f}")
    else:
        print(f"  경고: 예상치 못한 Confusion Matrix 형태: {cm_ensemble.shape}")
        # 이 경우 모든 지표를 출력하는 것이 어려울 수 있습니다.
        print(f"  Accuracy: {accuracy_score(y_test, y_pred_ensemble):.4f}")


    # --- 4. 모델 저장 ---
    # 각 종목별로 모델을 저장합니다. 파일명에 티커를 포함시킵니다.
    joblib.dump(voting_clf_hard, os.path.join(model_save_path, f'ensemble_voting_classifier_hard_{ticker}.pkl'))
    joblib.dump(rf_model, os.path.join(model_save_path, f'random_forest_classifier_{ticker}.pkl'))
    joblib.dump(lgbm_model, os.path.join(model_save_path, f'lightgbm_classifier_{ticker}.pkl'))
    print(f"  {ticker} 모델이 '{model_save_path}'에 저장되었습니다.")
    print(f"--- {ticker} 모델 학습 및 평가 완료 ---")

print("\n모든 종목에 대한 모델 학습 및 평가 완료.")