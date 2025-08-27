# 퀀트 트레이딩 전략 백테스팅 시스템

## 프로젝트 개요

이 프로젝트는 파이썬으로 개발된 종합적인 주식 트레이딩 전략 백테스팅 시스템입니다. 기술적 지표를 활용하는 규칙 기반 전략과 머신러닝(기계 학습) 기반 전략을 모두 설계하고, 과거 데이터를 통해 성과를 테스트 및 평가할 수 있습니다. 다양한 주식 유니버스, 기간, 전략 파라미터에 대한 실험을 용이하게 하도록 모듈화되고 설정 가능하게 설계되었습니다. (프로젝트를 쉽게 활용해 볼 수 있는 streamlit app을 구현해 놓았으니, tradingBot3 저장소를 이용하거나,https://tradingbot3-aacevydlbvfcxoqpfal5nj.streamlit.app 을 이용하세요)

## 주요 기능

* **듀얼 전략 프레임워크**:
    * 일반적인 기술적 지표(예: EMA 교차, RSI 임계값)를 사용하는 **규칙 기반 트레이딩 전략** 구현 및 테스트 지원.
    * 모델 학습, 평가, 예측을 위한 전체 파이프라인을 갖춘 **머신러닝 기반 트레이딩 전략** 지원.
* **머신러닝 파이프라인**:
    * `yfinance` 및 `pandas`를 사용한 다수 종목 주가 데이터 자동 수집 및 전처리.
    * `pandas-ta`를 활용하여 ML 모델의 피처로 사용될 다양한 기술적 지표(EMA, RSI, MACD, ATR 등) 계산.
    * `RandomForestClassifier`, `LGBMClassifier` 및 이들을 결합한 `VotingClassifier` (앙상블) 등 다중 분류 모델 학습 및 평가.
    * 학습된 모델을 `.pkl` 파일로 저장하고 백테스팅 시 불러와 활용하여 모델 지속성 확보.
    * 학습 기간과 테스트 기간을 분리하여 모델의 일반화 성능을 평가하는 **Out-of-Sample (표본 외) 테스트** 방법론 적용.
* **종합 백테스팅 엔진 (`backtester.py`)**:
    * 다중 종목에 대한 동시 백테스팅 지원.
    * 생성된 신호(매수, 매도, 보유)에 기반한 거래 시뮬레이션.
    * 초기 자본금, 거래 비용(수수료), 슬리피지(선택 사항, 현재 `config.py`에 정의되어 있으나 `backtester.py` 적용 로직 추가 필요, 배당금을 계산안했으므로, 슬리피지도 배제하고 테스트) 등 설정 가능.
    * 포지션 관리 기능:
        * 익절(Take Profit, TP) 및 손절(Stop Loss, SL) 메커니즘.
        * 포지션 최대 보유 기간(Max Hold Days) 설정.
        * 최대 동시 보유 가능 종목 수 및 종목당 최대 투자 비중 제어.
* **성과 분석 및 시각화 (`performance_analyzer.py`)**:
    * 주요 성과 지표 계산: 총수익률(Total Return), 연 복리 성장률(CAGR), 최대 낙폭(Max Drawdown, MDD), 샤프 지수(Sharpe Ratio) 등.
    * 선택된 기준 지수(벤치마크 티커)의 단순 보유 전략 대비 포트폴리오 가치 변화 시각화.
* **모듈화 및 설정 기반 설계**:
    * 데이터 처리, 전략 로직, ML 모델 학습, 백테스팅, 성과 분석 등 기능별로 파이썬 모듈(`.py` 파일)이 잘 분리되어 있음.
    * 중앙 설정 파일(`config.py`)을 통해 주식 유니버스, 날짜 범위, 기술 지표 파라미터, 백테스팅 설정 등을 쉽게 조정 가능.(특정기간을 학습한 모델을 이용해서 테스트를 진행할 경우에는, 모델의 정확도를 알아보기 위해 조정X)
* **ML 전략의 위험 관리 관찰 (사용자 강조 사항)**:
    * 머신러닝 모델은 특정 사용자 지정 백테스트(특정 데이터셋 및 기간, 시장 하락기 포함)에서 시장 벤치마크 대비 손실을 줄이며 위험을 완화하는 경향을 보였습니다. *(이 관찰은 해당 테스트의 특정 설정, 주식 유니버스, 시장 기간에 따라 달라질 수 있습니다.)*
      

## 사용된 기술

* Python 3.10 이상
* Pandas
* NumPy
* Scikit-learn
* LightGBM
* Joblib
* yfinance
* Pandas-TA
* Matplotlib

## 프로젝트 구조

tradingBot/

├── data/                    # 다운로드된 CSV 주가 데이터 저장

├── ml_model/                # 학습된 .pkl ML 모델 파일 저장

├── backtest_logs/           # 백테스트 거래 기록 CSV 저장
│
├── config.py                # 데이터, 지표, 백테스팅 관련 설정

├── data_handler.py          # 데이터 수집, 전처리 및 지표 계산
│
├── train_evaluate_ml.py     # ML 모델 학습 및 평가

├── strategy.py              # 규칙 기반 트레이딩 전략 로직

├── ml_strategy.py           # ML 기반 트레이딩 전략 로직 (학습된 모델 사용)
│
├── backtester.py            # 핵심 백테스팅 엔진

├── performance_analyzer.py  # 성과 지표 계산 및 시각화
│
├── run_backtest.py          # 규칙 기반 전략 백테스트 실행 스크립트

├── run_ml_backtest.py       # ML 전략 백테스트 실행 스크립트
│
├── requirements.txt         # 파이썬 패키지 의존성 목록

└── README.md                # 프로젝트 설명


## 설치 및 실행 방법

1.  **저장소 복제 (해당되는 경우):**
    ```bash
    git clone <저장소_URL>
    cd tradingBot
    ```
2.  **파이썬 가상 환경 생성 및 활성화:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    ```
3.  **필요 패키지 설치:**
    ```bash
    pip install -r requirements.txt
    ```

## 실행 가이드

1.  **`config.py` 설정**:
    * `STOCK_UNIVERSE`에 원하는 티커 목록을 설정합니다.
    * 데이터 수집을 위한 `START_DATE`와 `END_DATE`를 정의합니다.
        * ML Out-of-Sample 테스트 시: 먼저 **학습 기간**으로 설정합니다.
    * 기술 지표 파라미터 (예: `EMA_SHORT_PERIOD`, `RSI_PERIOD`)를 조정합니다.
    * 백테스팅 파라미터 (예: `INITIAL_CAPITAL`, `TAKE_PROFIT_PERCENT`, `STOP_LOSS_PERCENT`, `MAX_HOLD_DAYS`)를 설정합니다.

2.  **ML 모델 학습 (ML 전략 사용 시)**:
    * `config.py`가 **학습 기간**으로 설정되었는지 확인합니다.
    * 학습 스크립트를 실행합니다:
        ```bash
        python train_evaluate_ml.py
        ```
    * 학습된 모델들이 `ml_model/` 디렉토리에 저장됩니다.

3.  **백테스트 실행**:
    * **ML 전략 (Out-of-Sample 테스트 시)**:
        * `config.py`의 `START_DATE`와 `END_DATE`를 원하는 **테스트 기간**으로 수정합니다 (이 기간은 학습 기간 이후여야 하며 겹치지 않아야 함).
        * ML 백테스트 스크립트를 실행합니다:
            ```bash
            python run_ml_backtest.py
            ```
    * **규칙 기반 전략**:
        * `config.py`의 `START_DATE`와 `END_DATE`를 원하는 백테스팅 기간으로 설정합니다.
        * 규칙 기반 백테스트 스크립트를 실행합니다:
            ```bash
            python run_backtest.py
            ```

## 예시 결과(추가할것)@@@@@@@@@



예시:
"ML 기반 전략을 사용하여 2002년-2016년 데이터로 학습 후, 2017년(3월~11월 유효기간)을 Out-of-Sample 기간으로 설정하여 미국 주요 기술주 및 ETF 7개 종목에 대해 테스트한 결과 (TP 70%, SL 10% 적용):"
* **총수익률 (%):** [예: 53.22%]
* **CAGR (%):** [예: 88.44%]
* **최대 낙폭 (%):** [예: -16.17%]
* **샤프 지수:** [예: 2.27]
* **총 거래 횟수:** [예: 6회]



<img width="1710" height="1107" alt="Image" src="https://github.com/user-attachments/assets/3e3fb57a-7a42-4fb4-851a-7aa213701466" />


<img width="1710" height="1107" alt="Image" src="https://github.com/user-attachments/assets/b0d1bac8-e962-4667-b3dc-f3b573400cbb" />



## 한계점 및 향후 개선 방향

* `backtester.py`의 슬리피지 모델링을 더욱 현실적으로 개선할 수 있습니다. (만약 아직 이전 조언대로 적용하지 않으셨다면)
* 현재 시스템은 모든 거래가 신호 발생일의 '종가'에 체결된다고 가정합니다. 실제 트레이딩을 위해서는 T+1 실행 모델(Day T 데이터로 신호 생성 후, Day T+1 시가에 거래)로 개선하는 것이 가치 있을 것입니다.
* 더 다양한 ML 모델 탐색, 고급 피처 엔지니어링, 정교한 하이퍼파라미터 최적화 기법 적용.
* 증권사 API를 통한 실시간 자동매매 기능 연동 모듈 개발.

---
