# Daily-bar strategy backtests

조정 OHLCV에서 규칙 신호와 RF/LightGBM 분류 신호를 만들고 수량·현금·수수료를 추적합니다.

## 설치와 실행

Python 3.11 이상을 사용합니다. Yahoo Finance 다운로드에는 인터넷 연결이 필요합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_backtest.py
```

Windows PowerShell에서는 가상환경 활성화 명령을 `.\.venv\Scripts\Activate.ps1`로 바꿉니다.

## 계산 방식

- 지표는 현재와 과거 관측만 사용합니다. 초기 지표 구간은 제거하고 미래 값으로 채우지 않습니다.
- 상승 타깃은 다음 h개 관측의 최고가로 계산합니다. 아직 끝나지 않은 미래 라벨은 결측으로 유지합니다.
- 종목별로 신호를 계산합니다. 오늘 종가로 생성한 주문은 이후 해당 종목의 관측 시가에 실행합니다.
- 익절·손절은 종가에서 관찰하고 다음 관측 시가에 청산합니다. 일중 가격 경로와 보장된 손절 체결은 모델링하지 않습니다.
- 현금, 정수 주식 수량과 편도 수수료를 추적합니다. 휴장일은 마지막 관측 가격으로 평가하고, 데이터 구간 종료 시 각 종목의 마지막 관측 종가로 청산합니다.
- ML 학습/검증은 시간 순서로 나누고 라벨이 다음 구간을 침범하는 표본을 제외합니다. 모델의 feature schema와 마지막 validation label 날짜를 metadata에 저장합니다.
- ML 백테스트는 학습 및 보고된 검증 구간이 모두 끝난 이후 날짜만 사용합니다. 모델이 없거나 feature schema가 맞지 않으면 오류를 표시합니다.

## ML 실행

```bash
python train_evaluate_ml.py
python run_ml_backtest.py
```

학습 명령은 준비 데이터의 앞 80% 이내에서 학습/검증하고, 나머지 기간을 포트폴리오 평가에 남깁니다. `config.py`에서 기간, 종목, 비율을 설정합니다. 비율 `.03`은 3%입니다.

## 테스트

```bash
python -m pip install pytest
python -m pytest -q
```

회귀 테스트는 미래 최고가 라벨, 지표 인과성, 종목 간 신호 분리, 다음 시가 체결, 최종 수수료, 휴장일 평가와 ML 구간 분리를 검사합니다. 시세 다운로드는 mock으로 검사합니다.

## 범위

일별 연구 시뮬레이션입니다. 실시간 데이터, 주문 전송, 부분 체결, 계좌 동기화, 시장 충격, 차입, 환전은 포함하지 않습니다. 실거래 수익이나 미래 예측 우위는 입증하지 않았습니다. 현재 예제 유니버스는 선택된 미국 종목이며 생존 편향 없는 시장 전체 데이터가 아닙니다.
