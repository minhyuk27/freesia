# Freesia: Crypto Market Microstructure & High-Frequency Quantitative Research Platform

Freesia는 가상자산(Binance USDT-M Futures)의 고빈도(HFT) 시장 미시구조(Market Microstructure) 분석, 오더북 불균형(Order Flow Imbalance, OFI) 계산 및 체계적 알고리즘 트레이딩 전략 연구를 위해 설계된 고성능 퀀트 리서치·실행 플랫폼입니다.

---

## 1. 프로젝트 개요 (Overview)

- **목적**: 틱(Tick) 및 L2 호가창(Order Book) 레벨의 고빈도 데이터를 실시간 수집·가공하고, 정보성 거래 흐름(Informed Trading)을 계량화하여 단기 알파(Alpha) 발굴 및 실시간 페이퍼 트레이딩으로 검증
- **대상 자산**: Binance Futures BTCUSDT, ETHUSDT
- **개발 형태**: 1인 단독 설계 및 풀스택 리서치/시스템 개발 (기여도 100%)
- **핵심 특징**:
  - WebSocket 기반 비동기 L2 호가창 스트리밍 및 실시간 OFI 델타 산출
  - 120개 이상의 미시구조 팩터 및 단기 예측 전략 체계적 백테스팅
  - Purged / Combinatorial Purged Cross-Validation (CPCV)을 통한 백테스트 과적합(Overfitting) 방지
  - Linux `flock` 기반의 경량 프로세스 동시성 제어 및 내결함성(Fault Tolerance) 설계

---

## 2. 시스템 아키텍처 (Architecture)

```
[Binance Futures WebSocket] ──(L2 Orderbook)──┐
                                              ▼
[Binance REST API] ─────────(Klines/Funding)─►[Data Ingestion Engine]
                                                    │
                                                    ▼
[Real-time OFI Engine] ◄─────────────── [Tick / Book Processor]
        │                                           │
        ▼                                           ▼
[Live Paper Trader] ◄── [Signal Pipeline] ──► [Local Storage / Parquet]
        │                                           │
        ▼                                           ▼
[Execution & Risk Engine]                    [Research & Backtester]
                                              - VPIN / Toxic Flow
                                              - Basis Term Structure
                                              - Avellaneda-Stoikov MM
```

---

## 3. 핵심 기술 스택 (Tech Stack)

| 영역 | 기술 스택 | 설명 |
|---|---|---|
| **Language & Runtime** | Python 3.11, Asyncio | 고빈도 데이터 비동기 파이프라인 구현 |
| **Data Processing** | Polars, Pandas, NumPy, SciPy | 대용량 L2 틱 데이터 및 분봉 고속 연산 |
| **API & Streaming** | Binance Futures API, WebSockets | 저지연 실시간 시세 및 체결 피드 수집 |
| **Quant Analysis** | Statsmodels, Scikit-learn | 시계열 통계 검정, Regime Switching, 머신러닝 메타라벨링 |
| **Testing & Reliability**| Pytest | 비용/마진 모델 및 파이프라인 회귀 테스트 |
| **System & Deploy** | Linux Systemd, POSIX Flock, Crontab | 단일 인스턴스 락 및 서비스 무중단 자동 복구 |

---

## 4. 핵심 모듈 구성 (Project Structure)

```
freesia/
├── freesia/                  # 코어 패키지
│   ├── costs.py              # Binance 선물 티어별 수수료, 마진 및 청산 모델
│   ├── rest.py               # 바이낸스 비동기 REST API 클라이언트
│   ├── notify.py             # 텔레그램 실시간 포지션 및 장애 알림
│   └── store.py              # Parquet 및 HDF5 고속 스토리지 관리자
├── research/                 # 퀀트 전략 및 미시구조 팩터 연구 (120+ 연구 스크립트)
│   ├── r01~r135_*.py         # VPIN 독성 검정, Basis 기간구조, 마켓메이킹 백테스트
├── scripts/                  # 과거 데이터 백필 및 시계열 수집 데몬
│   ├── get_bookdepth.py      # 실시간 호가 깊이 수집기
│   └── get_agg1s.py          # 1초 집계 체결 틱 수집기
└── tests/                    # 비용/청산가 모델 및 파이프라인 단위 테스트 (Pytest)
```

---

## 5. 주요 엔지니어링 과제 및 해결 방안 (Engineering Highlights)

1. **대용량 L2 호가창 원형 큐 기반 초저지연 연산**:
   - 실시간 바이낸스 L2 Depth 스트림에서 체결 틱과 호가 잔량 변화를 분리하여 Order Flow Imbalance(OFI)를 밀리초 단위로 갱신하는 고속 파이프라인을 구축했습니다.
2. **백테스트 과적합(Overfitting) 방지**:
   - 금융 시계열의 자기상관성으로 인한 누수(Leakage)를 방지하기 위해 Purged / Combinatorial Purged Cross-Validation(CPCV)을 적용하여 표본외(OOS) 일반화 성능을 정밀 검증했습니다.
3. **비용 및 청산 모델의 엄격한 반영**:
   - VIP 티어별 메이커/테이커 수수료 및 격리마진 유지증거금률(MMR)을 반영한 엄격한 실전 비용 모델(`freesia/costs.py`)을 설계하여 가상 백테스트의 수익 왜곡을 차단했습니다.

---

## 6. 빠른 시작 (Quickstart)

### 설치 및 환경 설정
```bash
git clone https://github.com/minhyuk27/freesia.git
cd freesia

# 가상환경 생성 및 의존성 설치
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt pytest

# 환경변수 템플릿 복사
cp .env.example .env
```

### 단위 테스트 실행
```bash
PYTHONPATH=. pytest tests/
```

---

## 7. 라이선스 (License)

본 프로젝트는 [MIT License](LICENSE)를 따릅니다.
