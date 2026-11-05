# Freesia: Crypto Market Microstructure & High-Frequency Quantitative Research Platform

Freesia는 가상자산(Binance USDT-M Futures)의 고빈도(HFT) 시장 미시구조(Market Microstructure) 분석, 오더북 불균형(Order Flow Imbalance, OFI) 계산 및 체계적 알고리즘 트레이딩 전략 연구를 위해 설계된 고성능 퀀트 리서치·실행 플랫폼입니다.

---

## 1. 프로젝트 개요 (Overview)

- **목적**: 틱(Tick) 및 L2 호가창(Order Book) 레벨의 고빈도 데이터를 실시간 수집·가공하고, 정보성 거래 흐름(Informed Trading)을 계량화하여 단기 알파(Alpha)를 발굴 및 실시간 페이퍼 트레이딩으로 검증
- **대상 자산**: Binance Futures BTCUSDT, ETHUSDT
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
| **System & Deploy** | Linux Systemd, POSIX Flock, Crontab | 단일 인스턴스 락 및 서비스 무중단 자동 복구 |

---

## 4. 핵심 모듈 구성 (Project Structure)

```
freesia/
├── freesia/                  # 코어 패키지
│   ├── book.py               # L2 오더북 스냅샷 및 델타 빌더
│   ├── client.py             # 바이낸스 비동기 API 클라이언트
│   └── ofi.py                # Order Flow Imbalance 실시간 계산 모듈
├── research/                 # 120+ 체계적 퀀트 전략 연구 및 백테스트
│   ├── r118_engine.py        # 다중 전략 롱·숏 복합 백테스팅 엔진
│   ├── r122_edge_factory.py  # 팩터 발굴 및 시그널 결합 파이프라인
│   ├── r126_vpin.py          # VPIN(거래 흐름 독성 지표) 연구
│   └── ...                   # 모멘텀, 변동성, 펀딩비 차익, 시장조성 연구
├── scripts/                  # 실시간 운영 스크립트
│   ├── live_book.py          # 실시간 L2 호가 수집 및 저장 데몬
│   ├── live_paper.py         # 실시간 시그널 생성 및 페이퍼 트레이딩 엔진
│   ├── update_rest.py        # 5분 주기 분봉 및 펀딩비 갱신
│   └── validate.py           # 데이터 무결성 및 누락 검증 자동화
└── requirements.txt
```

---

## 5. 주요 엔지니어링 과제 및 해결 방안 (Engineering Highlights)

1. **초저지연 L2 오더북 재구성 및 메모리 누수 방지**:
   - WebSocket 연결 끊김 및 네트워크 지터를 견디기 위해 지수 백오프(Exponential Backoff) 자동 재연결 로직 구현
   - 깊은 오더북 갱신 시 Python 딕셔너리 재할당 오버헤드를 줄이고 원형 큐 구조로 고속 슬라이싱
2. **백테스트 과적합(Overfitting) 차단**:
   - 금융 시계열의 시계열 상관성을 고려하여 Marcos López de Prado의 Purged K-Fold 및 CPCV 기법을 적용, 룩어헤드 편향(Look-ahead bias) 완전 제거
3. **무중단 장기 가동성 보장**:
   - POSIX `flock`을 활용하여 중복 프로세스 실행을 하드웨어 레벨에서 차단하고, 프로세스 비정상 종료 시 잠금이 즉시 해제되도록 구성
