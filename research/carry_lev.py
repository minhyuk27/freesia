"""캐리(현물 롱 + 무기한 숏)에 레버리지를 얼마나 걸 수 있나 — 숏 다리 청산이 제약.
숏 perp 청산가 = 진입가 × (1 + 1/L - mmr). 현물 담보를 함께 쓰는 경우와 분리 증거금 경우를 모두 본다."""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from freesia import store  # noqa: E402

MMR = 0.004
TE = pd.Timestamp("2024-01-01", tz="UTC")

for sym in ("BTCUSDT", "ETHUSDT"):
    k = store.load("klines_1m", sym).set_index("ts")
    h = k.high.resample("1h").max(); c = k.close.resample("1h").last()
    print(f"\n=== {sym} 숏 다리 최대 상승 폭(진입 후 N일 내 고가/진입가 - 1) ===", flush=True)
    rows = []
    for d in (1, 3, 7, 30):
        w = 24 * d
        mx = h[::-1].rolling(w).max()[::-1] / c - 1
        mx = mx.dropna()
        rows.append((d, float(mx.quantile(0.5)) * 100, float(mx.quantile(0.95)) * 100, float(mx.quantile(0.99)) * 100, float(mx.max()) * 100))
    print(pd.DataFrame(rows, columns=["보유일", "중앙%", "95%", "99%", "최대%"]).round(2).to_string(index=False), flush=True)
    # 재증거금 없이 L배일 때 청산 확률 (숏 다리 단독 증거금)
    print("숏 다리 단독 증거금 L배 — 청산(=상승 1/L-0.4% 도달) 확률", flush=True)
    rows = []
    for L in (2, 3, 5, 10, 20):
        dist = 1 / L - MMR
        for d in (7, 30):
            w = 24 * d
            mx = (h[::-1].rolling(w).max()[::-1] / c - 1).dropna()
            rows.append((L, dist * 100, d, float((mx >= dist).mean()) * 100))
    print(pd.DataFrame(rows, columns=["L", "청산거리%", "보유일", "청산확률%"]).round(2).to_string(index=False), flush=True)

# 캐리 수익률과 레버리지 조합
print("\n=== 캐리 수익률 × 레버리지 (현물+숏 합산 자본 1, 숏 다리에 배분한 증거금 비율 m) ===", flush=True)
for sym in ("BTCUSDT", "ETHUSDT"):
    f = store.load("funding", sym).set_index("ts").rate
    pr = store.load("premium_1m", sym).set_index("ts").close
    prem = pr.reindex(f.index, method="nearest", tolerance=pd.Timedelta("5min"))
    carry = (f - prem.diff()).dropna()
    ppy = 3 * 365
    tr = carry.index < TE; te = ~tr
    print(f"\n  {sym} 명목 1 기준 캐리: 학습 연 {carry[tr].mean()*ppy*100:.2f}%, 테스트 연 {carry[te].mean()*ppy*100:.2f}%", flush=True)
    rows = []
    for m in (0.5, 0.35, 0.25, 0.15):        # 숏 다리 증거금 비율 → 명목 배수 = 1/(현물1 + 증거금m) 형태
        # 자본 1을 현물 x, 숏 증거금 m·x 로 나누면 x(1+m)=1 → x = 1/(1+m), 명목 = x, 숏 레버리지 = 1/m
        x = 1 / (1 + m)
        L = 1 / m
        rows.append((m, L, x, carry[tr].mean() * ppy * x * 100, carry[te].mean() * ppy * x * 100))
    print(pd.DataFrame(rows, columns=["숏증거금비율", "숏레버리지", "명목", "학습연%", "테스트연%"]).round(2).to_string(index=False), flush=True)
