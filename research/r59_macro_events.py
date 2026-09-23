"""R59 — 방향 G: 예정 이벤트(FOMC 성명·CPI 발표) 전후의 방향.

일정: FOMC 성명 14:00 ET (연준 달력, 2021~2026), CPI 08:30 ET (BLS 발표 목록, 2021-02~2026-09).
  ※ BLS 목록 요약에서 2024-01-15(미국 공휴일)은 오류로 판단해 실제 발표일 2024-01-11로 교정.
측정 (BTC·ETH, 1분봉):
  [1] 발표 전 24시간 → 발표 15분 전 수익 (선행 드리프트, 논문: FOMC 전날 +0.96%)
  [2] 발표 후 첫 15분 방향 → 이후 12시간 ±1%/±1.5% 먼저 도달이 같은 방향인 비율 (추종 vs 되돌림)
  [3] 발표 후 첫 15분 이동 크기 상위 절반만
표본이 작다(FOMC 약 46회, CPI 약 67회): 결과는 방향성 참고용.
사용: python research/r59_macro_events.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
from r47_x30_frame import outcome, CT, CM
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
FOMC = """2021-01-27 2021-03-17 2021-04-28 2021-06-16 2021-07-28 2021-09-22 2021-11-03 2021-12-15
2022-01-26 2022-03-16 2022-05-04 2022-06-15 2022-07-27 2022-09-21 2022-11-02 2022-12-14
2023-02-01 2023-03-22 2023-05-03 2023-06-14 2023-07-26 2023-09-20 2023-11-01 2023-12-13
2024-01-31 2024-03-20 2024-05-01 2024-06-12 2024-07-31 2024-09-18 2024-11-07 2024-12-18
2025-01-29 2025-03-19 2025-05-07 2025-06-18 2025-07-30 2025-09-17 2025-10-29 2025-12-10
2026-01-28 2026-03-18 2026-04-29 2026-06-17 2026-07-29 2026-09-16""".split()
CPI = """2021-02-10 2021-03-10 2021-04-13 2021-05-12 2021-06-10 2021-07-13 2021-08-11 2021-09-14 2021-10-13 2021-11-10 2021-12-10
2022-01-12 2022-02-10 2022-03-10 2022-04-12 2022-05-11 2022-06-10 2022-07-13 2022-08-10 2022-09-13 2022-10-13 2022-11-10 2022-12-13
2023-01-12 2023-02-14 2023-03-14 2023-04-12 2023-05-10 2023-06-13 2023-07-12 2023-08-10 2023-09-13 2023-10-12 2023-11-14 2023-12-12
2024-01-11 2024-02-13 2024-03-12 2024-04-10 2024-05-15 2024-06-12 2024-07-11 2024-08-14 2024-09-11 2024-10-10 2024-11-13 2024-12-11
2025-01-15 2025-02-12 2025-03-12 2025-04-10 2025-05-13 2025-06-11 2025-07-15 2025-08-12 2025-09-11 2025-10-24 2025-12-18
2026-01-13 2026-02-13 2026-03-11 2026-04-10 2026-05-12 2026-06-10 2026-07-14 2026-08-12 2026-09-11""".split()


def times(days, hhmm):
    return [pd.Timestamp(f"{d} {hhmm}", tz="America/New_York").tz_convert("UTC") for d in days]


EV = {"FOMC": times(FOMC, "14:00"), "CPI": times(CPI, "08:30")}
for s in ("BTCUSDT", "ETHUSDT"):
    k = store.load("klines_1m", s).set_index("ts")
    o, h, l, c = (k[x].values for x in ("open", "high", "low", "close"))
    P(f"\n===== {s}")
    for name, ts in EV.items():
        pos = k.index.get_indexer(ts)
        pos = pos[(pos > 1440) & (pos < len(c) - 800)]
        pre = np.log(c[pos - 15] / c[pos - 1440])
        first = np.log(c[pos + 14] / o[pos])
        side = np.sign(first).astype(np.int64)
        P(f"  {name} n={len(pos)} | 발표 전 24시간 평균 {pre.mean()*100:+.2f}% (양수 비율 {(pre>0).mean():.2f}) | 첫 15분 |이동| 중앙 {np.median(np.abs(first))*100:.2f}%")
        for d in (0.01, 0.015):
            res = {}
            for sd in (1, -1):
                m = side == sd
                if m.sum() == 0: continue
                _, kind, _ = outcome(o, h, l, c, h, l, (pos[m] + 15).astype(np.int64), sd, d, 0.99, 720, CT, CM)
                res[sd] = kind
            kk = np.concatenate([res.get(1, np.array([], np.int8)), res.get(-1, np.array([], np.int8))])
            dec = kk[kk != 0]
            big = np.abs(first) >= np.median(np.abs(first))
            kb = []
            for sd in (1, -1):
                m = (side == sd) & big
                if m.sum() == 0: continue
                _, kind, _ = outcome(o, h, l, c, h, l, (pos[m] + 15).astype(np.int64), sd, d, 0.99, 720, CT, CM)
                kb.append(kind)
            kb = np.concatenate(kb); db = kb[kb != 0]
            P(f"    ±{d*100:.1f}%: 첫 15분 방향 추종 적중 {(dec==1).mean():.2f} (결판 {len(dec)}/{len(kk)}) | 큰 첫 움직임만 {(db==1).mean():.2f} ({len(db)})")
