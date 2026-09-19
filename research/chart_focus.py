"""1차 탐색에서 살아남은 '수축확장(스퀴즈)' 계열 정밀 검증: 분해(ablation), 기하 고원, 연도별, 워크포워드, 비용, 위약.
사용: python research/chart_focus.py"""
import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from freesia.bt import core  # noqa: E402
from chart_combo import techniques, barrier, run_combo, stat, COST  # noqa: E402

TE = pd.Timestamp("2024-01-01", tz="UTC")
rng = np.random.default_rng(23)
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)


def ev(M, names, combo, rl, rs, hold, split, cost=COST):
    idx = np.array([names.index(x) for x in combo] + [0] * (4 - len(combo)), np.int64)
    a, b, c, d, e, f = run_combo(M, idx, len(combo), rl, rs, hold, cost, split)
    m1, t1, n1 = stat(a, b, c); m2, t2, n2 = stat(d, e, f)
    return m1 * 1e4, t1, n1, m2 * 1e4, t2, n2


DATA = {}
for tf in ("1h", "2h", "4h"):
    for sym in ("BTCUSDT", "ETHUSDT"):
        b = core.bars(sym, tf) if tf != "2h" else core.bars(sym, "1h").resample("2h").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "quote_volume": "sum"}).dropna()
        names, M = techniques(b)
        DATA[(tf, sym)] = (b, names, M, int((b.index < TE).sum()))

BASE = ["수축확장"]
ADD = ["캔들패턴", "하이킨아시", "수퍼트렌드", "돌파", "볼륨프로파일", "앵커드VWAP", "피봇", "FVG", "이치모쿠", "구조", "피보나치", "다이버전스", "회귀채널", "이중바닥천정"]

P("=" * 84 + "\n[1] 분해 — 수축확장 단독부터 하나씩 추가 (4h, 목표·손절 1.5%, 12봉)")
for sym in ("BTCUSDT", "ETHUSDT"):
    b, names, M, split = DATA[("4h", sym)]
    rl, rs, hold = barrier(b.high.values, b.low.values, b.close.values, 0.015, 0.015, 12)
    rows = []
    for extra in [[]] + [[a] for a in ADD]:
        c = BASE + extra
        r = ev(M, names, c, rl, rs, hold, split)
        rows.append((" + ".join(c), *r))
    d = pd.DataFrame(rows, columns=["조합", "학습bp", "학습t", "학습n", "테스트bp", "테스트t", "테스트n"])
    P(f"\n  {sym}")
    P(d.sort_values("테스트t", ascending=False).round(2).to_string(index=False))

P("\n" + "=" * 84 + "\n[2] 기하 고원 — 수축확장+캔들패턴+하이킨아시+수퍼트렌드")
C = ["수축확장", "캔들패턴", "하이킨아시", "수퍼트렌드"]
rows = []
for tf in ("1h", "2h", "4h"):
    for tp in (0.010, 0.015, 0.020, 0.030):
        for H in (8, 12, 24, 48):
            for sym in ("BTCUSDT", "ETHUSDT"):
                b, names, M, split = DATA[(tf, sym)]
                rl, rs, hold = barrier(b.high.values, b.low.values, b.close.values, tp, tp, H)
                r = ev(M, names, C, rl, rs, hold, split)
                rows.append((tf, tp * 100, H, sym[:3], *r))
d = pd.DataFrame(rows, columns=["봉", "목표%", "H", "종목", "학습bp", "학습t", "학습n", "테스트bp", "테스트t", "테스트n"])
d = d[(d.학습n >= 25) & (d.테스트n >= 15)]
pv = d.pivot_table(index=["봉", "목표%", "H"], columns="종목", values=["테스트bp", "테스트t", "테스트n"])
P(pv.round(2).to_string())
P(f"\n  유효 설정 {len(d)//2}개 중 두 종목 모두 테스트 양수: {int((d.groupby(['봉','목표%','H']).테스트bp.min() > 0).sum())}개")

P("\n" + "=" * 84 + "\n[3] 연도별 (4h, 1.5%, 12봉)")
for sym in ("BTCUSDT", "ETHUSDT"):
    b, names, M, split = DATA[("4h", sym)]
    rl, rs, hold = barrier(b.high.values, b.low.values, b.close.values, 0.015, 0.015, 12)
    idx = np.array([names.index(x) for x in C], np.int64)
    rows = []
    for y in range(2020, 2027):
        s = int((b.index < pd.Timestamp(f"{y}-01-01", tz="UTC")).sum())
        e = int((b.index < pd.Timestamp(f"{y+1}-01-01", tz="UTC")).sum())
        a1, b1, c1, d1, e1, f1 = run_combo(M[:, s:e], idx, 4, rl[s:e], rs[s:e], hold[s:e], COST, e - s)
        m, t, n = stat(a1, b1, c1)
        rows.append((y, m * 1e4, t, n, n / 52))
    P(f"\n  {sym}")
    P(pd.DataFrame(rows, columns=["연", "평균bp", "t", "거래수", "주당거래"]).round(2).to_string(index=False))

P("\n" + "=" * 84 + "\n[4] 앵커드 워크포워드 — 매년 직전까지 학습해 최적 조합 선택 → 그 해 평가 (4h, 1.5%, 12봉)")
cands = [BASE + list(x) for r in (1, 2, 3) for x in itertools.combinations(ADD, r)]
P(f"  후보 조합 {len(cands)}개")
for sym in ("BTCUSDT", "ETHUSDT"):
    b, names, M, split = DATA[("4h", sym)]
    rl, rs, hold = barrier(b.high.values, b.low.values, b.close.values, 0.015, 0.015, 12)
    rows = []
    for y in range(2022, 2027):
        s = int((b.index < pd.Timestamp(f"{y}-01-01", tz="UTC")).sum())
        e = int((b.index < pd.Timestamp(f"{y+1}-01-01", tz="UTC")).sum())
        best, bt = None, -9
        for c in cands:
            idx = np.array([names.index(x) for x in c] + [0] * (4 - len(c)), np.int64)
            a1, b1, c1, _, _, _ = run_combo(M[:, :s], idx, len(c), rl[:s], rs[:s], hold[:s], COST, s)
            m, t, n = stat(a1, b1, c1)
            if n >= 30 and t > bt:
                bt, best = t, c
        if best is None:
            continue
        idx = np.array([names.index(x) for x in best] + [0] * (4 - len(best)), np.int64)
        a1, b1, c1, _, _, _ = run_combo(M[:, s:e], idx, len(best), rl[s:e], rs[s:e], hold[s:e], COST, e - s)
        m, t, n = stat(a1, b1, c1)
        rows.append((y, " + ".join(best), bt, m * 1e4, t, n))
    P(f"\n  {sym}")
    P(pd.DataFrame(rows, columns=["연", "선택 조합", "학습t", "그해bp", "그해t", "거래"]).round(2).to_string(index=False))

P("\n" + "=" * 84 + "\n[5] 비용 민감도 & 위약 (4h, 1.5%, 12봉, 조합=수축확장+캔들패턴+하이킨아시+수퍼트렌드)")
for sym in ("BTCUSDT", "ETHUSDT"):
    b, names, M, split = DATA[("4h", sym)]
    rl, rs, hold = barrier(b.high.values, b.low.values, b.close.values, 0.015, 0.015, 12)
    row = []
    for cst in (0.0006, 0.0012, 0.0020, 0.0030):
        r = ev(M, names, C, rl, rs, hold, split, cst)
        row.append((cst * 1e4, r[0], r[3], r[4]))
    P(f"\n  {sym} 비용: " + " | ".join(f"{a:.0f}bp→학습{b_:.0f}/테스트{c:.0f}bp(t{d_:.2f})" for a, b_, c, d_ in row))
    idx = np.array([names.index(x) for x in C], np.int64)
    pl = []
    for _ in range(300):
        Mp = M.copy()
        for x in C:
            j = names.index(x)
            Mp[j] = np.roll(M[j], rng.integers(100, M.shape[1] - 100))
        a1, b1, c1, d1, e1, f1 = run_combo(Mp, idx, 4, rl, rs, hold, COST, split)
        m2, t2, n2 = stat(d1, e1, f1)
        if n2 >= 10:
            pl.append(m2 * 1e4)
    r = ev(M, names, C, rl, rs, hold, split)
    P(f"  위약 300회 테스트 평균bp: 평균 {np.mean(pl):.1f}, 95% {np.percentile(pl,95):.1f}, 99% {np.percentile(pl,99):.1f}, 최대 {np.max(pl):.1f} | 실제 {r[3]:.1f}")
    P(f"  → 위약 대비 백분위 {(np.array(pl) < r[3]).mean()*100:.1f}%")
