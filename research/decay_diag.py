"""최근 엣지 붕괴 진단: 다리별·깊이별·목표가별로 어디가 죽었나. 최근 국면 재최적화 가능성.
사용: python research/decay_diag.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from tp_robust import build, sim, cagr, D, ALL
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
EPS = 0.0002
LEGS = [(s, b) for s in ("BTCUSDT", "ETHUSDT") for b in (60, 240, 480)]
Y = lambda ts: ts.year


def tbl(E, by):
    df = pd.DataFrame([(e[4], e[2], e[6], e[5]) for e in E], columns=["ts", "r", "rv", "leg"]).set_index("ts")
    df["n"] = df.r/df.rv
    g = df.groupby([df.index.year, by] if by == "leg" else df.index.year)
    return df, g


P("[1] 다리별 연도별 정규화 수익(수익/rv 평균) — 어느 다리가 죽었나")
E, _ = build(LEGS, (5.0, 6.5, 8.0), 2.0, eps=EPS)
df = pd.DataFrame([(e[4], e[2], e[6], f"{e[5][0][:3]}{e[5][1]}") for e in E],
                  columns=["ts", "r", "rv", "leg"]).set_index("ts")
df["n"] = df.r/df.rv
pv = df.pivot_table(index=df.index.year, columns="leg", values="n", aggfunc="mean").round(2)
P(pv.to_string())
P("\n  진입 건수:")
P(df.pivot_table(index=df.index.year, columns="leg", values="r", aggfunc="size").to_string())

P("\n[2] 진입 깊이별 연도별 평균 수익(bp) — 얕은 딥이 죽고 깊은 딥은 살았나")
for k in (2.0, 3.0, 4.0, 5.0, 6.5, 8.0, 10.0, 13.0):
    E2, _ = build(LEGS, (k, k*1.3, k*1.6), 2.0, eps=EPS)
    d2 = pd.DataFrame([(e[4], e[2]) for e in E2], columns=["ts", "r"]).set_index("ts")
    g = d2.groupby(d2.index.year).r
    row = " ".join(f"{y}:{v*1e4:+6.1f}" for y, v in g.mean().items())
    P(f"  k={k:5.1f} n={len(E2):5d}  {row}")

P("\n[3] 목표가 배수 kg별 연도별 평균 수익(bp)")
for kg in (0, 1, 2, 3, 5, 8):
    E2, _ = build(LEGS, (5.0, 6.5, 8.0), float(kg), eps=EPS)
    d2 = pd.DataFrame([(e[4], e[2]) for e in E2], columns=["ts", "r"]).set_index("ts")
    g = d2.groupby(d2.index.year).r
    P(f"  kg={kg:2d}  " + " ".join(f"{y}:{v*1e4:+6.1f}" for y, v in g.mean().items()))

P("\n[4] 최근 2년(2025~)만으로 재최적화 — 살아있는 조합이 있나")
best = []
for k in (3.0, 5.0, 8.0, 12.0):
    for kg in (1.0, 2.0, 3.0, 5.0):
        E2, _ = build(LEGS, (k, k*1.3, k*1.6), kg, eps=EPS)
        d2 = pd.DataFrame([(e[4], e[2]) for e in E2], columns=["ts", "r"]).set_index("ts")
        rec = d2[d2.index >= pd.Timestamp("2025-01-01", tz="UTC")].r
        old = d2[d2.index < pd.Timestamp("2025-01-01", tz="UTC")].r
        best.append((k, kg, len(old), old.mean()*1e4, len(rec), rec.mean()*1e4, (rec > 0).mean()))
bd = pd.DataFrame(best, columns=["깊이k", "목표kg", "n(~24)", "평균bp(~24)", "n(25~)", "평균bp(25~)", "승률(25~)"]).round(2)
P(bd.sort_values("평균bp(25~)", ascending=False).to_string(index=False))

P("\n[5] 숏 방향도 같은 구조로 죽었나 — 시가 위 지정가 매도 (대칭 검증)")
import numba
@numba.njit(cache=True)
def evs(o, h, l, c, B, x1, x2, x3, kg, rv, fm, ft, eps):
    n = len(c)//B
    rr = np.zeros(n); tt = np.zeros(n, np.int64); k = 0
    for b in range(n):
        if np.isnan(x1[b]) or x1[b] <= 0: continue
        s = b*B; e = s+B; O = o[s]; C = c[e-1]
        p1 = O*(1+x1[b]); p2 = O*(1+x2[b]); p3 = O*(1+x3[b])
        g = rv[b]*kg if kg > 0 else 0.0
        cnt = 0.0; ws = 0.0; f1 = -1; hit = -1; px = 0.0
        for t in range(s, e):
            if cnt < 1.0 and h[t] >= p1*(1+eps):
                cnt += 1.0; ws += p1
                if f1 < 0: f1 = t
            if cnt < 2.0 and h[t] >= p2*(1+eps) and p2 > p1: cnt += 1.0; ws += p2
            if cnt < 3.0 and h[t] >= p3*(1+eps) and p3 > p2: cnt += 1.0; ws += p3
            if cnt == 0.0: continue
            if g > 0.0 and t > f1:
                T = (ws/cnt)*(1.0-g)
                if l[t] <= T*(1-eps): hit = t; px = T; break
        if f1 < 0: continue
        fee = (fm+fm) if hit >= 0 else (fm+ft)
        if hit < 0: px = C
        avg = ws/cnt
        rr[k] = ((1.0 - px/avg) - fee/1e4)*(cnt/3.0); tt[k] = f1; k += 1
    return rr[:k], tt[:k]

rows = []
for key in LEGS:
    d = D[key]; rv = d["rv"]; B = d["B"]
    r_, t_ = evs(d["o"], d["h"], d["l"], d["c"], B, np.asarray(rv*5.0, float), np.asarray(rv*6.5, float),
                 np.asarray(rv*8.0, float), 2.0, np.asarray(rv, float), 2.0, 5.0, EPS)
    ts = d["idx"]
    for r, a in zip(r_, t_): rows.append((ts[int(a)], r))
ds = pd.DataFrame(rows, columns=["ts", "r"]).set_index("ts").sort_index()
g = ds.groupby(ds.index.year).r
P("  숏 " + " ".join(f"{y}:{v*1e4:+6.1f}bp" for y, v in g.mean().items()) + f"  (총 n={len(ds)})")
P("  롱 " + " ".join(f"{y}:{v*1e4:+6.1f}bp" for y, v in df.groupby(df.index.year).r.mean().items()))
