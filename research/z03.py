"""z0.3 — 깊은 딥 + 긴 봉. 1h 다리 제거, 12h·1d 추가, 진입 깊이 확대.
z0.2 대비 변경: (a) 다리 = 4h·8h·12h·1d (1h 제거), (b) 깊이 k 확대, (c) 목표가 청산 kg.
공유자본 + 변동성타겟 + 교차마진 청산판정. 체결은 관통 요구(eps).
사용: python research/z03.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from lib_dip import bars
import tp_robust as TR
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
EPS = 0.0002
TE = pd.Timestamp("2024-01-01", tz="UTC")
REC = pd.Timestamp("2025-01-01", tz="UTC")

for s in ("BTCUSDT", "ETHUSDT"):
    for b in (240, 480, 720, 1440):
        if (s, b) in TR.D: continue
        d = bars(s, b)
        d["rv"] = (pd.Series(np.log(d["bc"])).diff().rolling(20).std()).shift(1).values
        TR.D[(s, b)] = d

SETS = {
    "4h+8h": [(s, b) for s in ("BTCUSDT", "ETHUSDT") for b in (240, 480)],
    "4h+8h+12h": [(s, b) for s in ("BTCUSDT", "ETHUSDT") for b in (240, 480, 720)],
    "4h+8h+12h+1d": [(s, b) for s in ("BTCUSDT", "ETHUSDT") for b in (240, 480, 720, 1440)],
    "8h+12h+1d": [(s, b) for s in ("BTCUSDT", "ETHUSDT") for b in (480, 720, 1440)],
    "z0.2(1h+4h+8h)": [(s, b) for s in ("BTCUSDT", "ETHUSDT") for b in (60, 240, 480)],
}

P("[1] 다리 구성 × 깊이 × 목표가 — 거래 단위 (전기간 / 2025년 이후)")
rows = []
for nm, lg in SETS.items():
    for k in (5.0, 6.5, 8.0, 10.0):
        for kg in (0.0, 2.0, 5.0):
            E, _ = TR.build(lg, (k, k*1.25, k*1.5), kg, eps=EPS)
            if len(E) < 60: continue
            d = pd.DataFrame([(e[4], e[2]) for e in E], columns=["ts", "r"]).set_index("ts")
            rec = d[d.index >= REC].r; old = d[d.index < REC].r
            rows.append((nm, k, kg, len(E), old.mean()*1e4, len(rec), rec.mean()*1e4, (rec > 0).mean()))
t = pd.DataFrame(rows, columns=["다리", "깊이k", "목표kg", "n전체", "평균bp(~24)", "n(25~)", "평균bp(25~)", "승률(25~)"]).round(2)
P(t.sort_values("평균bp(25~)", ascending=False).head(22).to_string(index=False))

P("\n[2] 유망 조합의 연도별 평균 bp")
CAND = [("4h+8h+12h+1d", 8.0, 5.0), ("4h+8h+12h+1d", 6.5, 5.0), ("4h+8h+12h", 8.0, 5.0),
        ("8h+12h+1d", 6.5, 5.0), ("4h+8h", 8.0, 5.0), ("4h+8h+12h+1d", 8.0, 2.0)]
for nm, k, kg in CAND:
    E, _ = TR.build(SETS[nm], (k, k*1.25, k*1.5), kg, eps=EPS)
    d = pd.DataFrame([(e[4], e[2]) for e in E], columns=["ts", "r"]).set_index("ts")
    g = d.groupby(d.index.year).r
    P(f"  {nm:16s} k={k:4.1f} kg={kg:.0f} n={len(E):5d}  " +
      " ".join(f"{y}:{v*1e4:+6.1f}" for y, v in g.mean().items()))

P("\n[3] 자본곡선 — 공유자본·변동성타겟 (청산 0인 것만)")
best = None
for nm, k, kg in CAND:
    E, _ = TR.build(SETS[nm], (k, k*1.25, k*1.5), kg, eps=EPS)
    P(f"  -- {nm} k={k} kg={kg:.0f} (n={len(E)}) --")
    for w in (3.0, 5.0, 8.0, 12.0, 20.0):
        for Nm in (5.0, 8.0, 12.0, 20.0):
            if Nm < w: continue
            eq, liq = TR.sim(E, w, Nm)
            if liq or eq.iloc[-1] <= 1e-6: continue
            c = TR.cagr(eq); mdd = float((eq/eq.cummax()-1).min())*100
            te = eq[eq.index >= TE]; tc = TR.cagr(te) if len(te) > 30 else np.nan
            rc = eq[eq.index >= REC]; rcc = TR.cagr(rc) if len(rc) > 30 else np.nan
            P(f"     w={w:4.0f} Nmax={Nm:4.0f}  전체 연 {c:+8.0f}% MDD {mdd:6.1f}% | 24~ {tc:+8.0f}% | 25~ {rcc:+8.0f}%")
            score = rcc if np.isfinite(rcc) else -999
            if best is None or score > best[0]: best = (score, c, mdd, tc, nm, k, kg, w, Nm, eq)

if best:
    _, c, mdd, tc, nm, k, kg, w, Nm, eq = best
    P(f"\n★ z0.3 최적(최근 성과 기준): {nm} 깊이 k={k} 목표 kg={kg:.0f} w={w:.0f} Nmax={Nm:.0f}")
    P(f"   전체 연 {c:+.0f}% MDD {mdd:.1f}% | 2024~ 연 {tc:+.0f}% | 2025~ 연 {best[0]:+.0f}%")
    yr = eq.resample("YE").last()/eq.resample("YE").first()-1
    mon = eq.resample("ME").last()/eq.resample("ME").first()-1
    P("   연도별: " + " | ".join(f"{i.year} {v*100:+.0f}%" for i, v in yr.items()))
    P("   최근 12개월: " + " | ".join(f"{i.strftime('%y-%m')} {v*100:+.0f}%" for i, v in mon.tail(12).items()))
    l12 = mon.tail(12)
    P(f"   최근 12개월 누적 {(np.prod(1+l12)-1)*100:+.1f}% 양수 {int((l12>0).sum())}/12 | 전체 월승률 {float((mon>0).mean()):.3f} (n={len(mon)})")
