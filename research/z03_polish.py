"""z0.3 마감 검증: 종목 가중 · 더 긴 봉(2d·3d) · 절대하한 정밀 · 상위거래 제외 강건성 · 3단 성과표."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
import exit_tier as X
from absfloor import build_f, met
from freesia import store
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
DEP = {10.0: 1.0, 14.0: 3.0, 20.0: 6.0}
BL6 = (240, 360, 480, 720, 960, 1440)


def tiers(E, lab):
    out = []
    for w, Nm, tag in ((1.0, 1.0, "무레버"), (3.0, 5.0, "중간"), (5.0, 5.0, "중간+"), (5.0, 8.0, "공격")):
        eq, fl, liq = X.sim(E, w, Nm); m = met(eq, fl, liq)
        out.append((tag, m, liq))
    P(f"  {lab}")
    for tag, m, liq in out:
        if m is None: P(f"     {tag:6s} 파산"); continue
        P(f"     {tag:6s} 연 {m[0]:+7.0f}% MDD {m[1]:6.1f}% 최악 {m[2]:6.1f}% | 24~ {m[3]:+7.0f}% | 25~ {m[4]:+7.0f}%")
    return out


P("[1] 종목별 분해 (10·14·20, 5x/5x)")
for only in ("BTCUSDT", "ETHUSDT", None):
    E = build_f(DEP, {}, bars=BL6)
    if only: E = [e for e in E if e[5][0] == only]
    r = np.array([e[2] for e in E])
    eq, fl, liq = X.sim(E, 5.0, 5.0); m = met(eq, fl, liq)
    P(f"  {only or '둘 다':8s} n={len(r):4d} 평균 {r.mean()*1e4:+6.0f}bp 승률 {(r>0).mean():.3f} | "
      f"연 {m[0]:+6.0f}% 최악 {m[2]:6.1f}% 24~ {m[3]:+6.0f}% 25~ {m[4]:+6.0f}%")

P("\n[2] 종목 가중 (BTC:ETH)")
for bw, ew in ((1, 1), (1, 1.5), (1, 2), (1.5, 1), (2, 1)):
    E = build_f(DEP, {}, bars=BL6)
    E = [(a, b, r, m, ts, k, rv, wg*(bw if k[0] == "BTCUSDT" else ew)) for (a, b, r, m, ts, k, rv, wg) in E]
    eq, fl, liq = X.sim(E, 5.0, 5.0); m = met(eq, fl, liq)
    P(f"  BTC{bw}:ETH{ew}  연 {m[0]:+6.0f}% 최악 {m[2]:6.1f}% | 24~ {m[3]:+6.0f}% | 25~ {m[4]:+6.0f}% 청산 {liq}")

P("\n[3] 더 긴 봉 추가 (2d·3d)")
for s in ("BTCUSDT", "ETHUSDT"):
    for B in (2880, 4320):
        if (s, B) not in X.CACHE: X.CACHE[(s, B)] = X.prep(s, B)
for bars, lab in ((BL6, "6종 4h~1d"), (BL6+(2880,), "7종 +2d"), (BL6+(2880, 4320), "8종 +2d·3d")):
    E = build_f(DEP, {}, bars=bars)
    r = np.array([e[2] for e in E])
    eq, fl, liq = X.sim(E, 5.0, 5.0); m = met(eq, fl, liq)
    P(f"  {lab:12s} n={len(r):4d} 평균 {r.mean()*1e4:+6.0f}bp | 연 {m[0]:+6.0f}% 최악 {m[2]:6.1f}% 24~ {m[3]:+6.0f}% 25~ {m[4]:+6.0f}% 청산 {liq}")

P("\n[4] 절대 하한 정밀 (깊은층, 6종 봉)")
for f1 in (0.0, 0.02, 0.04, 0.05, 0.06, 0.08, 0.10):
    E = build_f(DEP, {10.0: f1, 14.0: f1*1.4, 20.0: f1*2.0}, bars=BL6)
    r = np.array([e[2] for e in E])
    eq, fl, liq = X.sim(E, 5.0, 5.0); m = met(eq, fl, liq)
    eq0, fl0, l0 = X.sim(E, 1.0, 1.0); m0 = met(eq0, fl0, l0)
    P(f"  하한 {f1*100:5.1f}%: n={len(r):4d} 평균 {r.mean()*1e4:+6.0f}bp 승률 {(r>0).mean():.3f} | "
      f"무레버 {m0[0]:+5.0f}% | 5x/5x 연 {m[0]:+6.0f}% 최악 {m[2]:6.1f}% 24~ {m[3]:+6.0f}% 25~ {m[4]:+6.0f}%")

P("\n[5] 강건성 — 상위 수익 거래를 제외하면")
E = build_f(DEP, {}, bars=BL6)
r = np.array([e[2] for e in E])
order = np.argsort(-r)
for drop in (0, 1, 2, 3, 5, 10, 20):
    keep = set(range(len(E))) - set(order[:drop].tolist())
    E2 = [e for i, e in enumerate(E) if i in keep]
    r2 = np.array([e[2] for e in E2])
    eq, fl, liq = X.sim(E2, 5.0, 5.0); m = met(eq, fl, liq)
    eq0, fl0, l0 = X.sim(E2, 1.0, 1.0); m0 = met(eq0, fl0, l0)
    P(f"  상위 {drop:2d}건 제외: n={len(r2):4d} 평균 {r2.mean()*1e4:+6.0f}bp | 무레버 {m0[0]:+5.0f}% | 5x/5x 연 {m[0]:+6.0f}% 24~ {m[3]:+6.0f}% 25~ {m[4]:+6.0f}%")

P("\n[6] 강건성 — 연도 하나씩 제외 (해당 연도 거래를 뺀 나머지로 재평가)")
for y in range(2020, 2027):
    E2 = [e for e in E if e[4].year != y]
    if len(E2) < 50: continue
    r2 = np.array([e[2] for e in E2])
    eq, fl, liq = X.sim(E2, 5.0, 5.0); m = met(eq, fl, liq)
    P(f"  {y} 제외: n={len(r2):4d} 평균 {r2.mean()*1e4:+6.0f}bp | 5x/5x 연 {m[0]:+6.0f}% 최악 {m[2]:6.1f}%")

P("\n[7] 최종 3단 성과표")
BEST = build_f(DEP, {10.0: 0.05, 14.0: 0.07, 20.0: 0.10}, bars=BL6)
r = np.array([e[2] for e in BEST]); ts = pd.DatetimeIndex([e[4] for e in BEST])
P(f"  구성: 봉 4h·6h·8h·12h·16h·1d, 깊이 max(rv×10, 5%) / max(rv×14, 7%) / max(rv×20, 10%), 가중 1:3:6")
P(f"  거래 {len(r)}건 평균 {r.mean()*1e4:+.0f}bp 중앙 {np.median(r)*1e4:+.0f}bp 승률 {(r>0).mean():.3f} 거래월 {len(set(ts.to_period('M')))}/81")
tiers(BEST, "")
eq, fl, liq = X.sim(BEST, 5.0, 5.0); m = met(eq, fl, liq)
yr = m[5].resample("YE").last()/m[5].resample("YE").first()-1
P("   연도별(중간+): " + " | ".join(f"{i.year} {v*100:+.0f}%" for i, v in yr.items()))
