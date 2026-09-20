"""z0.3 성과를 세 레버리지 구간으로 — 무레버리지 / 중간 / 최대공격.
무레버리지 = 동시 총명목 ≤ 자본(1x). 변동성타겟 배율도 1을 넘지 않게 고정.
사용: python research/three_tiers.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from exit_tier import build, sim, RAW, TE, REC
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
DEP = {10.0: 1.0, 14.0: 3.0, 20.0: 6.0}
DEP2 = {10.0: 1.0, 14.0: 4.0, 20.0: 10.0}


def full(eq, fl, liq):
    y = (eq.index[-1]-eq.index[0]).days/365
    cg = (eq.iloc[-1]**(1/y)-1)*100
    mdd = float((eq/eq.cummax()-1).min())*100
    fld = float((fl.reindex(eq.index).ffill()/eq.cummax()-1).min())*100
    sg = lambda t0: (lambda s: ((s.iloc[-1]/s.iloc[0])**(365/max((s.index[-1]-s.index[0]).days,1))-1)*100 if len(s) > 60 else np.nan)(eq[eq.index >= t0])
    yr = eq.resample("YE").last()/eq.resample("YE").first()-1
    return dict(cagr=cg, mdd=mdd, floor=fld, t24=sg(TE), t25=sg(REC), mult=eq.iloc[-1],
                yr={i.year: v*100 for i, v in yr.items()}, liq=liq)


E = build(DEP, 0); E2 = build(DEP2, 0)
TIERS = [
    ("① 무레버리지 (총명목 ≤ 자본 1x)", E, 1.0, 1.0),
    ("   참고: 총명목 ≤ 2x",           E, 2.0, 2.0),
    ("② 중간 (진입 3x, 총 5x)",        E, 3.0, 5.0),
    ("   중간+ (진입 5x, 총 5x)",       E, 5.0, 5.0),
    ("③ 공격 (진입 5x, 총 8x)",        E, 5.0, 8.0),
    ("④ 최대공격 (가중 1:4:10, 5x/8x)", E2, 5.0, 8.0),
    ("   최대공격+ (가중 1:4:10, 8x/12x)", E2, 8.0, 12.0),
]
P("=" * 108)
P(f"{'구간':34s} {'전체연':>8s} {'최종배수':>9s} {'MDD':>7s} {'최악여유':>8s} {'2024~':>8s} {'2025~':>8s} {'청산':>4s}")
P("=" * 108)
rows = {}
for lab, Ex, w, Nm in TIERS:
    eq, fl, liq = sim(Ex, w, Nm)
    if eq is None or eq.iloc[-1] <= 1e-6:
        P(f"{lab:34s} {'파산':>8s}  (청산 {liq})"); continue
    d = full(eq, fl, liq); rows[lab] = d
    P(f"{lab:34s} {d['cagr']:+7.0f}% {d['mult']:8.1f}배 {d['mdd']:6.1f}% {d['floor']:7.1f}% {d['t24']:+7.0f}% {d['t25']:+7.0f}% {liq:4d}")
P("=" * 108)
P("\n연도별 (%)")
P(f"{'구간':34s} " + " ".join(f"{y:>7d}" for y in range(2020, 2027)))
for lab, d in rows.items():
    P(f"{lab:34s} " + " ".join(f"{d['yr'].get(y, float('nan')):+7.0f}" for y in range(2020, 2027)))

P("\n[원금별 환산 — 6.7년 누적, 무레버리지 vs 공격]")
for cap in (1000, 5000, 10000, 20000):
    line = f"  자본 {cap:>6,}만원: "
    for lab in ("① 무레버리지 (총명목 ≤ 자본 1x)", "② 중간 (진입 3x, 총 5x)", "③ 공격 (진입 5x, 총 8x)", "④ 최대공격 (가중 1:4:10, 5x/8x)"):
        if lab in rows:
            line += f"{lab[0]} {cap*rows[lab]['mult']:>11,.0f}만원  "
    P(line)

P("\n[레버리지 기여 분해]")
if "① 무레버리지 (총명목 ≤ 자본 1x)" in rows:
    b = rows["① 무레버리지 (총명목 ≤ 자본 1x)"]["cagr"]
    P(f"  전략 자체의 엣지(무레버리지): 연 {b:+.0f}%")
    for lab, d in rows.items():
        if lab.startswith("①"): continue
        P(f"  {lab:34s} 연 {d['cagr']:+7.0f}%  (배수 {d['cagr']/b:5.2f}x, 최악여유 {d['floor']:6.1f}%)")

P("\n[거래 단위 — 레버리지와 무관한 원지표]")
r = np.array([e[2] for e in E])
P(f"  n={len(r)} 평균 {r.mean()*1e4:+.0f}bp 중앙 {np.median(r)*1e4:+.0f}bp 승률 {(r>0).mean():.3f} "
  f"최악 {r.min()*1e4:+.0f}bp 표준편차 {r.std()*1e4:.0f}bp")
