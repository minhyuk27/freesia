"""국면별 신호 지도 — 어느 1차 신호가 어느 국면에서 강한가. 국면 전환 전략.
사용: python research/regime_map.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from meta_strong import attach, FEAT, SYM
from anchor import build_a
from z06b import build as build_z6
from z07_dev import build as build_z7
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import exit_tier as X
from absfloor import met
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
T0 = pd.Timestamp("2021-06-01", tz="UTC")
TE = pd.Timestamp("2024-01-01", tz="UTC"); REC = pd.Timestamp("2025-01-01", tz="UTC")
IR = FEAT[SYM[0]]["cols"].index("rvratio15_240")

SIGS = {
    "z0.3 깊은딥(4h~1d)": [x for x in build_a("open", ext=0.10) if x[4] >= T0],
    "z0.6 유동성충격(5분)": [x for x in build_z6(0.005, 1.0, 30, 240, 30) if x[4] >= T0],
    "z0.7A 평활괴리(30분)": [x for x in build_z7(0.001, 30, 30, 30, mkt=1) if x[4] >= T0],
    "z0.7B 평활괴리(240분)": [x for x in build_z7(0.005, 15, 240, 30, mkt=0, kx=2.0) if x[4] >= T0],
}


def regime_split(E):
    AV = attach(E)
    if not AV: return None
    rv = np.array([a[1][IR] for a in AV])
    ok = np.isfinite(rv)
    AV = [AV[i] for i in np.where(ok)[0]]; rv = rv[ok]
    med = np.nanmedian(rv)
    lo = [a[0] for i, a in enumerate(AV) if rv[i] <= med]
    hi = [a[0] for i, a in enumerate(AV) if rv[i] > med]
    return lo, hi


def quick(E, lab):
    if len(E) < 40: P(f"    {lab:26s} 표본 부족({len(E)})"); return None
    r = np.array([x[2] for x in E])
    out = f"    {lab:26s} n={len(E):5d} {r.mean()*1e4:+7.1f}bp 승률{(r>0).mean():.3f} |"
    best = None
    for w, Nm, tag in ((1.0, 1.0, "무레버"), (2.0, 3.0, "저레버"), (3.0, 5.0, "중간")):
        eq, fl, liq = X.sim(E, w, Nm); m = met(eq, fl, liq)
        out += f" {tag} {'파산' if m is None else f'{m[0]:+7.1f}%'}"
        if m and (best is None or m[0] > best[0]): best = (m[0], tag)
    P(out)
    return best


P("[1] 신호 × 국면 지도")
for nm, E in SIGS.items():
    P(f"\n  ══ {nm} (전체 {len(E)}건) ══")
    quick(E, "전체")
    sp = regime_split(E)
    if sp is None: continue
    lo, hi = sp
    quick(lo, "저변동 국면")
    quick(hi, "고변동 국면")

P("\n[2] 국면 전환 전략 — 저변동에선 A신호, 고변동에선 B신호")
def pick(E, low=True):
    AV = attach(E)
    if not AV: return []
    rv = np.array([a[1][IR] for a in AV])
    ok = np.isfinite(rv); AV = [AV[i] for i in np.where(ok)[0]]; rv = rv[ok]
    med = np.nanmedian(rv)
    return [a[0] for i, a in enumerate(AV) if (rv[i] <= med) == low]

names = list(SIGS.keys())
best_pairs = []
for i, a in enumerate(names):
    for j, b in enumerate(names):
        if i == j: continue
        E = sorted(pick(SIGS[a], True) + pick(SIGS[b], False), key=lambda z: z[0])
        r = quick(E, f"저변동={a[:8]} / 고변동={b[:8]}")
        if r: best_pairs.append((r[0], a, b))

P("\n[3] 최고 조합 상세 + 메타 필터 추가")
best_pairs.sort(reverse=True)
for sc_, a, b in best_pairs[:3]:
    E = sorted(pick(SIGS[a], True) + pick(SIGS[b], False), key=lambda z: z[0])
    P(f"\n  ── 저변동={a} / 고변동={b} ──")
    for w, Nm, tag in ((1.0, 1.0, "무레버"), (2.0, 3.0, "저레버"), (3.0, 5.0, "중간"), (4.0, 6.0, "중간+")):
        eq, fl, liq = X.sim(E, w, Nm); m = met(eq, fl, liq)
        if m is None: P(f"     {tag:8s} 파산"); continue
        mon = (m[5].resample("ME").last()/m[5].resample("ME").first()-1).dropna()
        P(f"     {tag:8s} 연 {m[0]:+8.1f}% MDD {m[1]:6.1f}% 최악 {m[2]:6.1f}% | 24~ {m[3]:+8.1f}% | 25~ {m[4]:+8.1f}% | 월승률 {float((mon>0).mean()):.2f}")
    df = pd.DataFrame([(x[4], x[2]) for x in E], columns=["ts", "r"]).set_index("ts")
    g = df.groupby(df.index.year).r
    P("     연도별: " + " ".join(f"{y}{v*1e4:+6.0f}×{n}" for (y, v), (_, n) in zip(g.mean().items(), g.size().items())))

P("\n[4] 전 신호 합집합 (국면 무관, 단순 합산)")
E_all = sorted(sum(SIGS.values(), []), key=lambda z: z[0])
quick(E_all, "전 신호 합집합")
E_lo = sorted(sum([pick(E, True) for E in SIGS.values()], []), key=lambda z: z[0])
quick(E_lo, "전 신호 저변동만")
E_hi = sorted(sum([pick(E, False) for E in SIGS.values()], []), key=lambda z: z[0])
quick(E_hi, "전 신호 고변동만")
