"""국면 공간 커버리지 지도 — 시장무관 수익을 위한 진단.
2차원 국면(변동성 × 추세) 9칸에서 각 전략이 어디를 담당하고 어디가 비어 있나.
그리고 '꾸준함'을 직접 측정(주간 수익 분포·연속 손실·시장 상관).
사용: python research/regime_grid.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from meta_strong import attach, FEAT, SYM
from anchor import build_a
from z0_6b import build as build_z6
from z0_7_dev import build as build_z7
from freesia import store
import exit_tier as X
from absfloor import met
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
T0 = pd.Timestamp("2021-06-01", tz="UTC")
cols = FEAT[SYM[0]]["cols"]
IR = cols.index("rvratio15_240")      # 변동성 국면
IE = cols.index("er60")               # 추세 강도

# 시장 기준선
MK = {}
for s in SYM:
    k = store.load("klines_1m", s).set_index("ts")
    MK[s] = k.close
btc_d = MK["BTCUSDT"].resample("D").last()
mkt_w = btc_d.resample("W").last().pct_change().dropna()
mkt_m = btc_d.resample("ME").last().pct_change().dropna()

SIGS = {
    "z0.3 깊은딥": [x for x in build_a("open", ext=0.10) if x[4] >= T0],
    "z0.6 유동성충격": [x for x in build_z6(0.005, 1.0, 30, 240, 30) if x[4] >= T0],
    "z0.7B 평활괴리": [x for x in build_z7(0.005, 15, 240, 30, mkt=0, kx=2.0) if x[4] >= T0],
}

P("[1] 국면 공간 정의 — 전체 시간의 분포")
# 15분 격자 전체에서 국면 분포 측정
allf = []
for s in SYM:
    f = FEAT[s]
    v = f["X"][:, [IR, IE]]
    m = np.isfinite(v).all(axis=1)
    allf.append(v[m])
A = np.vstack(allf)
rv_q = np.nanquantile(A[:, 0], [0.33, 0.67])
er_q = np.nanquantile(A[:, 1], [0.33, 0.67])
P(f"  변동성비율(rvratio15_240) 33/67분위: {rv_q[0]:.3f} / {rv_q[1]:.3f}")
P(f"  추세강도(er60) 33/67분위: {er_q[0]:.3f} / {er_q[1]:.3f}")
def cell(rv, er):
    i = 0 if rv <= rv_q[0] else (1 if rv <= rv_q[1] else 2)
    j = 0 if er <= er_q[0] else (1 if er <= er_q[1] else 2)
    return i, j
RVL = ["저변동", "중변동", "고변동"]; ERL = ["횡보", "중간", "추세"]
cnt = np.zeros((3, 3))
for r_, e_ in A[::20]:
    i, j = cell(r_, e_); cnt[i, j] += 1
cnt = cnt/cnt.sum()*100
P("\n  시간 비중(%):")
P("           " + "".join(f"{l:>10s}" for l in ERL))
for i in range(3):
    P(f"  {RVL[i]:>8s} " + "".join(f"{cnt[i,j]:9.1f}%" for j in range(3)))

P("\n[2] 전략 × 국면 — 어디를 담당하나 (거래당 bp / 건수)")
GRID = {}
for nm, E in SIGS.items():
    AV = attach(E)
    if not AV: continue
    res = np.full((3, 3), np.nan); nn = np.zeros((3, 3), int)
    for a in AV:
        v = a[1]
        if not (np.isfinite(v[IR]) and np.isfinite(v[IE])): continue
        i, j = cell(v[IR], v[IE])
        nn[i, j] += 1
    vals = {}
    for a in AV:
        v = a[1]
        if not (np.isfinite(v[IR]) and np.isfinite(v[IE])): continue
        i, j = cell(v[IR], v[IE])
        vals.setdefault((i, j), []).append(a[0][2])
    P(f"\n  ══ {nm} ══")
    P("           " + "".join(f"{l:>16s}" for l in ERL))
    for i in range(3):
        row = f"  {RVL[i]:>8s} "
        for j in range(3):
            v = vals.get((i, j), [])
            row += f"{np.mean(v)*1e4:+8.1f}bp/{len(v):<5d}" if len(v) >= 10 else f"{'—':>15s}"
        P(row)
    GRID[nm] = vals

P("\n[3] 커버리지 공백 — 어느 칸에 아무도 없나")
P("           " + "".join(f"{l:>16s}" for l in ERL))
for i in range(3):
    row = f"  {RVL[i]:>8s} "
    for j in range(3):
        good = [nm for nm, v in GRID.items() if len(v.get((i, j), [])) >= 20 and np.mean(v[(i, j)]) > 0.001]
        row += f"{(','.join(n[:4] for n in good) if good else '★공백'):>16s}"
    P(row)

P("\n[4] '꾸준함' 직접 측정 — 주간 수익 분포와 시장 상관")
P(f"  {'전략':24s} {'주승률':>7s} {'주중앙':>8s} {'최대연속손실주':>12s} {'거래없는주':>10s} {'시장상관(주)':>12s} {'시장상관(월)':>12s}")
for nm, E in SIGS.items():
    eq, fl, liq = X.sim(E, 2.0, 3.0); m = met(eq, fl, liq)
    if m is None: continue
    s = m[5]
    wk = (s.resample("W").last()/s.resample("W").first()-1).dropna()
    mo = (s.resample("ME").last()/s.resample("ME").first()-1).dropna()
    zero = (wk == 0).mean()
    neg = (wk < 0).astype(int)
    run = 0; mx = 0
    for x in neg:
        run = run+1 if x else 0
        mx = max(mx, run)
    j = pd.concat([wk.rename("s"), mkt_w.rename("m")], axis=1).dropna()
    jm = pd.concat([mo.rename("s"), mkt_m.rename("m")], axis=1).dropna()
    P(f"  {nm:24s} {float((wk>0).mean()):6.2f} {wk.median()*100:+7.2f}% {mx:11d}주 {zero*100:9.0f}% "
      f"{j.s.corr(j.m):+11.3f} {jm.s.corr(jm.m):+11.3f}")

P("\n[5] 시장 하락 주/월에 어떻게 되나 (진짜 시장무관인가)")
for nm, E in SIGS.items():
    eq, fl, liq = X.sim(E, 2.0, 3.0); m = met(eq, fl, liq)
    if m is None: continue
    s = m[5]
    wk = (s.resample("W").last()/s.resample("W").first()-1).dropna()
    j = pd.concat([wk.rename("s"), mkt_w.rename("m")], axis=1).dropna()
    dn = j[j.m < -0.05]; up = j[j.m > 0.05]; flat = j[j.m.abs() <= 0.02]
    P(f"  {nm:24s} 시장 -5%이하 주({len(dn):3d}): {dn.s.mean()*100:+6.2f}% 승률 {float((dn.s>0).mean()):.2f} | "
      f"횡보 주({len(flat):3d}): {flat.s.mean()*100:+6.2f}% 승률 {float((flat.s>0).mean()):.2f} | "
      f"+5%이상({len(up):3d}): {up.s.mean()*100:+6.2f}%")
