"""간극 해부: 산술평균 +707bp × 8배 = 연 1300배인데 실제는 +252%. 왜인가.
(a) 기하평균 vs 산술평균 (b) 동시보유로 인한 명목 희석 (c) 변동성타겟 축소 (d) 켈리 최적점
그리고 빈도-품질 프런티어 재탐색(프리미엄 증폭 포함).
사용: python research/kelly_gap.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
import exit_tier as X
from absfloor import met
from anchor import build_a
from freesia import store
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
SYM = ("BTCUSDT", "ETHUSDT")
TE = pd.Timestamp("2024-01-01", tz="UTC"); REC = pd.Timestamp("2025-01-01", tz="UTC")
Z = build_a("open", ext=0.10)
r = np.array([x[2] for x in Z])
YRS = 6.7

P("[1] 산술 vs 기하 — 명목 배수별 순차 복리 (겹침 무시, 한 번에 하나씩)")
P(f"  {'명목N':>6s} {'산술 거래당':>11s} {'기하 거래당':>11s} {'파산거래':>8s} {'연환산':>12s}")
for N in (1, 2, 3, 5, 8, 10, 12, 15, 20):
    x = 1.0 + N*r
    bust = int((x <= 0).sum())
    xc = np.clip(x, 1e-9, None)
    g = np.exp(np.mean(np.log(xc))) - 1.0
    tot = np.prod(xc)
    ann = (tot**(1/YRS)-1)*100 if bust == 0 else -100.0
    P(f"  {N:6d} {N*r.mean()*100:+10.1f}% {g*100:+10.1f}% {bust:8d} {ann:+11.0f}%")
P("  → 산술평균은 꼬리에 끌려가고, 복리는 기하평균이 지배한다. 손실 거래가 배수만큼 증폭된다.")

P("\n[2] 켈리 — 성장률 최대화 명목 배수")
grid = np.arange(0.5, 25, 0.25)
gl = []
for N in grid:
    x = 1.0 + N*r
    if (x <= 0).any(): gl.append(-np.inf); continue
    gl.append(np.mean(np.log(x)))
gl = np.array(gl)
i = int(np.nanargmax(gl))
P(f"  켈리 최적 N* = {grid[i]:.2f} (거래당 로그성장 {gl[i]:.4f} → 연 {(np.exp(gl[i]*len(r)/YRS)-1)*100:+.0f}%)")
P(f"  하프켈리 N={grid[i]/2:.2f} → 연 {(np.exp(np.mean(np.log(1+grid[i]/2*r))*len(r)/YRS)-1)*100:+.0f}%")
lim = r.min()
P(f"  이론 상한: 최악 거래 {lim*1e4:+.0f}bp 이므로 N < {-1/lim:.1f} 이어야 단일 거래 파산 없음")

P("\n[3] 실제 시뮬과의 차이 — 어디서 잃는가")
eq, fl, liq = X.sim(Z, 8.0, 8.0); m = met(eq, fl, liq)
P(f"  실제 시뮬 (w=8, Nmax=8): 연 {m[0]:+.0f}%")
# 동시보유·변동성타겟이 실제로 얼마나 명목을 깎는지 측정
cap = 1.0; open_ = []; rvh = {}; used = []
for a, b, rr, mm, ts, key, rvb, wg in Z:
    for p in [p for p in open_ if p[0] <= a]:
        cap += p[1]*p[2]; open_.remove(p)
    rh = rvh.setdefault(key, [])
    f = float(np.clip(np.median(rh)/rvb, 0.2, 3.0)) if len(rh) >= 30 else 0.2
    rh.append(rvb)
    want = cap*8.0*wg*f
    avail = max(cap*8.0 - sum(p[1] for p in open_), 0.0)
    n_ = min(want, avail)
    used.append((n_/cap, 8.0*wg*f, f, avail/cap))
    open_.append((b, n_, rr, mm))
u = pd.DataFrame(used, columns=["실제명목배수", "희망명목배수", "변동성계수", "여유배수"])
P(f"  희망 명목배수 중앙 {u.희망명목배수.median():.2f} → 실제 체결 명목배수 중앙 {u.실제명목배수.median():.2f}")
P(f"  변동성 계수 중앙 {u.변동성계수.median():.2f} (1.0이면 축소 없음) | 0.2에 붙은 비율 {(u.변동성계수<=0.21).mean():.1%}")
P(f"  Nmax 여유 부족으로 깎인 비율 {(u.실제명목배수 < u.희망명목배수*0.99).mean():.1%}")

P("\n[4] 변동성타겟 상한을 풀면 (clip 상한 hi 조정)")
def sim_hi(E, w, Nm, lo, hi):
    return X.sim(E, w, Nm, lo=lo, hi=hi)
for lo, hi in ((0.2, 3.0), (0.5, 3.0), (1.0, 1.0), (0.5, 5.0), (1.0, 5.0), (0.2, 10.0)):
    best = None
    for w in (3, 5, 8, 12, 20):
        for Nm in (5, 8, 12, 20):
            if Nm < w: continue
            eq, fl, liq = sim_hi(Z, float(w), float(Nm), lo, hi); mm2 = met(eq, fl, liq)
            if mm2 is None or liq: continue
            if best is None or mm2[0] > best[0]: best = (mm2[0], mm2[2], mm2[3], mm2[4], w, Nm)
    P(f"  clip({lo},{hi}): " + (f"최대 w={best[4]} Nmax={best[5]} 연 {best[0]:+7.0f}% 최악 {best[1]:6.1f}% 24~ {best[2]:+7.0f}% 25~ {best[3]:+7.0f}%" if best else "무청산 없음"))

P("\n[5] 빈도-품질 프런티어 재탐색 (프리미엄 증폭 포함, 최대 무청산 수익 기준)")
PM = {s: store.load("premium_1m", s).set_index("ts").close for s in SYM}
def pqs(E):
    out = []
    for a, b, rr, m2, ts, key, rv, wg in E:
        ser = PM[key[0]]; i = ser.index.searchsorted(ts)
        if i <= 1: out.append(np.nan); continue
        pv = float(ser.iloc[i-1]); past = ser.iloc[max(0, i-1-525600):i-1]
        out.append(float((past < pv).mean()) if len(past) > 10000 else np.nan)
    return np.array(out)

from absfloor import build_f
BL6 = (240, 360, 480, 720, 960, 1440)
FR = [
    ("얕음 6·8·11", {6.0: 1.0, 8.0: 2.0, 11.0: 4.0}, {6.0: .03, 8.0: .04, 11.0: .06}),
    ("중간 8·11·15", {8.0: 1.0, 11.0: 2.0, 15.0: 4.0}, {8.0: .04, 11.0: .05, 15.0: .07}),
    ("기준 10·14·20", {10.0: 1.0, 14.0: 3.0, 20.0: 6.0}, {10.0: .05, 14.0: .07, 20.0: .10}),
    ("깊음 12·16·22", {12.0: 1.0, 16.0: 3.0, 22.0: 6.0}, {12.0: .06, 16.0: .08, 22.0: .12}),
]
for lab, dep, flo in FR:
    E = build_f(dep, flo, bars=BL6)
    q = pqs(E)
    E = [(a, b, rr, m2, ts, key, rv, wg*(1.5 if (np.isfinite(q[j]) and q[j] <= 0.10) else 1.0))
         for j, (a, b, rr, m2, ts, key, rv, wg) in enumerate(E)]
    rr2 = np.array([x[2] for x in E])
    best = None
    for w in (3, 5, 8, 12, 20):
        for Nm in (5, 8, 12, 20):
            if Nm < w: continue
            eq, fl, liq = X.sim(E, float(w), float(Nm)); mm2 = met(eq, fl, liq)
            if mm2 is None or liq: continue
            if best is None or mm2[0] > best[0]: best = (mm2[0], mm2[2], mm2[3], mm2[4], w, Nm, mm2[5])
    if best:
        mo = (best[6].resample("ME").last()/best[6].resample("ME").first()-1).dropna()
        mo = mo[mo != 0]
        P(f"  {lab:14s} n={len(rr2):4d}({len(rr2)/6.7:.0f}/년) 거래당 {rr2.mean()*1e4:+5.0f}bp | "
          f"최대 w={best[4]:2d}/{best[5]:2d} 연 {best[0]:+7.0f}% 최악 {best[1]:6.1f}% | 25~ {best[3]:+7.0f}% | "
          f"거래월 {len(mo)}/81 월중앙 {mo.median()*100:+.0f}%")
    else:
        P(f"  {lab:14s} n={len(rr2):4d} 무청산 구성 없음")
