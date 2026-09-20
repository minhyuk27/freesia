"""진입 깊이에 절대 하한 도입: depth = max(rv×k, floor%).
저변동 국면의 무익한 얕은 체결을 걸러내고, 조용한 달에도 거래가 생기게 한다.
사용: python research/absfloor.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from exit_tier import RAW, SYM, BL, prep, CACHE, ev, sim, EPS, TE, REC
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)


def build_f(depths, floors, eps=EPS, fm=2.0, ft=5.0, bars=BL):
    """depths: {k: wgt}. floors: {k: 절대하한(비율)}  실제 깊이 = max(rv*k, floor)"""
    E = []
    for s in SYM:
        d = RAW[s]
        for B in bars:
            if (s, B) not in CACHE: CACHE[(s, B)] = prep(s, B)
            st, rv = CACHE[(s, B)]
            for k, wgt in depths.items():
                x = np.maximum(rv*k, floors.get(k, 0.0))
                x = np.where(np.isnan(rv), np.nan, x)
                a_, b_, r_, m_ = ev(d["o"], d["h"], d["l"], d["c"], st.astype(np.int64), B,
                                    np.asarray(x, float), 0, np.zeros(len(rv)), 0, eps, fm, ft)
                for a, bb, r, m in zip(a_, b_, r_, m_):
                    E.append((int(a), int(bb), float(r), float(m), d["idx"][int(a)], (s, B, k),
                              float(rv[int(a)//B]), wgt))
    E.sort(key=lambda z: z[0])
    return E


def met(eq, fl, liq):
    if eq is None or eq.iloc[-1] <= 1e-6: return None
    y = (eq.index[-1]-eq.index[0]).days/365
    sg = lambda t0: (lambda s: ((s.iloc[-1]/s.iloc[0])**(365/max((s.index[-1]-s.index[0]).days,1))-1)*100 if len(s) > 60 else np.nan)(eq[eq.index >= t0])
    return ((eq.iloc[-1]**(1/y)-1)*100, float((eq/eq.cummax()-1).min())*100,
            float((fl.reindex(eq.index).ffill()/eq.cummax()-1).min())*100, sg(TE), sg(REC), eq)


P("[1] 단일 깊이 k에 절대 하한을 걸었을 때 거래당 수익 (4h·8h, 두 종목 합산)")
P(f"  {'k':>5s} {'하한%':>6s} {'n':>6s} {'평균bp':>8s} {'중앙bp':>8s} {'승률':>6s} {'거래있는달':>8s}")
for k in (3.0, 4.0, 5.0, 7.0):
    for fl_ in (0.0, 0.005, 0.01, 0.015, 0.02, 0.03):
        E = build_f({k: 1.0}, {k: fl_}, bars=(240, 480))
        if len(E) < 30: continue
        r = np.array([e[2] for e in E])
        ts = pd.DatetimeIndex([e[4] for e in E])
        nm = len(set(ts.to_period("M")))
        P(f"  {k:5.1f} {fl_*100:6.1f} {len(r):6d} {r.mean()*1e4:+8.1f} {np.median(r)*1e4:+8.1f} {(r>0).mean():6.3f} {nm:6d}/71")

P("\n[2] 깊은 딥에도 하한을 걸면 (z0.3 기본 10·14·20)")
for fl_ in (0.0, 0.01, 0.02, 0.03, 0.05):
    E = build_f({10.0: 1.0, 14.0: 3.0, 20.0: 6.0},
                {10.0: fl_, 14.0: fl_*1.4, 20.0: fl_*2.0})
    r = np.array([e[2] for e in E]); ts = pd.DatetimeIndex([e[4] for e in E])
    eq, fl2, liq = sim(E, 5.0, 5.0); m = met(eq, fl2, liq)
    eq0, fl0, liq0 = sim(E, 1.0, 1.0); m0 = met(eq0, fl0, liq0)
    P(f"  하한 {fl_*100:4.1f}%: n={len(r):4d} 평균 {r.mean()*1e4:+6.0f}bp 승률 {(r>0).mean():.3f} 거래월 {len(set(ts.to_period('M'))):2d}/71 | "
      f"무레버 연 {m0[0]:+5.0f}% | 5x/5x 연 {m[0]:+6.0f}% 최악 {m[2]:6.1f}% 25~ {m[4]:+6.0f}%")

P("\n[3] 얕은층 + 깊은층 결합 — 조용한 달을 얕은층이 메운다")
COMB = {
    "z0.3 단독 (10·14·20)":        ({10.0: 1.0, 14.0: 3.0, 20.0: 6.0}, {}),
    "+얕은 k=5 하한1.5%":           ({5.0: 0.5, 10.0: 1.0, 14.0: 3.0, 20.0: 6.0}, {5.0: 0.015}),
    "+얕은 k=5 하한2%":             ({5.0: 0.5, 10.0: 1.0, 14.0: 3.0, 20.0: 6.0}, {5.0: 0.02}),
    "+얕은 k=4 하한2%":             ({4.0: 0.5, 10.0: 1.0, 14.0: 3.0, 20.0: 6.0}, {4.0: 0.02}),
    "+얕은 k=5 하한2% 가중1.0":       ({5.0: 1.0, 10.0: 1.0, 14.0: 3.0, 20.0: 6.0}, {5.0: 0.02}),
    "+얕은 k=4,7 하한2%/3%":        ({4.0: 0.5, 7.0: 0.8, 10.0: 1.0, 14.0: 3.0, 20.0: 6.0}, {4.0: 0.02, 7.0: 0.03}),
    "전층 하한 (4·7·10·14·20)":     ({4.0: 0.5, 7.0: 0.8, 10.0: 1.0, 14.0: 3.0, 20.0: 6.0},
                                    {4.0: 0.02, 7.0: 0.03, 10.0: 0.04, 14.0: 0.05, 20.0: 0.07}),
}
best = None
for lab, (dep, flo) in COMB.items():
    E = build_f(dep, flo)
    r = np.array([e[2] for e in E]); ts = pd.DatetimeIndex([e[4] for e in E])
    nm = len(set(ts.to_period("M")))
    P(f"  ── {lab}  n={len(r)} 평균 {r.mean()*1e4:+.0f}bp 승률 {(r>0).mean():.3f} 거래월 {nm}/71 ──")
    for w, Nm, tag in ((1.0, 1.0, "무레버"), (3.0, 5.0, "중간"), (5.0, 5.0, "중간+"), (5.0, 8.0, "공격")):
        eq, fl2, liq = sim(E, w, Nm); m = met(eq, fl2, liq)
        if m is None: P(f"     {tag:6s} 파산"); continue
        P(f"     {tag:6s} 연 {m[0]:+7.0f}% MDD {m[1]:6.1f}% 최악 {m[2]:6.1f}% | 24~ {m[3]:+7.0f}% | 25~ {m[4]:+7.0f}%")
        if tag == "중간+" and m[2] > -70 and (best is None or m[0] > best[0]):
            best = (m[0], lab, dep, flo, m)
if best:
    P(f"\n★ 중간+ 구간 최고(최악>−70%): {best[1]} → 연 {best[0]:+.0f}%")
    eq = best[4][5]
    mon = eq.resample("ME").last()/eq.resample("ME").first()-1
    P(f"   월승률 {float((mon>0).mean()):.3f} | 최근 12개월 " + " ".join(f"{v*100:+.0f}" for v in mon.tail(12)))
