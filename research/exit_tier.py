"""출구 규칙 × 더 깊은 단계. 자본 회전과 깊이 프리미엄을 동시에 노린다.
출구: A 봉종가 / B 시가회복 / C 진입가+g / D 시간제한 / E 시가회복 or 봉종가(둘중 먼저)
단계: rv×8·12 에 더해 rv×16·20·26 추가(더 깊을수록 큰 가중)
사용: python research/exit_tier.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
EPS = 0.0002; MMR = 0.004
TE = pd.Timestamp("2024-01-01", tz="UTC"); REC = pd.Timestamp("2025-01-01", tz="UTC")
SYM = ("BTCUSDT", "ETHUSDT"); BL = (240, 360, 480, 720, 960, 1440)
RAW = {}
for s in SYM:
    k = store.load("klines_1m", s).set_index("ts")
    RAW[s] = dict(o=k.open.values, h=k.high.values, l=k.low.values, c=k.close.values, idx=k.index)


def prep(s, B):
    c = RAW[s]["c"]; n = len(c)//B
    st = np.arange(n)*B
    rv = pd.Series(np.log(c[st+B-1])).diff().rolling(20).std().shift(1).values
    return st, rv


@numba.njit(cache=True)
def ev(o, h, l, c, st, B, x, mode, g, tmax, eps, fm, ft):
    """mode 0=봉종가 1=시가회복 2=진입가+g 3=시간제한 4=시가회복or종가"""
    n = len(st)
    ei = np.full(n, -1, np.int64); ex = np.full(n, -1, np.int64)
    rr = np.zeros(n); mm = np.zeros(n); k = 0
    for b in range(n):
        if np.isnan(x[b]) or x[b] <= 0: continue
        s0 = st[b]; e = s0+B
        if e > len(c): break
        O = o[s0]; T = O*(1-x[b])
        f = -1; lo_ = 1e18; hit = -1; px = 0.0; mk = 0
        lim = e
        if mode == 3:
            lim = e
        for t in range(s0, e):
            if f < 0:
                if l[t] <= T*(1-eps): f = t
                else: continue
            if l[t] < lo_: lo_ = l[t]
            if t == f: continue
            if mode == 3 and t-f >= tmax:
                hit = t; px = c[t]; mk = 0; break
            if (mode == 1 or mode == 4) and h[t] >= O*(1+eps):
                hit = t; px = O; mk = 1; break
            if mode == 2:
                TP = T*(1.0+g[b])
                if h[t] >= TP*(1+eps): hit = t; px = TP; mk = 1; break
        if f < 0: continue
        if hit < 0:
            if mode == 1:
                hit = e-1; px = c[e-1]; mk = 0
            else:
                hit = e-1; px = c[e-1]; mk = 0
        fee = (fm+fm) if mk == 1 else (fm+ft)
        ei[k] = f; ex[k] = hit
        rr[k] = (px/T-1.0) - fee/1e4
        mm[k] = 1.0 - lo_/T
        k += 1
    return ei[:k], ex[:k], rr[:k], mm[:k]


CACHE = {}
def build(depths, mode, kg=0.0, tmax=0, eps=EPS, fm=2.0, ft=5.0):
    E = []
    for s in SYM:
        d = RAW[s]
        for B in BL:
            key0 = (s, B)
            if key0 not in CACHE: CACHE[key0] = prep(s, B)
            st, rv = CACHE[key0]
            for k, wgt in depths.items():
                a_, b_, r_, m_ = ev(d["o"], d["h"], d["l"], d["c"], st.astype(np.int64), B,
                                    np.asarray(rv*k, float), mode, np.asarray(rv*kg, float),
                                    tmax, eps, fm, ft)
                for a, bb, r, m in zip(a_, b_, r_, m_):
                    E.append((int(a), int(bb), float(r), float(m), d["idx"][int(a)], (s, B, k),
                              float(rv[int(a)//B]), wgt))
    E.sort(key=lambda z: z[0])
    return E


def sim(E, w, Nmax, lo=0.2, hi=3.0, warm=30):
    cap = 1.0; open_ = []; rvh = {}; liq = 0; ts_l = []; eq_l = []; fl_l = []
    for a, b, r, m, ts, key, rvb, wgt in E:
        for p in [p for p in open_ if p[0] <= a]:
            cap += p[1]*p[2]; open_.remove(p)
        if cap <= 1e-9: break
        if not np.isfinite(rvb) or rvb <= 0: continue
        rh = rvh.setdefault(key, [])
        f = float(np.clip(np.median(rh)/rvb, lo, hi)) if len(rh) >= warm else lo
        rh.append(rvb)
        n_ = min(cap*w*wgt*f, max(cap*Nmax - sum(p[1] for p in open_), 0.0))
        if n_ <= 1e-9: continue
        if n_*m + sum(p[1]*p[3] for p in open_) >= cap*(1-MMR):
            cap = 0.0; liq += 1; break
        open_.append((b, n_, r, m))
        ts_l.append(ts); eq_l.append(cap); fl_l.append(cap - sum(p[1]*p[3] for p in open_))
    for p in open_: cap += p[1]*p[2]
    if not ts_l: return None, None, liq
    ts_l.append(E[-1][4]); eq_l.append(cap); fl_l.append(cap)
    eq = pd.Series(eq_l, index=pd.DatetimeIndex(ts_l)).groupby(level=0).last().resample("D").last().ffill()
    fl = pd.Series(fl_l, index=pd.DatetimeIndex(ts_l)).groupby(level=0).min().resample("D").min().ffill()
    return eq, fl, liq


def show(eq, fl, liq, lab, quiet=False):
    if eq is None or eq.iloc[-1] <= 1e-6:
        if not quiet: P(f"  {lab:32s} 파산 (청산 {liq})")
        return None
    y = (eq.index[-1]-eq.index[0]).days/365
    cg = (eq.iloc[-1]**(1/y)-1)*100
    mdd = float((eq/eq.cummax()-1).min())*100
    fld = float((fl.reindex(eq.index).ffill()/eq.cummax()-1).min())*100
    sg = lambda t0: (lambda s: ((s.iloc[-1]/s.iloc[0])**(365/max((s.index[-1]-s.index[0]).days,1))-1)*100 if len(s) > 60 else np.nan)(eq[eq.index >= t0])
    if not quiet:
        P(f"  {lab:32s} 연 {cg:+7.0f}% MDD {mdd:6.1f}%(최악 {fld:6.1f}%) | 24~ {sg(TE):+7.0f}% | 25~ {sg(REC):+7.0f}% | 청산 {liq}")
    return cg, mdd, fld, sg(TE), sg(REC), eq


D0 = {8.0: 1.0, 12.0: 3.0}
P("[1] 출구 규칙 비교 (깊이 8·12(1:3), w=3 Nmax=5)")
EXITS = [("A 봉종가", 0, 0.0, 0), ("B 시가회복→종가", 1, 0.0, 0), ("C 진입가+rv×3", 2, 3.0, 0),
         ("C 진입가+rv×6", 2, 6.0, 0), ("C 진입가+rv×10", 2, 10.0, 0),
         ("D 60분 제한", 3, 0.0, 60), ("D 240분 제한", 3, 0.0, 240), ("D 720분 제한", 3, 0.0, 720)]
for lab, md, kg, tm in EXITS:
    E = build(D0, md, kg, tm)
    r = np.array([e[2] for e in E]); hd = np.array([e[1]-e[0] for e in E])
    P(f"  {lab:16s} n={len(E):4d} 평균 {r.mean()*1e4:+6.0f}bp 승률 {(r>0).mean():.3f} 보유중앙 {np.median(hd):5.0f}분")
    eq, fl, liq = sim(E, 3.0, 5.0); show(eq, fl, liq, "")

P("\n[2] 더 깊은 단계 추가 (출구 A 봉종가, w=3 Nmax=5)")
TIERS = {
    "8·12 (1:3)":              {8.0: 1.0, 12.0: 3.0},
    "8·12·16 (1:2:4)":         {8.0: 1.0, 12.0: 2.0, 16.0: 4.0},
    "8·12·16·20 (1:2:4:6)":    {8.0: 1.0, 12.0: 2.0, 16.0: 4.0, 20.0: 6.0},
    "10·14·20 (1:3:6)":        {10.0: 1.0, 14.0: 3.0, 20.0: 6.0},
    "12·16·22 (1:3:6)":        {12.0: 1.0, 16.0: 3.0, 22.0: 6.0},
    "8·12·16·20·26":           {8.0: 1.0, 12.0: 2.0, 16.0: 3.0, 20.0: 5.0, 26.0: 8.0},
    "깊은쪽만 16·22 (1:3)":      {16.0: 1.0, 22.0: 3.0},
}
best = None
for lab, dep in TIERS.items():
    E = build(dep, 0)
    r = np.array([e[2] for e in E])
    P(f"  {lab:22s} n={len(E):4d} 평균 {r.mean()*1e4:+6.0f}bp 승률 {(r>0).mean():.3f}")
    for w in (2.0, 3.0, 5.0):
        for Nm in (3.0, 5.0, 8.0):
            if Nm < w: continue
            eq, fl, liq = sim(E, w, Nm)
            rr = show(eq, fl, liq, f"     w={w:.0f} Nmax={Nm:.0f}", quiet=(liq > 0 or eq is None))
            if rr and rr[2] > -70 and (best is None or rr[0] > best[0]): best = (*rr, lab, dep, w, Nm)

if best:
    cg, mdd, fld, t24, t25, eq, lab, dep, w, Nm = best
    P(f"\n★ 안전제약(최악>−70%) 통과 최적: {lab}, w={w:.0f}, Nmax={Nm:.0f}")
    P(f"   연 {cg:+.0f}% MDD {mdd:.1f}% 최악 {fld:.1f}% | 24~ {t24:+.0f}% | 25~ {t25:+.0f}%")
    yr = eq.resample("YE").last()/eq.resample("YE").first()-1
    mon = eq.resample("ME").last()/eq.resample("ME").first()-1
    P("   연도별: " + " | ".join(f"{i.year} {v*100:+.0f}%" for i, v in yr.items()))
    P("   최근 12개월: " + " | ".join(f"{i.strftime('%y-%m')} {v*100:+.1f}%" for i, v in mon.tail(12).items()))
    l12 = mon.tail(12)
    P(f"   최근 12개월 {(np.prod(1+l12)-1)*100:+.1f}% 양수 {int((l12>0).sum())}/12 | 월승률 {float((mon>0).mean()):.3f}")
