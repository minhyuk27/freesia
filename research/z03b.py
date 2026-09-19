"""z0.3b — 깊이를 독립 다리로 분리(얕은·중간·깊은 동시 운용, 깊이별 명목 가중).
측정 수정: 일별 자본곡선 + MAE 기준 최악 곡선으로 MDD를 보수적으로 측정.
사용: python research/z03b.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from lib_dip import bars
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
EPS = 0.0002; MMR = 0.004
TE = pd.Timestamp("2024-01-01", tz="UTC"); REC = pd.Timestamp("2025-01-01", tz="UTC")
SYM = ("BTCUSDT", "ETHUSDT"); BARS = (240, 480, 720, 1440)
D = {}
for s in SYM:
    for b in BARS:
        d = bars(s, b)
        d["rv"] = (pd.Series(np.log(d["bc"])).diff().rolling(20).std()).shift(1).values
        D[(s, b)] = d


@numba.njit(cache=True)
def ev1(o, h, l, c, B, x, kg, rv, fm, ft, eps):
    """단일 깊이 x. 지정가 매수 1회 → 목표가(메이커) 또는 종가(테이커) 청산."""
    n = len(c)//B
    ei = np.full(n, -1, np.int64); ex = np.full(n, -1, np.int64)
    rr = np.zeros(n); mm = np.zeros(n)
    k = 0
    for b in range(n):
        if np.isnan(x[b]) or x[b] <= 0: continue
        s = b*B; e = s+B; O = o[s]; C = c[e-1]
        T0 = O*(1-x[b]); g = rv[b]*kg if kg > 0 else 0.0
        f = -1; lo_ = 1e18; hit = -1
        for t in range(s, e):
            if f < 0:
                if l[t] <= T0*(1-eps): f = t
                else: continue
            if l[t] < lo_: lo_ = l[t]
            if g > 0.0 and t > f:
                TP = T0*(1.0+g)
                if h[t] >= TP*(1+eps): hit = t; break
        if f < 0: continue
        if hit < 0: hit = e-1; px = C; fee = fm+ft
        else: px = T0*(1.0+g); fee = fm+fm
        ei[k] = f; ex[k] = hit
        rr[k] = (px/T0-1.0) - fee/1e4
        mm[k] = 1.0 - lo_/T0
        k += 1
    return ei[:k], ex[:k], rr[:k], mm[:k]


def build(depths, kg, fm=2.0, ft=5.0, eps=EPS):
    """depths: {k: 명목가중}. 각 (심볼,봉,k)가 독립 다리."""
    E = []
    for s in SYM:
        for b in BARS:
            d = D[(s, b)]; rv = d["rv"]; B = d["B"]; ts = d["idx"]
            for k, wgt in depths.items():
                a_, b_, r_, m_ = ev1(d["o"], d["h"], d["l"], d["c"], B, np.asarray(rv*k, float),
                                     float(kg), np.asarray(rv, float), fm, ft, eps)
                key = (s, b, k)
                for a, bb, r, m in zip(a_, b_, r_, m_):
                    E.append((int(a), int(bb), float(r), float(m), ts[int(a)], key, float(rv[int(a)//B]), wgt))
    E.sort(key=lambda z: z[0])
    return E


def sim(E, w, Nmax, lo=0.2, hi=3.0, warm=30, mmr=MMR):
    cap = 1.0; open_ = []; rvh = {}; liq = 0
    ts_l = []; eq_l = []; floor_l = []
    for a, b, r, m, ts, key, rvb, wgt in E:
        for p in [p for p in open_ if p[0] <= a]:
            cap += p[1]*p[2]; open_.remove(p)
        if cap <= 1e-9: break
        if not np.isfinite(rvb) or rvb <= 0: continue
        rh = rvh.setdefault(key, [])
        f = float(np.clip(np.median(rh)/rvb, lo, hi)) if len(rh) >= warm else lo
        rh.append(rvb)
        notional = min(cap*w*wgt*f, max(cap*Nmax - sum(p[1] for p in open_), 0.0))
        if notional <= 1e-9: continue
        if notional*m + sum(p[1]*p[3] for p in open_) >= cap*(1-mmr):
            cap = 0.0; liq += 1; break
        open_.append((b, notional, r, m, key))
        ts_l.append(ts); eq_l.append(cap + sum(p[1]*p[2] for p in open_)*0)
        floor_l.append(cap - sum(p[1]*p[3] for p in open_))
    for p in open_: cap += p[1]*p[2]
    if not ts_l: return None, None, liq
    ts_l.append(E[-1][4]); eq_l.append(cap); floor_l.append(cap)
    eq = pd.Series(eq_l, index=pd.DatetimeIndex(ts_l)).groupby(level=0).last()
    fl = pd.Series(floor_l, index=pd.DatetimeIndex(ts_l)).groupby(level=0).min()
    dly = eq.resample("D").last().ffill()
    fld = pd.concat([eq, fl]).groupby(level=0).min().resample("D").min().ffill()
    return dly, fld, liq


def rep(dly, fld, liq, label, quiet=False):
    if dly is None or dly.iloc[-1] <= 1e-6:
        if not quiet: P(f"  {label:30s} 파산 (청산 {liq})")
        return None
    yrs = (dly.index[-1]-dly.index[0]).days/365
    cg = (dly.iloc[-1]**(1/yrs)-1)*100
    mdd = float((dly/dly.cummax()-1).min())*100
    mddw = float((fld/dly.cummax()-1).min())*100
    def seg(t0):
        s = dly[dly.index >= t0]
        if len(s) < 60: return np.nan
        return ((s.iloc[-1]/s.iloc[0])**(365/(s.index[-1]-s.index[0]).days)-1)*100
    if not quiet:
        P(f"  {label:30s} 연 {cg:+8.0f}% MDD {mdd:6.1f}%(최악 {mddw:6.1f}%) | 24~ {seg(TE):+8.0f}% | 25~ {seg(REC):+8.0f}% | 청산 {liq}")
    return cg, mdd, mddw, seg(TE), seg(REC), dly, fld


P("[1] 깊이 구성 비교 (kg=5, w=5, Nmax=8)")
CFG = {
    "얕은만 k=5":            {5.0: 1.0},
    "중간만 k=8":            {8.0: 1.0},
    "깊은만 k=12":           {12.0: 1.0},
    "균등 5·8·12":          {5.0: 1.0, 8.0: 1.0, 12.0: 1.0},
    "깊이가중 5·8·12 (1:2:4)": {5.0: 1.0, 8.0: 2.0, 12.0: 4.0},
    "깊이가중 5·8·12 (1:3:9)": {5.0: 1.0, 8.0: 3.0, 12.0: 9.0},
    "깊은쪽 8·12 (1:3)":     {8.0: 1.0, 12.0: 3.0},
    "넓게 4·6·8·10·13":      {4.0: 0.5, 6.0: 1.0, 8.0: 2.0, 10.0: 3.0, 13.0: 5.0},
}
res = {}
for nm, dep in CFG.items():
    E = build(dep, 5.0)
    dly, fld, liq = sim(E, 5.0, 8.0)
    r = rep(dly, fld, liq, f"{nm} (n={len(E)})")
    if r: res[nm] = (r, dep)

P("\n[2] 상위 구성의 명목 스윕")
best = None
for nm in ("깊이가중 5·8·12 (1:3:9)", "깊이가중 5·8·12 (1:2:4)", "넓게 4·6·8·10·13", "깊은쪽 8·12 (1:3)"):
    if nm not in res: continue
    dep = res[nm][1]; E = build(dep, 5.0)
    P(f"  -- {nm} --")
    for w in (2.0, 3.0, 5.0, 8.0, 12.0):
        for Nm in (5.0, 8.0, 12.0, 20.0):
            if Nm < w: continue
            dly, fld, liq = sim(E, w, Nm)
            r = rep(dly, fld, liq, f"   w={w:.0f} Nmax={Nm:.0f}", quiet=(liq > 0 or dly is None))
            if r and (best is None or r[0] > best[0]): best = (*r, nm, w, Nm)

P("\n[3] 목표가 kg 재확인 (최적 구성)")
if best:
    nm, w, Nm = best[7], best[8], best[9]
    dep = res[nm][1]
    for kg in (0.0, 2.0, 3.0, 5.0, 8.0):
        E = build(dep, kg)
        dly, fld, liq = sim(E, w, Nm)
        rep(dly, fld, liq, f"  kg={kg:.0f}")

if best:
    cg, mdd, mddw, t24, t25, dly, fld, nm, w, Nm = best
    P(f"\n★ z0.3 최적: {nm}, w={w:.0f}, Nmax={Nm:.0f}")
    P(f"   전체 연 {cg:+.0f}% | MDD {mdd:.1f}% (최악 {mddw:.1f}%) | 2024~ {t24:+.0f}% | 2025~ {t25:+.0f}%")
    yr = dly.resample("YE").last()/dly.resample("YE").first()-1
    mon = dly.resample("ME").last()/dly.resample("ME").first()-1
    P("   연도별: " + " | ".join(f"{i.year} {v*100:+.0f}%" for i, v in yr.items()))
    P("   최근 12개월: " + " | ".join(f"{i.strftime('%y-%m')} {v*100:+.1f}%" for i, v in mon.tail(12).items()))
    l12 = mon.tail(12)
    P(f"   최근 12개월 누적 {(np.prod(1+l12)-1)*100:+.1f}% 양수 {int((l12>0).sum())}/12 | 전체 월승률 {float((mon>0).mean()):.3f} (n={len(mon)})")
