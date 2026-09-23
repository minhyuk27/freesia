"""z0.7 — 평활 괴리 반전 (단독 전략). 가격이 인과적 EMA에서 극단적으로 벌어지면 반전 매수.
임계는 확장창 분위수 → 룩어헤드 없음. 진입 방식 두 가지(시장가/지정가) 비교.
사용: python research/z0_7_dev.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from smooth_dir import ema, D, SYM
import exit_tier as X
from absfloor import met
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
TE = pd.Timestamp("2024-01-01", tz="UTC"); REC = pd.Timestamp("2025-01-01", tz="UTC")

for s in SYM:
    d = D[s]
    c = d["c"]
    for n in (10, 15, 30):
        d[f"dev{n}"] = (c - ema(c, n))/c
    lr = pd.Series(np.log(c)).diff()
    d["rvr"] = (lr.rolling(30).std()/lr.rolling(1440).std()).values
    d["qvz"] = (pd.Series(d["qv"]).rolling(30).mean()/pd.Series(d["qv"]).rolling(1440).mean()).values

_TH = {}
def eqth(v, q, key, warm=200000, step=10080):
    ck = (key, q)
    if ck in _TH: return _TH[ck]
    n = len(v); th = np.full(n, np.nan)
    for i in range(warm, n, step):
        th[i:i+step] = np.nanquantile(v[:i], q)
    _TH[ck] = th
    return th


@numba.njit(cache=True)
def run(o, h, l, c, dev, th, cond, use_cond, hold, cool, mkt, kx, rv, W, fm, ft, eps):
    """mkt=1: t+1 시가 시장가 진입. mkt=0: c[t]*(1-rv*kx) 지정가, W분 유효."""
    n = len(c)
    ei = np.full(n, -1, np.int64); ex = np.full(n, -1, np.int64)
    rr = np.zeros(n); mm = np.zeros(n); k = 0; last = -10**9
    for t in range(n-1):
        if np.isnan(th[t]) or np.isnan(dev[t]): continue
        if dev[t] > th[t]: continue
        if use_cond == 1 and (np.isnan(cond[t]) or cond[t] < 1.2): continue
        if t - last < cool: continue
        last = t
        if mkt == 1:
            f = t+1; px = o[f]; fee = ft + ft
        else:
            if np.isnan(rv[t]) or rv[t] <= 0: continue
            T = c[t]*(1.0 - rv[t]*kx); f = -1
            lim = t+1+W
            if lim >= n: break
            for u in range(t+1, lim):
                if l[u] <= T*(1.0-eps): f = u; break
            if f < 0: continue
            px = T; fee = fm + ft
        e = f + hold
        if e >= n: break
        lo_ = 1e18
        for u in range(f, e+1):
            if l[u] < lo_: lo_ = l[u]
        ei[k] = f; ex[k] = e
        rr[k] = (c[e]/px - 1.0) - fee/1e4
        mm[k] = 1.0 - lo_/px
        k += 1
    return ei[:k], ex[:k], rr[:k], mm[:k]


def build(q, dn, hold, cool, mkt=1, kx=1.0, W=30, cond=False, fm=2.0, ft=5.0, eps=0.0002, syms=SYM):
    E = []
    for s in syms:
        d = D[s]
        dev = d[f"dev{dn}"]
        th = eqth(dev, q, key=f"d{dn}{s}")
        lr = pd.Series(np.log(d["c"])).diff()
        rv = lr.rolling(300).std().shift(1).values*np.sqrt(5)
        cd = np.nan_to_num(d["rvr"], nan=0.0)
        a_, b_, r_, m_ = run(d["o"], d["h"], d["l"], d["c"], dev, th, cd, 1 if cond else 0,
                             hold, cool, 1 if mkt else 0, kx, np.asarray(rv, float), W, fm, ft, eps)
        for a, bb, r, m in zip(a_, b_, r_, m_):
            E.append((int(a), int(bb), float(r), float(m), d["idx"][int(a)], (s, dn, hold), 0.01, 1.0))
    E.sort(key=lambda z: z[0])
    return E


def tiers(E, lab, quiet=False):
    r = np.array([x[2] for x in E])
    if len(r) < 50: return None
    out = f"  {lab:40s} n={len(r):5d}({len(r)/6.7:5.0f}/년) {r.mean()*1e4:+6.1f}bp 승률{(r>0).mean():.3f} |"
    res = {}
    for w, Nm, tag in ((1.0, 1.0, "무레버"), (2.0, 3.0, "저레버"), (3.0, 5.0, "중간"), (5.0, 8.0, "공격")):
        eq, fl, liq = X.sim(E, w, Nm); m = met(eq, fl, liq)
        res[tag] = m
        out += f" {tag} {'파산' if m is None else f'{m[0]:+6.1f}%'}"
    if not quiet: P(out)
    return res


P("[1] 시장가 진입 — 괴리 분위 × EMA길이 × 보유")
for q in (0.001, 0.005, 0.01):
    for dn in (10, 15, 30):
        for hold in (30, 60, 120):
            tiers(build(q, dn, hold, 30, mkt=1), f"q={q} EMA{dn} 보유{hold}분 시장가")

P("\n[2] 지정가 대기 진입 (더 아래 rv5×kx, 유효 30분)")
for q in (0.005, 0.01, 0.02):
    for dn in (15,):
        for kx in (0.5, 1.0, 2.0):
            for hold in (60, 120, 240):
                tiers(build(q, dn, hold, 30, mkt=0, kx=kx), f"q={q} EMA{dn} k={kx} 보유{hold}분 지정가")

P("\n[3] 고변동 조건 추가 (rv비율 > 1.2)")
for q in (0.005, 0.01):
    for hold in (60, 120):
        for mkt in (1, 0):
            tiers(build(q, 15, hold, 30, mkt=mkt, kx=1.0, cond=True),
                  f"q={q} 보유{hold}분 {'시장가' if mkt else '지정가'} +고변동")

P("\n[4] 종목별 분해 (최적 후보)")
for s in SYM:
    for q, dn, hold, mkt, kx in ((0.005, 15, 120, 0, 1.0), (0.001, 15, 60, 1, 1.0)):
        tiers(build(q, dn, hold, 30, mkt=mkt, kx=kx, syms=(s,)),
              f"{s[:3]} q={q} 보유{hold}분 {'시장가' if mkt else '지정가'}")
