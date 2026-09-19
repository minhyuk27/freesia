"""저변동 역행 되돌림 공통 엔진 — 룩어헤드 완전 제거판.
규칙: 봉 b 진입 결정에 쓰는 모든 정보는 봉 b-1 종료 시점까지의 것만 사용.
"""
import numpy as np, pandas as pd, numba
from freesia import store

B_DEF = 240


def bars(sym, B):
    k = store.load("klines_1m", sym).set_index("ts")
    o, h, l, c, v = k.open.values, k.high.values, k.low.values, k.close.values, k.quote_volume.values
    n = len(c) // B
    bo = o[::B][:n]; bc = c[B - 1::B][:n]
    bh = np.array([h[i * B:(i + 1) * B].max() for i in range(n)])
    bl = np.array([l[i * B:(i + 1) * B].min() for i in range(n)])
    bv = np.array([v[i * B:(i + 1) * B].sum() for i in range(n)])
    bi = k.index[::B][:n]
    return dict(o=o, h=h, l=l, c=c, v=v, idx=k.index, n=n, bo=bo, bc=bc, bh=bh, bl=bl, bv=bv, bi=bi, B=B)


def lagged(series, shift=1):
    """봉 b에서 쓸 수 있도록 1봉 지연. 모든 조건 계산의 마지막에 반드시 적용."""
    return pd.Series(series).shift(shift)


@numba.njit(cache=True)
def _run(o, h, l, c, B, xarr, cond, buf, fee_bp):
    n = len(c) // B
    ret = np.full(n, np.nan); mae = np.full(n, np.nan); xs = np.full(n, np.nan)
    tin = np.zeros(n, np.int32); hold = np.zeros(n, np.int32)
    for b in range(n):
        if not cond[b] or np.isnan(xarr[b]) or xarr[b] <= 0:
            continue
        s = b * B; e = s + B
        O = o[s]; x = xarr[b]
        trig = O * (1.0 - x); fill = trig * (1.0 - buf)
        j = -1
        for t in range(s, e):
            if l[t] <= fill:
                j = t; break
        if j < 0:
            continue
        lo_ = trig
        for t in range(j, e):
            if l[t] < lo_:
                lo_ = l[t]
        ret[b] = (c[e - 1] / trig - 1.0) * 1e4 - fee_bp
        mae[b] = (1.0 - lo_ / trig) * 1e4
        xs[b] = x * 1e4; tin[b] = j - s; hold[b] = e - 1 - j
    return ret, mae, xs, tin, hold


def run(D, xarr, cond, buf=0.0, fee_bp=7.0):
    return _run(D["o"], D["h"], D["l"], D["c"], D["B"],
                np.asarray(xarr, float), np.asarray(cond, np.bool_), buf, fee_bp)


def summarize(ret, bi, TE, label=""):
    ok = ~np.isnan(ret)
    if ok.sum() < 20:
        return None
    ts = bi[ok]; r = ret[ok]
    tr = ts < TE; te = ~tr
    out = {"label": label, "n": int(ok.sum())}
    for nm, m in (("tr", tr), ("te", te)):
        if m.sum() < 15:
            out[nm] = None; continue
        x = r[m]
        out[nm] = dict(mean=x.mean(), med=float(np.median(x)), win=float((x > 0).mean()),
                       n=int(m.sum()), t=x.mean() / x.std() * np.sqrt(len(x)) if x.std() > 0 else 0,
                       wk=m.sum() / max((ts[m][-1] - ts[m][0]).days / 7, 1))
    return out
