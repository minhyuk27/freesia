"""z0.2 확장·검증 — 다리 확장(1h/12h/1d), 숏 방향, 워크포워드(정확 자본배분), 가중치.
사용: python research/z02_extend.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from lib_dip import bars
TE = pd.Timestamp("2024-01-01", tz="UTC")
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
ALL_LEGS = [(s, b) for s in ("BTCUSDT", "ETHUSDT") for b in (60, 240, 480, 720, 1440)]
D = {}
for key in ALL_LEGS:
    d = bars(key[0], key[1]); s = pd.Series
    d["rv"] = (s(np.log(d["bc"])).diff().rolling(20).std()).shift(1).values
    D[key] = d


@numba.njit(cache=True)
def leg3(o, h, l, c, B, x1, x2, x3, fee, N, side):
    n = len(c) // B
    pnl = np.zeros(n); mae = np.zeros(n); used = np.zeros(n)
    for b in range(n):
        if np.isnan(x1[b]) or x1[b] <= 0:
            continue
        s = b * B; e = s + B
        O = o[s]; C = c[e - 1]
        if side == 1:
            t1 = O * (1 - x1[b]); t2 = O * (1 - x2[b]); t3 = O * (1 - x3[b])
        else:
            t1 = O * (1 + x1[b]); t2 = O * (1 + x2[b]); t3 = O * (1 + x3[b])
        f1 = -1; f2 = -1; f3 = -1
        for t in range(s, e):
            if side == 1:
                if f1 < 0 and l[t] <= t1:
                    f1 = t
                if f2 < 0 and l[t] <= t2:
                    f2 = t
                if f3 < 0 and l[t] <= t3:
                    f3 = t
            else:
                if f1 < 0 and h[t] >= t1:
                    f1 = t
                if f2 < 0 and h[t] >= t2:
                    f2 = t
                if f3 < 0 and h[t] >= t3:
                    f3 = t
        if f1 < 0:
            continue
        tot = 0.0; cnt = 0.0; wavg = 0.0
        for tt, ff in ((t1, f1), (t2, f2), (t3, f3)):
            if ff >= 0:
                rr = (C / tt - 1.0) if side == 1 else (1.0 - C / tt)
                tot += (rr - fee / 1e4) / 3.0; cnt += 1.0; wavg += tt
        avg = wavg / cnt
        ex = t1
        for t in range(f1, e):
            if side == 1:
                if l[t] < ex:
                    ex = l[t]
            else:
                if h[t] > ex:
                    ex = h[t]
        adverse = (1.0 - ex / avg) if side == 1 else (ex / avg - 1.0)
        pnl[b] = tot * N
        mae[b] = adverse * (cnt / 3.0) * N
        used[b] = cnt / 3.0
    return pnl, mae, used


def port(legs, split, N, fee=7.0, side=1, end=None):
    k1, k2, k3 = split
    w = 1.0 / len(legs)
    eqs = []
    for key in legs:
        d = D[key]
        p, m, u = leg3(d["o"], d["h"], d["l"], d["c"], d["B"],
                       np.asarray(d["rv"] * k1, float), np.asarray(d["rv"] * k2, float),
                       np.asarray(d["rv"] * k3, float), fee, float(N), side)
        nz = u > 0
        ser = pd.Series(p[nz], index=d["bi"][nz])
        mse = pd.Series(m[nz], index=d["bi"][nz])
        ser = ser.where(mse < 1.0, -1.0)
        if end is not None:
            ser = ser[ser.index < end]
        eqs.append(ser)
    idx = sorted(set().union(*[s.index for s in eqs]))
    cur = np.full(len(eqs), w)
    hist = []
    for t in idx:
        for i, s_ in enumerate(eqs):
            if t in s_.index:
                v = s_.loc[t]
                v = float(v.iloc[0]) if hasattr(v, "iloc") and not np.isscalar(v) else float(v)
                cur[i] = max(cur[i] * (1 + v), 0.0)
        hist.append(cur.sum())
    return pd.Series(hist, index=pd.DatetimeIndex(idx))


def rep(eq, label):
    if len(eq) < 60 or eq.iloc[-1] <= 1e-6:
        P(f"  {label:34s} 파산 또는 표본 부족"); return None
    yrs = (eq.index[-1] - eq.index[0]).days / 365
    te = eq[eq.index >= TE]
    r = dict(full=(eq.iloc[-1] ** (1 / yrs) - 1) * 100, fmdd=float((eq / eq.cummax() - 1).min()) * 100,
             n=len(eq), wk=len(eq) / ((eq.index[-1] - eq.index[0]).days / 7))
    if len(te) > 30:
        y2 = (te.index[-1] - te.index[0]).days / 365
        r["test"] = ((te.iloc[-1] / te.iloc[0]) ** (1 / y2) - 1) * 100
        r["tmdd"] = float((te / te.cummax() - 1).min()) * 100
    P(f"  {label:34s} 전체 연 {r['full']:+7.1f}% MDD {r['fmdd']:6.1f}% | 테스트 연 {r.get('test', float('nan')):+7.1f}% MDD {r.get('tmdd', float('nan')):6.1f}% | 진입 {r['n']:5d} 주{r['wk']:5.2f}")
    return r


SP = (4.0, 5.5, 7.0)
P("[1] 다리 구성 확장 (분할 4.0/5.5/7.0, 명목 3배, 롱)")
SETS = {
    "4h+8h (기존)": [(s, b) for s in ("BTCUSDT", "ETHUSDT") for b in (240, 480)],
    "1h+4h+8h": [(s, b) for s in ("BTCUSDT", "ETHUSDT") for b in (60, 240, 480)],
    "4h+8h+12h": [(s, b) for s in ("BTCUSDT", "ETHUSDT") for b in (240, 480, 720)],
    "4h+8h+12h+1d": [(s, b) for s in ("BTCUSDT", "ETHUSDT") for b in (240, 480, 720, 1440)],
    "전체 5봉": ALL_LEGS,
    "ETH만 4h+8h+12h": [("ETHUSDT", b) for b in (240, 480, 720)],
    "BTC만 4h+8h+12h": [("BTCUSDT", b) for b in (240, 480, 720)],
}
for nm, lg in SETS.items():
    rep(port(lg, SP, 3), nm)

P("\n[2] 숏 방향 (상승 후 매도) — 같은 구조")
for nm, lg in list(SETS.items())[:5]:
    rep(port(lg, SP, 3, side=-1), nm + " 숏")

P("\n[3] 롱+숏 동시 (각 절반 자본)")
for nm, lg in list(SETS.items())[:5]:
    eqL = port(lg, SP, 3, side=1); eqS = port(lg, SP, 3, side=-1)
    idx = eqL.index.union(eqS.index)
    comb = (eqL.reindex(idx).ffill().fillna(1.0) + eqS.reindex(idx).ffill().fillna(1.0)) / 2
    rep(comb, nm + " 롱+숏")

P("\n[4] 명목 정밀 탐색 (전체 5봉, 롱)")
for N in (1, 2, 3, 4, 5, 6):
    rep(port(ALL_LEGS, SP, N), f"전체5봉 명목 {N}배")

P("\n[5] 워크포워드 (정확 자본배분) — 매년 직전까지로 (분할,명목) 선택")
CAND = [((3.0, 4.5, 6.0), 2), ((4.0, 5.5, 7.0), 2), ((4.0, 5.5, 7.0), 3),
        ((5.0, 6.5, 8.0), 3), ((5.0, 6.5, 8.0), 4), ((2.5, 4.0, 5.5), 2)]
LG = [(s, b) for s in ("BTCUSDT", "ETHUSDT") for b in (240, 480, 720)]
rows = []
for y in range(2022, 2027):
    sD = pd.Timestamp(f"{y}-01-01", tz="UTC"); eD = pd.Timestamp(f"{y+1}-01-01", tz="UTC")
    best = None
    for sp, N in CAND:
        e_ = port(LG, sp, N, end=sD)
        if len(e_) < 100 or e_.iloc[-1] <= 1e-6:
            continue
        yrs = (e_.index[-1] - e_.index[0]).days / 365
        g = (e_.iloc[-1] ** (1 / yrs) - 1)
        if best is None or g > best[0]:
            best = (g, sp, N)
    if best is None:
        continue
    _, sp, N = best
    e_ = port(LG, sp, N)
    seg = e_[(e_.index >= sD) & (e_.index < eD)]
    if len(seg) < 15:
        continue
    rows.append((y, str(sp), N, (seg.iloc[-1] / seg.iloc[0] - 1) * 100,
                 float((seg / seg.cummax() - 1).min()) * 100, len(seg)))
P("  " + pd.DataFrame(rows, columns=["연", "선택 분할", "명목", "그해%", "그해MDD%", "진입"]).round(1).to_string(index=False).replace("\n", "\n  "))
