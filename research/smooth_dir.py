"""주름 펴기(인과적 평활) 기반 방향 예측 — 3분~1시간.
모든 필터는 t 시점까지의 정보만 사용(인과적). 양방향 필터는 룩어헤드이므로 금지.
사용: python research/smooth_dir.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
SYM = ("BTCUSDT", "ETHUSDT")
D = {}
for s in SYM:
    k = store.load("klines_1m", s).set_index("ts")
    D[s] = dict(c=k.close.values.astype(np.float64), o=k.open.values.astype(np.float64),
                h=k.high.values.astype(np.float64), l=k.low.values.astype(np.float64),
                qv=k.quote_volume.values.astype(np.float64), idx=k.index)
    P(f"{s}: {len(k):,}분")

HOR = (1, 3, 5, 15, 30, 60)

P("\n[0] 손익분기 정확도 — 구간별 평균 절대이동과 필요 정확도")
P(f"  {'종목':6s} {'구간':>5s} {'평균|이동|':>10s} {'메이커4bp':>10s} {'메+테7bp':>10s} {'테이커10bp':>10s}")
for s in SYM:
    c = D[s]["c"]
    for h in HOR:
        r = pd.Series(c).pct_change(h).dropna()
        m = r.abs().mean()*1e4
        P(f"  {s[:3]:6s} {h:4d}분 {m:9.1f}bp " + " ".join(f"{(0.5+cost/(2*m))*100:9.1f}%" for cost in (4, 7, 10)))


# ── 인과적 필터들 ────────────────────────────────────────────────
@numba.njit(cache=True)
def ema(x, n):
    a = 2.0/(n+1.0); out = np.empty(len(x)); out[0] = x[0]
    for i in range(1, len(x)): out[i] = a*x[i] + (1-a)*out[i-1]
    return out


@numba.njit(cache=True)
def wma(x, n):
    out = np.full(len(x), np.nan)
    w = np.arange(1.0, n+1.0); sw = w.sum()
    for i in range(n-1, len(x)):
        s = 0.0
        for j in range(n): s += x[i-n+1+j]*w[j]
        out[i] = s/sw
    return out


@numba.njit(cache=True)
def kama(x, n, fast, slow):
    """Kaufman 적응형 이동평균 — 효율비가 높으면 빠르게, 낮으면 느리게"""
    out = np.full(len(x), np.nan); out[n] = x[n]
    fs = 2.0/(fast+1.0); ss = 2.0/(slow+1.0)
    for i in range(n+1, len(x)):
        chg = abs(x[i]-x[i-n])
        vol = 0.0
        for j in range(n): vol += abs(x[i-j]-x[i-j-1])
        er = chg/vol if vol > 0 else 0.0
        sc = (er*(fs-ss)+ss)**2
        out[i] = out[i-1] + sc*(x[i]-out[i-1])
    return out


@numba.njit(cache=True)
def kalman(x, q, r):
    """국소 수준 칼만 필터 (인과적). q=과정분산, r=관측분산"""
    n = len(x); out = np.empty(n); p = 1.0
    out[0] = x[0]
    for i in range(1, n):
        pp = p + q
        kg = pp/(pp+r)
        out[i] = out[i-1] + kg*(x[i]-out[i-1])
        p = (1-kg)*pp
    return out


@numba.njit(cache=True)
def sg_causal(x, n, deg):
    """편측 다항 적합 — 직전 n개로 deg차 다항 적합 후 끝점 값과 기울기. 인과적."""
    val = np.full(len(x), np.nan); slp = np.full(len(x), np.nan)
    t = np.empty(n)
    for j in range(n): t[j] = float(j)
    for i in range(n-1, len(x)):
        # 최소제곱 (deg=1 또는 2)
        sx = 0.0; sy = 0.0; sxx = 0.0; sxy = 0.0
        for j in range(n):
            sx += t[j]; sy += x[i-n+1+j]; sxx += t[j]*t[j]; sxy += t[j]*x[i-n+1+j]
        den = n*sxx - sx*sx
        if den == 0: continue
        b = (n*sxy - sx*sy)/den
        a = (sy - b*sx)/n
        val[i] = a + b*(n-1)
        slp[i] = b
    return val, slp


@numba.njit(cache=True)
def med_filt(x, n):
    out = np.full(len(x), np.nan)
    buf = np.empty(n)
    for i in range(n-1, len(x)):
        for j in range(n): buf[j] = x[i-n+1+j]
        out[i] = np.median(buf)
    return out


def fw(c, h): return pd.Series(c).shift(-h).values/c - 1.0


def evaluate(name, sig, c, horizons=HOR, nq=10):
    """신호 분위별 이후 수익(bp)과 방향 정확도"""
    out = []
    v = pd.Series(sig)
    q = v.rank(pct=True).values
    for h in horizons:
        F = fw(c, h)
        ok = np.isfinite(F) & np.isfinite(sig)
        hi = ok & (q >= 0.9); lo = ok & (q <= 0.1)
        if hi.sum() < 1000 or lo.sum() < 1000: continue
        a = F[hi].mean()*1e4; b = F[lo].mean()*1e4
        acc_hi = (F[hi] > 0).mean(); acc_lo = (F[lo] < 0).mean()
        out.append((h, a, b, a-b, acc_hi, acc_lo, (acc_hi+acc_lo)/2))
    return out


P("\n" + "="*112)
P("[1] 필터별 기울기 신호 — 상위10%/하위10% 이후 수익(bp)과 방향 정확도")
P("="*112)
for s in SYM:
    c = D[s]["c"]
    P(f"\n  ══ {s} ══")
    FILT = []
    for n in (3, 5, 10, 15, 30, 60):
        e = ema(c, n)
        FILT.append((f"EMA{n} 기울기", np.diff(e, prepend=e[0])))
        v, sl = sg_causal(c, n, 1)
        FILT.append((f"편측선형적합{n} 기울기", sl))
        FILT.append((f"가격-EMA{n} 괴리", c - e))
    for n in (10, 20, 30):
        km = kama(c, n, 2, 30)
        FILT.append((f"KAMA{n} 기울기", np.diff(km, prepend=km[0])))
    for q_, r_ in ((0.001, 1.0), (0.01, 1.0), (0.1, 1.0)):
        kf = kalman(c, q_, r_)
        FILT.append((f"칼만 q={q_} 기울기", np.diff(kf, prepend=kf[0])))
    for n in (5, 15, 30):
        mf = med_filt(c, n)
        FILT.append((f"중앙값{n} 대비 가격", c - np.nan_to_num(mf, nan=c[0])))
    # Hull MA (인과적): WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    for n in (10, 30):
        h1 = wma(c, max(n//2, 2)); h2 = wma(c, n)
        raw = 2*np.nan_to_num(h1) - np.nan_to_num(h2)
        hull = wma(raw, max(int(np.sqrt(n)), 2))
        FILT.append((f"HullMA{n} 기울기", np.diff(np.nan_to_num(hull), prepend=0.0)))

    P(f"    {'신호':>24s} {'구간':>4s} {'상위10%':>9s} {'하위10%':>9s} {'차이':>8s} {'정확도':>7s}")
    best = []
    for nm, sig in FILT:
        res = evaluate(nm, sig, c)
        for h, a, b, d, ah, al, acc in res:
            if h in (3, 15, 30, 60):
                best.append((abs(d), nm, h, a, b, d, acc))
    best.sort(reverse=True)
    for _, nm, h, a, b, d, acc in best[:14]:
        P(f"    {nm:>24s} {h:3d}분 {a:+8.2f} {b:+8.2f} {d:+7.2f} {acc*100:6.2f}%")
