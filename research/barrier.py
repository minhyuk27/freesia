"""±X% 배리어 — 어디에 먼저 닿는가. 기저율·시간·예측가능성·손익분기.
사용: python research/barrier.py"""
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
    D[s] = dict(o=k.open.values, h=k.high.values, l=k.low.values, c=k.close.values,
                qv=k.quote_volume.values, tb=k.taker_buy_quote_volume.values, idx=k.index)
    P(f"{s}: {len(k):,}분")


@numba.njit(cache=True)
def first_touch(h, l, c, up, dn, maxmin, step):
    """각 시점에서 +up / -dn 중 어디에 먼저 닿는지. 1=위, -1=아래, 0=미도달"""
    n = len(c)
    out = np.zeros(n//step, np.int8)
    tt = np.zeros(n//step, np.int32)
    k = 0
    for i in range(0, n-maxmin, step):
        p = c[i]
        U = p*(1.0+up); Dn = p*(1.0-dn)
        r = 0; t = 0
        for j in range(i+1, i+maxmin+1):
            hi = h[j] >= U; lo = l[j] <= Dn
            if hi and lo:
                r = -1; t = j-i; break        # 같은 분이면 보수적으로 아래 먼저
            if hi:
                r = 1; t = j-i; break
            if lo:
                r = -1; t = j-i; break
        out[k] = r; tt[k] = t; k += 1
    return out[:k], tt[:k]


P("\n" + "="*110)
P("[1] 기저율 — ±X% 중 위쪽에 먼저 닿을 확률과 소요 시간")
P("="*110)
P(f"  {'종목':6s} {'배리어':>7s} {'최대시간':>8s} {'판정률':>7s} {'위먼저':>7s} {'중앙시간':>9s} {'평균시간':>9s} {'손익분기(테10bp)':>16s}")
BASE = {}
for s in SYM:
    d = D[s]
    for x in (0.002, 0.005, 0.01, 0.02, 0.05):
        for mx, mxl in ((60, "1시간"), (1440, "1일"), (10080, "7일")):
            if x >= 0.02 and mx < 1440: continue
            r, t = first_touch(d["h"], d["l"], d["c"], x, x, mx, 15)
            dec = r != 0
            if dec.sum() < 100: continue
            up = (r[dec] == 1).mean()
            be = 0.5 + 0.001/(2*x)
            BASE[(s, x, mx)] = (r, t, up)
            P(f"  {s[:3]:6s} ±{x*100:5.2f}% {mxl:>8s} {dec.mean()*100:6.1f}% {up*100:6.2f}% "
              f"{np.median(t[dec]):8.0f}분 {t[dec].mean():8.0f}분 {be*100:15.1f}%")

P("\n  → 손익분기: p* = 0.5 + 비용/(2×배리어). ±1%·테이커10bp면 55.0%, 메이커4bp면 52.0%")

P("\n" + "="*110)
P("[2] 예측 가능성 — 최근 시장 상태로 '위 먼저'를 예측할 수 있나 (±1%, 1일)")
P("="*110)
for s in SYM:
    d = D[s]
    key = (s, 0.01, 1440)
    if key not in BASE: continue
    r, t, base_up = BASE[key]
    idx = d["idx"][::15][:len(r)]
    c = d["c"][::15][:len(r)]
    qv = pd.Series(d["qv"]).rolling(60).sum().values[::15][:len(r)]
    lr = pd.Series(np.log(d["c"])).diff()
    P(f"\n  ══ {s} (기저 위먼저 {base_up*100:.2f}%) ══")
    feats = {}
    for w in (15, 60, 240, 1440, 4320):
        feats[f"수익{w}분"] = pd.Series(d["c"]).pct_change(w).values[::15][:len(r)]
    feats["rv60/1440"] = (lr.rolling(60).std()/lr.rolling(1440).std()).values[::15][:len(r)]
    feats["거래량비"] = (pd.Series(d["qv"]).rolling(60).mean()/pd.Series(d["qv"]).rolling(1440).mean()).values[::15][:len(r)]
    feats["OFI60"] = (2*pd.Series(d["tb"]).rolling(60).sum()/pd.Series(d["qv"]).rolling(60).sum()-1).values[::15][:len(r)]
    hh = pd.Series(d["h"]).rolling(1440).max().values[::15][:len(r)]
    ll = pd.Series(d["l"]).rolling(1440).min().values[::15][:len(r)]
    feats["일중위치"] = (c-ll)/np.maximum(hh-ll, 1e-9)
    dec = r != 0
    P(f"    {'특징':>14s} {'하위10% 위먼저':>14s} {'상위10% 위먼저':>14s} {'차이':>8s} {'최대|초과|':>10s}")
    for nm, v in feats.items():
        ok = dec & np.isfinite(v)
        if ok.sum() < 1000: continue
        q = pd.Series(np.where(ok, v, np.nan)).rank(pct=True).values
        lo = ok & (q <= 0.10); hi = ok & (q >= 0.90)
        a = (r[lo] == 1).mean()*100; b = (r[hi] == 1).mean()*100
        exc = max(abs(a-55.0), abs(b-55.0))
        P(f"    {nm:>14s} {a:13.2f}% {b:13.2f}% {b-a:+7.2f} {exc:9.1f}%p")

P("\n" + "="*110)
P("[3] 배리어 크기별 예측력 — 큰 배리어일수록 문턱이 낮아진다")
P("="*110)
for s in SYM:
    d = D[s]
    lr = pd.Series(np.log(d["c"])).diff()
    v = pd.Series(d["c"]).pct_change(1440).values          # 직전 1일 수익
    P(f"\n  ══ {s} ══")
    P(f"    {'배리어':>8s} {'문턱(테10bp)':>13s} {'기저':>7s} {'추세하위10%':>11s} {'추세상위10%':>11s} {'최대':>7s} {'통과':>6s}")
    for x in (0.005, 0.01, 0.02, 0.05):
        key = (s, x, 1440 if x < 0.02 else 10080)
        if key not in BASE: continue
        r, t, base_up = BASE[key]
        vv = v[::15][:len(r)]
        dec = (r != 0) & np.isfinite(vv)
        if dec.sum() < 1000: continue
        q = pd.Series(np.where(dec, vv, np.nan)).rank(pct=True).values
        lo = dec & (q <= 0.10); hi = dec & (q >= 0.90)
        a = (r[lo] == 1).mean()*100; b = (r[hi] == 1).mean()*100
        be = (0.5 + 0.001/(2*x))*100
        mx = max(a, b, 100-a, 100-b)
        P(f"    ±{x*100:5.2f}% {be:12.1f}% {base_up*100:6.2f}% {a:10.2f}% {b:10.2f}% {mx:6.2f}% "
          f"{'예' if mx > be else '아니오':>6s}")

P("\n" + "="*110)
P("[4] 비대칭 배리어 — +1%/-0.5% 처럼 다르게 잡으면")
P("="*110)
for s in SYM:
    d = D[s]
    P(f"\n  ══ {s} ══")
    P(f"    {'위':>6s} {'아래':>6s} {'판정률':>7s} {'위먼저':>7s} {'기대값bp(테10bp)':>17s}")
    for up in (0.005, 0.01, 0.02):
        for dn in (0.005, 0.01, 0.02):
            r, t = first_touch(d["h"], d["l"], d["c"], up, dn, 1440, 30)
            dec = r != 0
            if dec.sum() < 500: continue
            pu = (r[dec] == 1).mean()
            ev = (pu*up - (1-pu)*dn)*1e4 - 10
            P(f"    +{up*100:4.2f}% -{dn*100:4.2f}% {dec.mean()*100:6.1f}% {pu*100:6.2f}% {ev:+16.1f}")
