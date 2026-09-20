"""1분~30분 초단기 전면 탐색 — 깊이 프리미엄·마이크로 되돌림·주문흐름·갭.
방향 예측이 아니라 '지정가로 받는' 구조. 전부 인과적.
사용: python research/micro.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
import exit_tier as X
from absfloor import met
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
SYM = ("BTCUSDT", "ETHUSDT")
TE = pd.Timestamp("2024-01-01", tz="UTC"); REC = pd.Timestamp("2025-01-01", tz="UTC")
D = {}
for s in SYM:
    k = store.load("klines_1m", s).set_index("ts")
    c = k.close.values
    lr = pd.Series(np.log(c)).diff()
    D[s] = dict(o=k.open.values, h=k.high.values, l=k.low.values, c=c,
                qv=k.quote_volume.values, tb=k.taker_buy_quote_volume.values,
                idx=k.index, rv1=lr.rolling(300).std().shift(1).values)
    P(f"{s}: {len(k):,}분 | 1분 rv 중앙 {np.nanmedian(D[s]['rv1'])*1e4:.1f}bp")


@numba.njit(cache=True)
def depth_run(o, h, l, c, B, x, H, fee_m, fee_t, eps, cool):
    """B분봉 시가 대비 x(비율) 아래 지정가 → H봉 뒤 종가 청산. H=0이면 같은 봉 종가."""
    n = len(c)//B
    ei = np.full(n, -1, np.int64); ex = np.full(n, -1, np.int64)
    rr = np.zeros(n); mm = np.zeros(n); k = 0; last = -10**9
    for b in range(n-H-1):
        if b - last < cool: continue
        if np.isnan(x[b]) or x[b] <= 0: continue
        s0 = b*B; e0 = s0+B
        if e0 > len(c): break
        T = o[s0]*(1.0-x[b])
        f = -1
        for t in range(s0, e0):
            if l[t] <= T*(1.0-eps): f = t; break
        if f < 0: continue
        last = b
        ee = (b+H+1)*B - 1
        if ee >= len(c): break
        lo_ = 1e18
        for t in range(f, ee+1):
            if l[t] < lo_: lo_ = l[t]
        ei[k] = f; ex[k] = ee
        rr[k] = (c[ee]/T - 1.0) - (fee_m+fee_t)/1e4
        mm[k] = 1.0 - lo_/T
        k += 1
    return ei[:k], ex[:k], rr[:k], mm[:k]


def build_depth(s, B, kx, H, fee_m=2.0, fee_t=5.0, eps=0.0002, cool=0, absbp=None):
    d = D[s]
    nb = len(d["c"])//B
    if absbp is not None:
        x = np.full(nb, absbp/1e4)
    else:
        rvb = pd.Series(d["rv1"]).iloc[::B].values[:nb]*np.sqrt(B)
        x = rvb*kx
    a_, b_, r_, m_ = depth_run(d["o"], d["h"], d["l"], d["c"], B, np.asarray(x, float),
                               H, fee_m, fee_t, eps, cool)
    ts = d["idx"]
    return [(int(a), int(bb), float(r), float(m), ts[int(a)], (s, B, kx), 0.01, 1.0)
            for a, bb, r, m in zip(a_, b_, r_, m_)]


def rep(E, lab, full=False, quiet_small=True):
    if len(E) < 100:
        if not quiet_small: P(f"  {lab:46s} 표본부족({len(E)})")
        return None
    r = np.array([x[2] for x in E])
    out = f"  {lab:46s} n={len(E):6d}({len(E)/6.7:5.0f}/년) {r.mean()*1e4:+6.1f}bp 승률{(r>0).mean():.3f} |"
    best = None
    for w, Nm, tag in ((1.0, 1.0, "무레버"), (2.0, 3.0, "저레버"), (3.0, 5.0, "중간")):
        eq, fl, liq = X.sim(E, w, Nm); m = met(eq, fl, liq)
        out += f" {tag}{'파산' if m is None else f'{m[0]:+7.1f}%'}"
        if m and (best is None or m[0] > best[0]): best = (m[0], m[1], m[2], m[3], m[4], tag, m[5])
    P(out)
    if full and best:
        mon = (best[6].resample("ME").last()/best[6].resample("ME").first()-1).dropna()
        P(f"      최고 {best[5]}: MDD {best[1]:.0f}% 최악 {best[2]:.0f}% 24~ {best[3]:+.0f}% 25~ {best[4]:+.0f}% 월승률 {float((mon>0).mean()):.2f}")
        df = pd.DataFrame([(x[4], x[2]) for x in E], columns=["ts", "r"]).set_index("ts")
        g = df.groupby(df.index.year).r
        P("      연도별 " + " ".join(f"{y}{v*1e4:+5.0f}×{n}" for (y, v), (_, n) in zip(g.mean().items(), g.size().items())))
    return best


P("\n" + "="*140)
P("[1] 초단기 깊이 프리미엄 — 봉 길이 × 깊이배수 (같은 봉 종가 청산)")
P("="*140)
res = []
for B in (1, 2, 3, 5, 10, 15, 30):
    for kx in (1.0, 2.0, 3.0, 5.0, 8.0):
        E = sorted(sum([build_depth(s, B, kx, 0) for s in SYM], []), key=lambda z: z[0])
        b = rep(E, f"{B}분봉 깊이 rv×{kx} 같은봉 청산")
        if b: res.append((b[0], B, kx, 0))

P("\n" + "="*140)
P("[2] 보유 연장 — 다음 봉들까지")
P("="*140)
for B in (1, 3, 5, 15):
    for kx in (2.0, 3.0, 5.0):
        for H in (1, 3, 10):
            E = sorted(sum([build_depth(s, B, kx, H) for s in SYM], []), key=lambda z: z[0])
            b = rep(E, f"{B}분봉 rv×{kx} +{H}봉 보유(총 {B*(H+1)}분)")
            if b: res.append((b[0], B, kx, H))

P("\n" + "="*140)
P("[3] 절대 깊이(bp) — rv 무관 고정")
P("="*140)
for B in (1, 3, 5, 15):
    for ab in (5, 10, 20, 50, 100):
        E = sorted(sum([build_depth(s, B, 0, 0, absbp=ab) for s in SYM], []), key=lambda z: z[0])
        rep(E, f"{B}분봉 절대깊이 {ab}bp 같은봉 청산")

P("\n" + "="*140)
P("[4] 메이커 양방향 가정 (청산도 지정가)")
P("="*140)
for B in (1, 3, 5):
    for kx in (2.0, 3.0, 5.0):
        E = sorted(sum([build_depth(s, B, kx, 0, fee_t=2.0) for s in SYM], []), key=lambda z: z[0])
        rep(E, f"{B}분봉 rv×{kx} 메이커왕복4bp")

P("\n" + "="*140)
P("[5] 상위 후보 상세")
P("="*140)
res.sort(reverse=True)
seen = set()
for sc_, B, kx, H in res[:6]:
    key = (B, kx, H)
    if key in seen: continue
    seen.add(key)
    E = sorted(sum([build_depth(s, B, kx, H) for s in SYM], []), key=lambda z: z[0])
    rep(E, f"{B}분봉 rv×{kx} +{H}봉", full=True)
