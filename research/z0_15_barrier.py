"""z0.15 — 배리어 우선도달 전략. 추세로 방향을 정하고 ±X% 배리어까지 보유.
진입: 지정가(메이커) 또는 시장가. 확률 추정은 확장창(룩어헤드 없음).
사용: python research/z0_15_barrier.py"""
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
    D[s] = dict(o=k.open.values, h=k.high.values, l=k.low.values, c=c, idx=k.index,
                rv=lr.rolling(1440).std().shift(1).values)


@numba.njit(cache=True)
def run_bar(o, h, l, c, sig, th_hi, th_lo, up, dn, maxmin, step, mkt, kx, rv, W, fm, ft, eps, cool):
    """sig>=th_hi 롱. mkt=1 시장가, 0이면 현재가×(1-rv*kx) 지정가(W분 유효).
    청산: +up 먼저면 승, -dn 먼저면 패, maxmin 초과면 종가."""
    n = len(c)
    ei = np.full(n//step+1, -1, np.int64); ex = np.full(n//step+1, -1, np.int64)
    rr = np.zeros(n//step+1); mm = np.zeros(n//step+1)
    k = 0; last = -10**9
    for i in range(1440, n-maxmin-W-1, step):
        if np.isnan(sig[i]) or np.isnan(th_hi[i]): continue
        if sig[i] < th_hi[i]: continue
        if i - last < cool: continue
        last = i
        if mkt == 1:
            f = i+1; px = o[f]; fee = ft+ft
        else:
            if np.isnan(rv[i]) or rv[i] <= 0: continue
            T = c[i]*(1.0-rv[i]*kx)
            f = -1
            for u in range(i+1, i+1+W):
                if l[u] <= T*(1.0-eps): f = u; break
            if f < 0: continue
            px = T; fee = fm+ft
        U = px*(1.0+up); Dn = px*(1.0-dn)
        e = -1; res = 0.0; lo_ = 1e18
        for j in range(f, f+maxmin+1):
            if j >= n: break
            if l[j] < lo_: lo_ = l[j]
            hi_ = h[j] >= U; lo2 = l[j] <= Dn
            if hi_ and lo2:
                e = j; res = -dn; break
            if lo2:
                e = j; res = -dn; break
            if hi_:
                e = j; res = up; break
        if e < 0:
            e = min(f+maxmin, n-1); res = c[e]/px - 1.0
        ei[k] = f; ex[k] = e
        rr[k] = res - fee/1e4
        mm[k] = 1.0 - lo_/px
        k += 1
    return ei[:k], ex[:k], rr[:k], mm[:k]


def expq(v, q, step=1440, warm=100000):
    n = len(v); th = np.full(n, np.nan)
    for i in range(warm, n, step):
        x = v[:i]; x = x[np.isfinite(x)]
        if len(x) > 5000: th[i:i+step] = np.nanquantile(x, q)
    return th


def build(s, sig_w, q, up, dn, maxh, mkt=True, kx=1.0, W=60, step=60,
          fm=2.0, ft=5.0, eps=0.0002, cool=60):
    d = D[s]
    sig = pd.Series(d["c"]).pct_change(sig_w).values
    th = expq(sig, q)
    a_, b_, r_, m_ = run_bar(d["o"], d["h"], d["l"], d["c"], np.asarray(sig, float),
                             np.asarray(th, float), np.asarray(th, float), up, dn,
                             maxh, step, 1 if mkt else 0, kx, np.asarray(d["rv"], float),
                             W, fm, ft, eps, cool)
    ts = d["idx"]
    return [(int(a), int(bb), float(r), float(m), ts[int(a)], (s, up), 0.01, 1.0)
            for a, bb, r, m in zip(a_, b_, r_, m_)]


def rep(E, lab, full=False):
    if len(E) < 60: P(f"  {lab:48s} 표본부족({len(E)})"); return None
    r = np.array([x[2] for x in E])
    out = f"  {lab:48s} n={len(E):5d}({len(E)/6.7:4.0f}/년) {r.mean()*1e4:+6.1f}bp 승률{(r>0).mean():.3f} |"
    best = None
    for w, Nm, tag in ((1.0, 1.0, "무레버"), (2.0, 3.0, "저레버"), (3.0, 5.0, "중간"), (4.0, 6.0, "중간+")):
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


P("[1] 시장가 진입 — 추세창 × 분위 × 배리어")
res = []
for sw in (1440, 4320):
    for q in (0.80, 0.90):
        for bx, mh in ((0.01, 1440), (0.02, 2880), (0.05, 10080)):
            E = sorted(sum([build(s, sw, q, bx, bx, mh, mkt=True) for s in SYM], []), key=lambda z: z[0])
            b = rep(E, f"추세{sw}분 상위{(1-q)*100:.0f}% ±{bx*100:.0f}% 시장가")
            if b: res.append((b[0], sw, q, bx, mh, True, 1.0))

P("\n[2] 지정가 진입 (메이커) — 문턱이 낮아진다")
for sw in (1440, 4320):
    for q in (0.80, 0.90):
        for bx, mh in ((0.01, 1440), (0.02, 2880)):
            for kx in (0.3, 0.6):
                E = sorted(sum([build(s, sw, q, bx, bx, mh, mkt=False, kx=kx, W=120) for s in SYM], []), key=lambda z: z[0])
                b = rep(E, f"추세{sw}분 상위{(1-q)*100:.0f}% ±{bx*100:.0f}% 지정가k={kx}")
                if b: res.append((b[0], sw, q, bx, mh, False, kx))

P("\n[3] 상위 후보 상세")
res.sort(reverse=True)
for sc_, sw, q, bx, mh, mkt, kx in res[:4]:
    E = sorted(sum([build(s, sw, q, bx, bx, mh, mkt=mkt, kx=kx, W=120) for s in SYM], []), key=lambda z: z[0])
    rep(E, f"추세{sw} q{q} ±{bx*100:.0f}% {'시장가' if mkt else f'지정가k{kx}'}", full=True)

P("\n[4] 최선 후보 비용 민감도 + 플라시보")
@numba.njit(cache=True)
def rnd(o, c, st, hd, fee):
    out = np.zeros(len(st))
    for i in range(len(st)):
        e = st[i]+hd[i]
        if e >= len(c): e = len(c)-1
        out[i] = (c[e]/o[st[i]]-1.0) - fee/1e4
    return out
if res:
    _, sw, q, bx, mh, mkt, kx = res[0]
    for fm, ft, eps, lab in ((2, 5, 0.0002, "기준"), (3, 6, 0.0005, "보수"), (4, 7, 0.001, "비관"), (5, 8, 0.002, "최악")):
        E = sorted(sum([build(s, sw, q, bx, bx, mh, mkt=mkt, kx=kx, W=120, fm=fm, ft=ft, eps=eps) for s in SYM], []), key=lambda z: z[0])
        rep(E, f"  {lab} 메{fm}/테{ft}")
    E = sorted(sum([build(s, sw, q, bx, bx, mh, mkt=mkt, kx=kx, W=120) for s in SYM], []), key=lambda z: z[0])
    r = np.array([x[2] for x in E]); hd = np.array([x[1]-x[0] for x in E])
    rng = np.random.default_rng(41); nul = []
    for _ in range(300):
        acc = []
        for s in SYM:
            o = D[s]["o"]; c = D[s]["c"]; N = len(c)
            cnt = max(len(E)//2, 1)
            acc.append(rnd(o, c, rng.integers(0, N-15000, cnt).astype(np.int64),
                           rng.choice(hd, cnt).astype(np.int64), 7.0))
        nul.append(np.concatenate(acc).mean()*1e4)
    nul = np.array(nul)
    P(f"  플라시보 300회: 위약 평균 {nul.mean():+.1f}bp 최대 {nul.max():+.1f} | 실제 {r.mean()*1e4:+.1f}bp → 백분위 {(nul < r.mean()*1e4).mean()*100:.1f}%")
