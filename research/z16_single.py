"""z1.6 — 단일 포지션 ±1% 배리어. 매 진입 시 '최근 결판난 사례'로 확률 추정.
룩어헤드 방어: 시점 t에서는 t 이전에 '결판난' 배리어 결과만 사용한다.
사용: python research/z16_single.py"""
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
BAR = 0.01
MAXH = 4320          # 3일 내 미결판이면 종가 청산
D = {}
for s in SYM:
    k = store.load("klines_1m", s).set_index("ts")
    c = k.close.values
    lr = pd.Series(np.log(c)).diff()
    D[s] = dict(o=k.open.values, h=k.high.values, l=k.low.values, c=c, idx=k.index,
                rv=lr.rolling(1440).std().shift(1).values)
    P(f"{s}: {len(k):,}분")


@numba.njit(cache=True)
def touch_all(h, l, c, bar, maxh, step):
    """step분마다 ±bar 우선도달 결과와 결판 시각. out: 1/-1/0, res: 결판 시각(절대 분)"""
    n = len(c)
    m = n//step
    out = np.zeros(m, np.int8); res = np.zeros(m, np.int64)
    for k in range(m):
        i = k*step
        p = c[i]; U = p*(1.0+bar); Dn = p*(1.0-bar)
        r = 0; rt = i+maxh
        end = i+maxh
        if end >= n: end = n-1
        for j in range(i+1, end+1):
            hi = h[j] >= U; lo = l[j] <= Dn
            if lo:
                r = -1; rt = j; break
            if hi:
                r = 1; rt = j; break
        out[k] = r; res[k] = rt
    return out, res


P("\n결과 사전계산 중(±1%)...")
for s in SYM:
    d = D[s]
    o5, r5 = touch_all(d["h"], d["l"], d["c"], BAR, MAXH, 5)
    d["o5"], d["r5"] = o5, r5
    dec = o5 != 0
    P(f"  {s}: 표본 {len(o5):,} 판정률 {dec.mean()*100:.1f}% 위먼저 {(o5[dec]==1).mean()*100:.2f}% "
      f"중앙 결판 {np.median((r5[dec]-np.arange(len(o5))[dec]*5)):.0f}분")


@numba.njit(cache=True)
def recent_p(out5, res5, t, win, step):
    """시각 t 기준, 최근 win분 안에 '결판난' 사례들의 위먼저 비율. 룩어헤드 없음."""
    up = 0; tot = 0
    kmax = t//step
    k = kmax
    while k >= 0:
        rt = res5[k]
        if rt <= t and rt >= t-win:
            if out5[k] == 1: up += 1; tot += 1
            elif out5[k] == -1: tot += 1
        if k*step < t-win-4320: break
        k -= 1
    if tot < 10: return -1.0, tot
    return up/tot, tot


@numba.njit(cache=True)
def run_single(o, h, l, c, out5, res5, bar, maxh, win, step, th, mkt, kx, rv, W,
               fm, ft, eps, start, minobs):
    """단일 포지션. flat일 때만 진입 판단. 확률이 th 이상이면 롱, 1-th 이하면 숏."""
    n = len(c)
    cap_i = np.zeros(200000, np.int64); cap_x = np.zeros(200000, np.int64)
    rr = np.zeros(200000); mm = np.zeros(200000); sd = np.zeros(200000, np.int8)
    k = 0
    t = start
    while t < n - maxh - W - 2:
        p, tot = recent_p(out5, res5, t, win, step)
        if p < 0.0 or tot < minobs:
            t += 60; continue
        side = 0
        if p >= th: side = 1
        elif p <= 1.0-th: side = -1
        if side == 0:
            t += 60; continue
        # 진입
        if mkt == 1:
            f = t+1; px = o[f]; fee = ft+ft
        else:
            if np.isnan(rv[t]) or rv[t] <= 0:
                t += 60; continue
            T = c[t]*(1.0-rv[t]*kx) if side == 1 else c[t]*(1.0+rv[t]*kx)
            f = -1
            for u in range(t+1, min(t+1+W, n)):
                if side == 1 and l[u] <= T*(1.0-eps): f = u; break
                if side == -1 and h[u] >= T*(1.0+eps): f = u; break
            if f < 0:
                t += W; continue
            px = T; fee = fm+ft
        U = px*(1.0+bar); Dn = px*(1.0-bar)
        e = -1; ret = 0.0; lo_ = 1e18; hi_ = -1e18
        end = min(f+maxh, n-1)
        for j in range(f, end+1):
            if l[j] < lo_: lo_ = l[j]
            if h[j] > hi_: hi_ = h[j]
            tu = h[j] >= U; td = l[j] <= Dn
            if tu and td:
                e = j; ret = -bar if side == 1 else -bar; break
            if td:
                e = j; ret = -bar if side == 1 else bar; break
            if tu:
                e = j; ret = bar if side == 1 else -bar; break
        if e < 0:
            e = end
            ret = (c[e]/px-1.0) if side == 1 else (1.0-c[e]/px)
        cap_i[k] = f; cap_x[k] = e
        rr[k] = ret - fee/1e4
        mm[k] = (1.0-lo_/px) if side == 1 else (hi_/px-1.0)
        sd[k] = side
        k += 1
        t = e+1
        if k >= 199990: break
    return cap_i[:k], cap_x[:k], rr[:k], mm[:k], sd[:k]


def build(s, win, th, mkt=True, kx=0.5, W=120, fm=2.0, ft=5.0, eps=0.0002, minobs=20):
    d = D[s]
    a_, b_, r_, m_, sd = run_single(d["o"], d["h"], d["l"], d["c"], d["o5"], d["r5"],
                                    BAR, MAXH, win, 5, th, 1 if mkt else 0, kx,
                                    np.asarray(d["rv"], float), W, fm, ft, eps, 200000, minobs)
    ts = d["idx"]
    return [(int(a), int(bb), float(r), float(m), ts[int(a)], (s, win, int(x)), 0.01, 1.0)
            for a, bb, r, m, x in zip(a_, b_, r_, m_, sd)]


def rep(E, lab, full=False):
    if len(E) < 50: P(f"  {lab:46s} 표본부족({len(E)})"); return None
    r = np.array([x[2] for x in E]); sd = np.array([x[5][2] for x in E])
    out = f"  {lab:46s} n={len(E):5d}({len(E)/6.7:4.0f}/년) {r.mean()*1e4:+6.1f}bp 승률{(r>0).mean():.3f} 롱{(sd==1).mean()*100:3.0f}% |"
    best = None
    for w, Nm, tag in ((1.0, 1.0, "무레버"), (5.0, 5.0, "5배"), (10.0, 10.0, "10배"),
                       (20.0, 20.0, "20배"), (40.0, 40.0, "40배")):
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


P("\n[1] 최근 관측창 × 확률 문턱 (시장가 진입)")
res = []
for win in (1440, 4320, 10080, 43200):
    for th in (0.52, 0.55, 0.58, 0.62):
        E = sorted(sum([build(s, win, th, mkt=True) for s in SYM], []), key=lambda z: z[0])
        b = rep(E, f"최근{win/1440:.0f}일 문턱{th:.2f} 시장가")
        if b: res.append((b[0], win, th, True, 0.5))

P("\n[2] 지정가 진입 (문턱이 53.5%로 낮아진다)")
for win in (4320, 10080, 43200):
    for th in (0.52, 0.55, 0.58):
        for kx in (0.3, 0.6):
            E = sorted(sum([build(s, win, th, mkt=False, kx=kx) for s in SYM], []), key=lambda z: z[0])
            b = rep(E, f"최근{win/1440:.0f}일 문턱{th:.2f} 지정가k={kx}")
            if b: res.append((b[0], win, th, False, kx))

P("\n[3] 상위 후보 상세")
res.sort(reverse=True)
for sc_, win, th, mkt, kx in res[:4]:
    E = sorted(sum([build(s, win, th, mkt=mkt, kx=kx) for s in SYM], []), key=lambda z: z[0])
    rep(E, f"최근{win/1440:.0f}일 문턱{th:.2f} {'시장가' if mkt else f'지정가k{kx}'}", full=True)
