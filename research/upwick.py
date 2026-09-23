"""위꼬리 비대칭 단일 전략 — 긴 위꼬리 뒤에 오르는 현상.
가설: 위꼬리 = 매수 관심의 증거(급등 시도). 매도 소진이 아니라 매수 존재를 뜻한다.
사용: python research/upwick.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from z0_13_break import bars, D, SYM
import exit_tier as X
from absfloor import met
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
TE = pd.Timestamp("2024-01-01", tz="UTC"); REC = pd.Timestamp("2025-01-01", tz="UTC")


def feats(s, B):
    b = bars(s, B)
    rng = np.maximum((b["h"]-b["l"])/b["c"], 1e-9)
    up = (b["h"]-np.maximum(b["c"], b["o"]))/b["c"]
    dn = (np.minimum(b["c"], b["o"])-b["l"])/b["c"]
    body = (b["c"]-b["o"])/b["c"]
    b["upr"] = up/rng; b["dnr"] = dn/rng; b["rng"] = rng; b["body"] = body
    b["asym"] = (up-dn)/rng                     # 위꼬리-아래꼬리 비대칭
    b["vz"] = b["qv"]/pd.Series(b["qv"]).rolling(200).mean().values
    lr = pd.Series(np.log(b["c"])).diff()
    b["rv"] = lr.rolling(50).std().shift(1).values
    b["rvr"] = (lr.rolling(10).std()/lr.rolling(100).std()).shift(1).values
    return b


def expq(v, q, step=200, warm=400):
    n = len(v); th = np.full(n, np.nan)
    for i in range(warm, n, step):
        x = v[:i]; x = x[np.isfinite(x)]
        if len(x) > 100: th[i:i+step] = np.nanquantile(x, q)
    return th


def build(s, B, key, q, H, mkt=True, kx=1.0, W=1, fee_m=2.0, fee_t=5.0, eps=0.0002,
          vol_min=None, cool=1, above=True):
    b = feats(s, B); d = D[s]; cl = b["c"]; n = len(cl)
    v = b[key]
    th = expq(v, q if above else 1-q)
    E = []; last = -10**9
    for i in range(500, n-H-W-1):
        if not np.isfinite(th[i]) or not np.isfinite(v[i]): continue
        if above and v[i] < th[i]: continue
        if (not above) and v[i] > th[i]: continue
        if vol_min is not None and not (np.isfinite(b["vz"][i]) and b["vz"][i] >= vol_min): continue
        if i - last < cool: continue
        last = i
        if mkt:
            a = int(b["st"][i]) + B - 1
            px = cl[i]; fee = fee_t + fee_t
        else:
            if not np.isfinite(b["rv"][i]) or b["rv"][i] <= 0: continue
            T = cl[i]*(1.0 - b["rv"][i]*kx)
            a0 = int(b["st"][i]) + B; a1 = int(b["st"][i+W]) + B - 1
            if a1 >= len(d["c"]): break
            a = -1
            for u in range(a0, a1+1):
                if d["l"][u] <= T*(1-eps): a = u; break
            if a < 0: continue
            px = T; fee = fee_m + fee_t
        e_i = i + (0 if mkt else W) + H
        if e_i >= n: break
        bb = int(b["st"][e_i]) + B - 1
        if bb >= len(d["c"]) or bb <= a: continue
        lo_ = d["l"][a:bb+1].min()
        E.append((a, bb, float(cl[e_i]/px - 1.0 - fee/1e4), float(max(1.0-lo_/px, 0)),
                  d["idx"][a], (s, key, B), 0.01, 1.0))
    return E


def rep(E, lab, full=False):
    if len(E) < 60: P(f"  {lab:50s} 표본부족({len(E)})"); return None
    r = np.array([x[2] for x in E])
    out = f"  {lab:50s} n={len(E):5d}({len(E)/6.7:4.0f}/년) {r.mean()*1e4:+6.1f}bp 승률{(r>0).mean():.3f} |"
    best = None
    for w, Nm, tag in ((1.0, 1.0, "무레버"), (2.0, 3.0, "저레버"), (3.0, 5.0, "중간"), (5.0, 8.0, "공격")):
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


P("[1] 위꼬리 비율 — 봉 × 분위 × 보유 (시장가)")
for B in (60, 240, 480):
    for q in (0.90, 0.95, 0.99):
        for H in (1, 2, 4):
            E = sorted(sum([build(s, B, "upr", q, H) for s in SYM], []), key=lambda z: z[0])
            rep(E, f"{B}분 위꼬리비율 상위{(1-q)*100:.0f}% 보유{H}봉 시장가")

P("\n[2] 위꼬리-아래꼬리 비대칭 지표")
for B in (60, 240):
    for q in (0.90, 0.95, 0.99):
        for H in (1, 2, 4):
            E = sorted(sum([build(s, B, "asym", q, H) for s in SYM], []), key=lambda z: z[0])
            rep(E, f"{B}분 비대칭 상위{(1-q)*100:.0f}% 보유{H}봉 시장가")

P("\n[3] 거래량 조건 추가")
for B in (60, 240):
    for vm in (1.0, 1.5, 2.0):
        for H in (2, 4):
            E = sorted(sum([build(s, B, "upr", 0.95, H, vol_min=vm) for s in SYM], []), key=lambda z: z[0])
            rep(E, f"{B}분 위꼬리 상위5% 거래량>{vm}배 보유{H}봉")

P("\n[4] 지정가 진입으로 전환")
for B in (60, 240):
    for kx in (0.5, 1.0, 2.0):
        for H in (2, 4):
            E = sorted(sum([build(s, B, "upr", 0.95, H, mkt=False, kx=kx, W=1) for s in SYM], []), key=lambda z: z[0])
            rep(E, f"{B}분 위꼬리 상위5% 지정가k={kx} 보유{H}봉")

P("\n[5] 대조 — 아래꼬리 상위(같은 조건)")
for B in (60, 240):
    for H in (2, 4):
        E = sorted(sum([build(s, B, "dnr", 0.95, H) for s in SYM], []), key=lambda z: z[0])
        rep(E, f"{B}분 아래꼬리 상위5% 보유{H}봉 [대조]")

P("\n[6] 상위 후보 상세")
for B, key, q, H, mkt, kx in ((240, "upr", 0.95, 4, True, 1.0), (240, "asym", 0.95, 4, True, 1.0),
                              (60, "upr", 0.95, 4, False, 1.0), (480, "upr", 0.95, 2, True, 1.0)):
    E = sorted(sum([build(s, B, key, q, H, mkt=mkt, kx=kx) for s in SYM], []), key=lambda z: z[0])
    rep(E, f"{B}분 {key} 상위{(1-q)*100:.0f}% H{H} {'시장가' if mkt else f'지정가k{kx}'}", full=True)
