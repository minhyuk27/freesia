"""기준점 변경 + 보유 연장.
A) 딥 기준을 '봉 시가' 대신 '최근 N봉 고가'로 → 며칠에 걸친 완만한 하락을 포착
B) 봉 후반부에 체결된 건 다음 봉 종가까지 보유 연장 → 되돌릴 시간 확보
사용: python research/anchor.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
import exit_tier as X
from absfloor import met
from exit_tier import RAW, SYM, prep, CACHE
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
BL6 = (240, 360, 480, 720, 960, 1440)
DEP = {10.0: 1.0, 14.0: 3.0, 20.0: 6.0}
FLO = {10.0: 0.05, 14.0: 0.07, 20.0: 0.10}


@numba.njit(cache=True)
def ev_anchor(o, h, l, c, st, B, x, anchor, ext, eps, fm, ft):
    """anchor: 봉별 기준가 배열(봉 시가이거나 최근고가). ext>0이면 봉 후반 ext 비율 이후 체결 시 다음 봉 종가까지."""
    n = len(st)
    ei = np.full(n, -1, np.int64); ex = np.full(n, -1, np.int64)
    rr = np.zeros(n); mm = np.zeros(n); k = 0
    for j in range(n):
        if np.isnan(x[j]) or x[j] <= 0 or np.isnan(anchor[j]) or anchor[j] <= 0: continue
        s0 = st[j]; e = s0+B
        if e > len(c): break
        T = anchor[j]*(1-x[j])
        f = -1; lo_ = 1e18
        for t in range(s0, e):
            if f < 0:
                if l[t] <= T*(1-eps): f = t
                else: continue
            if l[t] < lo_: lo_ = l[t]
        if f < 0: continue
        end = e-1
        if ext > 0.0 and (f - s0) > B*(1.0-ext):
            e2 = e + B
            if e2 <= len(c):
                end = e2-1
                for t in range(e, e2):
                    if l[t] < lo_: lo_ = l[t]
        ei[k] = f; ex[k] = end
        rr[k] = (c[end]/T - 1.0) - (fm+ft)/1e4
        mm[k] = 1.0 - lo_/T
        k += 1
    return ei[:k], ex[:k], rr[:k], mm[:k]


def build_a(mode, look=0, ext=0.0, depths=DEP, floors=FLO, bars=BL6, eps=0.0002, fm=2.0, ft=5.0):
    """mode: 'open' 봉 시가 / 'high' 최근 look봉 고가(직전봉까지) / 'max' 둘 중 높은 쪽"""
    E = []
    for s in SYM:
        d = RAW[s]
        for B in bars:
            if (s, B) not in CACHE: CACHE[(s, B)] = prep(s, B)
            st, rv = CACHE[(s, B)]
            nb = len(st)
            bo = d["o"][st]
            bh = np.array([d["h"][st[i]:st[i]+B].max() if st[i]+B <= len(d["h"]) else np.nan for i in range(nb)])
            if mode == "open":
                anc = bo
            else:
                hh = pd.Series(bh).rolling(look).max().shift(1).values
                anc = hh if mode == "high" else np.maximum(hh, bo)
            for k, wgt in depths.items():
                xx = np.maximum(rv*k, floors.get(k, 0.0)); xx = np.where(np.isnan(rv), np.nan, xx)
                a_, b_, r_, m_ = ev_anchor(d["o"], d["h"], d["l"], d["c"], st.astype(np.int64), B,
                                           np.asarray(xx, float), np.asarray(anc, float), ext, eps, fm, ft)
                for a, bb, r, m in zip(a_, b_, r_, m_):
                    E.append((int(a), int(bb), float(r), float(m), d["idx"][int(a)], (s, B, k), float(rv[int(a)//B]), wgt))
    E.sort(key=lambda z: z[0])
    return E


def tiers(E, lab, quiet_bankrupt=False):
    r = np.array([x[2] for x in E]); ts = pd.DatetimeIndex([x[4] for x in E])
    P(f"  {lab}  n={len(r)} 평균 {r.mean()*1e4:+6.0f}bp 승률 {(r>0).mean():.3f} 거래월 {len(set(ts.to_period('M')))}/81")
    out = {}
    for w, Nm, tag in ((1.0, 1.0, "무레버"), (3.0, 5.0, "중간"), (5.0, 8.0, "공격")):
        eq, fl, liq = X.sim(E, w, Nm); m = met(eq, fl, liq)
        if m is None: P(f"     {tag:6s} 파산"); continue
        out[tag] = m
        P(f"     {tag:6s} 연 {m[0]:+7.0f}% 최악 {m[2]:6.1f}% | 24~ {m[3]:+7.0f}% | 25~ {m[4]:+7.0f}%")
    return out


P("[기준] 봉 시가 기준 (현재 z0.3)")
base = tiers(build_a("open"), "")

P("\n[A] 기준점 = 최근 N봉 고가")
for look in (2, 3, 5, 10, 20):
    tiers(build_a("high", look), f"최근{look}봉 고가")

P("\n[A2] 기준점 = max(최근 N봉 고가, 봉 시가) — 둘 중 보수적인 쪽")
for look in (3, 5, 10):
    tiers(build_a("max", look), f"max(최근{look}봉고가, 시가)")

P("\n[B] 보유 연장 — 봉 후반 ext 비율 이후 체결 시 다음 봉 종가까지")
for ext in (0.0, 0.1, 0.25, 0.5, 1.0):
    tiers(build_a("open", ext=ext), f"후반 {ext*100:.0f}% 체결분 연장")

P("\n[C] 두 기준을 독립 다리로 동시 운용 (시가 기준 + 최근5봉고가 기준)")
E1 = build_a("open")
E2 = build_a("high", 5)
key2 = lambda k: (k[0], k[1], k[2]+100)
E2b = [(a, b, r, m, ts, key2(k), rv, wg) for (a, b, r, m, ts, k, rv, wg) in E2]
for wgt in (0.3, 0.5, 1.0):
    EM = sorted(E1 + [(a, b, r, m, ts, k, rv, wg*wgt) for (a, b, r, m, ts, k, rv, wg) in E2b], key=lambda z: z[0])
    tiers(EM, f"결합 (고가기준 가중 {wgt})")

P("\n[D] 최근고가 기준의 성격 — 깊이별 거래당 수익 (최근5봉 고가, 4h·8h)")
for k in (5.0, 8.0, 10.0, 14.0, 20.0):
    E = build_a("high", 5, depths={k: 1.0}, floors={k: 0.05}, bars=(240, 480))
    if len(E) < 20: continue
    r = np.array([x[2] for x in E]); ts = pd.DatetimeIndex([x[4] for x in E])
    P(f"  k={k:5.1f}: n={len(r):5d} 평균 {r.mean()*1e4:+7.0f}bp 중앙 {np.median(r)*1e4:+6.0f}bp 승률 {(r>0).mean():.3f} 거래월 {len(set(ts.to_period('M'))):2d}/81")
