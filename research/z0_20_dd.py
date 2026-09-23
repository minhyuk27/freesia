"""z0.20 자본곡선 기반 배수 조절 — 낙폭 중에는 줄이고 회복하면 키운다.
거래 단위 연속손실 감축은 실패했으나, 낙폭은 국면 정보를 담고 있을 수 있다.
확정 구성: BTC 상위20% 시장가 ±1.5% 쿨60 + 변동성×신호계단."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R/"research"))
import z0_20_lib as L


def P(*a, end="\n"):
    print(" ".join(str(x) for x in a), end=end, flush=True)


S = "BTCUSDT"; Y = L.YRS
d = L.D[S]; idx = d["idx"]; rv = d["rv"]; sig = d["sig"]
warm = 200000; _st = 5000
med = np.full(len(rv), np.nan)
for i in range(warm, len(rv), _st):
    x = rv[:i]; x = x[np.isfinite(x)]
    if len(x) >= 1000: med[i:i+_st] = np.median(x)
sq = np.full(len(sig), np.nan); _sw = warm//L.STEP; _ss = _st//L.STEP
for i in range(_sw, len(sig), _ss):
    x = sig[:i]; x = x[np.isfinite(x)]
    if len(x) < 1000: continue
    sq[i:i+_ss] = np.searchsorted(np.sort(x), sig[i:i+_ss])/len(x)


@numba.njit(cache=True)
def tr_at(o, h, l, c, starts, up, dn, maxh, cool):
    n = len(c); K = len(starts)
    ei = np.zeros(K, np.int64); ex = np.zeros(K, np.int64); rr = np.zeros(K)
    k = 0; free = 0
    for si in range(K):
        t = starts[si]
        if t < free or t >= n-maxh-2: continue
        f = t+1; px = o[f]; U = px*(1.0+up); Dn = px*(1.0-dn)
        e = -1; res = 0.0; end = min(f+maxh, n-1)
        for j in range(f, end+1):
            if l[j] <= Dn: e = j; res = -dn; break
            if h[j] >= U: e = j; res = up; break
        if e < 0: e = end; res = c[e]/px-1.0
        ei[k] = f; ex[k] = e; rr[k] = res; k += 1
        free = e+1+cool
    return ei[:k], ex[:k], rr[:k]


def sig_starts(q):
    th = np.quantile(d["fin"], 1-q)
    ok = np.where(np.isfinite(sig) & (sig >= th))[0]*L.STEP
    return ok[ok >= warm].astype(np.int64)


Q, BAR, COOL = 0.20, 0.015, 60
ei, ex, rr = tr_at(d["o"], d["h"], d["l"], d["c"], sig_starts(Q), BAR, BAR, L.MAXH, COOL)
P(f"확정 구성 — {len(ei)}건 (연 {len(ei)/Y:.0f}건), 승률 {np.mean(rr>0)*100:.2f}%\n")

SZ = np.ones(len(ei))
for i, a in enumerate(ei):
    mu = 1.0
    r_ = rv[a-1]; mm = med[a]
    if np.isfinite(r_) and np.isfinite(mm) and r_ > 0: mu *= float(np.clip(mm/r_, 0.1, 4.0))
    s_ = sq[(a-1)//L.STEP]
    if np.isfinite(s_): mu *= (2.0 if s_ >= 0.97 else (1.0 if s_ >= 0.90 else 0.5))
    SZ[i] = mu


def sim(lev, dd_mode="none", dd_k=1.0, dd_floor=0.2, dd_cut=0.30, dn=BAR, yr=None, fee=10.0):
    """dd_mode: none / linear(낙폭 비례 축소) / cut(임계 넘으면 고정 축소) / sqrt"""
    cap = 1e7; peak = cap; cur = []
    for (a, b, res, mu) in zip(ei, ex, rr, SZ):
        if yr is not None and idx[a].year != yr: continue
        dd = 1.0 - cap/peak
        m = mu
        if dd_mode == "linear": m *= max(dd_floor, 1.0 - dd_k*dd)
        elif dd_mode == "cut":  m *= (dd_floor if dd > dd_cut else 1.0)
        elif dd_mode == "sqrt": m *= max(dd_floor, (1.0-dd)**dd_k)
        lv = lev*m
        if lv <= 0: cur.append(cap); continue
        liq = 1.0/lv - L.MMR; r = res
        if r <= -dn and liq < dn: r = -1.0/lv
        pnl = cap*lv*(r-fee/1e4)
        if pnl < -cap: pnl = -cap
        cap += pnl
        if cap > peak: peak = cap
        cur.append(cap)
        if cap <= 1e5: break
    if not cur: return None
    eq = np.array(cur)
    if eq[-1] <= 2e5: return None
    yy = Y if yr is None else 1.0
    pk = np.maximum.accumulate(np.concatenate([[1e7], eq]))[1:]
    return dict(ann=((eq[-1]/1e7)**(1/yy)-1)*100, mdd=(eq/pk-1).min()*100, fin=eq[-1]/1e7)


def cell(r):
    return f"{r['ann']:+8.1f}%/{r['mdd']:6.1f}%" if r else "         파산   "


P("="*126); P("[1] 낙폭 비례 축소 (linear: 배수 ×= max(floor, 1−k×낙폭))"); P("="*126)
P(f"  {'배수':>5s} {'조절없음':>17s} {'k=1.0':>17s} {'k=1.5':>17s} {'k=2.0':>17s} {'k=3.0':>17s}")
for lev in (1, 3, 5, 8, 12, 20, 30):
    row = [cell(sim(lev))]
    for k in (1.0, 1.5, 2.0, 3.0):
        row.append(cell(sim(lev, "linear", k, 0.2)))
    P(f"  {lev:4d}x " + " ".join(f"{x:>17s}" for x in row))

P("\n"+"="*126); P("[2] 임계 컷 (낙폭 > cut 이면 배수 ×floor)"); P("="*126)
P(f"  {'배수':>5s} " + " ".join(f"{'cut'+str(int(c*100))+'%':>17s}" for c in (0.20, 0.30, 0.40, 0.50)))
for lev in (3, 5, 8, 12, 20):
    row = [cell(sim(lev, "cut", 1.0, 0.3, c)) for c in (0.20, 0.30, 0.40, 0.50)]
    P(f"  {lev:4d}x " + " ".join(f"{x:>17s}" for x in row))

P("\n"+"="*126); P("[3] 제곱근형 (배수 ×= (1−낙폭)^k)"); P("="*126)
P(f"  {'배수':>5s} " + " ".join(f"{'k='+str(k):>17s}" for k in (1, 2, 3, 5)))
for lev in (3, 5, 8, 12, 20, 30):
    row = [cell(sim(lev, "sqrt", k, 0.1)) for k in (1, 2, 3, 5)]
    P(f"  {lev:4d}x " + " ".join(f"{x:>17s}" for x in row))

P("\n"+"="*126); P("[4] 최고 조합 찾기 — 연수익 상위 12"); P("="*126)
res = []
for lev in (1, 2, 3, 5, 8, 12, 16, 20, 25, 30, 40):
    for mode, k, fl, cut in [("none", 0, 0, 0)] + \
            [("linear", k, fl, 0) for k in (1.0, 1.5, 2.0, 3.0) for fl in (0.1, 0.2, 0.3)] + \
            [("cut", 0, fl, c) for fl in (0.2, 0.3, 0.5) for c in (0.2, 0.3, 0.4, 0.5)] + \
            [("sqrt", k, 0.1, 0) for k in (1, 2, 3, 5, 8)]:
        r = sim(lev, mode, k, fl, cut)
        if r: res.append((r["ann"], lev, mode, k, fl, cut, r["mdd"], r["fin"]))
res.sort(reverse=True)
P(f"  {'연수익%':>9s} {'MDD%':>8s} {'배수':>5s} {'방식':>8s} {'k':>5s} {'floor':>6s} {'cut':>5s} {'최종배수':>11s}")
for a, lev, mode, k, fl, cut, mdd, fin in res[:12]:
    P(f"  {a:9.1f} {mdd:8.1f} {lev:4d}x {mode:>8s} {k:5.1f} {fl:6.2f} {cut:5.2f} {fin:11.2f}")

P("\n"+"="*126); P("[5] 최종 3단 레버리지 (최고 조절 방식 적용) + 연도별"); P("="*126)
if res:
    _, _, mode, k, fl, cut, _, _ = res[0]
    P(f"  조절 방식: {mode} (k={k}, floor={fl}, cut={cut})")
    P(f"\n  {'구간':>12s} {'배수':>6s} {'연수익%':>10s} {'MDD%':>8s} {'최종배수':>11s}")
    for lb, lev in (("무레버리지", 1), ("중간", 3), ("중간+", 5), ("공격", 8),
                    ("최대공격", 12), ("극단", 20), ("초극단", 30)):
        r = sim(lev, mode, k, fl, cut)
        P(f"  {lb:>12s} {lev:5d}x " + (f"{r['ann']:10.1f} {r['mdd']:8.1f} {r['fin']:11.2f}" if r else "        파산"))
    for lv in (1, 3, 5, 8):
        out = []
        for yr in range(2020, 2027):
            r = sim(lv, mode, k, fl, cut, yr=yr)
            out.append(f"{yr} {(r['fin']-1)*100:+7.0f}%" if r else f"{yr}   파산")
        P(f"  연도별({lv}x): " + " | ".join(out))
    P("\n  비용 민감도 (최고 구성):")
    for fee in (4, 7, 10, 15, 20, 30):
        row = []
        for lv in (1, 3, 5):
            r = sim(lv, mode, k, fl, cut, fee=fee)
            row.append(f"{r['ann']:+8.1f}%" if r else "     파산")
        P(f"    왕복 {fee:2d}bp: " + " ".join(row))
