"""z0.20 시뮬 2 — 현실적 비용구조(메이커 진입·메이커 익절·테이커 손절) + 비대칭 배리어.
사용: python research/z0_20_sim2.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from z0_20_sim import SIG, D, SYM, STEP, MAXH, MMR
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
TE = pd.Timestamp("2024-01-01", tz="UTC"); REC = pd.Timestamp("2025-01-01", tz="UTC")


@numba.njit(cache=True)
def sim2(o, h, l, c, sig, th_hi, up, dn, maxh, step, lev, frac, mmr,
         fee_m, fee_t, entry_k, rv, W, eps, start):
    """entry_k>0: 현재가 아래 rv*entry_k 지정가(메이커). 0이면 시장가(테이커).
    익절 +up는 지정가(메이커), 손절 -dn는 시장가(테이커)."""
    n = len(c)
    cap = 1.0
    liq_d = 1.0/lev - mmr
    eq_t = np.zeros(400000, np.int64); eq_v = np.zeros(400000)
    k = 0; nw = 0; nl = 0; nq = 0; nskip = 0
    t = start
    while t < n - maxh - W - 2:
        if cap <= 1e-9: break
        i = t//step
        if i >= len(sig): break
        sv = sig[i]
        if not np.isfinite(sv) or sv < th_hi: t += step; continue
        # 진입
        if entry_k > 0.0:
            if np.isnan(rv[t]) or rv[t] <= 0: t += step; continue
            T = c[t]*(1.0-rv[t]*entry_k)
            f = -1
            for u in range(t+1, min(t+1+W, n)):
                if l[u] <= T*(1.0-eps): f = u; break
            if f < 0: t += W; nskip += 1; continue
            px = T; fee_in = fee_m
        else:
            f = t+1; px = o[f]; fee_in = fee_t
        U = px*(1.0+up); Dn = px*(1.0-dn); LQ = px*(1.0-liq_d)
        e = -1; res = 0.0; fee_out = fee_t
        end = min(f+maxh, n-1)
        for j in range(f, end+1):
            if liq_d < dn and l[j] <= LQ:
                e = j; res = -1.0/lev; nq += 1; fee_out = fee_t; break
            if l[j] <= Dn:
                e = j; res = -dn; nl += 1; fee_out = fee_t; break
            if h[j] >= U:
                e = j; res = up; nw += 1; fee_out = fee_m; break
        if e < 0:
            e = end; res = c[e]/px - 1.0
            fee_out = fee_t
            if res > 0: nw += 1
            else: nl += 1
        pnl = cap*frac*lev*(res - (fee_in+fee_out)/1e4)
        if pnl < -cap*frac: pnl = -cap*frac
        cap += pnl
        if cap < 0.0: cap = 0.0
        eq_t[k] = e; eq_v[k] = cap; k += 1
        t = e+1
        if k >= 399990: break
    return eq_t[:k], eq_v[:k], nw, nl, nq, nskip


for s in SYM:
    d = D[s]
    lr = pd.Series(np.log(d["c"])).diff()
    d["rv"] = lr.rolling(1440).std().shift(1).values


def run(s, q_hi, lev, frac, up, dn, entry_k=0.0, W=240, fee_m=2.0, fee_t=5.0):
    d = D[s]; sg = SIG[s]
    fin = sg[np.isfinite(sg)]
    th = np.quantile(fin, 1-q_hi)
    t_, v_, w_, l_, q_, sk = sim2(d["o"], d["h"], d["l"], d["c"], sg, th, up, dn, MAXH, STEP,
                                  float(lev), float(frac), MMR, fee_m, fee_t,
                                  float(entry_k), np.asarray(d["rv"], float), W, 0.0002, 300000)
    if len(v_) < 30: return None
    eq = pd.Series(v_, index=d["idx"][t_]).resample("D").last().ffill()
    if eq.iloc[-1] <= 1e-9: return dict(bust=True, n=len(v_), q=q_)
    y = (eq.index[-1]-eq.index[0]).days/365
    sg2 = lambda t0: (lambda x: ((x.iloc[-1]/x.iloc[0])**(365/max((x.index[-1]-x.index[0]).days, 1))-1)*100 if len(x) > 60 else np.nan)(eq[eq.index >= t0])
    mon = (eq.resample("ME").last()/eq.resample("ME").first()-1).dropna()
    return dict(bust=False, cagr=(eq.iloc[-1]**(1/y)-1)*100, mdd=float((eq/eq.cummax()-1).min())*100,
                t24=sg2(TE), t25=sg2(REC), n=len(v_), w=w_, l=l_, q=q_, sk=sk,
                wr=w_/max(w_+l_+q_, 1)*100, mw=float((mon > 0).mean()), eq=eq, mult=eq.iloc[-1])


P("[1] 현실적 비용구조 — 메이커 익절 반영 (±1%, 시장가 진입)")
P(f"  {'분위':>6s} {'레버':>5s} {'자본':>5s} {'거래':>6s} {'승률':>6s} {'전체연':>9s} {'MDD':>8s} {'24~':>9s} {'25~':>9s} {'월승률':>6s}")
best = []
for q in (0.01, 0.02, 0.05, 0.10):
    for lev in (1, 3, 5, 8, 12, 20):
        for frac in (1.0, 0.5):
            r = run("BTCUSDT", q, lev, frac, 0.01, 0.01)
            if r is None or r["bust"]:
                if r: P(f"  {q*100:5.0f}% {lev:4d}x {frac*100:4.0f}% {r['n']:6d} {'파산':>25s}")
                continue
            P(f"  {q*100:5.0f}% {lev:4d}x {frac*100:4.0f}% {r['n']:6d} {r['wr']:5.1f}% {r['cagr']:+8.1f}% "
              f"{r['mdd']:7.1f}% {r['t24']:+8.1f}% {r['t25']:+8.1f}% {r['mw']:5.2f}")
            best.append((r["cagr"], "시장가", q, lev, frac, 0.01, 0.01, r))

P("\n[2] 지정가 진입 (메이커 양쪽) — 비용 4bp")
P(f"  {'분위':>6s} {'깊이k':>6s} {'레버':>5s} {'자본':>5s} {'거래':>6s} {'승률':>6s} {'전체연':>9s} {'MDD':>8s} {'25~':>9s}")
for q in (0.02, 0.05, 0.10):
    for ek in (0.2, 0.5, 1.0):
        for lev in (3, 5, 8, 12):
            r = run("BTCUSDT", q, lev, 1.0, 0.01, 0.01, entry_k=ek)
            if r is None or r["bust"]: continue
            P(f"  {q*100:5.0f}% {ek:5.1f} {lev:4d}x {100:4.0f}% {r['n']:6d} {r['wr']:5.1f}% "
              f"{r['cagr']:+8.1f}% {r['mdd']:7.1f}% {r['t25']:+8.1f}%")
            best.append((r["cagr"], f"지정가k{ek}", q, lev, 1.0, 0.01, 0.01, r))

P("\n[3] 비대칭 배리어 — 손익비 개선")
P(f"  {'분위':>6s} {'익절':>6s} {'손절':>6s} {'레버':>5s} {'거래':>6s} {'승률':>6s} {'전체연':>9s} {'MDD':>8s} {'25~':>9s}")
for q in (0.02, 0.05):
    for up, dn in ((0.01, 0.005), (0.01, 0.0075), (0.015, 0.01), (0.02, 0.01), (0.005, 0.01), (0.0075, 0.01)):
        for lev in (3, 5, 8):
            r = run("BTCUSDT", q, lev, 1.0, up, dn)
            if r is None or r["bust"]: continue
            P(f"  {q*100:5.0f}% {up*100:5.2f}% {dn*100:5.2f}% {lev:4d}x {r['n']:6d} {r['wr']:5.1f}% "
              f"{r['cagr']:+8.1f}% {r['mdd']:7.1f}% {r['t25']:+8.1f}%")
            best.append((r["cagr"], "비대칭", q, lev, 1.0, up, dn, r))

P("\n[4] 최고 구성 상세")
best.sort(key=lambda x: -x[0])
for cg, mode, q, lev, frac, up, dn, r in best[:6]:
    P(f"\n  ── {mode} 상위{q*100:.0f}% {lev}배 자본{frac*100:.0f}% 익절{up*100:.2f}%/손절{dn*100:.2f}% ──")
    P(f"     거래 {r['n']} 승률 {r['wr']:.1f}% | 연 {r['cagr']:+.1f}% 최종 {r['mult']:.3g}배 MDD {r['mdd']:.1f}% "
      f"| 24~ {r['t24']:+.1f}% 25~ {r['t25']:+.1f}% 월승률 {r['mw']:.2f}")
    yr = r["eq"].resample("YE").last()/r["eq"].resample("YE").first()-1
    P("     연도별: " + " | ".join(f"{i.year} {v*100:+.0f}%" for i, v in yr.items()))
