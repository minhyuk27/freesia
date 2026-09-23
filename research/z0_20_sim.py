"""z0.20 자본·레버리지 조합 시뮬레이션 — 단일 포지션, 롱전용/롱숏, 레버리지·자본비율 스윕.
청산: 레버리지 L의 청산거리 = 1/L - mmr. 그보다 배리어가 멀면 청산이 먼저 온다.
사용: python research/z0_20_sim.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from z0_20_final import build_sigs, exp_z
from z0_20_adaptive import D, SYM, STEP
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
BAR = 0.01; MAXH = 4320; MMR = 0.004
TE = pd.Timestamp("2024-01-01", tz="UTC"); REC = pd.Timestamp("2025-01-01", tz="UTC")

NAMES = ["고저비대칭4320", "절대값8640", "OFI1440"]
SIG = {}
for s in SYM:
    S = build_sigs(s)
    Z = [exp_z(np.asarray(S[n], float)) for n in NAMES]
    n = min(min(len(z) for z in Z), len(D[s]["y"]))
    SIG[s] = np.nanmean(np.vstack([z[:n] for z in Z]), axis=0)
    P(f"{s}: 신호 {n:,}")


@numba.njit(cache=True)
def sim(h, l, c, sig, th_hi, th_lo, bar, maxh, step, lev, frac, mmr,
        side_mode, fee_bp, start):
    """단일 포지션. side_mode: 1=롱만, 0=롱숏.
    청산거리 = 1/lev - mmr. 배리어(bar)보다 가까우면 청산이 먼저."""
    n = len(c)
    cap = 1.0
    liq_d = 1.0/lev - mmr
    eq_t = np.zeros(400000, np.int64); eq_v = np.zeros(400000)
    k = 0
    nwin = 0; nlose = 0; nliq = 0
    t = start
    while t < n - maxh - 2:
        if cap <= 1e-9: break
        i = t//step
        if i >= len(sig): break
        sv = sig[i]
        if not np.isfinite(sv): t += step; continue
        side = 0
        if sv >= th_hi: side = 1
        elif side_mode == 0 and sv <= th_lo: side = -1
        if side == 0: t += step; continue
        px = c[t]
        U = px*(1.0+bar); Dn = px*(1.0-bar)
        # 청산가
        if side == 1: LQ = px*(1.0-liq_d)
        else: LQ = px*(1.0+liq_d)
        e = -1; res = 0.0
        end = min(t+maxh, n-1)
        for j in range(t+1, end+1):
            hi_ = h[j] >= U; lo_ = l[j] <= Dn
            if side == 1:
                if l[j] <= LQ and liq_d < bar:
                    e = j; res = -1.0/lev; nliq += 1; break     # 청산(증거금 전액)
                if lo_: e = j; res = -bar; nlose += 1; break
                if hi_: e = j; res = bar; nwin += 1; break
            else:
                if h[j] >= LQ and liq_d < bar:
                    e = j; res = -1.0/lev; nliq += 1; break
                if hi_: e = j; res = -bar; nlose += 1; break
                if lo_: e = j; res = bar; nwin += 1; break
        if e < 0:
            e = end
            r_ = (c[e]/px-1.0) if side == 1 else (1.0-c[e]/px)
            res = r_
            if r_ > 0: nwin += 1
            else: nlose += 1
        # 자본 반영: 투입 = cap*frac, 명목 = 투입*lev
        pnl = cap*frac*lev*(res - fee_bp/1e4)
        # 손실이 투입액을 넘지 않게(격리)
        if pnl < -cap*frac: pnl = -cap*frac
        cap += pnl
        if cap < 0.0: cap = 0.0
        eq_t[k] = e; eq_v[k] = cap; k += 1
        t = e+1
        if k >= 399990: break
    return eq_t[:k], eq_v[:k], nwin, nlose, nliq


def run(s, q_hi, lev, frac, side_mode, fee=10.0):
    d = D[s]; sg = SIG[s]
    fin = sg[np.isfinite(sg)]
    if len(fin) < 1000: return None
    th_hi = np.quantile(fin, 1-q_hi); th_lo = np.quantile(fin, q_hi)
    t_, v_, w_, l_, lq = sim(d["h"], d["l"], d["c"], sg, th_hi, th_lo, BAR, MAXH, STEP,
                             float(lev), float(frac), MMR, side_mode, fee, 300000)
    if len(v_) < 30: return None
    idx = d["idx"][t_]
    eq = pd.Series(v_, index=idx).resample("D").last().ffill()
    if eq.iloc[-1] <= 1e-9:
        return dict(bust=True, n=len(v_), w=w_, l=l_, lq=lq)
    y = (eq.index[-1]-eq.index[0]).days/365
    sgf = lambda t0: (lambda x: ((x.iloc[-1]/x.iloc[0])**(365/max((x.index[-1]-x.index[0]).days, 1))-1)*100 if len(x) > 60 else np.nan)(eq[eq.index >= t0])
    mon = (eq.resample("ME").last()/eq.resample("ME").first()-1).dropna()
    return dict(bust=False, cagr=(eq.iloc[-1]**(1/y)-1)*100,
                mdd=float((eq/eq.cummax()-1).min())*100, t24=sgf(TE), t25=sgf(REC),
                n=len(v_), w=w_, l=l_, lq=lq, wr=w_/max(w_+l_+lq, 1)*100,
                mw=float((mon > 0).mean()), eq=eq, mult=eq.iloc[-1])


P("\n" + "="*140)
P("[1] 롱 전용 — 진입분위 × 레버리지 × 자본비율 (BTC)")
P("="*140)
P(f"  {'분위':>6s} {'레버':>5s} {'자본비':>6s} {'거래':>6s} {'승률':>6s} {'청산':>5s} {'전체연':>10s} {'MDD':>8s} {'24~':>9s} {'25~':>9s} {'월승률':>6s}")
best = []
for q in (0.01, 0.02, 0.05):
    for lev in (1, 5, 10, 15, 20, 30, 50, 71, 100):
        for frac in (1.0, 0.5, 0.3):
            r = run("BTCUSDT", q, lev, frac, 1)
            if r is None: continue
            if r["bust"]:
                P(f"  {q*100:5.0f}% {lev:4d}x {frac*100:5.0f}% {r['n']:6d} {'':6s} {r['lq']:5d} {'파산':>10s}")
                continue
            P(f"  {q*100:5.0f}% {lev:4d}x {frac*100:5.0f}% {r['n']:6d} {r['wr']:5.1f}% {r['lq']:5d} "
              f"{r['cagr']:+9.1f}% {r['mdd']:7.1f}% {r['t24']:+8.1f}% {r['t25']:+8.1f}% {r['mw']:5.2f}")
            best.append((r["cagr"], q, lev, frac, 1, r))

P("\n" + "="*140)
P("[2] 롱숏 양방향 (BTC)")
P("="*140)
P(f"  {'분위':>6s} {'레버':>5s} {'자본비':>6s} {'거래':>6s} {'승률':>6s} {'청산':>5s} {'전체연':>10s} {'MDD':>8s} {'24~':>9s} {'25~':>9s}")
for q in (0.02, 0.05):
    for lev in (1, 5, 10, 15, 20, 30):
        for frac in (1.0, 0.5):
            r = run("BTCUSDT", q, lev, frac, 0)
            if r is None: continue
            if r["bust"]:
                P(f"  {q*100:5.0f}% {lev:4d}x {frac*100:5.0f}% {r['n']:6d} {'':6s} {r['lq']:5d} {'파산':>10s}"); continue
            P(f"  {q*100:5.0f}% {lev:4d}x {frac*100:5.0f}% {r['n']:6d} {r['wr']:5.1f}% {r['lq']:5d} "
              f"{r['cagr']:+9.1f}% {r['mdd']:7.1f}% {r['t24']:+8.1f}% {r['t25']:+8.1f}%")
            best.append((r["cagr"], q, lev, frac, 0, r))

P("\n" + "="*140)
P("[3] 최고 구성 상세")
P("="*140)
best.sort(key=lambda x: -x[0])
for cg, q, lev, frac, sm, r in best[:5]:
    P(f"\n  ── 분위 상위{q*100:.0f}% / {lev}배 / 자본 {frac*100:.0f}% / {'롱전용' if sm else '롱숏'} ──")
    P(f"     거래 {r['n']} (승 {r['w']} 패 {r['l']} 청산 {r['lq']}) 승률 {r['wr']:.1f}%")
    P(f"     전체 연 {r['cagr']:+.1f}% | 최종 {r['mult']:.4g}배 | MDD {r['mdd']:.1f}% | 24~ {r['t24']:+.1f}% | 25~ {r['t25']:+.1f}% | 월승률 {r['mw']:.2f}")
    yr = r["eq"].resample("YE").last()/r["eq"].resample("YE").first()-1
    P("     연도별: " + " | ".join(f"{i.year} {v*100:+.0f}%" for i, v in yr.items()))
