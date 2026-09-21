"""z2 청산 후 재진입 — 청산당한 자리에서 남은 자본으로 다시 들어가 반등을 먹는 구조.

근거: z0.3·z1.4에서 '깊은 딥은 되돌림 프리미엄이 있다'가 검증됨.
청산은 가격이 liq거리만큼 빠졌다는 뜻이므로, 그 자리가 오히려 진입 자리일 수 있다.
청산될 때마다 진입가가 낮아져 반등 시 수익률이 커진다.

위험: 마틴게일. 연속 실패하면 손실이 누적된다. 최대 재진입 N으로 제한한다.

모드:
  new  = 재진입가 기준으로 다시 +bar 목표 (각 포지션 독립)
  recv = 최초 진입가 회복을 목표 (원금 회복형)
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R/"research"))
import z2_lib as L
from z2_final import build_sigs, exp_z


def P(*a, end="\n"):
    print(" ".join(str(x) for x in a), end=end, flush=True)


S = "BTCUSDT"; Y = L.YRS; MMR = L.MMR
d = L.D[S]; idx = d["idx"]; rv = d["rv"]
O, H, Lo, C = d["o"], d["h"], d["l"], d["c"]
STEP = L.STEP; MAXH = L.MAXH
warm = 200000; _st = 5000
med = np.full(len(rv), np.nan)
for i in range(warm, len(rv), _st):
    x = rv[:i]; x = x[np.isfinite(x)]
    if len(x) >= 1000: med[i:i+_st] = np.median(x)

SG = build_sigs(S)
names = list(SG.keys())
Z = {n: exp_z(np.asarray(SG[n], float)) for n in names}
mlen = min(min(len(Z[n]) for n in names), len(C)//STEP)
PICK = [n for n in names if n.endswith("4320")]
SIG = np.nanmean(np.vstack([Z[n][:mlen] for n in PICK]), axis=0)
SQ = np.full(len(SIG), np.nan)
for i in range(warm//STEP, len(SIG), _st//STEP):
    x = SIG[:i]; x = x[np.isfinite(x)]
    if len(x) < 1000: continue
    SQ[i:i+_st//STEP] = np.searchsorted(np.sort(x), SIG[i:i+_st//STEP])/len(x)

MU = np.zeros(len(C))
for t in range(warm, len(C)):
    mu = 1.0
    r_ = rv[t]; mm = med[t]
    if np.isfinite(r_) and np.isfinite(mm) and r_ > 0:
        x = mm/r_
        mu *= 0.1 if x < 0.1 else (4.0 if x > 4.0 else x)
    j = t//STEP
    s_ = SQ[j] if j < len(SQ) else np.nan
    if np.isfinite(s_): mu *= (2.0 if s_ >= 0.97 else (1.0 if s_ >= 0.90 else 0.5))
    MU[t] = mu

BAR = 0.015; Q = 0.16; COOL = 60
TH = float(np.quantile(SIG[np.isfinite(SIG)], 1-Q))
NEND = len(C)


P("="*126); P("[0] 먼저 사실 확인 — 하락 후 반등 확률 (신호 진입 후 -X% 찍은 시점부터)"); P("="*126)


@numba.njit(cache=True)
def bounce_stats(o, h, l, c, sigv, th, drop, bar, maxh, step, start, n_end):
    """신호 진입 후 -drop을 찍은 경우, 그 시점부터 +bar 먼저 닿는 비율."""
    n_hit = 0; n_up = 0; n_sig = 0
    t = start
    while t < n_end - maxh - 2:
        i = t//step
        if i >= len(sigv): break
        if not np.isfinite(sigv[i]) or sigv[i] < th: t += step; continue
        n_sig += 1
        f = t+1; px = o[f]; D = px*(1.0-drop); U0 = px*(1.0+bar)
        e = -1
        for j in range(f, min(f+maxh, n_end-1)+1):
            if l[j] <= D: e = j; break
            if h[j] >= U0: e = -2; break
        if e >= 0:                      # 하락 먼저
            n_hit += 1
            px2 = D; U = px2*(1.0+bar); D2 = px2*(1.0-bar)
            for j in range(e, min(e+maxh, n_end-1)+1):
                if l[j] <= D2: break
                if h[j] >= U: n_up += 1; break
            t = e+1+60
        else:
            t += step*4
    return n_sig, n_hit, n_up


P(f"  {'하락폭':>8s} {'도달건':>7s} {'그후 +1.5% 먼저':>16s} {'반등확률%':>10s}")
for drop in (0.005, 0.0075, 0.01, 0.015, 0.02, 0.03):
    ns, nh, nu = bounce_stats(O, H, Lo, C, SIG, TH, drop, BAR, MAXH, STEP, warm, NEND)
    P(f"  {drop*100:7.2f}% {nh:7d} {nu:16d} {nu/max(nh,1)*100:10.2f}")
P("  (기저: 무조건 진입 시 +1.5% 먼저 닿을 확률 약 52.7%)")


@numba.njit(cache=True)
def walk_re(o, h, l, c, sigv, th, mu, bar, maxh, step, cool, lev, frac, mmr,
            fee_bp, slip_bp, start, n_end, max_re, mode, ruin):
    """mode: 0=재진입가 기준 +bar, 1=최초 진입가 회복 목표"""
    cap = 1.0; peak = 1.0; mdd = 0.0
    t = start; ntr = 0; nliq = 0; nwin = 0; nrec = 0
    max_chain = 0
    while t < n_end - maxh - 2:
        i = t//step
        if i >= len(sigv): break
        if not np.isfinite(sigv[i]) or sigv[i] < th: t += step; continue
        m = mu[t]
        if m <= 0.0: t += step; continue
        lv = lev*m
        liq = 1.0/lv - mmr
        stop = bar if bar < liq else liq
        j0 = t+1
        px0 = o[j0]
        cur = j0; px = px0; tries = 0; deadline = j0 + maxh
        if deadline > n_end-1: deadline = n_end-1
        chain = 0
        while True:
            U = px0*(1.0+bar) if (mode == 1 and tries > 0) else px*(1.0+bar)
            D = px*(1.0-stop)
            e = -1; res = 0.0; liqd = False
            for j in range(cur, deadline+1):
                if l[j] <= D:
                    e = j; liqd = (stop < bar - 1e-12)
                    res = -stop - (slip_bp/1e4 if liqd else 0.0)
                    break
                if h[j] >= U:
                    e = j; res = U/px - 1.0; break
            if e < 0:
                e = deadline; res = c[e]/px - 1.0
            margin = cap*frac
            pnl = margin*lv*(res - fee_bp/1e4)
            if pnl < -margin: pnl = -margin
            cap += pnl
            if cap > peak: peak = cap
            dd = cap/peak - 1.0
            if dd < mdd: mdd = dd
            ntr += 1
            if res > 0: nwin += 1
            if liqd:
                nliq += 1
                tries += 1; chain += 1
                if tries <= max_re and e < deadline - 5 and cap > ruin:
                    px = D; cur = e+1
                    continue
            if chain > max_chain: max_chain = chain
            if tries > 0 and res > 0: nrec += 1
            break
        t = e + 1 + cool
        if cap < ruin: break
    return cap, mdd, ntr, nliq, nwin, nrec, max_chain


def sim(lev, frac, max_re=0, mode=0, fee=10.0, slip=50.0):
    cap, mdd, ntr, nliq, nwin, nrec, mx = walk_re(
        O, H, Lo, C, SIG, TH, MU, BAR, MAXH, STEP, COOL, float(lev), float(frac),
        MMR, fee, slip, warm, NEND, max_re, mode, 0.01)
    if ntr < 30: return None
    return dict(ann=(max(cap, 1e-12)**(1/Y)-1)*100, mdd=mdd*100, fin=cap, n=ntr,
                nliq=nliq, wr=nwin/ntr*100, nrec=nrec, mx=mx, ruin=cap < 0.01)


P("\n"+"="*126); P("[1] 재진입 횟수 × 배율 (자본비 10%, mode=재진입가 기준)"); P("="*126)
P(f"  {'배율':>6s} " + " ".join(f"{'재진입'+str(k):>17s}" for k in (0, 1, 2, 3, 5, 10)))
BEST = []
for lev in (12, 20, 30, 50, 75, 100, 150):
    row = []
    for k in (0, 1, 2, 3, 5, 10):
        v = sim(lev, 0.10, max_re=k)
        if v is None or v["ruin"]: row.append("       파산      "); continue
        row.append(f"{v['ann']:+7.1f}%/{v['mdd']:6.1f}%")
        BEST.append((v["ann"], lev, 0.10, k, 0, v))
    P(f"  {lev:5d}x " + " ".join(f"{x:>17s}" for x in row))

P("\n[2] 같은 표, 자본비 20%")
P(f"  {'배율':>6s} " + " ".join(f"{'재진입'+str(k):>17s}" for k in (0, 1, 2, 3, 5, 10)))
for lev in (12, 20, 30, 50, 75, 100):
    row = []
    for k in (0, 1, 2, 3, 5, 10):
        v = sim(lev, 0.20, max_re=k)
        if v is None or v["ruin"]: row.append("       파산      "); continue
        row.append(f"{v['ann']:+7.1f}%/{v['mdd']:6.1f}%")
        BEST.append((v["ann"], lev, 0.20, k, 0, v))
    P(f"  {lev:5d}x " + " ".join(f"{x:>17s}" for x in row))

P("\n[3] 자본비 5% (작게 여러 번)")
P(f"  {'배율':>6s} " + " ".join(f"{'재진입'+str(k):>17s}" for k in (0, 2, 5, 10, 20)))
for lev in (30, 50, 75, 100, 150, 200):
    row = []
    for k in (0, 2, 5, 10, 20):
        v = sim(lev, 0.05, max_re=k)
        if v is None or v["ruin"]: row.append("       파산      "); continue
        row.append(f"{v['ann']:+7.1f}%/{v['mdd']:6.1f}%")
        BEST.append((v["ann"], lev, 0.05, k, 0, v))
    P(f"  {lev:5d}x " + " ".join(f"{x:>17s}" for x in row))

P("\n"+"="*126); P("[4] 원금 회복형(mode=1) — 최초 진입가 회복을 목표로"); P("="*126)
P(f"  {'배율':>6s} {'자본비':>6s} " + " ".join(f"{'재진입'+str(k):>17s}" for k in (1, 2, 3, 5)))
for lev, fr in ((30, 0.1), (50, 0.1), (50, 0.05), (100, 0.05), (100, 0.1)):
    row = []
    for k in (1, 2, 3, 5):
        v = sim(lev, fr, max_re=k, mode=1)
        if v is None or v["ruin"]: row.append("       파산      "); continue
        row.append(f"{v['ann']:+7.1f}%/{v['mdd']:6.1f}%")
        BEST.append((v["ann"], lev, fr, k, 1, v))
    P(f"  {lev:5d}x {fr*100:5.0f}% " + " ".join(f"{x:>17s}" for x in row))

P("\n"+"="*126); P("[5] 파산 없는 최고 구성 상위 12"); P("="*126)
BEST.sort(reverse=True)
P(f"  {'연수익%':>9s} {'MDD%':>8s} {'배율':>6s} {'자본비':>6s} {'명목':>6s} {'재진입':>6s} {'모드':>5s} "
  f"{'거래':>6s} {'청산':>6s} {'회복승':>6s} {'최장연쇄':>8s} {'최종배수':>11s}")
for a, lev, fr, k, md, v in BEST[:12]:
    P(f"  {a:9.1f} {v['mdd']:8.1f} {lev:5d}x {fr*100:5.0f}% {lev*fr:5.1f}x {k:6d} "
      f"{'회복' if md else '독립':>5s} {v['n']:6d} {v['nliq']:6d} {v['nrec']:6d} {v['mx']:8d} {v['fin']:11.1f}")

P("\n"+"="*126); P("[6] 재진입 구조 vs 단순 구조 — 같은 명목에서 비교"); P("="*126)
P(f"  {'명목':>6s} {'단순(저배율)':>18s} {'고배율+재진입5':>18s} {'고배율+재진입10':>18s}")
for nom in (1.0, 2.0, 3.0, 4.0, 5.0):
    a = sim(max(1, round(nom)), min(1.0, nom/max(1, round(nom))), max_re=0)
    lev_hi = 50; fr = nom/lev_hi
    b = sim(lev_hi, fr, max_re=5)
    c_ = sim(lev_hi, fr, max_re=10)
    f_ = lambda v: ("      파산       " if (v is None or v["ruin"]) else f"{v['ann']:+7.1f}%/{v['mdd']:6.1f}%")
    P(f"  {nom:5.1f}x {f_(a):>18s} {f_(b):>18s} {f_(c_):>18s}")

if BEST:
    a, lev, fr, k, md, v = BEST[0]
    P("\n"+"="*126); P(f"[7] 최고 구성 상세 — {lev}배 × 자본{fr*100:.0f}% × 재진입{k}회 ({'회복형' if md else '독립형'})")
    P("="*126)
    P(f"  거래 {v['n']}건 (연 {v['n']/Y:.0f}건), 청산 {v['nliq']}건 ({v['nliq']/v['n']*100:.1f}%), "
      f"청산 후 회복 성공 {v['nrec']}건, 최장 연쇄 청산 {v['mx']}회")
    P(f"  연 {v['ann']:+.1f}%  MDD {v['mdd']:.1f}%  최종 {v['fin']:.1f}배")
    P("\n  비용·슬리피지 민감도:")
    for fee, slip in ((10, 0), (10, 50), (10, 100), (15, 50), (20, 100)):
        vv = sim(lev, fr, max_re=k, mode=md, fee=fee, slip=slip)
        P(f"    수수료 {fee}bp 청산슬립 {slip}bp: " +
          ("파산" if (vv is None or vv["ruin"]) else f"{vv['ann']:+7.1f}%  MDD {vv['mdd']:6.1f}%"))
