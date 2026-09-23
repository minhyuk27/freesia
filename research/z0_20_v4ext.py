"""z0.24 확장 — ①ETH 재시도(손절 넓은 구조로) ②보유시간 상한 ③추적 익절 ④시간 기반 손절.

ETH는 대칭 ±1.5%에서 전 구성 음수였다. 그러나 z0.24의 메커니즘은 '반등을 버텨서 먹기'이고
이건 종목 특성보다 시장 구조에 가까우므로 다시 확인할 가치가 있다."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R/"research"))
import z0_20_lib as L
from z0_20_final import build_sigs, exp_z


def P(*a, end="\n"):
    print(" ".join(str(x) for x in a), end=end, flush=True)


Y = L.YRS; MMR = L.MMR; STEP = L.STEP
warm = 200000; _st = 5000
COOL = 60
DAT = {}
for S in ("BTCUSDT", "ETHUSDT"):
    d = L.D[S]; rv = d["rv"]
    med = np.full(len(rv), np.nan)
    for i in range(warm, len(rv), _st):
        x = rv[:i]; x = x[np.isfinite(x)]
        if len(x) >= 1000: med[i:i+_st] = np.median(x)
    SG = build_sigs(S)
    names = list(SG.keys())
    Zs = {n: exp_z(np.asarray(SG[n], float)) for n in names}
    mlen = min(min(len(Zs[n]) for n in names), len(d["c"])//STEP)
    sig = np.nanmean(np.vstack([Zs[n][:mlen] for n in names if n.endswith("4320")]), axis=0)
    sq = np.full(len(sig), np.nan)
    for i in range(warm//STEP, len(sig), _st//STEP):
        x = sig[:i]; x = x[np.isfinite(x)]
        if len(x) < 1000: continue
        sq[i:i+_st//STEP] = np.searchsorted(np.sort(x), sig[i:i+_st//STEP])/len(x)
    mu = np.zeros(len(d["c"]))
    for t in range(warm, len(d["c"])):
        m = 1.0
        r_ = rv[t]; mm = med[t]
        if np.isfinite(r_) and np.isfinite(mm) and r_ > 0:
            x = mm/r_
            m *= 0.1 if x < 0.1 else (4.0 if x > 4.0 else x)
        j = t//STEP
        s_ = sq[j] if j < len(sq) else np.nan
        if np.isfinite(s_): m *= (2.0 if s_ >= 0.97 else (1.0 if s_ >= 0.90 else 0.5))
        mu[t] = m
    DAT[S] = dict(o=d["o"], h=d["h"], l=d["l"], c=d["c"], idx=d["idx"], sig=sig, mu=mu)
    P(f"{S} 준비 완료")
P("")


@numba.njit(cache=True)
def walk(o, h, l, c, sigv, th, mu, up, dn, maxh, step, cool, nom, mmr,
         fee_bp, start, n_end, ruin, trail, tstop_min, tstop_lvl):
    """trail>0: 최고가 대비 trail만큼 하락 시 청산(추적 익절, 이익 구간에서만)
       tstop_min>0: tstop_min분 경과 시 수익이 tstop_lvl 미만이면 청산"""
    cap = 1.0; peak = 1.0; mdd = 0.0
    ntr = 0; nwin = 0; sum_r = 0.0; sum_hold = 0.0
    t = start
    while t < n_end - maxh - 2:
        i = t//step
        if i >= len(sigv): break
        if not np.isfinite(sigv[i]) or sigv[i] < th: t += step; continue
        m = mu[t]
        if m <= 0.0: t += step; continue
        lv = nom*m
        liq = 1.0/lv - mmr
        stop = dn if dn < liq else liq
        f = t+1; px = o[f]
        U = px*(1.0+up); D = px*(1.0-stop)
        e = -1; res = 0.0; hi = px
        end = f+maxh
        if end > n_end-1: end = n_end-1
        for j in range(f, end+1):
            if h[j] > hi: hi = h[j]
            if l[j] <= D: e = j; res = -stop; break
            if trail <= 0.0 and h[j] >= U: e = j; res = up; break
            if trail > 0.0:
                if hi >= px*(1.0+up):
                    tl = hi*(1.0-trail)
                    if l[j] <= tl: e = j; res = tl/px-1.0; break
            if tstop_min > 0 and (j-f) >= tstop_min:
                if c[j]/px-1.0 < tstop_lvl:
                    e = j; res = c[j]/px-1.0; break
        if e < 0:
            e = end; res = c[e]/px-1.0
        pnl = cap*lv*(res - fee_bp/1e4)
        if pnl < -cap: pnl = -cap
        cap += pnl
        if cap > peak: peak = cap
        dd = cap/peak-1.0
        if dd < mdd: mdd = dd
        ntr += 1; sum_r += res; sum_hold += (e-f)
        if res > 0: nwin += 1
        t = e+1+cool
        if cap < ruin: break
    return cap, mdd, ntr, nwin, sum_r, sum_hold


def sim(S, q, up, dn, nom=1.0, maxh=4320, fee=10.0, trail=0.0, ts=0, tl=0.0):
    D_ = DAT[S]
    fin = D_["sig"][np.isfinite(D_["sig"])]
    th = float(np.quantile(fin, 1-q))
    cap, mdd, ntr, nwin, sr, sh = walk(D_["o"], D_["h"], D_["l"], D_["c"], D_["sig"], th, D_["mu"],
                                       up, dn, maxh, STEP, COOL, float(nom), MMR, fee,
                                       warm, len(D_["c"]), 0.01, trail, ts, tl)
    if ntr < 30: return None
    return dict(ann=(max(cap, 1e-12)**(1/Y)-1)*100, mdd=mdd*100, fin=cap, n=ntr,
                wr=nwin/ntr*100, bp=sr/ntr*1e4, hold=sh/ntr, ruin=cap < 0.01)


def cell(v):
    return "      파산      " if (v is None or v["ruin"]) else f"{v['ann']:+7.1f}%/{v['wr']:5.1f}%"


P("="*128); P("[1] ETH 재시도 — 익절 × 손절 격자 (상위25%, 명목 1배). 셀 = 연수익%/승률%"); P("="*128)
for S in ("BTCUSDT", "ETHUSDT"):
    P(f"\n  ══ {S} ══")
    P(f"  {'익절':>7s} " + " ".join(f"{'손절'+str(x*100)+'%':>16s}" for x in (0.02, 0.025, 0.03, 0.04, 0.05)))
    for up in (0.01, 0.0125, 0.015, 0.02, 0.025):
        row = [cell(sim(S, 0.25, up, dn)) for dn in (0.02, 0.025, 0.03, 0.04, 0.05)]
        P(f"  {up*100:6.2f}% " + " ".join(f"{x:>16s}" for x in row))

P("\n"+"="*128); P("[2] ETH 분위 스윕 (익절1.5/손절3.0)"); P("="*128)
P(f"  {'분위':>6s} {'연거래':>6s} {'승률%':>7s} {'거래당bp':>9s} {'1배%':>9s} {'2배%':>9s} {'3배%':>9s}")
for q in (0.10, 0.16, 0.25, 0.35, 0.50):
    v = sim("ETHUSDT", q, 0.015, 0.03)
    if v is None: continue
    r = []
    for nom in (1, 2, 3):
        vv = sim("ETHUSDT", q, 0.015, 0.03, nom=nom)
        r.append("    파산" if (vv is None or vv["ruin"]) else f"{vv['ann']:+9.1f}")
    P(f"  {q*100:5.0f}% {v['n']/Y:6.0f} {v['wr']:7.2f} {v['bp']:+9.1f} " + " ".join(r))

P("\n"+"="*128); P("[3] 보유시간 상한 (BTC 상위25% 익절1.5/손절3.0, 명목 2배)"); P("="*128)
P(f"  {'상한':>9s} {'연거래':>6s} {'승률%':>7s} {'평균보유':>9s} {'거래당bp':>9s} {'연수익%':>10s} {'MDD%':>8s}")
for mh in (240, 720, 1440, 2880, 4320, 8640, 20160):
    v = sim("BTCUSDT", 0.25, 0.015, 0.03, nom=2.0, maxh=mh)
    if v is None: continue
    P(f"  {mh:6d}분 {v['n']/Y:6.0f} {v['wr']:7.2f} {v['hold']:8.0f}분 {v['bp']:+9.1f} "
      + ("      파산" if v["ruin"] else f"{v['ann']:10.1f} {v['mdd']:8.1f}"))

P("\n"+"="*128); P("[4] 추적 익절 — +1.5% 도달 후 고점 대비 trail% 하락 시 청산"); P("="*128)
P(f"  {'trail':>7s} {'연거래':>6s} {'승률%':>7s} {'거래당bp':>9s} {'1배%':>9s} {'2배%':>9s} {'3배%':>9s}")
for tr in (0.0, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02):
    v = sim("BTCUSDT", 0.25, 0.015, 0.03, trail=tr)
    if v is None: continue
    r = []
    for nom in (1, 2, 3):
        vv = sim("BTCUSDT", 0.25, 0.015, 0.03, nom=nom, trail=tr)
        r.append("    파산" if (vv is None or vv["ruin"]) else f"{vv['ann']:+9.1f}")
    lab = "없음" if tr == 0 else f"{tr*100:.2f}%"
    P(f"  {lab:>7s} {v['n']/Y:6.0f} {v['wr']:7.2f} {v['bp']:+9.1f} " + " ".join(r))

P("\n"+"="*128); P("[5] 시간 기반 손절 — N분 경과 시 수익 < lvl 이면 청산"); P("="*128)
P(f"  {'경과':>8s} {'기준':>7s} {'연거래':>6s} {'승률%':>7s} {'거래당bp':>9s} {'2배%':>9s}")
for ts in (60, 240, 720, 1440):
    for tl in (-0.01, 0.0, 0.005):
        v = sim("BTCUSDT", 0.25, 0.015, 0.03, nom=2.0, ts=ts, tl=tl)
        if v is None: continue
        P(f"  {ts:6d}분 {tl*100:+6.2f}% {v['n']/Y:6.0f} {v['wr']:7.2f} {v['bp']:+9.1f} "
          + ("    파산" if v["ruin"] else f"{v['ann']:+9.1f}"))

P("\n"+"="*128); P("[6] BTC+ETH 병합 운용 (단일 포지션, 겹치면 건너뜀)"); P("="*128)


def events(S, q, up, dn):
    D_ = DAT[S]
    fin = D_["sig"][np.isfinite(D_["sig"])]
    th = float(np.quantile(fin, 1-q))
    n_end = len(D_["c"]); o, h, l, c = D_["o"], D_["h"], D_["l"], D_["c"]
    out = []; t = warm; free = warm
    while t < n_end-4320-2:
        i = t//STEP
        if i >= len(D_["sig"]) or not np.isfinite(D_["sig"][i]) or D_["sig"][i] < th:
            t += STEP; continue
        m = D_["mu"][t]
        if m <= 0: t += STEP; continue
        f = t+1; px = o[f]; U = px*(1+up); Dn = px*(1-dn)
        e = -1; res = 0.0; end = min(f+4320, n_end-1)
        for j in range(f, end+1):
            if l[j] <= Dn: e = j; res = -dn; break
            if h[j] >= U: e = j; res = up; break
        if e < 0: e = end; res = c[e]/px-1.0
        out.append((D_["idx"][f], D_["idx"][e], res, m))
        t = e+1+COOL
    return out


B = events("BTCUSDT", 0.25, 0.015, 0.03)
E = events("ETHUSDT", 0.25, 0.015, 0.03)
P(f"  BTC {len(B)}건 (연 {len(B)/Y:.0f}), ETH {len(E)}건 (연 {len(E)/Y:.0f})")
al = sorted(B+E, key=lambda x: x[0])
merged = []; last = None
for ev in al:
    if last is not None and ev[0] <= last: continue
    merged.append(ev); last = ev[1]
P(f"  병합(단일 포지션) {len(merged)}건 (연 {len(merged)/Y:.0f}건, 이틀당 {len(merged)/Y/182.5:.2f}건)")
P(f"\n  {'구성':>14s} {'연거래':>6s} {'승률%':>7s} {'1배%':>9s} {'2배%':>9s} {'3배%':>9s} {'4배%':>9s}")
for lab, ev in (("BTC 단독", B), ("ETH 단독", E), ("BTC+ETH", merged)):
    row = []
    for nom in (1, 2, 3, 4):
        cap = 1.0; peak = 1.0; mdd = 0.0; dead = False
        for (a, b, res, m) in ev:
            lv = nom*m
            pnl = cap*lv*(res-10.0/1e4)
            if pnl < -cap: pnl = -cap
            cap += pnl
            if cap > peak: peak = cap
            mdd = min(mdd, cap/peak-1)
            if cap < 0.01: dead = True; break
        row.append("    파산" if dead else f"{(cap**(1/Y)-1)*100:+9.1f}")
    wr = np.mean([x[2] > 0 for x in ev])*100
    P(f"  {lab:>14s} {len(ev)/Y:6.0f} {wr:7.2f} " + " ".join(row))
