"""z2 다단 진입(scale-in) — 청산을 거치지 않고 반등을 먹는 구조.

발견: 신호 진입 후 -1% 찍은 자리에서의 승률이 60.07%로 초기 진입 52.7%보다 높다.
청산은 이 반등을 먹기 위한 수단이 아니다(청산되면 증거금을 잃고 나가므로 오히려 반대).
저배율로 버티다가 하락 단계마다 추가 진입해 평균단가를 낮추고 반등에서 전량 청산한다.

구조:
  신호 → 1차 진입(명목 w1)
  -d1% → 2차 진입(명목 w2), -d2% → 3차 진입(명목 w3) ...
  청산: 평균단가 +bar 도달 시 전량 / 최초가 기준 -stop 도달 시 전량 손절 / 시간만료

위험: 물타기는 지는 자리에 더 넣는 구조다. 손절을 반드시 둔다."""
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
SIG = np.nanmean(np.vstack([Z[n][:mlen] for n in names if n.endswith("4320")]), axis=0)
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


@numba.njit(cache=True)
def walk_si(o, h, l, c, sigv, th, mu, bar, stop, maxh, step, cool, base_lev, mmr,
            steps, wts, fee_bp, start, n_end, ruin, tp_mode):
    """steps: 추가진입 하락폭 배열(최초가 대비). wts: 각 단계 명목 가중(1차 포함).
    tp_mode: 0=평균단가 +bar, 1=최초가 +bar"""
    cap = 1.0; peak = 1.0; mdd = 0.0
    t = start; ntr = 0; nwin = 0; nliq = 0
    nfill = np.zeros(8, np.int64)
    while t < n_end - maxh - 2:
        i = t//step
        if i >= len(sigv): break
        if not np.isfinite(sigv[i]) or sigv[i] < th: t += step; continue
        m = mu[t]
        if m <= 0.0: t += step; continue
        j0 = t+1
        px0 = o[j0]
        nqty = wts[0]              # 명목 가중 합
        navg = px0*wts[0]          # 가중 가격 합
        k = 1                      # 다음 추가진입 단계
        deadline = j0+maxh
        if deadline > n_end-1: deadline = n_end-1
        e = -1; ret = 0.0
        SL = px0*(1.0-stop)
        for j in range(j0, deadline+1):
            # 추가 진입 (하락 단계 도달)
            while k < len(steps) and l[j] <= px0*(1.0-steps[k]):
                pk = px0*(1.0-steps[k])
                navg += pk*wts[k]; nqty += wts[k]
                nfill[k] += 1
                k += 1
            avg = navg/nqty
            TP = (avg if tp_mode == 0 else px0)*(1.0+bar)
            if l[j] <= SL:
                e = j; ret = (SL/avg-1.0)*nqty; break
            if h[j] >= TP:
                e = j; ret = (TP/avg-1.0)*nqty; nwin += 1; break
        if e < 0:
            e = deadline; avg = navg/nqty; ret = (c[e]/avg-1.0)*nqty
        # 비용: 진입 단계 수 × 왕복 절반씩 (명목 가중)
        cost = fee_bp/1e4*nqty
        lv = base_lev*m
        # 청산 점검: 총 명목 대비 최대 역행이 청산거리를 넘으면 청산
        maxnom = nqty*lv
        liq = 1.0/maxnom - mmr if maxnom > 0 else 9.9
        if liq < stop:
            ret = -stop*nqty
            nliq += 1
        pnl = cap*lv*(ret - cost)
        if pnl < -cap: pnl = -cap
        cap += pnl
        if cap > peak: peak = cap
        dd = cap/peak-1.0
        if dd < mdd: mdd = dd
        ntr += 1
        t = e+1+cool
        if cap < ruin: break
    return cap, mdd, ntr, nwin, nliq, nfill


def sim(lev, steps, wts, stop=0.025, bar=BAR, fee=10.0, tp_mode=0):
    st = np.array(steps, float); wt = np.array(wts, float)
    cap, mdd, ntr, nwin, nliq, nf = walk_si(O, H, Lo, C, SIG, TH, MU, bar, stop, MAXH, STEP,
                                            COOL, float(lev), MMR, st, wt, fee, warm, NEND,
                                            0.01, tp_mode)
    if ntr < 30: return None
    return dict(ann=(max(cap, 1e-12)**(1/Y)-1)*100, mdd=mdd*100, fin=cap, n=ntr,
                wr=nwin/ntr*100, nliq=nliq, fills=nf[:len(steps)].copy(), ruin=cap < 0.01)


def cell(v):
    return "       파산      " if (v is None or v["ruin"]) else f"{v['ann']:+7.1f}%/{v['mdd']:6.1f}%"


P("="*126); P("[1] 단계 수 × 손절폭 (명목 1배 기준, 각 단계 동일 가중, 평균단가 +1.5% 익절)"); P("="*126)
LADDERS = [
    ("단일(비교)", [0.0], [1.0]),
    ("2단 -1%", [0.0, 0.01], [1.0, 1.0]),
    ("2단 -0.75%", [0.0, 0.0075], [1.0, 1.0]),
    ("3단 -0.75/-1.5%", [0.0, 0.0075, 0.015], [1.0, 1.0, 1.0]),
    ("3단 -1/-2%", [0.0, 0.01, 0.02], [1.0, 1.0, 1.0]),
    ("4단 -0.5/-1/-1.5%", [0.0, 0.005, 0.01, 0.015], [1.0, 1.0, 1.0, 1.0]),
    ("3단 가중 1:2:3", [0.0, 0.01, 0.02], [1.0, 2.0, 3.0]),
    ("3단 가중 3:2:1", [0.0, 0.01, 0.02], [3.0, 2.0, 1.0]),
]
P(f"  {'사다리':>20s} " + " ".join(f"{'손절'+str(s*100)+'%':>17s}" for s in (0.02, 0.025, 0.03, 0.04, 0.06)))
BEST = []
for lab, st, wt in LADDERS:
    nom = sum(wt)
    lev = 1.0/nom                 # 총 명목을 1배로 정규화
    row = []
    for stop in (0.02, 0.025, 0.03, 0.04, 0.06):
        v = sim(lev, st, wt, stop=stop)
        row.append(cell(v))
        if v and not v["ruin"]: BEST.append((v["ann"], lab, st, wt, stop, lev, v))
    P(f"  {lab:>20s} " + " ".join(f"{x:>17s}" for x in row))

P("\n"+"="*126); P("[2] 최적 사다리에 레버리지 — 명목 1~6배"); P("="*126)
BEST.sort(reverse=True)
if BEST:
    _, lab, st, wt, stop, _, _ = BEST[0]
    P(f"  사다리: {lab}, 손절 {stop*100:.1f}%")
    P(f"  {'명목':>6s} {'연수익%':>10s} {'MDD%':>8s} {'승률%':>7s} {'거래':>6s} {'청산':>6s} {'최종배수':>11s}")
    for nom in (1, 1.5, 2, 3, 4, 5, 6, 8):
        v = sim(nom/sum(wt), st, wt, stop=stop)
        if v is None or v["ruin"]: P(f"  {nom:5.1f}x        파산"); continue
        P(f"  {nom:5.1f}x {v['ann']:10.1f} {v['mdd']:8.1f} {v['wr']:7.2f} {v['n']:6d} {v['nliq']:6d} {v['fin']:11.1f}")

P("\n"+"="*126); P("[3] 익절 기준 — 평균단가 vs 최초가"); P("="*126)
P(f"  {'사다리':>20s} {'평균단가 기준':>18s} {'최초가 기준':>18s}")
for lab, st, wt in LADDERS[1:]:
    lev = 1.0/sum(wt)
    a = sim(lev, st, wt, stop=0.03, tp_mode=0)
    b = sim(lev, st, wt, stop=0.03, tp_mode=1)
    P(f"  {lab:>20s} {cell(a):>18s} {cell(b):>18s}")

P("\n"+"="*126); P("[4] 익절폭 고원 (최적 사다리, 명목 3배)"); P("="*126)
if BEST:
    _, lab, st, wt, stop, _, _ = BEST[0]
    P(f"  {'익절':>7s} {'연수익%':>10s} {'MDD%':>8s} {'승률%':>7s} {'거래':>6s}")
    for bar in (0.0075, 0.01, 0.0125, 0.015, 0.02, 0.025, 0.03):
        v = sim(3.0/sum(wt), st, wt, stop=stop, bar=bar)
        if v is None or v["ruin"]: P(f"  {bar*100:6.2f}%        파산"); continue
        P(f"  {bar*100:6.2f}% {v['ann']:10.1f} {v['mdd']:8.1f} {v['wr']:7.2f} {v['n']:6d}")

P("\n"+"="*126); P("[5] 상위 8개 구성"); P("="*126)
P(f"  {'연수익%':>9s} {'MDD%':>8s} {'승률%':>7s} {'손절':>6s} {'사다리':>22s} {'단계별 체결수':>22s}")
for a, lab, st, wt, stop, lev, v in BEST[:8]:
    P(f"  {a:9.1f} {v['mdd']:8.1f} {v['wr']:7.2f} {stop*100:5.1f}% {lab:>22s}   {str(list(v['fills'])):>20s}")

P("\n"+"="*126); P("[6] 최고 구성 3단 레버리지 + 연도별 + 비용"); P("="*126)
if BEST:
    _, lab, st, wt, stop, _, _ = BEST[0]
    P(f"  구성: {lab} / 손절 {stop*100:.1f}% / 익절 평균단가 +1.5%")
    P(f"\n  {'구간':>12s} {'명목':>6s} {'연수익%':>10s} {'MDD%':>8s} {'최종배수':>11s}")
    for lb, nom in (("무레버리지", 1.0), ("중간", 2.0), ("중간+", 3.0), ("공격", 4.0), ("최대공격", 5.0)):
        v = sim(nom/sum(wt), st, wt, stop=stop)
        P(f"  {lb:>12s} {nom:5.1f}x " + ("       파산" if (v is None or v["ruin"])
                                          else f"{v['ann']:10.1f} {v['mdd']:8.1f} {v['fin']:11.1f}"))
    P("\n  비용 민감도(명목 3배):")
    for fee in (4, 7, 10, 15, 20, 30):
        v = sim(3.0/sum(wt), st, wt, stop=stop, fee=fee)
        P(f"    왕복 {fee:2d}bp: " + ("파산" if (v is None or v["ruin"]) else f"{v['ann']:+7.1f}%  MDD {v['mdd']:6.1f}%"))
