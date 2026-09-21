"""z2 격리마진 — 거래마다 실제 배율로 청산거리를 계산하는 정합 시뮬.

이전 판(z2_iso.py)의 오류: 사이징으로 배율이 거래마다 달라지는데
하단 배리어를 기본 배율 하나로 고정해 거래를 생성했다. 서로 다른 전략을 섞은 꼴.

여기서는 한 함수 안에서 순차적으로:
  진입후보 t → 사이징 승수 mu[t] → 실제 배율 lv = lev*mu → 청산거리 = 1/lv - mmr
  → 실효 하단 = min(손절, 청산거리) → 그 하단으로 전방 스캔 → 자본 갱신
청산 시 손실은 증거금 전액(+청산 슬리피지)."""
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
STEP = L.STEP; MAXH = L.MAXH
warm = 200000; _st = 5000
med = np.full(len(rv), np.nan)
for i in range(warm, len(rv), _st):
    x = rv[:i]; x = x[np.isfinite(x)]
    if len(x) >= 1000: med[i:i+_st] = np.median(x)

SG = build_sigs(S)
names = list(SG.keys())
Z = {n: exp_z(np.asarray(SG[n], float)) for n in names}
mlen = min(min(len(Z[n]) for n in names), len(d["c"])//STEP)
PICK = [n for n in names if n.endswith("4320")]
SIG = np.nanmean(np.vstack([Z[n][:mlen] for n in PICK]), axis=0)
SQ = np.full(len(SIG), np.nan)
for i in range(warm//STEP, len(SIG), _st//STEP):
    x = SIG[:i]; x = x[np.isfinite(x)]
    if len(x) < 1000: continue
    SQ[i:i+_st//STEP] = np.searchsorted(np.sort(x), SIG[i:i+_st//STEP])/len(x)

# 분 단위 사이징 승수 (자본 무관, 시점 t 정보만)
MU = np.zeros(len(d["c"]))
for t in range(warm, len(d["c"])):
    mu = 1.0
    r_ = rv[t-1]; mm = med[t]
    if np.isfinite(r_) and np.isfinite(mm) and r_ > 0:
        x = mm/r_
        mu *= 0.1 if x < 0.1 else (4.0 if x > 4.0 else x)
    j = (t-1)//STEP
    s_ = SQ[j] if j < len(SQ) else np.nan
    if np.isfinite(s_): mu *= (2.0 if s_ >= 0.97 else (1.0 if s_ >= 0.90 else 0.5))
    MU[t] = mu
P(f"신호: 5계열 × 창4320 균등 | 사이징 승수 중앙 {np.median(MU[warm:]):.2f} "
  f"평균 {MU[warm:].mean():.2f} 90% {np.percentile(MU[warm:],90):.2f}\n")

fin = SIG[np.isfinite(SIG)]


@numba.njit(cache=True)
def walk(o, h, l, c, sigv, th, mu, bar, maxh, step, cool, lev, frac, mmr,
         fee_bp, liq_slip_bp, start, n_end, ruin_frac):
    """정합 시뮬: 거래마다 실제 배율로 청산거리를 정한다."""
    cap = 1.0; peak = 1.0; mdd = 0.0
    t = start; ntr = 0; nliq = 0; nwin = 0
    sum_stop = 0.0
    while t < n_end - maxh - 2:
        i = t//step
        if i >= len(sigv): break
        if not np.isfinite(sigv[i]) or sigv[i] < th:
            t += step; continue
        m = mu[t]
        if m <= 0.0: t += step; continue
        lv = lev*m
        liq = 1.0/lv - mmr
        stop = bar if bar < liq else liq          # 실효 하단
        f = t+1; px = o[f]
        U = px*(1.0+bar); D = px*(1.0-stop)
        e = -1; res = 0.0; was_liq = False
        end = f+maxh
        if end > n_end-1: end = n_end-1
        for j in range(f, end+1):
            if l[j] <= D:
                e = j
                if stop < bar - 1e-12:            # 청산이 먼저 온 경우
                    res = -stop - liq_slip_bp/1e4
                    was_liq = True
                else:
                    res = -stop
                break
            if h[j] >= U:
                e = j; res = bar; nwin += 1; break
        if e < 0:
            e = end; res = c[e]/px - 1.0
        margin = cap*frac
        pnl = margin*lv*(res - fee_bp/1e4)
        if pnl < -margin: pnl = -margin           # 격리: 증거금까지만
        cap += pnl
        if cap > peak: peak = cap
        dd = cap/peak - 1.0
        if dd < mdd: mdd = dd
        ntr += 1; sum_stop += stop
        if was_liq: nliq += 1
        t = e + 1 + cool
        if cap < ruin_frac: break
    return cap, mdd, ntr, nliq, nwin, sum_stop


BAR = 0.015; Q = 0.16; COOL = 60
TH = float(np.quantile(fin, 1-Q))
NEND = len(d["c"])


def sim(lev, frac, fee=10.0, slip=0.0, yr=None):
    a, b, c_, e = d["o"], d["h"], d["l"], d["c"]
    cap, mdd, ntr, nliq, nwin, sstop = walk(a, b, c_, e, SIG, TH, MU, BAR, MAXH, STEP, COOL,
                                            float(lev), float(frac), MMR, fee, slip,
                                            warm, NEND, 0.01)
    if ntr < 30: return None
    return dict(ann=(max(cap, 1e-12)**(1/Y)-1)*100, mdd=mdd*100, fin=cap, n=ntr,
                nliq=nliq, wr=nwin/ntr*100, stop=sstop/ntr, ruin=cap < 0.01)


P("="*126); P("[1] 정합 시뮬 — 배율 × 자본비율 (연수익% / 최종배수 / 청산건수)"); P("="*126)
FR = (1.0, 0.5, 0.3, 0.2, 0.1, 0.05)
P(f"  {'배율':>6s} " + " ".join(f"{'자본'+str(int(f*100))+'%':>18s}" for f in FR))
ALL = []
for lev in (1, 3, 5, 8, 12, 20, 30, 50, 75, 100, 150):
    row = []
    for fr in FR:
        v = sim(lev, fr)
        if v is None: row.append("         -        "); continue
        if v["ruin"]: row.append("       파산       "); continue
        row.append(f"{v['ann']:+7.1f}%/{v['fin']:7.1f}x/{v['nliq']:3d}")
        ALL.append((v["ann"], lev, fr, v))
    P(f"  {lev:5d}x " + " ".join(f"{x:>18s}" for x in row))

P("\n"+"="*126); P("[2] 실효 손절·승률·청산 — 배율이 올라가면 무슨 일이 일어나나 (자본100% 기준)"); P("="*126)
P(f"  {'배율':>6s} {'평균실효손절':>12s} {'승률%':>7s} {'거래수':>6s} {'청산건':>6s} {'청산비율%':>9s} {'연수익%':>10s} {'MDD%':>8s}")
for lev in (1, 5, 12, 20, 30, 40, 50, 60, 75, 100, 150):
    v = sim(lev, 1.0)
    if v is None: continue
    tag = "파산" if v["ruin"] else f"{v['ann']:10.1f}"
    P(f"  {lev:5d}x {v['stop']*100:11.3f}% {v['wr']:7.2f} {v['n']:6d} {v['nliq']:6d} "
      f"{v['nliq']/v['n']*100:9.1f} {tag:>10s} {v['mdd']:8.1f}")

P("\n"+"="*126); P("[3] 명목(배율×자본비율) 등가성 — 격리가 실제로 다른 결과를 주는가"); P("="*126)
P(f"  {'명목':>6s} " + " ".join(f"{'L'+str(l)+'x':>17s}" for l in (3, 5, 10, 20, 40, 80)))
for nom in (1, 2, 3, 4, 5, 6, 8, 10):
    row = []
    for lv in (3, 5, 10, 20, 40, 80):
        fr = nom/lv
        if fr > 1.0: row.append("        -        "); continue
        v = sim(lv, fr)
        if v is None: row.append("        -        "); continue
        row.append("      파산       " if v["ruin"] else f"{v['ann']:+7.1f}%/{v['mdd']:6.1f}%")
    P(f"  {nom:5.1f}x " + " ".join(f"{x:>17s}" for x in row))

P("\n"+"="*126); P("[4] 청산 슬리피지를 넣으면 (격리 청산은 불리하게 체결된다)"); P("="*126)
P(f"  {'배율':>6s} {'자본비':>6s} " + " ".join(f"{'슬립'+str(s)+'bp':>15s}" for s in (0, 20, 50, 100)))
for lev, fr in ((20, 0.5), (30, 0.3), (50, 0.2), (50, 0.1), (75, 0.1), (100, 0.1)):
    row = []
    for sl in (0, 20, 50, 100):
        v = sim(lev, fr, slip=sl)
        row.append("     파산     " if (v is None or v["ruin"]) else f"{v['ann']:+7.1f}%/{v['mdd']:5.0f}%")
    P(f"  {lev:5d}x {fr*100:5.0f}% " + " ".join(f"{x:>15s}" for x in row))

P("\n"+"="*126); P("[5] 파산 없는 최고 — 명목 상한 찾기"); P("="*126)
ALL.sort(reverse=True)
P(f"  {'연수익%':>10s} {'MDD%':>8s} {'배율':>6s} {'자본비':>6s} {'명목':>6s} {'청산건':>6s} {'승률%':>7s} {'최종배수':>12s}")
for a, lev, fr, v in ALL[:12]:
    P(f"  {a:10.1f} {v['mdd']:8.1f} {lev:5d}x {fr*100:5.0f}% {lev*fr:5.1f}x {v['nliq']:6d} {v['wr']:7.2f} {v['fin']:12.1f}")

P("\n"+"="*126); P("[6] 3단 레버리지 확정 + 연도별 (청산 슬리피지 50bp 포함)"); P("="*126)
P(f"  {'구간':>14s} {'구성':>20s} {'연수익%':>10s} {'MDD%':>8s} {'최종배수':>11s} {'청산':>6s}")
TIER = [("무레버리지", 1, 1.0), ("중간", 3, 1.0), ("중간+", 5, 1.0),
        ("공격", 8, 1.0), ("최대공격", 12, 1.0)]
if ALL: TIER.append(("탐색 최고", ALL[0][1], ALL[0][2]))
for lb, lev, fr in TIER:
    v = sim(lev, fr, slip=50.0)
    cfg = f"{lev}x × 자본{fr*100:.0f}%"
    if v is None or v["ruin"]: P(f"  {lb:>14s} {cfg:>20s}        파산"); continue
    P(f"  {lb:>14s} {cfg:>20s} {v['ann']:10.1f} {v['mdd']:8.1f} {v['fin']:11.1f} {v['nliq']:6d}")
