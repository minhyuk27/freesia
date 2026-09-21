"""z2.4 — 익절 좁게·손절 넓게. '반등을 먹으려면 잘리지 않아야 한다'의 구현.

발견: 익절 +1.5% / 손절 -3.0%가 대칭 ±1.5%보다 훨씬 낫다(무레버 +17.2% → +32.2%, 승률 52.7% → 68.0%).
이유: 신호 후 -1% 찍은 자리의 반등 확률이 60.07%인데, 손절 1.5%면 그 반등을 먹기 전에 잘린다.

여기서: 익절·손절 격자 전면 탐색 + 빈도 맞추기(이틀 1건) + 플라시보·비용·고원 확정 검증."""
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
COOL = 60; NEND = len(C)


@numba.njit(cache=True)
def walk(o, h, l, c, sigv, th, mu, up, dn, maxh, step, cool, nom, mmr,
         fee_bp, slip_bp, start, n_end, ruin, rnd_starts):
    """nom = 목표 총 명목(사이징 승수 곱해짐). rnd_starts: 길이>0이면 그 시점들에서 진입(플라시보)."""
    cap = 1.0; peak = 1.0; mdd = 0.0
    ntr = 0; nwin = 0; nliq = 0; sum_r = 0.0
    use_rnd = len(rnd_starts) > 0
    si = 0; t = start; free = start
    while True:
        if use_rnd:
            if si >= len(rnd_starts): break
            t = rnd_starts[si]; si += 1
            if t < free or t >= n_end-maxh-2: continue
        else:
            if t >= n_end-maxh-2: break
            i = t//step
            if i >= len(sigv): break
            if not np.isfinite(sigv[i]) or sigv[i] < th: t += step; continue
        m = mu[t]
        if m <= 0.0:
            t += step; continue
        lv = nom*m
        liq = 1.0/lv - mmr
        stop = dn if dn < liq else liq
        f = t+1; px = o[f]
        U = px*(1.0+up); D = px*(1.0-stop)
        e = -1; res = 0.0; wasliq = False
        end = f+maxh
        if end > n_end-1: end = n_end-1
        for j in range(f, end+1):
            if l[j] <= D:
                e = j; wasliq = stop < dn-1e-12
                res = -stop - (slip_bp/1e4 if wasliq else 0.0); break
            if h[j] >= U:
                e = j; res = up; nwin += 1; break
        if e < 0:
            e = end; res = c[e]/px-1.0
        pnl = cap*lv*(res - fee_bp/1e4)
        if pnl < -cap: pnl = -cap
        cap += pnl
        if cap > peak: peak = cap
        dd = cap/peak-1.0
        if dd < mdd: mdd = dd
        ntr += 1; sum_r += res
        if wasliq: nliq += 1
        free = e+1+cool
        t = free
        if cap < ruin: break
    return cap, mdd, ntr, nwin, nliq, sum_r


EMPTY = np.zeros(0, np.int64)


def sim(q, up, dn, nom=1.0, fee=10.0, slip=50.0, rnd=EMPTY, yr=None):
    th = float(np.quantile(SIG[np.isfinite(SIG)], 1-q))
    cap, mdd, ntr, nwin, nliq, sr = walk(O, H, Lo, C, SIG, th, MU, up, dn, MAXH, STEP, COOL,
                                         float(nom), MMR, fee, slip, warm, NEND, 0.01, rnd)
    if ntr < 30: return None
    return dict(ann=(max(cap, 1e-12)**(1/Y)-1)*100, mdd=mdd*100, fin=cap, n=ntr,
                wr=nwin/ntr*100, nliq=nliq, bp=sr/ntr*1e4, ruin=cap < 0.01)


def cell(v):
    return "      파산      " if (v is None or v["ruin"]) else f"{v['ann']:+7.1f}%/{v['wr']:5.1f}%"


P("="*128); P("[1] 익절 × 손절 격자 (상위16%, 명목 1배) — 연수익%/승률%"); P("="*128)
UPS = (0.0075, 0.01, 0.0125, 0.015, 0.02, 0.025)
DNS = (0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.07)
P(f"  {'익절':>7s} " + " ".join(f"{'손절'+str(x*100)+'%':>16s}" for x in DNS))
GRID = []
for up in UPS:
    row = []
    for dn in DNS:
        v = sim(0.16, up, dn)
        row.append(cell(v))
        if v and not v["ruin"]: GRID.append((v["ann"], up, dn, 0.16, v))
    P(f"  {up*100:6.2f}% " + " ".join(f"{x:>16s}" for x in row))

P("\n"+"="*128); P("[2] 빈도 맞추기 — 분위 × 손절 (익절 1.5%, 명목 1배). 셀 = 연수익%/연거래"); P("="*128)
P(f"  {'분위':>6s} " + " ".join(f"{'손절'+str(x*100)+'%':>16s}" for x in (0.025, 0.03, 0.04, 0.05)))
CAND = []
for q in (0.16, 0.20, 0.25, 0.30, 0.40, 0.50):
    row = []
    for dn in (0.025, 0.03, 0.04, 0.05):
        v = sim(q, 0.015, dn)
        if v is None or v["ruin"]: row.append("      파산      "); continue
        row.append(f"{v['ann']:+7.1f}%/{v['n']/Y:4.0f}")
        if v["n"]/Y >= 182.5: CAND.append((v["ann"], q, 0.015, dn, v))
    P(f"  {q*100:5.0f}% " + " ".join(f"{x:>16s}" for x in row))

P("\n  연 182.5건(이틀당 1건) 이상 중 상위:")
CAND.sort(reverse=True)
for a, q, up, dn, v in CAND[:6]:
    P(f"    상위{q*100:.0f}% 익절{up*100:.2f}/손절{dn*100:.2f}%: 연 {a:+7.1f}%  MDD {v['mdd']:6.1f}%  "
      f"연거래 {v['n']/Y:4.0f}  승률 {v['wr']:5.2f}%  거래당 {v['bp']:+5.1f}bp")

if not CAND:
    P("    없음 → 분위를 더 넓히거나 쿨다운을 줄여야 한다")
    CAND = sorted(GRID, reverse=True)[:1]
    CAND = [(CAND[0][0], CAND[0][3], CAND[0][1], CAND[0][2], CAND[0][4])]

A, Q, UP, DN, V = CAND[0]
P(f"\n  → 채택: 상위{Q*100:.0f}% 익절{UP*100:.2f}% 손절{DN*100:.2f}% (연 {V['n']/Y:.0f}건, "
  f"이틀당 {V['n']/Y/182.5:.2f}건)")

P("\n"+"="*128); P("[3] 3단 레버리지 + 연도별"); P("="*128)
P(f"  {'구간':>12s} {'명목':>6s} {'연수익%':>10s} {'MDD%':>8s} {'최종배수':>11s} {'청산':>6s}")
for lb, nom in (("무레버리지", 1.0), ("중간", 2.0), ("중간+", 3.0), ("공격", 4.0), ("최대공격", 5.0), ("극단", 6.0)):
    v = sim(Q, UP, DN, nom=nom)
    P(f"  {lb:>12s} {nom:5.1f}x " + ("       파산" if (v is None or v["ruin"])
                                      else f"{v['ann']:10.1f} {v['mdd']:8.1f} {v['fin']:11.1f} {v['nliq']:6d}"))


def yearly(nom):
    th = float(np.quantile(SIG[np.isfinite(SIG)], 1-Q))
    out = []
    for yr in range(2020, 2027):
        m = (idx.year == yr)
        ii = np.where(m)[0]
        if len(ii) < 50000: out.append(f"{yr}  n/a"); continue
        a, b = max(ii[0], warm), ii[-1]
        if a >= b-MAXH: out.append(f"{yr}  n/a"); continue
        cap, mdd, ntr, nwin, nliq, sr = walk(O, H, Lo, C, SIG, th, MU, UP, DN, MAXH, STEP, COOL,
                                             float(nom), MMR, 10.0, 50.0, a, b, 0.01, EMPTY)
        out.append(f"{yr} {(cap-1)*100:+7.0f}%" if ntr >= 5 else f"{yr}  n/a")
    return out


for nom in (1.0, 2.0, 3.0):
    P(f"  연도별({nom:.0f}배): " + " | ".join(yearly(nom)))

P("\n"+"="*128); P("[4] 플라시보 300회 (무작위 시점, 같은 규칙·같은 건수)"); P("="*128)
rng = np.random.default_rng(31)
th = float(np.quantile(SIG[np.isfinite(SIG)], 1-Q))
ncand = int((np.isfinite(SIG) & (SIG >= th)).sum())
real = sim(Q, UP, DN); real3 = sim(Q, UP, DN, nom=3.0)
pb, p1, p3 = [], [], []
for it in range(300):
    st = np.sort(rng.integers(warm, NEND-MAXH-3, size=ncand)).astype(np.int64)
    v1 = sim(Q, UP, DN, rnd=st); v3 = sim(Q, UP, DN, nom=3.0, rnd=st)
    if v1 is None: continue
    pb.append(v1["bp"]); p1.append(v1["ann"])
    p3.append(v3["ann"] if (v3 and not v3["ruin"]) else -99.9)
pb, p1, p3 = map(np.array, (pb, p1, p3))
P(f"  거래당bp  실제 {real['bp']:+7.2f} | 위약 평균 {pb.mean():+6.2f} 최대 {pb.max():+6.2f} "
  f"→ 백분위 {np.mean(pb < real['bp'])*100:5.1f}%")
P(f"  1배 연%   실제 {real['ann']:+7.1f} | 위약 평균 {p1.mean():+6.1f} 최대 {p1.max():+6.1f} "
  f"→ 백분위 {np.mean(p1 < real['ann'])*100:5.1f}%")
if real3 and not real3["ruin"]:
    P(f"  3배 연%   실제 {real3['ann']:+7.1f} | 위약 평균 {p3.mean():+6.1f} 최대 {p3.max():+6.1f} "
      f"→ 백분위 {np.mean(p3 < real3['ann'])*100:5.1f}%")

P("\n"+"="*128); P("[5] 비용·슬리피지 민감도"); P("="*128)
for fee in (4, 7, 10, 15, 20, 25, 30):
    row = []
    for nom in (1.0, 2.0, 3.0):
        v = sim(Q, UP, DN, nom=nom, fee=fee)
        row.append("     파산" if (v is None or v["ruin"]) else f"{v['ann']:+8.1f}%")
    P(f"  왕복 {fee:2d}bp: 1배{row[0]}  2배{row[1]}  3배{row[2]}")

P("\n"+"="*128); P("[6] 파라미터 고원 (명목 2배)"); P("="*128)
P("  익절 고원(손절 고정):")
for up in (0.01, 0.0125, 0.015, 0.0175, 0.02, 0.025):
    v = sim(Q, up, DN, nom=2.0)
    P(f"    익절 {up*100:5.2f}%: " + ("파산" if (v is None or v["ruin"])
                                     else f"연 {v['ann']:+7.1f}%  승률 {v['wr']:5.2f}%  연거래 {v['n']/Y:4.0f}"))
P("  손절 고원(익절 고정):")
for dn in (0.02, 0.025, 0.03, 0.04, 0.05, 0.07, 0.10):
    v = sim(Q, UP, dn, nom=2.0)
    P(f"    손절 {dn*100:5.2f}%: " + ("파산" if (v is None or v["ruin"])
                                     else f"연 {v['ann']:+7.1f}%  승률 {v['wr']:5.2f}%  연거래 {v['n']/Y:4.0f}"))
P("  분위 고원:")
for q in (0.10, 0.16, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70, 1.0):
    v = sim(q, UP, DN, nom=2.0)
    P(f"    상위 {q*100:5.0f}%: " + ("파산" if (v is None or v["ruin"])
                                    else f"연 {v['ann']:+7.1f}%  승률 {v['wr']:5.2f}%  연거래 {v['n']/Y:4.0f}"))
