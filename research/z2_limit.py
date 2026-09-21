"""z2.5 — 지정가 대기 진입. 시장가로 쫓지 말고 아래에 깔고 기다린다.

근거: 신호 후 -1% 찍은 자리의 승률 60.07% > 진입 직후 52.7%.
z0.6·z1.4에서 확립된 '시장가 추격 대신 지정가 대기'와 같은 발상.
덤으로 진입이 메이커가 되어 수수료도 준다.

진입: 신호 봉 종가 × (1-k) 에 지정가, W분 유효. 미체결이면 취소.
청산: 익절 +1.5%(진입가 기준), 손절 -3.0%(진입가 기준)."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R/"research"))
import z2_lib as L
from z2_final import build_sigs, exp_z


def P(*a, end="\n"):
    print(" ".join(str(x) for x in a), end=end, flush=True)


S = "BTCUSDT"; Y = L.YRS; MMR = L.MMR; STEP = L.STEP
d = L.D[S]; idx = d["idx"]; rv = d["rv"]
O, H, Lo, C = d["o"], d["h"], d["l"], d["c"]
warm = 200000; _st = 5000; COOL = 60; MAXH = 8640
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
    m = 1.0
    r_ = rv[t]; mm = med[t]
    if np.isfinite(r_) and np.isfinite(mm) and r_ > 0:
        x = mm/r_
        m *= 0.1 if x < 0.1 else (4.0 if x > 4.0 else x)
    j = t//STEP
    s_ = SQ[j] if j < len(SQ) else np.nan
    if np.isfinite(s_): m *= (2.0 if s_ >= 0.97 else (1.0 if s_ >= 0.90 else 0.5))
    MU[t] = m
NEND = len(C)


@numba.njit(cache=True)
def walk(o, h, l, c, sigv, th, mu, up, dn, maxh, step, cool, nom, mmr,
         fee_in, fee_out, k_abs, k_rv, rvv, W, eps, start, n_end, ruin, rnd):
    """k_abs>0: 절대 깊이(비율). k_rv>0: rv 배수 깊이. 둘 다 0이면 시장가.
    fee_in/out: 진입·청산 수수료 bp"""
    cap = 1.0; peak = 1.0; mdd = 0.0
    ntr = 0; nwin = 0; nskip = 0; sum_r = 0.0
    use_rnd = len(rnd) > 0
    si = 0; t = start; free = start
    while True:
        if use_rnd:
            if si >= len(rnd): break
            t = rnd[si]; si += 1
            if t < free or t >= n_end-maxh-W-2: continue
        else:
            if t >= n_end-maxh-W-2: break
            i = t//step
            if i >= len(sigv): break
            if not np.isfinite(sigv[i]) or sigv[i] < th: t += step; continue
        m = mu[t]
        if m <= 0.0: t += step; continue
        if k_abs > 0.0 or k_rv > 0.0:
            depth = k_abs
            if k_rv > 0.0:
                if np.isnan(rvv[t]) or rvv[t] <= 0: t += step; continue
                depth = rvv[t]*k_rv
            T = c[t]*(1.0-depth)
            f = -1
            for u in range(t+1, min(t+1+W, n_end)):
                if l[u] <= T*(1.0-eps): f = u; break
            if f < 0:
                nskip += 1; t += W; continue
            px = T
        else:
            f = t+1; px = o[f]
        lv = nom*m
        liq = 1.0/lv - mmr
        stop = dn if dn < liq else liq
        U = px*(1.0+up); D = px*(1.0-stop)
        e = -1; res = 0.0
        end = f+maxh
        if end > n_end-1: end = n_end-1
        for j in range(f, end+1):
            if l[j] <= D: e = j; res = -stop; break
            if h[j] >= U: e = j; res = up; nwin += 1; break
        if e < 0: e = end; res = c[e]/px-1.0
        pnl = cap*lv*(res - (fee_in+fee_out)/1e4)
        if pnl < -cap: pnl = -cap
        cap += pnl
        if cap > peak: peak = cap
        dd = cap/peak-1.0
        if dd < mdd: mdd = dd
        ntr += 1; sum_r += res
        free = e+1+cool; t = free
        if cap < ruin: break
    return cap, mdd, ntr, nwin, nskip, sum_r


EMPTY = np.zeros(0, np.int64)


def sim(q, up=0.015, dn=0.03, nom=1.0, k_abs=0.0, k_rv=0.0, W=240,
        fee_in=2.0, fee_out=5.0, rnd=EMPTY):
    th = float(np.quantile(SIG[np.isfinite(SIG)], 1-q))
    cap, mdd, ntr, nwin, nskip, sr = walk(O, H, Lo, C, SIG, th, MU, up, dn, MAXH, STEP, COOL,
                                          float(nom), MMR, fee_in, fee_out, k_abs, k_rv,
                                          rv, W, 0.0002, warm, NEND, 0.01, rnd)
    if ntr < 30: return None
    return dict(ann=(max(cap, 1e-12)**(1/Y)-1)*100, mdd=mdd*100, fin=cap, n=ntr,
                wr=nwin/ntr*100, bp=sr/ntr*1e4, skip=nskip, ruin=cap < 0.01)


def cell(v):
    return "      파산      " if (v is None or v["ruin"]) else f"{v['ann']:+7.1f}%/{v['wr']:5.1f}%"


P("="*128); P("[1] 절대 깊이 지정가 × 분위 (익절1.5/손절3.0, 명목1배, 메이커진입 2bp+테이커청산 5bp)")
P("="*128)
P(f"  {'분위':>6s} {'시장가(테)':>17s} " + " ".join(f"{'깊이'+str(k*100)+'%':>17s}"
                                                for k in (0.0025, 0.005, 0.0075, 0.01, 0.015)))
BEST = []
for q in (0.16, 0.25, 0.35, 0.50):
    row = [cell(sim(q, fee_in=5.0))]
    for k in (0.0025, 0.005, 0.0075, 0.01, 0.015):
        v = sim(q, k_abs=k)
        row.append(cell(v))
        if v and not v["ruin"]: BEST.append((v["ann"], q, k, 0.0, v))
    P(f"  {q*100:5.0f}% " + " ".join(f"{x:>17s}" for x in row))

P("\n"+"="*128); P("[2] rv 비례 깊이 × 분위 (적응형 진입)"); P("="*128)
P(f"  {'분위':>6s} " + " ".join(f"{'rv×'+str(k):>17s}" for k in (0.5, 1.0, 1.5, 2.0, 3.0)))
for q in (0.16, 0.25, 0.35, 0.50):
    row = []
    for k in (0.5, 1.0, 1.5, 2.0, 3.0):
        v = sim(q, k_rv=k)
        row.append(cell(v))
        if v and not v["ruin"]: BEST.append((v["ann"], q, 0.0, k, v))
    P(f"  {q*100:5.0f}% " + " ".join(f"{x:>17s}" for x in row))

P("\n"+"="*128); P("[3] 상위 후보 — 빈도·미체결 포함"); P("="*128)
BEST.sort(reverse=True)
P(f"  {'연수익%':>9s} {'MDD%':>8s} {'승률%':>7s} {'연거래':>6s} {'이틀당':>7s} {'거래당bp':>9s} "
  f"{'미체결':>6s} {'분위':>6s} {'깊이':>12s}")
for a, q, ka, kr, v in BEST[:10]:
    dep = f"절대{ka*100:.2f}%" if ka > 0 else f"rv×{kr}"
    P(f"  {a:9.1f} {v['mdd']:8.1f} {v['wr']:7.2f} {v['n']/Y:6.0f} {v['n']/Y/182.5:7.2f} "
      f"{v['bp']:+9.1f} {v['skip']:6d} {q*100:5.0f}% {dep:>12s}")

CAND = [x for x in BEST if x[4]["n"]/Y >= 182.5]
if not CAND: CAND = BEST
A, Q, KA, KR, V = CAND[0]
dep = f"절대 {KA*100:.2f}%" if KA > 0 else f"rv×{KR}"
P(f"\n  → 채택: 상위{Q*100:.0f}% 지정가 {dep} (연 {V['n']/Y:.0f}건, 이틀당 {V['n']/Y/182.5:.2f}건)")

P("\n"+"="*128); P("[4] 3단 레버리지 + 연도별 + 시장가 대비"); P("="*128)
P(f"  {'구간':>12s} {'명목':>6s} {'지정가 연%':>12s} {'MDD%':>8s} | {'시장가 연%':>12s} {'MDD%':>8s}")
for lb, nom in (("무레버리지", 1.0), ("중간", 2.0), ("중간+", 3.0), ("공격", 4.0), ("최대공격", 5.0)):
    a = sim(Q, nom=nom, k_abs=KA, k_rv=KR)
    b = sim(0.25, nom=nom, fee_in=5.0)
    s1 = "        파산" if (a is None or a["ruin"]) else f"{a['ann']:12.1f} {a['mdd']:8.1f}"
    s2 = "        파산" if (b is None or b["ruin"]) else f"{b['ann']:12.1f} {b['mdd']:8.1f}"
    P(f"  {lb:>12s} {nom:5.1f}x {s1} | {s2}")


def yearly(nom):
    th = float(np.quantile(SIG[np.isfinite(SIG)], 1-Q))
    out = []
    for yr in range(2020, 2027):
        ii = np.where(idx.year == yr)[0]
        if len(ii) < 50000: out.append(f"{yr}  n/a"); continue
        a0, b0 = max(ii[0], warm), ii[-1]
        if a0 >= b0-MAXH: out.append(f"{yr}  n/a"); continue
        cap, mdd, ntr, nwin, nskip, sr = walk(O, H, Lo, C, SIG, th, MU, 0.015, 0.03, MAXH, STEP,
                                              COOL, float(nom), MMR, 2.0, 5.0, KA, KR, rv, 240,
                                              0.0002, a0, b0, 0.01, EMPTY)
        out.append(f"{yr} {(cap-1)*100:+7.0f}%" if ntr >= 5 else f"{yr}  n/a")
    return out


for nom in (1.0, 2.0, 3.0):
    P(f"  연도별({nom:.0f}배): " + " | ".join(yearly(nom)))

P("\n"+"="*128); P("[5] 플라시보 200회"); P("="*128)
rng = np.random.default_rng(41)
th = float(np.quantile(SIG[np.isfinite(SIG)], 1-Q))
ncand = int((np.isfinite(SIG) & (SIG >= th)).sum())
real = sim(Q, k_abs=KA, k_rv=KR); real3 = sim(Q, nom=3.0, k_abs=KA, k_rv=KR)
pb, p1, p3 = [], [], []
for it in range(200):
    st = np.sort(rng.integers(warm, NEND-MAXH-250, size=ncand)).astype(np.int64)
    v1 = sim(Q, k_abs=KA, k_rv=KR, rnd=st); v3 = sim(Q, nom=3.0, k_abs=KA, k_rv=KR, rnd=st)
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

P("\n"+"="*128); P("[6] 비용 가정별 (진입 메이커 / 청산 테이커)"); P("="*128)
for fi, fo, lab in ((0, 4, "메이커0/테4 (BNB·VIP)"), (2, 5, "메이커2/테5 (기본)"),
                    (2, 8, "메이커2/테8 (보수)"), (5, 10, "둘 다 테이커 비관"),
                    (5, 15, "최악")):
    row = []
    for nom in (1, 2, 3):
        v = sim(Q, nom=nom, k_abs=KA, k_rv=KR, fee_in=fi, fee_out=fo)
        row.append("    파산" if (v is None or v["ruin"]) else f"{v['ann']:+8.1f}%")
    P(f"  {lab:>22s}: 1배{row[0]} 2배{row[1]} 3배{row[2]}")

P("\n"+"="*128); P("[7] 지정가 유효시간 W 민감도"); P("="*128)
for W in (15, 30, 60, 120, 240, 480, 1440):
    v = sim(Q, nom=2.0, k_abs=KA, k_rv=KR, W=W)
    if v is None: continue
    P(f"  W {W:5d}분: 연거래 {v['n']/Y:4.0f}  미체결 {v['skip']:5d}  승률 {v['wr']:5.2f}%  "
      + ("파산" if v["ruin"] else f"연 {v['ann']:+7.1f}%  MDD {v['mdd']:6.1f}%"))
