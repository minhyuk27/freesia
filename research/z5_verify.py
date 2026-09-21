"""z5 검증 — '문턱 상위2% + ±3% 배리어 + 고켈리'가 실제로 월 +80%를 내는가.

켈리 공식은 월 +171%를 가리켰다. 그러나 그것은
  ①표본이 충분하고 ②승률이 시간에 걸쳐 안정적이고 ③복리 경로가 파산하지 않는다
는 전제다. 실제 자본 곡선으로 전부 확인한다."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R/"research"))
import z2_lib as L2
from z2_final import build_sigs, exp_z


def P(*a, end="\n"):
    print(" ".join(str(x) for x in a), end=end, flush=True)


S = "BTCUSDT"; Y = L2.YRS; MMR = L2.MMR; STEP = L2.STEP
d = L2.D[S]; idx = d["idx"]; rv = d["rv"]
O, H, Lo, C = d["o"], d["h"], d["l"], d["c"]
warm = 200000
SG = build_sigs(S)
names = list(SG.keys())
Z = {n: exp_z(np.asarray(SG[n], float)) for n in names}
mlen = min(min(len(Z[n]) for n in names), len(C)//STEP)
SIG = np.nanmean(np.vstack([Z[n][:mlen] for n in names if n.endswith("4320")]), axis=0)
NEND = len(C)


@numba.njit(cache=True)
def walk(o, h, l, c, sigv, th, up, dn, maxh, step, cool, nom, mmr, fee_bp,
         start, n_end, ruin, rnd):
    cap = 1.0; peak = 1.0; mdd = 0.0
    ntr = 0; nwin = 0; sum_r = 0.0; sum_h = 0.0
    ei = np.zeros(40000, np.int64); rr = np.zeros(40000)
    use_rnd = len(rnd) > 0
    si = 0; t = start; free = start
    while True:
        if use_rnd:
            if si >= len(rnd): break
            t = rnd[si]; si += 1
            if t < free or t >= n_end-maxh-2: continue
        else:
            if t >= n_end-maxh-2: break
            i = t//step
            if i >= len(sigv): break
            if th > 0.0 and (not np.isfinite(sigv[i]) or sigv[i] < th): t += step; continue
        liq = 1.0/nom - mmr
        stop = dn if dn < liq else liq
        f = t+1; px = o[f]
        U = px*(1.0+up); D = px*(1.0-stop)
        e = -1; res = 0.0
        end = min(f+maxh, n_end-1)
        for j in range(f, end+1):
            if l[j] <= D: e = j; res = -stop; break
            if h[j] >= U: e = j; res = up; nwin += 1; break
        if e < 0: e = end; res = c[e]/px-1.0
        pnl = cap*nom*(res - fee_bp/1e4)
        if pnl < -cap: pnl = -cap
        cap += pnl
        if cap > peak: peak = cap
        dd = cap/peak-1.0
        if dd < mdd: mdd = dd
        if ntr < 40000:
            ei[ntr] = f; rr[ntr] = res
        ntr += 1; sum_r += res; sum_h += (e-f)
        free = e+1+cool; t = free
        if cap < ruin: break
    return cap, mdd, ntr, nwin, sum_r, sum_h, ei[:min(ntr, 40000)], rr[:min(ntr, 40000)]


EMPTY = np.zeros(0, np.int64)


def sim(q=0.02, up=0.03, dn=0.03, nom=1.0, maxh=20160, cool=0, fee=10.0,
        start=None, end=None, rnd=EMPTY):
    th = float(np.quantile(SIG[np.isfinite(SIG)], 1-q)) if q < 1.0 else 0.0
    s0 = warm if start is None else start
    e0 = NEND if end is None else end
    cap, mdd, ntr, nwin, sr, sh, ei, rr = walk(O, H, Lo, C, SIG, th, up, dn, maxh, STEP,
                                               cool, float(nom), MMR, fee, s0, e0, 0.01, rnd)
    if ntr < 10: return None
    yy = Y if start is None else max((e0-s0)/525600, 0.05)
    ann = (max(cap, 1e-12)**(1/yy)-1)*100
    mon = ((1+ann/100)**(1/12)-1)*100 if ann > -99.99 else -100
    return dict(ann=ann, mon=mon, mdd=mdd*100, fin=cap, n=ntr, wr=nwin/ntr*100,
                bp=sr/ntr*1e4, hold=sh/ntr, ei=ei, rr=rr, ruin=cap < 0.01)


P("="*130); P("[1] 문턱 상위2% 구성의 실제 표본과 승률 — 켈리 계산의 전제를 확인"); P("="*130)
P(f"  {'문턱':>8s} {'배리어':>8s} {'거래수':>7s} {'연거래':>7s} {'승률%':>7s} {'표준오차%p':>10s} "
  f"{'거래당bp':>9s} {'평균보유':>9s}")
for q in (0.01, 0.02, 0.03, 0.05, 0.10):
    for bar in (0.02, 0.03, 0.04):
        v = sim(q=q, up=bar, dn=bar, nom=1.0)
        if v is None: continue
        se = np.sqrt(v["wr"]/100*(1-v["wr"]/100)/v["n"])*100
        P(f"  상위{q*100:4.1f}% ±{bar*100:5.2f}% {v['n']:7d} {v['n']/Y:7.1f} {v['wr']:7.2f} "
          f"{se:10.2f} {v['bp']:+9.1f} {v['hold']:8.0f}분")

P("\n"+"="*130); P("[2] 실제 자본 곡선 — 명목 배수 스윕 (상위2%, ±3%)"); P("="*130)
P(f"  {'명목':>7s} {'월%':>10s} {'연%':>14s} {'MDD%':>8s} {'최종배수':>14s} {'판정':>8s}")
BEST = None
for nom in (1, 2, 3, 4, 5, 6, 8, 9.15, 10, 12, 15, 20):
    v = sim(q=0.02, up=0.03, dn=0.03, nom=nom)
    if v is None: continue
    if v["ruin"]:
        P(f"  {nom:6.2f}x {'파산':>10s}"); continue
    P(f"  {nom:6.2f}x {v['mon']:10.2f} {v['ann']:14.1f} {v['mdd']:8.1f} {v['fin']:14.2f} "
      f"{'★목표달성' if v['mon'] >= 80 else '':>8s}")
    if BEST is None or v["mon"] > BEST[0]: BEST = (v["mon"], nom, v)

P("\n"+"="*130); P("[3] 연도별 — 승률이 시간에 걸쳐 안정적인가 (상위2% ±3%)"); P("="*130)
P(f"  {'연도':>6s} {'거래':>5s} {'승률%':>7s} {'거래당bp':>9s} " +
  " ".join(f"{'명목'+str(n)+'배':>12s}" for n in (1, 3, 9)))
for yr in range(2020, 2027):
    ii = np.where(idx.year == yr)[0]
    if len(ii) < 50000: continue
    a0, b0 = max(ii[0], warm), ii[-1]
    if a0 >= b0-20160: continue
    base = sim(q=0.02, up=0.03, dn=0.03, nom=1.0, start=a0, end=b0)
    if base is None: P(f"  {yr:6d} 표본부족"); continue
    row = []
    for nom in (1, 3, 9):
        v = sim(q=0.02, up=0.03, dn=0.03, nom=nom, start=a0, end=b0)
        row.append("        파산" if (v is None or v["ruin"]) else f"{(v['fin']-1)*100:+11.0f}%")
    P(f"  {yr:6d} {base['n']:5d} {base['wr']:7.2f} {base['bp']:+9.1f} " + " ".join(f"{x:>12s}" for x in row))

P("\n"+"="*130); P("[4] 플라시보 — 같은 건수를 무작위 시점에서 (200회)"); P("="*130)
rng = np.random.default_rng(77)
real = sim(q=0.02, up=0.03, dn=0.03, nom=1.0)
real9 = sim(q=0.02, up=0.03, dn=0.03, nom=9.15)
th2 = float(np.quantile(SIG[np.isfinite(SIG)], 0.98))
ncand = int((np.isfinite(SIG) & (SIG >= th2)).sum())
pb, pw_, p9 = [], [], []
for it in range(200):
    st = np.sort(rng.integers(warm, NEND-20160-3, size=ncand)).astype(np.int64)
    v = sim(q=0.02, up=0.03, dn=0.03, nom=1.0, rnd=st)
    v9 = sim(q=0.02, up=0.03, dn=0.03, nom=9.15, rnd=st)
    if v is None: continue
    pb.append(v["bp"]); pw_.append(v["wr"])
    p9.append(v9["mon"] if (v9 and not v9["ruin"]) else -100)
pb, pw_, p9 = map(np.array, (pb, pw_, p9))
P(f"  거래당bp 실제 {real['bp']:+7.2f} | 위약 평균 {pb.mean():+6.2f} 최대 {pb.max():+6.2f} "
  f"→ 백분위 {np.mean(pb < real['bp'])*100:5.1f}%")
P(f"  승률%   실제 {real['wr']:7.2f} | 위약 평균 {pw_.mean():6.2f} 최대 {pw_.max():6.2f} "
  f"→ 백분위 {np.mean(pw_ < real['wr'])*100:5.1f}%")
if real9 and not real9["ruin"]:
    P(f"  월%(9배) 실제 {real9['mon']:+7.2f} | 위약 평균 {p9.mean():+6.2f} 최대 {p9.max():+6.2f} "
      f"→ 백분위 {np.mean(p9 < real9['mon'])*100:5.1f}%")

P("\n"+"="*130); P("[5] 비용 내성 (상위2% ±3%)"); P("="*130)
for fee in (4, 7, 10, 15, 20, 30, 50):
    row = []
    for nom in (1, 3, 9.15):
        v = sim(q=0.02, up=0.03, dn=0.03, nom=nom, fee=fee)
        row.append("      파산" if (v is None or v["ruin"]) else f"{v['mon']:+9.2f}%")
    P(f"  왕복 {fee:2d}bp: 1배{row[0]}  3배{row[1]}  9.15배{row[2]}")

P("\n"+"="*130); P("[6] 문턱·배리어 고원 (명목 3배)"); P("="*130)
P(f"  {'문턱':>8s} " + " ".join(f"{'±'+str(b*100)+'%':>14s}" for b in (0.02, 0.025, 0.03, 0.04, 0.05)))
for q in (0.01, 0.015, 0.02, 0.03, 0.05, 0.08):
    row = []
    for bar in (0.02, 0.025, 0.03, 0.04, 0.05):
        v = sim(q=q, up=bar, dn=bar, nom=3.0)
        row.append("     파산     " if (v is None or v["ruin"])
                   else f"{v['mon']:+7.2f}%/{v['wr']:4.1f}")
    P(f"  상위{q*100:4.1f}% " + " ".join(f"{x:>14s}" for x in row))
P("  (셀 = 월% / 승률%)")

P("\n"+"="*130); P("[7] 최종 3단 레버리지"); P("="*130)
if BEST:
    P(f"  {'구간':>12s} {'명목':>7s} {'월%':>10s} {'연%':>14s} {'MDD%':>8s} {'6.7년 최종':>14s}")
    for lb, nom in (("무레버리지", 1.0), ("중간", 3.0), ("중간+", 5.0), ("공격", 7.0),
                    ("최대공격", 9.15), ("극단", 12.0)):
        v = sim(q=0.02, up=0.03, dn=0.03, nom=nom)
        P(f"  {lb:>12s} {nom:6.2f}x " + ("        파산" if (v is None or v["ruin"])
                                          else f"{v['mon']:10.2f} {v['ann']:14.1f} {v['mdd']:8.1f} {v['fin']:14.2f}"))
