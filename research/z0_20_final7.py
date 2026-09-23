"""z0.27 최종 통합 — 지정가 진입 + 신호 문턱 + 신호 사이징 + 넓은 손절 + 긴 보유.

정확한 플라시보: 구조·진입규칙은 그대로 두고 **신호 시계열만 블록 셔플**한다.
이러면 신호가 기여하는 몫만 정확히 분리된다(구조가 만드는 수익은 위약에도 남는다)."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R/"research"))
import z0_20_lib as L
from z0_20_final import build_sigs, exp_z


def P(*a, end="\n"):
    print(" ".join(str(x) for x in a), end=end, flush=True)


S = "BTCUSDT"; Y = L.YRS; MMR = L.MMR; STEP = L.STEP
d = L.D[S]; idx = d["idx"]; rv = d["rv"]
O, H, Lo, C = d["o"], d["h"], d["l"], d["c"]
warm = 200000
med = np.full(len(rv), np.nan)
for i in range(warm, len(rv), 5000):
    x = rv[:i]; x = x[np.isfinite(x)]
    if len(x) >= 1000: med[i:i+5000] = np.median(x)
SG = build_sigs(S)
names = list(SG.keys())
Z = {n: exp_z(np.asarray(SG[n], float)) for n in names}
mlen = min(min(len(Z[n]) for n in names), len(C)//STEP)
SIG = np.nanmean(np.vstack([Z[n][:mlen] for n in names if n.endswith("4320")]), axis=0)


def qseries(sig):
    sq = np.full(len(sig), np.nan)
    for i in range(warm//STEP, len(sig), 5000//STEP):
        x = sig[:i]; x = x[np.isfinite(x)]
        if len(x) < 1000: continue
        sq[i:i+5000//STEP] = np.searchsorted(np.sort(x), sig[i:i+5000//STEP])/len(x)
    return sq


SQ = qseries(SIG)
NEND = len(C)


@numba.njit(cache=True)
def walk(o, h, l, c, rv, med, sq, k, up, dn, maxh, step, cool, W, eps, nom, mmr,
         fee_in, fee_out, th, s_lo, s_mid, s_hi, start, n_end, ruin):
    cap = 1.0; peak = 1.0; mdd = 0.0
    ntr = 0; nwin = 0; nskip = 0; sum_r = 0.0; sum_m = 0.0; sum_h = 0.0
    t = start
    while t < n_end - maxh - W - 2:
        if np.isnan(rv[t]) or rv[t] <= 0: t += step; continue
        j0 = t//step
        s_ = sq[j0] if j0 < len(sq) else np.nan
        if th > 0.0:
            if not np.isfinite(s_) or s_ < th: t += step; continue
        T = c[t]*(1.0 - rv[t]*k)
        f = -1
        for u in range(t+1, min(t+1+W, n_end)):
            if l[u] <= T*(1.0-eps): f = u; break
        if f < 0:
            nskip += 1; t += W; continue
        px = T
        m = 1.0
        mm = med[t]
        if np.isfinite(mm) and rv[t] > 0:
            x = mm/rv[t]
            m *= 0.1 if x < 0.1 else (4.0 if x > 4.0 else x)
        if np.isfinite(s_):
            m *= s_hi if s_ >= 0.97 else (s_mid if s_ >= 0.90 else s_lo)
        lv = nom*m
        if lv <= 0: t += step; continue
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
        ntr += 1; sum_r += res; sum_m += m; sum_h += (e-f)
        t = e+1+cool
        if cap < ruin: break
    return cap, mdd, ntr, nwin, nskip, sum_r, sum_m, sum_h


def sim(sq=None, k=4.0, up=0.02, dn=0.03, nom=1.0, maxh=8640, cool=60, W=240, th=0.75,
        fee_in=2.0, fee_out=5.0, s_lo=0.5, s_mid=1.0, s_hi=2.0, start=None, end=None):
    if sq is None: sq = SQ
    s0 = warm if start is None else start
    e0 = NEND if end is None else end
    cap, mdd, ntr, nwin, nskip, sr, sm, sh = walk(O, H, Lo, C, rv, med, sq, k, up, dn, maxh, STEP,
                                                  cool, W, 0.0002, float(nom), MMR, fee_in, fee_out,
                                                  th, s_lo, s_mid, s_hi, s0, e0, 0.01)
    if ntr < 25: return None
    yy = Y if start is None else max((e0-s0)/525600, 0.1)
    return dict(ann=(max(cap, 1e-12)**(1/yy)-1)*100, mdd=mdd*100, fin=cap, n=ntr,
                wr=nwin/ntr*100, bp=sr/ntr*1e4, mult=sm/ntr, hold=sh/ntr,
                skip=nskip, ruin=cap < 0.01)


def cell(v):
    return "      파산      " if (v is None or v["ruin"]) else f"{v['ann']:+7.1f}%/{v['wr']:5.1f}%"


P("="*128); P("[1] 문턱 × 배리어 (지정가 rv×4, 사이징 0.5/1/2, 명목1배). 셀 = 연수익%/승률%"); P("="*128)
BARS = [(0.015, 0.03), (0.02, 0.03), (0.03, 0.03), (0.02, 0.04), (0.03, 0.04), (0.015, 0.04)]
P(f"  {'문턱':>8s} " + " ".join(f"{str(u*100)+'/'+str(dd*100):>16s}" for u, dd in BARS))
BEST = []
for th in (0.0, 0.50, 0.70, 0.75, 0.84, 0.90):
    row = []
    for up, dn in BARS:
        v = sim(th=th, up=up, dn=dn)
        row.append(cell(v))
        if v and not v["ruin"]: BEST.append((v["ann"], th, up, dn, v))
    P(f"  {th*100:6.0f}% " + " ".join(f"{x:>16s}" for x in row))

BEST.sort(reverse=True)
P("\n  상위 6개 (연 182.5건 이상만):")
CAND = [x for x in BEST if x[4]["n"]/Y >= 182.5]
show = CAND if CAND else BEST
P(f"  {'연수익%':>9s} {'MDD%':>8s} {'승률%':>7s} {'연거래':>6s} {'이틀당':>7s} {'거래당bp':>9s} "
  f"{'문턱':>6s} {'익절':>7s} {'손절':>7s}")
for a, th, up, dn, v in show[:6]:
    P(f"  {a:9.1f} {v['mdd']:8.1f} {v['wr']:7.2f} {v['n']/Y:6.0f} {v['n']/Y/182.5:7.2f} "
      f"{v['bp']:+9.1f} {th*100:5.0f}% {up*100:6.2f}% {dn*100:6.2f}%")

A, TH, UP, DN, V = show[0]
P(f"\n  → 채택: 문턱 상위{(1-TH)*100:.0f}% 익절{UP*100:.2f}% 손절{DN*100:.2f}% "
  f"(연 {V['n']/Y:.0f}건, 이틀당 {V['n']/Y/182.5:.2f}건)")

P("\n"+"="*128); P("[2] ★ 정확한 플라시보 — 신호 시계열만 블록 셔플 (구조는 그대로)"); P("="*128)
rng = np.random.default_rng(101)
real = sim(th=TH, up=UP, dn=DN); real2 = sim(th=TH, up=UP, dn=DN, nom=2.0)
n = len(SIG)
res_bp, res1, res2 = [], [], []
BLK = 20000  # 블록 길이(신호 자기상관 보존)
for it in range(200):
    nb = n//BLK + 1
    order = rng.permutation(nb)
    parts = []
    for b in order:
        parts.append(SIG[b*BLK:(b+1)*BLK])
    sh = np.concatenate(parts)[:n]
    sq = qseries(sh)
    v1 = sim(sq=sq, th=TH, up=UP, dn=DN)
    v2 = sim(sq=sq, th=TH, up=UP, dn=DN, nom=2.0)
    if v1 is None: continue
    res_bp.append(v1["bp"]); res1.append(v1["ann"])
    res2.append(v2["ann"] if (v2 and not v2["ruin"]) else -99.9)
res_bp, res1, res2 = map(np.array, (res_bp, res1, res2))
P(f"  블록 {BLK:,}분({BLK/1440:.0f}일) 셔플 {len(res1)}회")
P(f"  거래당bp  실제 {real['bp']:+7.2f} | 위약 평균 {res_bp.mean():+6.2f} 최대 {res_bp.max():+6.2f} "
  f"→ 백분위 {np.mean(res_bp < real['bp'])*100:5.1f}%")
P(f"  1배 연%   실제 {real['ann']:+7.1f} | 위약 평균 {res1.mean():+6.1f} 최대 {res1.max():+6.1f} "
  f"→ 백분위 {np.mean(res1 < real['ann'])*100:5.1f}%")
if real2 and not real2["ruin"]:
    P(f"  2배 연%   실제 {real2['ann']:+7.1f} | 위약 평균 {res2.mean():+6.1f} 최대 {res2.max():+6.1f} "
      f"→ 백분위 {np.mean(res2 < real2['ann'])*100:5.1f}%")
P("\n  → 위약도 양수라면 그만큼은 '신호가 아니라 구조'(지정가+넓은손절+긴보유)가 버는 몫이다.")

P("\n"+"="*128); P("[3] 3단 레버리지 + 연도별"); P("="*128)
P(f"  {'구간':>12s} {'명목':>6s} {'연수익%':>10s} {'MDD%':>8s} {'최종배수':>11s}")
for lb, nom in (("무레버리지", 1.0), ("중간", 2.0), ("중간+", 3.0), ("공격", 4.0), ("최대공격", 5.0)):
    v = sim(th=TH, up=UP, dn=DN, nom=nom)
    P(f"  {lb:>12s} {nom:5.1f}x " + ("       파산" if (v is None or v["ruin"])
                                      else f"{v['ann']:10.1f} {v['mdd']:8.1f} {v['fin']:11.1f}"))
for nom in (1.0, 2.0, 3.0):
    out = []
    for yr in range(2020, 2027):
        ii = np.where(idx.year == yr)[0]
        if len(ii) < 50000: out.append(f"{yr}  n/a"); continue
        a0, b0 = max(ii[0], warm), ii[-1]
        if a0 >= b0-8640: out.append(f"{yr}  n/a"); continue
        v = sim(th=TH, up=UP, dn=DN, nom=nom, start=a0, end=b0)
        out.append(f"{yr} {(v['fin']-1)*100:+7.0f}%" if v else f"{yr}  n/a")
    P(f"  연도별({nom:.0f}배): " + " | ".join(out))

P("\n"+"="*128); P("[4] 비용 · 깊이 · 문턱 고원"); P("="*128)
P("  비용:")
for fi, fo, lab in ((0, 4, "메0/테4"), (2, 5, "메2/테5"), (2, 8, "메2/테8"), (5, 10, "테10/테10"), (8, 15, "최악")):
    r = []
    for nom in (1, 2, 3):
        v = sim(th=TH, up=UP, dn=DN, nom=nom, fee_in=fi, fee_out=fo)
        r.append("    파산" if (v is None or v["ruin"]) else f"{v['ann']:+8.1f}%")
    P(f"    {lab:>10s}: 1배{r[0]} 2배{r[1]} 3배{r[2]}")
P("\n  깊이 고원(명목2배):")
for k in (1.0, 2.0, 3.0, 4.0, 6.0, 8.0):
    v = sim(th=TH, up=UP, dn=DN, nom=2.0, k=k)
    if v is None: continue
    P(f"    rv×{k:<4.1f}: 연거래 {v['n']/Y:4.0f} 승률 {v['wr']:5.2f}% "
      + ("파산" if v["ruin"] else f"연 {v['ann']:+7.1f}%  MDD {v['mdd']:6.1f}%"))
P("\n  문턱 고원(명목2배):")
for th in (0.0, 0.3, 0.5, 0.6, 0.7, 0.75, 0.8, 0.84, 0.9, 0.95):
    v = sim(th=th, up=UP, dn=DN, nom=2.0)
    if v is None: continue
    P(f"    상위{(1-th)*100:5.0f}%: 연거래 {v['n']/Y:4.0f} 승률 {v['wr']:5.2f}% "
      + ("파산" if v["ruin"] else f"연 {v['ann']:+7.1f}%  MDD {v['mdd']:6.1f}%"))
P("\n  사이징 계단(명목2배, 실효 승수 함께):")
for lo, mi, hi in ((1.0, 1.0, 1.0), (0.75, 1.0, 1.5), (0.5, 1.0, 2.0), (0.3, 1.0, 3.0), (0.2, 1.0, 4.0)):
    v = sim(th=TH, up=UP, dn=DN, nom=2.0, s_lo=lo, s_mid=mi, s_hi=hi)
    if v is None: continue
    P(f"    {lo}/{mi}/{hi}: 실효 {2.0*v['mult']:4.2f}배  "
      + ("파산" if v["ruin"] else f"연 {v['ann']:+7.1f}%  MDD {v['mdd']:6.1f}%"))
