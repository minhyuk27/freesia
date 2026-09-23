"""z0.26 — 진입 문턱 없음 + 지정가 대기 + 신호 기반 사이징.

정정된 이해: z0.20 신호는 '언제 들어갈지'가 아니라 '얼마나 걸지'에서 가치가 있다.
 - 신호 완전 제거(R3) → 전부 음수
 - 신호를 진입 문턱으로만 사용(z0.24) → 1배 +31.2%
 - 신호를 사이징으로만 사용하고 진입은 상시 → 이 파일에서 검증

구조: 매 15분 c[t]×(1 − rv[t]×k) 지정가, W분 유효, 문턱 없음(상시).
      사이징 = 변동성목표 × 신호계단(신호 확장창 분위 기반).
      익절 +up / 손절 −dn, 최대 maxh 보유, 단일 포지션 + 쿨다운."""
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
SQ = np.full(len(SIG), np.nan)
for i in range(warm//STEP, len(SIG), 5000//STEP):
    x = SIG[:i]; x = x[np.isfinite(x)]
    if len(x) < 1000: continue
    SQ[i:i+5000//STEP] = np.searchsorted(np.sort(x), SIG[i:i+5000//STEP])/len(x)
NEND = len(C)
P(f"신호 분위 분포: 중앙 {np.nanmedian(SQ):.2f}, 결측 {np.isnan(SQ).sum()/len(SQ)*100:.0f}%\n")


@numba.njit(cache=True)
def walk(o, h, l, c, rv, med, sq, k, up, dn, maxh, step, cool, W, eps, nom, mmr,
         fee_in, fee_out, use_vol, use_sig, th, s_lo, s_mid, s_hi, start, n_end, ruin):
    """th>0이면 신호 분위가 th 미만인 시점은 진입하지 않는다(문턱).
    use_sig=1이면 신호 분위로 사이징(≥0.97→s_hi, ≥0.90→s_mid, else s_lo)."""
    cap = 1.0; peak = 1.0; mdd = 0.0
    ntr = 0; nwin = 0; nskip = 0; sum_r = 0.0; sum_m = 0.0
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
        if use_vol == 1:
            mm = med[t]
            if np.isfinite(mm) and rv[t] > 0:
                x = mm/rv[t]
                m *= 0.1 if x < 0.1 else (4.0 if x > 4.0 else x)
        if use_sig == 1 and np.isfinite(s_):
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
        ntr += 1; sum_r += res; sum_m += m
        t = e+1+cool
        if cap < ruin: break
    return cap, mdd, ntr, nwin, nskip, sum_r, sum_m


def sim(k=3.0, up=0.015, dn=0.03, nom=1.0, maxh=8640, cool=60, W=240, th=0.0,
        fee_in=2.0, fee_out=5.0, use_vol=1, use_sig=1,
        s_lo=0.5, s_mid=1.0, s_hi=2.0, start=None, end=None):
    s0 = warm if start is None else start
    e0 = NEND if end is None else end
    cap, mdd, ntr, nwin, nskip, sr, sm = walk(O, H, Lo, C, rv, med, SQ, k, up, dn, maxh, STEP,
                                              cool, W, 0.0002, float(nom), MMR, fee_in, fee_out,
                                              use_vol, use_sig, th, s_lo, s_mid, s_hi, s0, e0, 0.01)
    if ntr < 30: return None
    yy = Y if start is None else max((e0-s0)/525600, 0.1)
    return dict(ann=(max(cap, 1e-12)**(1/yy)-1)*100, mdd=mdd*100, fin=cap, n=ntr,
                wr=nwin/ntr*100, bp=sr/ntr*1e4, mult=sm/ntr, skip=nskip, ruin=cap < 0.01)


def cell(v):
    return "      파산      " if (v is None or v["ruin"]) else f"{v['ann']:+7.1f}%/{v['wr']:5.1f}%"


P("="*128); P("[1] 신호의 역할 분해 — 진입 문턱 vs 사이징 (지정가 rv×3, 익절1.5/손절3, 명목1배)")
P("="*128)
P(f"  {'구성':>28s} {'연거래':>6s} {'승률%':>7s} {'거래당bp':>9s} {'평균승수':>8s} {'1배%':>9s} {'2배%':>9s} {'3배%':>9s}")
CASES = [("신호 전혀 안 씀", 0.0, 0, 1),
         ("사이징만 신호(상시 진입)", 0.0, 1, 1),
         ("문턱만 신호(균등 사이징)", 0.75, 0, 1),
         ("문턱+사이징 둘 다", 0.75, 1, 1),
         ("사이징만(변동성목표도 제거)", 0.0, 1, 0)]
for lab, th, us, uv in CASES:
    v = sim(th=th, use_sig=us, use_vol=uv)
    if v is None: continue
    r = []
    for nom in (1, 2, 3):
        vv = sim(nom=nom, th=th, use_sig=us, use_vol=uv)
        r.append("    파산" if (vv is None or vv["ruin"]) else f"{vv['ann']:+9.1f}")
    P(f"  {lab:>28s} {v['n']/Y:6.0f} {v['wr']:7.2f} {v['bp']:+9.1f} {v['mult']:8.2f} " + " ".join(r))

P("\n"+"="*128); P("[2] 상시 진입 + 신호 사이징 — 깊이 × 손절 (익절1.5, 명목1배)"); P("="*128)
P(f"  {'깊이':>8s} " + " ".join(f"{'손절'+str(x*100)+'%':>16s}" for x in (0.02, 0.03, 0.04, 0.06)))
BEST = []
for k in (1.0, 2.0, 3.0, 4.0, 6.0):
    row = []
    for dn in (0.02, 0.03, 0.04, 0.06):
        v = sim(k=k, dn=dn)
        row.append(cell(v))
        if v and not v["ruin"]: BEST.append((v["ann"], k, 0.015, dn, v))
    P(f"  rv×{k:<5.1f} " + " ".join(f"{x:>16s}" for x in row))

P("\n[3] 익절 × 깊이 (손절3%)")
P(f"  {'깊이':>8s} " + " ".join(f"{'익절'+str(x*100)+'%':>16s}" for x in (0.01, 0.015, 0.02, 0.03)))
for k in (1.0, 2.0, 3.0, 4.0, 6.0):
    row = []
    for up in (0.01, 0.015, 0.02, 0.03):
        v = sim(k=k, up=up)
        row.append(cell(v))
        if v and not v["ruin"]: BEST.append((v["ann"], k, up, 0.03, v))
    P(f"  rv×{k:<5.1f} " + " ".join(f"{x:>16s}" for x in row))

BEST.sort(reverse=True)
P("\n  상위 8개:")
P(f"  {'연수익%':>9s} {'MDD%':>8s} {'승률%':>7s} {'연거래':>6s} {'이틀당':>7s} {'거래당bp':>9s} "
  f"{'깊이':>8s} {'익절':>7s} {'손절':>7s}")
for a, k, up, dn, v in BEST[:8]:
    P(f"  {a:9.1f} {v['mdd']:8.1f} {v['wr']:7.2f} {v['n']/Y:6.0f} {v['n']/Y/182.5:7.2f} "
      f"{v['bp']:+9.1f} rv×{k:<5.1f} {up*100:6.2f}% {dn*100:6.2f}%")

A, K, UP, DN, V = BEST[0]
P(f"\n  → 최고: rv×{K} 익절{UP*100:.2f}% 손절{DN*100:.2f}% "
  f"(연 {V['n']/Y:.0f}건, 이틀당 {V['n']/Y/182.5:.2f}건)")

P("\n"+"="*128); P("[4] 사이징 계단 값 — 신호를 얼마나 세게 쓸까"); P("="*128)
P(f"  {'계단(하/중/상)':>18s} {'연거래':>6s} {'승률%':>7s} {'평균승수':>8s} {'1배%':>9s} {'2배%':>9s} {'3배%':>9s}")
for lo, mi, hi in ((1.0, 1.0, 1.0), (0.75, 1.0, 1.5), (0.5, 1.0, 2.0), (0.3, 1.0, 3.0),
                   (0.2, 1.0, 4.0), (0.0, 1.0, 2.0), (0.5, 1.0, 4.0)):
    v = sim(k=K, up=UP, dn=DN, s_lo=lo, s_mid=mi, s_hi=hi)
    if v is None: continue
    r = []
    for nom in (1, 2, 3):
        vv = sim(k=K, up=UP, dn=DN, nom=nom, s_lo=lo, s_mid=mi, s_hi=hi)
        r.append("    파산" if (vv is None or vv["ruin"]) else f"{vv['ann']:+9.1f}")
    P(f"  {f'{lo}/{mi}/{hi}':>18s} {v['n']/Y:6.0f} {v['wr']:7.2f} {v['mult']:8.2f} " + " ".join(r))

P("\n"+"="*128); P("[5] 3단 레버리지 + 연도별 + z2.4 대비"); P("="*128)
P(f"  {'구간':>12s} {'명목':>6s} {'연수익%':>10s} {'MDD%':>8s} {'최종배수':>11s}")
for lb, nom in (("무레버리지", 1.0), ("중간", 2.0), ("중간+", 3.0), ("공격", 4.0), ("최대공격", 5.0)):
    v = sim(k=K, up=UP, dn=DN, nom=nom)
    P(f"  {lb:>12s} {nom:5.1f}x " + ("       파산" if (v is None or v["ruin"])
                                      else f"{v['ann']:10.1f} {v['mdd']:8.1f} {v['fin']:11.1f}"))
for nom in (1.0, 2.0, 3.0):
    out = []
    for yr in range(2020, 2027):
        ii = np.where(idx.year == yr)[0]
        if len(ii) < 50000: out.append(f"{yr}  n/a"); continue
        a0, b0 = max(ii[0], warm), ii[-1]
        if a0 >= b0-8640: out.append(f"{yr}  n/a"); continue
        v = sim(k=K, up=UP, dn=DN, nom=nom, start=a0, end=b0)
        out.append(f"{yr} {(v['fin']-1)*100:+7.0f}%" if v else f"{yr}  n/a")
    P(f"  연도별({nom:.0f}배): " + " | ".join(out))

P("\n"+"="*128); P("[6] 비용 내성 · 깊이 고원 · 쿨다운 · 보유상한"); P("="*128)
P("  비용:")
for fi, fo, lab in ((0, 4, "메0/테4"), (2, 5, "메2/테5"), (2, 8, "메2/테8"), (5, 10, "테10/테10"), (8, 15, "최악")):
    r = []
    for nom in (1, 2, 3):
        v = sim(k=K, up=UP, dn=DN, nom=nom, fee_in=fi, fee_out=fo)
        r.append("    파산" if (v is None or v["ruin"]) else f"{v['ann']:+8.1f}%")
    P(f"    {lab:>10s}: 1배{r[0]} 2배{r[1]} 3배{r[2]}")
P("\n  깊이 고원(명목2배):")
for k in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0):
    v = sim(k=k, up=UP, dn=DN, nom=2.0)
    if v is None: continue
    P(f"    rv×{k:<4.1f}: 연거래 {v['n']/Y:4.0f} 미체결 {v['skip']:5d} 승률 {v['wr']:5.2f}% "
      + ("파산" if v["ruin"] else f"연 {v['ann']:+7.1f}%  MDD {v['mdd']:6.1f}%"))
P("\n  쿨다운(명목2배):")
for cool in (0, 30, 60, 120, 240, 720):
    v = sim(k=K, up=UP, dn=DN, nom=2.0, cool=cool)
    if v is None: continue
    P(f"    {cool:4d}분: 연거래 {v['n']/Y:4.0f} 승률 {v['wr']:5.2f}% "
      + ("파산" if v["ruin"] else f"연 {v['ann']:+7.1f}%  MDD {v['mdd']:6.1f}%"))
P("\n  보유상한(명목2배):")
for mh in (1440, 2880, 4320, 8640, 20160):
    v = sim(k=K, up=UP, dn=DN, nom=2.0, maxh=mh)
    if v is None: continue
    P(f"    {mh:6d}분: 연거래 {v['n']/Y:4.0f} 승률 {v['wr']:5.2f}% "
      + ("파산" if v["ruin"] else f"연 {v['ann']:+7.1f}%"))
