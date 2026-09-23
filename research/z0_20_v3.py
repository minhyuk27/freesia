"""z0.23 확정 — 규칙 기반 신호(5계열 × 창4320 균등)로 재확정.
선택편향을 최소화한 구성으로 빈도·3단 레버리지·플라시보·비용·고원을 다시 확인한다."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R/"research"))
import z0_20_lib as L
from z0_20_final import build_sigs, exp_z


def P(*a, end="\n"):
    print(" ".join(str(x) for x in a), end=end, flush=True)


S = "BTCUSDT"; Y = L.YRS
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
W = 4320
PICK = [n for n in names if n.endswith(str(W))]
P(f"규칙 신호: {', '.join(PICK)}\n")
SIG = np.nanmean(np.vstack([Z[n][:mlen] for n in PICK]), axis=0)
SQ = np.full(len(SIG), np.nan)
_sw = warm//STEP; _ss = _st//STEP
for i in range(_sw, len(SIG), _ss):
    x = SIG[:i]; x = x[np.isfinite(x)]
    if len(x) < 1000: continue
    SQ[i:i+_ss] = np.searchsorted(np.sort(x), SIG[i:i+_ss])/len(x)


@numba.njit(cache=True)
def tr_at(o, h, l, c, starts, up, dn, maxh, cool):
    n = len(c); K = len(starts)
    ei = np.zeros(K, np.int64); ex = np.zeros(K, np.int64); rr = np.zeros(K)
    k = 0; free = 0
    for si in range(K):
        t = starts[si]
        if t < free or t >= n-maxh-2: continue
        f = t+1; px = o[f]; U = px*(1.0+up); D = px*(1.0-dn)
        e = -1; res = 0.0; end = min(f+maxh, n-1)
        for j in range(f, end+1):
            if l[j] <= D: e = j; res = -dn; break
            if h[j] >= U: e = j; res = up; break
        if e < 0: e = end; res = c[e]/px-1.0
        ei[k] = f; ex[k] = e; rr[k] = res; k += 1
        free = e+1+cool
    return ei[:k], ex[:k], rr[:k]


def starts_q(q):
    fin = SIG[np.isfinite(SIG)]
    th = np.quantile(fin, 1-q)
    st = np.where(np.isfinite(SIG) & (SIG >= th))[0]*STEP
    return st[st >= warm].astype(np.int64)


def run(ei, ex, rr, lev, bar=0.015, fee=10.0, yr=None, use_size=True, dd_k=0.0):
    cap = 1e7; peak = cap; cur = []
    for a, b, res in zip(ei, ex, rr):
        if yr is not None and idx[a].year != yr: continue
        mu = 1.0
        if use_size:
            r_ = rv[a-1]; mm = med[a]
            if np.isfinite(r_) and np.isfinite(mm) and r_ > 0: mu *= float(np.clip(mm/r_, 0.1, 4.0))
            s_ = SQ[(a-1)//STEP]
            if np.isfinite(s_): mu *= (2.0 if s_ >= 0.97 else (1.0 if s_ >= 0.90 else 0.5))
        if dd_k > 0: mu *= max(0.2, 1.0-dd_k*(1.0-cap/peak))
        lv = lev*mu
        if lv <= 0: cur.append(cap); continue
        liq = 1.0/lv-L.MMR; r = res
        if r <= -bar and liq < bar: r = -1.0/lv
        pnl = cap*lv*(r-fee/1e4)
        if pnl < -cap: pnl = -cap
        cap += pnl
        if cap > peak: peak = cap
        cur.append(cap)
        if cap <= 1e5: break
    if not cur: return None
    eq = np.array(cur)
    if eq[-1] <= 2e5: return None
    yy = Y if yr is None else 1.0
    pk = np.maximum.accumulate(np.concatenate([[1e7], eq]))[1:]
    return dict(ann=((eq[-1]/1e7)**(1/yy)-1)*100, mdd=(eq/pk-1).min()*100, fin=eq[-1]/1e7)


P("="*124); P("[1] 분위 스윕 — 이틀당 1건(연 182.5건) 이상 찾기"); P("="*124)
P(f"  {'분위':>6s} {'연거래':>6s} {'이틀당':>7s} {'승률%':>7s} {'거래당bp':>9s} {'1x%':>8s} {'3x%':>8s} {'5x%':>8s} {'8x%':>8s}")
CAND = []
for q in (0.13, 0.15, 0.16, 0.17, 0.18, 0.20, 0.22):
    ei, ex, rr = tr_at(d["o"], d["h"], d["l"], d["c"], starts_q(q), 0.015, 0.015, MAXH, 60)
    row = []
    for lev in (1, 3, 5, 8):
        v = run(ei, ex, rr, lev)
        row.append(f"{v['ann']:+8.1f}" if v else "    파산")
    P(f"  {q*100:5.0f}% {len(ei)/Y:6.0f} {len(ei)/Y/182.5:7.2f} {np.mean(rr>0)*100:7.2f} "
      f"{rr.mean()*1e4:+9.1f} " + " ".join(row))
    if len(ei)/Y >= 182.5: CAND.append((q, ei, ex, rr))

q, ei, ex, rr = CAND[0] if CAND else (0.15,)+tr_at(d["o"], d["h"], d["l"], d["c"], starts_q(0.15), 0.015, 0.015, MAXH, 60)
P(f"\n  → 채택 분위 상위{q*100:.0f}% ({len(ei)}건, 연 {len(ei)/Y:.0f}건, 이틀당 {len(ei)/Y/182.5:.2f}건)")

P("\n"+"="*124); P("[2] 3단 레버리지 (+ 낙폭 비례 축소 적용시)"); P("="*124)
P(f"  {'구간':>12s} {'배수':>6s} {'조절없음 연%':>13s} {'MDD%':>8s} | {'낙폭조절k1.5 연%':>16s} {'MDD%':>8s}")
for lb, lev in (("무레버리지", 1), ("중간", 3), ("중간+", 5), ("공격", 8), ("최대공격", 12), ("극단", 20)):
    a = run(ei, ex, rr, lev); b = run(ei, ex, rr, lev, dd_k=1.5)
    s1 = f"{a['ann']:13.1f} {a['mdd']:8.1f}" if a else f"{'파산':>13s} {'':>8s}"
    s2 = f"{b['ann']:16.1f} {b['mdd']:8.1f}" if b else f"{'파산':>16s} {'':>8s}"
    P(f"  {lb:>12s} {lev:5d}x {s1} | {s2}")

P("\n  연도별:")
for lv, dk in ((1, 0), (3, 0), (5, 0), (8, 1.5), (12, 1.5)):
    out = []
    for yr in range(2020, 2027):
        v = run(ei, ex, rr, lv, yr=yr, dd_k=dk)
        out.append(f"{yr} {(v['fin']-1)*100:+7.0f}%" if v else f"{yr}   파산")
    P(f"    {lv:2d}x" + (f"(조절{dk})" if dk else "       ") + ": " + " | ".join(out))

P("\n"+"="*124); P("[3] 플라시보 300회 — 무작위 시점 동일 규칙"); P("="*124)
rng = np.random.default_rng(23)
nlo, nhi = warm, len(d["c"])-MAXH-3
real_bp = rr.mean()*1e4
r1 = run(ei, ex, rr, 1)["ann"]; r3v = run(ei, ex, rr, 3); r3 = r3v["ann"] if r3v else -99
pb, p1, p3 = [], [], []
for it in range(300):
    s2 = np.sort(rng.integers(nlo, nhi, size=len(starts_q(q)))).astype(np.int64)
    a2, b2, c2 = tr_at(d["o"], d["h"], d["l"], d["c"], s2, 0.015, 0.015, MAXH, 60)
    if len(a2) < 50: continue
    pb.append(c2.mean()*1e4)
    v1 = run(a2, b2, c2, 1); v3 = run(a2, b2, c2, 3)
    p1.append(v1["ann"] if v1 else -99.9); p3.append(v3["ann"] if v3 else -99.9)
pb, p1, p3 = map(np.array, (pb, p1, p3))
P(f"  거래당bp  실제 {real_bp:+7.2f} | 위약 평균 {pb.mean():+6.2f} 최대 {pb.max():+6.2f} → 백분위 {np.mean(pb<real_bp)*100:5.1f}%")
P(f"  1x 연%    실제 {r1:+7.1f} | 위약 평균 {p1.mean():+6.1f} 최대 {p1.max():+6.1f} → 백분위 {np.mean(p1<r1)*100:5.1f}%")
P(f"  3x 연%    실제 {r3:+7.1f} | 위약 평균 {p3.mean():+6.1f} 최대 {p3.max():+6.1f} → 백분위 {np.mean(p3<r3)*100:5.1f}%")

P("\n"+"="*124); P("[4] 비용 민감도 · 배리어 고원 · 쿨다운 고원"); P("="*124)
P("  비용(왕복):")
for fee in (4, 7, 10, 15, 20, 25, 30):
    row = []
    for lv in (1, 3, 5):
        v = run(ei, ex, rr, lv, fee=fee)
        row.append(f"{v['ann']:+8.1f}%" if v else "     파산")
    P(f"    {fee:2d}bp  거래당 {rr.mean()*1e4-fee:+6.1f}bp  " + " ".join(row))
P("\n  배리어 고원(3배):")
for bar in (0.0125, 0.015, 0.0175, 0.02, 0.025):
    a2, b2, c2 = tr_at(d["o"], d["h"], d["l"], d["c"], starts_q(q), bar, bar, MAXH, 60)
    v = run(a2, b2, c2, 3, bar=bar)
    P(f"    ±{bar*100:5.2f}%  연거래 {len(a2)/Y:4.0f}  승률 {np.mean(c2>0)*100:5.2f}%  "
      + (f"연 {v['ann']:+7.1f}%  MDD {v['mdd']:6.1f}%" if v else "파산"))
P("\n  쿨다운 고원(3배):")
for cool in (0, 30, 60, 120, 240):
    a2, b2, c2 = tr_at(d["o"], d["h"], d["l"], d["c"], starts_q(q), 0.015, 0.015, MAXH, cool)
    v = run(a2, b2, c2, 3)
    P(f"    {cool:4d}분  연거래 {len(a2)/Y:4.0f}  승률 {np.mean(c2>0)*100:5.2f}%  "
      + (f"연 {v['ann']:+7.1f}%  MDD {v['mdd']:6.1f}%" if v else "파산"))

P("\n"+"="*124); P("[5] 이상치 내성 · 사이징 기여 · 거래 간격"); P("="*124)
o = np.argsort(-rr)
for k in (0, 10, 30, 60):
    keep = np.ones(len(rr), bool); keep[o[:k]] = False
    v1 = run(ei[keep], ex[keep], rr[keep], 1); v3 = run(ei[keep], ex[keep], rr[keep], 3)
    P(f"  상위 {k:2d}건 제외: 거래당 {rr[keep].mean()*1e4:+6.1f}bp  "
      f"1x {v1['ann'] if v1 else -99:+7.1f}%  3x " + (f"{v3['ann']:+7.1f}%" if v3 else "  파산"))
P("")
for lab, us in (("사이징 적용", True), ("사이징 없음(고정)", False)):
    row = []
    for lv in (1, 3, 5):
        v = run(ei, ex, rr, lv, use_size=us)
        row.append(f"{v['ann']:+8.1f}%" if v else "     파산")
    P(f"  {lab:>16s}: 1x{row[0]} 3x{row[1]} 5x{row[2]}")
hold = ex-ei; gap = ei[1:]-ex[:-1]
P(f"\n  보유 중앙 {np.median(hold):.0f}분 평균 {hold.mean():.0f}분 90% {np.percentile(hold,90):.0f}분")
P(f"  공백 중앙 {np.median(gap):.0f}분 평균 {gap.mean():.0f}분 90% {np.percentile(gap,90):.0f}분")
P(f"  노출 {hold.sum()/(len(d['c'])-warm)*100:.1f}%")
mm = pd.Series(1, index=pd.DatetimeIndex(idx[ei])).resample("ME").sum()
P(f"  월 거래 중앙 {mm.median():.0f} 평균 {mm.mean():.1f} 최소 {mm.min():.0f} | 0건 월 {int((mm==0).sum())}/{len(mm)}")
ww = pd.Series(1, index=pd.DatetimeIndex(idx[ei])).resample("W").sum()
P(f"  주 거래 중앙 {ww.median():.0f} | 0건 주 {int((ww==0).sum())}/{len(ww)} ({(ww==0).mean()*100:.0f}%)")
