"""z0.20 확정 검증 — 플라시보, 비용 민감도, 파라미터 고원, 워크포워드.
확정 구성: BTC 상위15% 시장가 ±1.5% 쿨60 + 변동성목표 × 신호계단(레버리지 중립)."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R/"research"))
import z0_20_lib as L


def P(*a, end="\n"):
    print(" ".join(str(x) for x in a), end=end, flush=True)


S = "BTCUSDT"; Y = L.YRS
d = L.D[S]; idx = d["idx"]; rv = d["rv"]; sig = d["sig"]
warm = 200000; step = 5000
med = np.full(len(rv), np.nan)
for i in range(warm, len(rv), step):
    x = rv[:i]; x = x[np.isfinite(x)]
    if len(x) >= 1000: med[i:i+step] = np.median(x)
sq = np.full(len(sig), np.nan); sw = warm//L.STEP; ss = step//L.STEP
for i in range(sw, len(sig), ss):
    x = sig[:i]; x = x[np.isfinite(x)]
    if len(x) < 1000: continue
    sq[i:i+ss] = np.searchsorted(np.sort(x), sig[i:i+ss])/len(x)

Q, UP, DN, COOL = 0.15, 0.015, 0.015, 60


@numba.njit(cache=True)
def tr_at(o, h, l, c, starts, up, dn, maxh, cool):
    """주어진 진입 후보 시점들에서 단일 포지션으로 순차 진입."""
    n = len(c); K = len(starts)
    ei = np.zeros(K, np.int64); ex = np.zeros(K, np.int64); rr = np.zeros(K)
    k = 0; free = 0
    for si in range(K):
        t = starts[si]
        if t < free or t >= n-maxh-2: continue
        f = t+1; px = o[f]; U = px*(1.0+up); Dn = px*(1.0-dn)
        e = -1; res = 0.0; end = min(f+maxh, n-1)
        for j in range(f, end+1):
            if l[j] <= Dn: e = j; res = -dn; break
            if h[j] >= U: e = j; res = up; break
        if e < 0: e = end; res = c[e]/px-1.0
        ei[k] = f; ex[k] = e; rr[k] = res; k += 1
        free = e+1+cool
    return ei[:k], ex[:k], rr[:k]


def sig_starts(q):
    th = np.quantile(d["fin"], 1-q)
    ok = np.where(np.isfinite(sig) & (sig >= th))[0]*L.STEP
    return ok[ok >= warm].astype(np.int64)


def evaluate(ei, ex, rr, lev, use_size=True, dn=DN, yr=None):
    cap = 1e7; cur = []
    for a, b, res in zip(ei, ex, rr):
        mu = 1.0
        if use_size:
            r_ = rv[a-1] if a > 0 else np.nan; mm = med[a]
            if np.isfinite(r_) and np.isfinite(mm) and r_ > 0:
                mu *= float(np.clip(mm/r_, 0.1, 4.0))
            s_ = sq[(a-1)//L.STEP] if (a-1)//L.STEP < len(sq) else np.nan
            if np.isfinite(s_):
                mu *= (2.0 if s_ >= 0.97 else (1.0 if s_ >= 0.90 else 0.5))
        lv = lev*mu
        if lv <= 0: cur.append(cap); continue
        if yr is not None and idx[a].year != yr: continue
        liq = 1.0/lv - L.MMR; r = res
        if r <= -dn and liq < dn: r = -1.0/lv
        pnl = cap*lv*(r-10.0/1e4)
        if pnl < -cap: pnl = -cap
        cap += pnl; cur.append(cap)
        if cap <= 1e5: break
    if not cur: return None
    eq = np.array(cur)
    if eq[-1] <= 2e5: return None
    yy = Y if yr is None else 1.0
    peak = np.maximum.accumulate(np.concatenate([[1e7], eq]))[1:]
    return dict(ann=((eq[-1]/1e7)**(1/yy)-1)*100, mdd=(eq/peak-1).min()*100, fin=eq[-1]/1e7)


st = sig_starts(Q)
ei, ex, rr = tr_at(d["o"], d["h"], d["l"], d["c"], st, UP, DN, L.MAXH, COOL)
P(f"확정 구성 — 진입후보 {len(st):,} → 실거래 {len(ei):,}건 (연 {len(ei)/Y:.0f}건)")
P(f"거래당 평균 {rr.mean()*1e4:+.1f}bp, 승률 {np.mean(rr>0)*100:.2f}%\n")

P("="*118); P("[1] 플라시보 — 무작위 시점에서 같은 횟수·같은 규칙 (500회)"); P("="*118)
rng = np.random.default_rng(7)
nlo = warm; nhi = len(d["c"])-L.MAXH-3
real1 = evaluate(ei, ex, rr, 1)["ann"]; real3 = evaluate(ei, ex, rr, 3)
real3 = real3["ann"] if real3 else -99
realbp = rr.mean()*1e4
pb, p1, p3 = [], [], []
for it in range(500):
    s2 = np.sort(rng.integers(nlo, nhi, size=len(st))).astype(np.int64)
    a2, b2, r2 = tr_at(d["o"], d["h"], d["l"], d["c"], s2, UP, DN, L.MAXH, COOL)
    if len(a2) < 50: continue
    pb.append(r2.mean()*1e4)
    v1 = evaluate(a2, b2, r2, 1); v3 = evaluate(a2, b2, r2, 3)
    p1.append(v1["ann"] if v1 else -99.9); p3.append(v3["ann"] if v3 else -99.9)
pb = np.array(pb); p1 = np.array(p1); p3 = np.array(p3)
P(f"  거래당bp   실제 {realbp:+7.2f} | 위약 평균 {pb.mean():+7.2f} 최대 {pb.max():+7.2f} "
  f"→ 백분위 {np.mean(pb < realbp)*100:5.1f}%")
P(f"  무레버 연%  실제 {real1:+7.1f} | 위약 평균 {p1.mean():+7.1f} 최대 {p1.max():+7.1f} "
  f"→ 백분위 {np.mean(p1 < real1)*100:5.1f}%")
P(f"  3배 연%    실제 {real3:+7.1f} | 위약 평균 {p3.mean():+7.1f} 최대 {p3.max():+7.1f} "
  f"→ 백분위 {np.mean(p3 < real3)*100:5.1f}%")

P("\n"+"="*118); P("[2] 비용 민감도 (왕복 bp) — 단조여야 건강하다"); P("="*118)
P(f"  {'왕복비용':>8s} {'거래당bp':>9s} {'1x 연%':>9s} {'3x 연%':>9s} {'5x 연%':>9s}")
for fee in (4, 7, 10, 15, 20, 30, 40):
    def ev2(lev):
        cap = 1e7; cur = []
        for a, b, res in zip(ei, ex, rr):
            mu = 1.0
            r_ = rv[a-1]; mm = med[a]
            if np.isfinite(r_) and np.isfinite(mm) and r_ > 0: mu *= float(np.clip(mm/r_, 0.1, 4.0))
            s_ = sq[(a-1)//L.STEP]
            if np.isfinite(s_): mu *= (2.0 if s_ >= 0.97 else (1.0 if s_ >= 0.90 else 0.5))
            lv = lev*mu
            if lv <= 0: continue
            liq = 1.0/lv-L.MMR; r = res
            if r <= -DN and liq < DN: r = -1.0/lv
            pnl = cap*lv*(r-fee/1e4)
            if pnl < -cap: pnl = -cap
            cap += pnl
            if cap <= 1e5: return None
        return ((cap/1e7)**(1/Y)-1)*100
    v = [ev2(x) for x in (1, 3, 5)]
    f_ = lambda x: f"{x:+9.1f}" if x is not None else "     파산"
    P(f"  {fee:7d}bp {rr.mean()*1e4-fee:+9.1f} {f_(v[0])} {f_(v[1])} {f_(v[2])}")

P("\n"+"="*118); P("[3] 파라미터 고원 — 배리어 × 분위 (3배, 변동성×계단)"); P("="*118)
P(f"  {'배리어':>7s} " + " ".join(f"{'상위'+str(int(q*100))+'%':>11s}" for q in (0.10, 0.15, 0.20, 0.25, 0.30)))
for bar in (0.010, 0.0125, 0.015, 0.0175, 0.020, 0.025):
    row = []
    for q in (0.10, 0.15, 0.20, 0.25, 0.30):
        a2, b2, r2 = tr_at(d["o"], d["h"], d["l"], d["c"], sig_starts(q), bar, bar, L.MAXH, COOL)
        v = evaluate(a2, b2, r2, 3, dn=bar)
        row.append(f"{v['ann']:+8.1f}%" if v else "    파산 ")
    P(f"  {bar*100:6.2f}% " + " ".join(f"{x:>11s}" for x in row))

P("\n"+"="*118); P("[4] 쿨다운 고원 (상위15% ±1.5%, 3배)"); P("="*118)
for cool in (0, 30, 60, 120, 240, 480):
    a2, b2, r2 = tr_at(d["o"], d["h"], d["l"], d["c"], sig_starts(Q), UP, DN, L.MAXH, cool)
    v = evaluate(a2, b2, r2, 3)
    P(f"  쿨 {cool:4d}분  거래 {len(a2):5d}(연 {len(a2)/Y:4.0f})  승률 {np.mean(r2>0)*100:5.2f}%  "
      + (f"연 {v['ann']:+7.1f}%  MDD {v['mdd']:6.1f}%" if v else "파산"))

P("\n"+"="*118); P("[5] 워크포워드 — 앞 3년으로 정하고 뒤에서 검증"); P("="*118)
for cut in (2022, 2023, 2024, 2025):
    m = np.array([idx[a].year >= cut for a in ei])
    if m.sum() < 30: continue
    v1 = evaluate(ei[m], ex[m], rr[m], 1); v3 = evaluate(ei[m], ex[m], rr[m], 3)
    yy = (idx[ei[m]][-1]-idx[ei[m]][0]).days/365
    a1 = ((v1["fin"])**(1/yy)-1)*100 if v1 else None
    a3 = ((v3["fin"])**(1/yy)-1)*100 if v3 else None
    P(f"  {cut}년 이후 ({m.sum():4d}건, {yy:.1f}년): 승률 {np.mean(rr[m]>0)*100:5.2f}%  "
      f"1x {a1:+7.1f}%  3x " + (f"{a3:+7.1f}%" if a3 else "  파산"))

P("\n"+"="*118); P("[6] 상위 거래 제외 내성"); P("="*118)
o = np.argsort(-rr)
for k in (0, 5, 10, 20, 50):
    keep = np.ones(len(rr), bool); keep[o[:k]] = False
    v1 = evaluate(ei[keep], ex[keep], rr[keep], 1); v3 = evaluate(ei[keep], ex[keep], rr[keep], 3)
    P(f"  상위 {k:2d}건 제외: 거래당 {rr[keep].mean()*1e4:+6.1f}bp  "
      f"1x {v1['ann'] if v1 else -99:+7.1f}%  3x " + (f"{v3['ann']:+7.1f}%" if v3 else "  파산"))
