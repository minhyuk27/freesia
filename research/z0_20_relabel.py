"""z0.20 라벨 재정렬 — 지금 신호는 ±1% 우선도달 라벨로 고른 지표인데 배리어는 ±1.5%다.
목표에 맞는 라벨(±1.5% 우선도달)로 지표를 다시 고르고 결합한다.
지표 선택은 앞 절반(학습)에서만 하고 뒤 절반에서 검증한다."""
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
d = L.D[S]; idx = d["idx"]
STEP = L.STEP; MAXH = L.MAXH


@numba.njit(cache=True)
def touch(h, l, c, bar, maxh, step):
    """±bar 중 어디에 먼저 닿는가. 1=위, 0=아래, -1=판정불가"""
    n = len(c); m = n//step
    y = np.full(m, -1, np.int8)
    for i in range(m):
        t = i*step
        if t+maxh >= n: break
        px = c[t]; U = px*(1.0+bar); D = px*(1.0-bar)
        for j in range(t+1, t+maxh):
            if l[j] <= D: y[i] = 0; break
            if h[j] >= U: y[i] = 1; break
    return y


P("지표 재계산 중...")
SG = build_sigs(S)
names = [k for k in SG.keys()]
P(f"  후보 지표 {len(names)}개\n")

Z = {}
for n in names:
    v = np.asarray(SG[n], float)
    z = exp_z(v)
    Z[n] = z

BARS = (0.01, 0.015, 0.02)
LAB = {}
for bar in BARS:
    LAB[bar] = touch(d["h"], d["l"], d["c"], bar, MAXH, STEP)

P("="*124); P("[1] 배리어별 단일 지표 정확도 — 라벨이 바뀌면 순위도 바뀌는가"); P("="*124)
half = None
ACC = {}
for bar in BARS:
    y = LAB[bar]
    rows = []
    for n in names:
        z = Z[n]; m = min(len(z), len(y))
        zz = z[:m]; yy = y[:m]
        ok = (yy >= 0) & np.isfinite(zz)
        if ok.sum() < 5000: continue
        half = ok.sum()//2
        th = np.quantile(zz[ok], 0.85)
        sel = ok & (zz >= th)
        if sel.sum() < 500: continue
        rows.append((np.mean(yy[sel] == 1)*100, n, sel.sum()))
    rows.sort(reverse=True)
    ACC[bar] = rows
    P(f"\n  ── 배리어 ±{bar*100:.1f}% (상위15% 진입 기준) ──")
    for a, n, k in rows[:8]:
        P(f"    {n:28s} {a:6.2f}%  (n={k})")

P("\n"+"="*124); P("[2] 학습 절반에서 고르고 검증 절반에서 확인 (배리어 ±1.5%)"); P("="*124)
bar = 0.015; y = LAB[bar]
m = min(min(len(Z[n]) for n in names), len(y))
yy = y[:m]
cut = m//2
tr = np.arange(m) < cut
te = ~tr
rows = []
for n in names:
    z = Z[n][:m]
    ok = (yy >= 0) & np.isfinite(z)
    o1 = ok & tr
    if o1.sum() < 3000: continue
    th = np.quantile(z[o1], 0.85)
    s1 = o1 & (z >= th)
    s2 = ok & te & (z >= th)
    if s1.sum() < 300 or s2.sum() < 300: continue
    rows.append((np.mean(yy[s1] == 1)*100, np.mean(yy[s2] == 1)*100, n, s2.sum()))
rows.sort(reverse=True)
P(f"  {'지표':>28s} {'학습%':>7s} {'검증%':>7s} {'격차':>7s} {'검증n':>7s}")
for a1, a2, n, k in rows[:12]:
    P(f"  {n:>28s} {a1:7.2f} {a2:7.2f} {a2-a1:+7.2f} {k:7d}")
P(f"\n  학습 상위 5개의 검증 평균: {np.mean([r[1] for r in rows[:5]]):.2f}%")
P(f"  전체 지표의 검증 평균:     {np.mean([r[1] for r in rows]):.2f}%")

P("\n"+"="*124); P("[3] 결합 — 학습 상위 K개 z평균 (선택은 학습 절반에서만)"); P("="*124)
P(f"  {'K':>4s} {'학습%':>8s} {'검증%':>8s} {'구성':>60s}")
best = None
for K in (1, 2, 3, 4, 5, 6, 8, 10):
    picks = [r[2] for r in rows[:K]]
    comb = np.nanmean(np.vstack([Z[n][:m] for n in picks]), axis=0)
    ok = (yy >= 0) & np.isfinite(comb)
    o1 = ok & tr
    th = np.quantile(comb[o1], 0.85)
    s1 = o1 & (comb >= th); s2 = ok & te & (comb >= th)
    a1 = np.mean(yy[s1] == 1)*100; a2 = np.mean(yy[s2] == 1)*100
    P(f"  {K:4d} {a1:8.2f} {a2:8.2f}   {','.join(p[:12] for p in picks)[:58]:>60s}")
    if best is None or a2 > best[0]: best = (a2, K, picks)

P("\n"+"="*124); P("[4] 현행 신호 vs 재선택 신호 — 같은 규칙으로 백테스트"); P("="*124)
CUR = ["고저비대칭4320", "절대값8640", "OFI1440"]
NEW = best[2]
P(f"  현행: {', '.join(CUR)}")
P(f"  재선택(K={best[1]}): {', '.join(NEW)}\n")


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


warm = 200000; _st = 5000
rv = d["rv"]
med = np.full(len(rv), np.nan)
for i in range(warm, len(rv), _st):
    x = rv[:i]; x = x[np.isfinite(x)]
    if len(x) >= 1000: med[i:i+_st] = np.median(x)


def bt(sigv, q, bar, cool=60, lev=3, yr=None, fee=10.0):
    sq = np.full(len(sigv), np.nan)
    _sw = warm//STEP; _ss = _st//STEP
    for i in range(_sw, len(sigv), _ss):
        x = sigv[:i]; x = x[np.isfinite(x)]
        if len(x) < 1000: continue
        sq[i:i+_ss] = np.searchsorted(np.sort(x), sigv[i:i+_ss])/len(x)
    fin = sigv[np.isfinite(sigv)]
    th = np.quantile(fin, 1-q)
    st = np.where(np.isfinite(sigv) & (sigv >= th))[0]*STEP
    st = st[st >= warm].astype(np.int64)
    ei, ex, rr = tr_at(d["o"], d["h"], d["l"], d["c"], st, bar, bar, MAXH, cool)
    cap = 1e7; cur = []
    for a, b, res in zip(ei, ex, rr):
        if yr is not None and idx[a].year != yr: continue
        mu = 1.0
        r_ = rv[a-1]; mm = med[a]
        if np.isfinite(r_) and np.isfinite(mm) and r_ > 0: mu *= float(np.clip(mm/r_, 0.1, 4.0))
        s_ = sq[(a-1)//STEP]
        if np.isfinite(s_): mu *= (2.0 if s_ >= 0.97 else (1.0 if s_ >= 0.90 else 0.5))
        lv = lev*mu
        if lv <= 0: continue
        liq = 1.0/lv-L.MMR; r = res
        if r <= -bar and liq < bar: r = -1.0/lv
        pnl = cap*lv*(r-fee/1e4)
        if pnl < -cap: pnl = -cap
        cap += pnl; cur.append(cap)
        if cap <= 1e5: break
    if not cur: return None
    eq = np.array(cur)
    if eq[-1] <= 2e5: return dict(ann=-99.9, mdd=-100.0, n=len(ei), wr=np.mean(rr > 0)*100, fin=0)
    yy_ = Y if yr is None else 1.0
    pk = np.maximum.accumulate(np.concatenate([[1e7], eq]))[1:]
    return dict(ann=((eq[-1]/1e7)**(1/yy_)-1)*100, mdd=(eq/pk-1).min()*100,
                n=len(ei), wr=np.mean(rr > 0)*100, fin=eq[-1]/1e7)


mm_ = min(min(len(Z[n]) for n in CUR+NEW), len(d["c"])//STEP)
SIG_CUR = np.nanmean(np.vstack([Z[n][:mm_] for n in CUR]), axis=0)
SIG_NEW = np.nanmean(np.vstack([Z[n][:mm_] for n in NEW]), axis=0)
P(f"  {'신호':>8s} {'분위':>6s} {'연거래':>6s} {'승률%':>7s} {'1x 연%':>9s} {'3x 연%':>9s} {'5x 연%':>9s}")
for lab, sg in (("현행", SIG_CUR), ("재선택", SIG_NEW)):
    for q in (0.15, 0.20, 0.25):
        v1 = bt(sg, q, 0.015, lev=1); v3 = bt(sg, q, 0.015, lev=3); v5 = bt(sg, q, 0.015, lev=5)
        f_ = lambda v: f"{v['ann']:+9.1f}" if v and v['ann'] > -99 else "     파산"
        P(f"  {lab:>8s} {q*100:5.0f}% {v3['n']/Y:6.0f} {v3['wr']:7.2f} {f_(v1)} {f_(v3)} {f_(v5)}")

P("\n  재선택 신호 연도별 (상위20%, 3배):")
out = []
for yr in range(2020, 2027):
    v = bt(SIG_NEW, 0.20, 0.015, lev=3, yr=yr)
    out.append(f"{yr} {(v['fin']-1)*100:+7.0f}%" if v and v['fin'] > 0 else f"{yr}   파산")
P("    " + " | ".join(out))
