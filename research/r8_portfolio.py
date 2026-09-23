"""R8 메커니즘 포트폴리오 — 천장을 올리는 이론적으로 유일한 길.

천장 분석 결과: 월수익 = exp(N_독립 × g)에서 N_독립이 제약이다.
같은 종목·같은 메커니즘의 거래는 상관되어 있어 동시 보유로 늘려도 N_독립이 안 는다.
**서로 다른 메커니즘**은 상관이 낮으므로 N_독립을 실제로 늘린다.

묶을 후보(전부 검증 완료된 것만):
  A. R4 스윙   — 신호 상위10%, ±4% 배리어, 평균보유 3,175분
  B. z0.14 초단기 딥 — 15분봉 rv×6·8·12 지정가 → 같은 봉 종가 (보유 ≤15분)
  C. z0.3 깊은 딥  — 4h~1d봉 rv×10·14·20 지정가 → 봉 종가 (보유 4~24시간)

보유시간대가 겹치지 않아 자본 충돌이 적고, 메커니즘이 달라 상관이 낮을 것으로 예상."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R/"research"))
import z0_20_lib as L2
from z0_20_final import build_sigs, exp_z


def P(*a, end="\n"):
    print(" ".join(str(x) for x in a), end=end, flush=True)


S = "BTCUSDT"; Y = L2.YRS; MMR = L2.MMR; STEP = L2.STEP
d = L2.D[S]; idx = d["idx"]; rv = d["rv"]
O, H, Lo, C = d["o"], d["h"], d["l"], d["c"]
N = len(C); warm = 200000
SG = build_sigs(S)
names = list(SG.keys())
Z = {n: exp_z(np.asarray(SG[n], float)) for n in names}
ml = min(min(len(Z[n]) for n in names), N//STEP)
SIG = np.nanmean(np.vstack([Z[n][:ml] for n in names if n.endswith("4320")]), axis=0)


@numba.njit(cache=True)
def ev_swing(o, h, l, c, sigv, th, bar, maxh, step, start, n_end):
    """A. z4 스윙: 신호 문턱 → 시장가 → ±bar 배리어"""
    K = 40000
    ea = np.zeros(K, np.int64); eb = np.zeros(K, np.int64); er = np.zeros(K)
    k = 0; t = start
    while t < n_end-maxh-2:
        i = t//step
        if i >= len(sigv) or not np.isfinite(sigv[i]) or sigv[i] < th: t += step; continue
        f = t+1; px = o[f]; U = px*(1+bar); D = px*(1-bar)
        e = -1; res = 0.0; end = min(f+maxh, n_end-1)
        for j in range(f, end+1):
            if l[j] <= D: e = j; res = -bar; break
            if h[j] >= U: e = j; res = bar; break
        if e < 0: e = end; res = c[e]/px-1.0
        ea[k] = f; eb[k] = e; er[k] = res; k += 1
        t = e+1
        if k >= K-2: break
    return ea[:k], eb[:k], er[:k]


@numba.njit(cache=True)
def ev_dip(o, h, l, c, rvv, bar_min, ks0, ks1, ks2, w0, w1, w2, start, n_end):
    """B/C. 딥: bar_min분봉 시가 대비 rv×k 아래 지정가 3층 → 같은 봉 종가 청산"""
    K = 120000
    ea = np.zeros(K, np.int64); eb = np.zeros(K, np.int64); er = np.zeros(K)
    k = 0
    nb = (n_end-start)//bar_min
    for b in range(nb):
        t0 = start + b*bar_min
        t1 = t0 + bar_min - 1
        if t1 >= n_end-1: break
        if np.isnan(rvv[t0]) or rvv[t0] <= 0: continue
        px0 = o[t0]
        tot = 0.0; wsum = 0.0; fill_t = -1
        for li in range(3):
            kk = ks0 if li == 0 else (ks1 if li == 1 else ks2)
            ww = w0 if li == 0 else (w1 if li == 1 else w2)
            T = px0*(1.0 - rvv[t0]*kk)
            got = -1
            for j in range(t0, t1+1):
                if l[j] <= T: got = j; break
            if got >= 0:
                tot += ww*(c[t1]/T - 1.0); wsum += ww
                if fill_t < 0 or got < fill_t: fill_t = got
        if wsum > 0:
            ea[k] = fill_t; eb[k] = t1; er[k] = tot/wsum; k += 1
            if k >= K-2: break
    return ea[:k], eb[:k], er[:k]


# D. ETH 숏 — z7에서 발견: BTC는 롱만, ETH는 숏만 산다(드리프트 차이)
E = L2.D["ETHUSDT"]
SGE = build_sigs("ETHUSDT")
ZE = {n: exp_z(np.asarray(SGE[n], float)) for n in SGE}
mlE = min(min(len(ZE[n]) for n in ZE), len(E["c"])//STEP)
SIGE = np.nanmean(np.vstack([ZE[n][:mlE] for n in ZE if n.endswith("4320")]), axis=0)


@numba.njit(cache=True)
def ev_short(o, h, l, c, sigv, th_lo, bar, maxh, step, start, n_end):
    """신호 하위 문턱에서 숏 진입 → ±bar 배리어(내려가면 이익)"""
    K = 40000
    ea = np.zeros(K, np.int64); eb = np.zeros(K, np.int64); er = np.zeros(K)
    k = 0; t = start
    while t < n_end-maxh-2:
        i = t//step
        if i >= len(sigv) or not np.isfinite(sigv[i]) or sigv[i] > th_lo: t += step; continue
        f = t+1; px = o[f]; U = px*(1-bar); D = px*(1+bar)
        e = -1; res = 0.0; end = min(f+maxh, n_end-1)
        for j in range(f, end+1):
            if h[j] >= D: e = j; res = -bar; break
            if l[j] <= U: e = j; res = bar; break
        if e < 0: e = end; res = px/c[e]-1.0
        ea[k] = f; eb[k] = e; er[k] = res; k += 1
        t = e+1
        if k >= K-2: break
    return ea[:k], eb[:k], er[:k]


TH10 = float(np.quantile(SIG[np.isfinite(SIG)], 0.90))
A = ev_swing(O, H, Lo, C, SIG, TH10, 0.04, 43200, STEP, warm, N)
B = ev_dip(O, H, Lo, C, rv, 15, 6.0, 8.0, 12.0, 1.0, 3.0, 6.0, warm, N)
Cc = ev_dip(O, H, Lo, C, rv, 240, 10.0, 14.0, 20.0, 1.0, 3.0, 6.0, warm, N)
THE = float(np.quantile(SIGE[np.isfinite(SIGE)], 0.10))
Dd = ev_short(E["o"], E["h"], E["l"], E["c"], SIGE, THE, 0.04, 43200, STEP, warm, len(E["c"]))
idxE = E["idx"]
P(f"A z4 스윙:     {len(A[0]):5d}건 (연 {len(A[0])/Y:6.1f}), 거래당 {A[2].mean()*1e4:+7.1f}bp, "
  f"승률 {np.mean(A[2]>0)*100:5.2f}%, 평균보유 {(A[1]-A[0]).mean():6.0f}분")
P(f"B z1.4 15분딥: {len(B[0]):5d}건 (연 {len(B[0])/Y:6.1f}), 거래당 {B[2].mean()*1e4:+7.1f}bp, "
  f"승률 {np.mean(B[2]>0)*100:5.2f}%, 평균보유 {(B[1]-B[0]).mean():6.0f}분")
P(f"C z0.3 4시간딥:{len(Cc[0]):5d}건 (연 {len(Cc[0])/Y:6.1f}), 거래당 {Cc[2].mean()*1e4:+7.1f}bp, "
  f"승률 {np.mean(Cc[2]>0)*100:5.2f}%, 평균보유 {(Cc[1]-Cc[0]).mean():6.0f}분")
P(f"D ETH숏 ±4%:  {len(Dd[0]):5d}건 (연 {len(Dd[0])/Y:6.1f}), 거래당 {Dd[2].mean()*1e4:+7.1f}bp, "
  f"승률 {np.mean(Dd[2]>0)*100:5.2f}%, 평균보유 {(Dd[1]-Dd[0]).mean():6.0f}분")

P("\n"+"="*130); P("[1] 월별 수익 상관 — 정말 독립인가"); P("="*130)


def monthly(ea, er, nom=1.0, fee=10.0, ix=None):
    ix = idx if ix is None else ix
    s = pd.Series(nom*(er-fee/1e4), index=pd.DatetimeIndex(ix[ea]))
    return s.resample("ME").sum()


ma, mb, mc = monthly(A[0], A[2]), monthly(B[0], B[2]), monthly(Cc[0], Cc[2])
md = monthly(Dd[0], Dd[2], ix=idxE)
df = pd.concat([ma, mb, mc, md], axis=1,
               keys=["A스윙", "B15분딥", "C4시간딥", "D ETH숏"]).fillna(0)
P("  월별 수익 상관행렬:")
cr = df.corr()
P("            " + " ".join(f"{c:>10s}" for c in cr.columns))
for i, r in cr.iterrows():
    P(f"  {i:>10s} " + " ".join(f"{r[c]:10.3f}" for c in cr.columns))
P(f"\n  월 승률: A {np.mean(ma>0)*100:.1f}%  B {np.mean(mb>0)*100:.1f}%  "
  f"C {np.mean(mc>0)*100:.1f}%  D {np.mean(md>0)*100:.1f}%")
P(f"  거래월 수: A {np.sum(ma!=0)} / B {np.sum(mb!=0)} / C {np.sum(mc!=0)} / D {np.sum(md!=0)} "
  f"(총 {len(df)}개월)")

P("\n"+"="*130); P("[2] 포트폴리오 — 자본을 나눠 동시 운용 (각 다리 독립 자본)"); P("="*130)


def port(legs, weights, nom, fee=10.0):
    """각 다리에 weights 비율의 자본을 주고 독립 운용, 월별 합산."""
    tot = None
    for (ea, eb, er), w in zip(legs, weights):
        ix = idxE if (ea is Dd[0]) else idx
        s = pd.Series(nom*w*(er-fee/1e4), index=pd.DatetimeIndex(ix[ea]))
        m = s.resample("ME").sum()
        tot = m if tot is None else tot.add(m, fill_value=0)
    eq = (1+tot).cumprod()
    if (eq <= 0.01).any(): return None
    yy = len(tot)/12
    ann = (eq.iloc[-1]**(1/yy)-1)*100
    peak = eq.cummax()
    return dict(ann=ann, mon=((1+ann/100)**(1/12)-1)*100, mdd=(eq/peak-1).min()*100,
                fin=eq.iloc[-1], wr=np.mean(tot > 0)*100, nmonth=int((tot != 0).sum()))


P(f"  {'구성':>22s} {'가중':>16s} " + " ".join(f"{'명목'+str(n)+'배':>13s}" for n in (1, 2, 3, 5, 8)))
CFG = [("A 단독", [A], [1.0]), ("B 단독", [B], [1.0]), ("C 단독", [Cc], [1.0]),
       ("D ETH숏 단독", [Dd], [1.0]),
       ("A+B 균등", [A, B], [0.5, 0.5]), ("A+C 균등", [A, Cc], [0.5, 0.5]),
       ("A+D 균등", [A, Dd], [0.5, 0.5]), ("B+D 균등", [B, Dd], [0.5, 0.5]),
       ("A+B+C 균등", [A, B, Cc], [1/3, 1/3, 1/3]),
       ("A+B+D 균등", [A, B, Dd], [1/3, 1/3, 1/3]),
       ("A+B+C+D 균등", [A, B, Cc, Dd], [0.25, 0.25, 0.25, 0.25]),
       ("A+B+D 2:1:1", [A, B, Dd], [0.5, 0.25, 0.25]),
       ("A+B+C+D 3:2:2:3", [A, B, Cc, Dd], [0.3, 0.2, 0.2, 0.3])]
BEST = None
for lab, legs, w in CFG:
    row = []
    for nom in (1, 2, 3, 5, 8):
        r = port(legs, w, nom)
        row.append("        파산" if r is None else f"{r['ann']:+12.1f}")
        if r and (BEST is None or r["ann"] > BEST[0]): BEST = (r["ann"], lab, legs, w, nom, r)
    P(f"  {lab:>22s} {'/'.join(f'{x:.2f}' for x in w):>16s} " + " ".join(f"{x:>13s}" for x in row))

P("\n"+"="*130); P("[3] 최고 포트폴리오 3단 레버리지 + 월 통계"); P("="*130)
if BEST:
    _, lab, legs, w, nom0, _ = BEST
    P(f"  구성: {lab}")
    P(f"  {'구간':>12s} {'명목':>6s} {'연수익%':>12s} {'월%':>9s} {'MDD%':>8s} {'월승률%':>8s} {'최종배수':>12s}")
    for lb, nom in (("무레버리지", 1), ("중간", 2), ("중간+", 3), ("공격", 5), ("최대공격", 8), ("극단", 12)):
        r = port(legs, w, nom)
        P(f"  {lb:>12s} {nom:5d}x " + ("        파산" if r is None
                                        else f"{r['ann']:12.1f} {r['mon']:9.2f} {r['mdd']:8.1f} "
                                             f"{r['wr']:8.1f} {r['fin']:12.2f}"))
    P("\n  연도별(명목 3배):")
    tot = None
    for (ea, eb, er), ww in zip(legs, w):
        ix = idxE if (ea is Dd[0]) else idx
        s = pd.Series(3*ww*(er-10.0/1e4), index=pd.DatetimeIndex(ix[ea]))
        m = s.resample("ME").sum()
        tot = m if tot is None else tot.add(m, fill_value=0)
    out = []
    for yr in range(2020, 2027):
        sub = tot[tot.index.year == yr]
        if len(sub) == 0: out.append(f"{yr}  n/a"); continue
        out.append(f"{yr} {((1+sub).prod()-1)*100:+7.0f}%")
    P("    " + " | ".join(out))

P("\n"+"="*130); P("[4] 독립 베팅 수의 실제 증가 — 포트폴리오가 천장을 올리나"); P("="*130)
P(f"  {'구성':>16s} {'연거래':>8s} {'평균보유':>9s} {'월승률%':>8s} {'월수익 표준편차':>14s} {'샤프(월)':>9s}")
for lab, legs, w in CFG:
    tot = None; ntr = 0; hsum = 0.0
    for (ea, eb, er), ww in zip(legs, w):
        ix = idxE if (ea is Dd[0]) else idx
        s = pd.Series(ww*(er-10.0/1e4), index=pd.DatetimeIndex(ix[ea]))
        m = s.resample("ME").sum()
        tot = m if tot is None else tot.add(m, fill_value=0)
        ntr += len(ea); hsum += (eb-ea).sum()
    sd = tot.std()
    P(f"  {lab:>16s} {ntr/Y:8.0f} {hsum/max(ntr,1):8.0f}분 {np.mean(tot>0)*100:8.1f} "
      f"{sd*100:13.2f}% {tot.mean()/max(sd,1e-9):9.3f}")
