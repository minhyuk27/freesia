"""R12 확정 검증 — 4시간봉 분rv×20 딥.

확인할 것: 플라시보(무작위 봉), 3단 레버리지, 변동성 타겟 사이징, 이상치 내성,
          비용/관통 고원, 그리고 R4 스윙·ETH숏과의 포트폴리오 효과."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R/"research"))
import z0_20_lib as L2
from z0_20_final import build_sigs, exp_z


def P(*a, end="\n"):
    print(" ".join(str(x) for x in a), end=end, flush=True)


S = "BTCUSDT"; Y = L2.YRS
d = L2.D[S]; idx = d["idx"]; rv = d["rv"]
O, H, Lo, C = d["o"], d["h"], d["l"], d["c"]
N = len(C); warm = 200000
med = np.full(N, np.nan)
for i in range(warm, N, 5000):
    x = rv[:i]; x = x[np.isfinite(x)]
    if len(x) >= 1000: med[i:i+5000] = np.median(x)


@numba.njit(cache=True)
def dip(o, h, l, c, rvv, medv, barmin, k0, k1, k2, w0, w1, w2,
        eps_bp, slip_bp, fee_bp, start, n_end, use_size, rnd_off):
    """use_size=1: 변동성목표 사이징(clip(med/rv,0.2,3)) 적용.
    rnd_off: 봉마다 진입 시각을 이만큼 밀어 플라시보로 쓴다(0이면 정상)."""
    K = 200000
    ea = np.zeros(K, np.int64); er = np.zeros(K); sz = np.zeros(K)
    k = 0
    nb = (n_end-start)//barmin
    for b in range(nb):
        t0 = start + b*barmin + rnd_off
        t1 = t0 + barmin - 1
        if t1 >= n_end-1 or t0 < start: continue
        if np.isnan(rvv[t0]) or rvv[t0] <= 0: continue
        px0 = o[t0]
        tot = 0.0; wsum = 0.0; ft = -1
        for li in range(3):
            kk = k0 if li == 0 else (k1 if li == 1 else k2)
            ww = w0 if li == 0 else (w1 if li == 1 else w2)
            T = px0*(1.0 - rvv[t0]*kk)
            trig = T*(1.0 - eps_bp/1e4)
            got = -1
            for j in range(t0, t1+1):
                if l[j] <= trig: got = j; break
            if got >= 0:
                fill = T*(1.0 + slip_bp/1e4)
                tot += ww*(c[t1]/fill - 1.0 - fee_bp/1e4); wsum += ww
                if ft < 0: ft = got
        if wsum > 0:
            m = 1.0
            if use_size == 1:
                mm = medv[t0]
                if np.isfinite(mm) and rvv[t0] > 0:
                    x = mm/rvv[t0]
                    m = 0.2 if x < 0.2 else (3.0 if x > 3.0 else x)
            ea[k] = ft; er[k] = tot/wsum; sz[k] = m; k += 1
            if k >= K-2: break
    return ea[:k], er[:k], sz[:k]


def stat(ea, er, sz=None, nom=1.0):
    if len(er) < 30: return None
    w = np.ones(len(er)) if sz is None else sz
    m = pd.Series(nom*w*er, index=pd.DatetimeIndex(idx[ea])).resample("ME").sum()
    eq = (1+m).cumprod()
    if (eq <= 0.01).any(): return None
    yy = len(m)/12
    ann = (eq.iloc[-1]**(1/yy)-1)*100
    return dict(ann=ann, mon=((1+ann/100)**(1/12)-1)*100,
                mdd=(eq/eq.cummax()-1).min()*100, n=len(er), wr=np.mean(er > 0)*100,
                bp=er.mean()*1e4, mwr=np.mean(m > 0)*100, mser=m)


BAR = 720; KS = (15.0, 21.0, 30.0); WS = (1.0, 3.0, 6.0)   # [3]에서 720분봉 깊이15가 최적
EPS, SLIP, FEE = 5.0, 10.0, 10.0
ea, er, sz = dip(O, H, Lo, C, rv, med, BAR, *KS, *WS, EPS, SLIP, FEE, warm, N, 1, 0)  # use_size=1
base = stat(ea, er)
P(f"기준 구성: 4시간봉 분rv×{KS} 가중{WS}, 관통{EPS:.0f}bp 슬립{SLIP:.0f}bp 수수료{FEE:.0f}bp")
P(f"  거래 {len(er):,}건 (연 {len(er)/Y:.0f}), 승률 {base['wr']:.2f}%, 거래당 {base['bp']:+.1f}bp, "
  f"무레버 연 {base['ann']:+.1f}%\n")

P("="*126); P("[1] 플라시보 — 무작위 봉 위상 300회 (같은 규칙, 진입 시각만 이동)"); P("="*126)
rng = np.random.default_rng(1234)
vals, anns = [], []
for it in range(300):
    off = int(rng.integers(1, BAR))
    ea2, er2, _ = dip(O, H, Lo, C, rv, med, BAR, *KS, *WS, EPS, SLIP, FEE, warm, N, 1, off)
    r = stat(ea2, er2)
    if r: vals.append(r["bp"]); anns.append(r["ann"])
vals, anns = np.array(vals), np.array(anns)
P(f"  거래당bp 실제 {base['bp']:+7.2f} | 위약 평균 {vals.mean():+6.2f} 최대 {vals.max():+6.2f} "
  f"→ 백분위 {np.mean(vals < base['bp'])*100:5.1f}%")
P(f"  연수익%  실제 {base['ann']:+7.1f} | 위약 평균 {anns.mean():+6.1f} 최대 {anns.max():+6.1f} "
  f"→ 백분위 {np.mean(anns < base['ann'])*100:5.1f}%")
P("  ※ 위상 이동은 같은 메커니즘을 다른 시각에 적용하는 것이라 '봉 경계가 특별한가'만 본다.")

P("\n  강한 대조군 — 딥 깊이를 0으로(시가 시장가 매수):")
ea0, er0, _ = dip(O, H, Lo, C, rv, med, BAR, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, FEE, warm, N, 0, 0)
r0 = stat(ea0, er0)
P(f"    깊이0: " + ("파산" if r0 is None else f"연 {r0['ann']:+.1f}%") +
  f"  거래당 {er0.mean()*1e4:+.1f}bp  승률 {np.mean(er0>0)*100:.2f}%  n={len(er0):,}")
P(f"    rv×20: 연 {base['ann']:+.1f}%  거래당 {base['bp']:+.1f}bp  승률 {base['wr']:.2f}%  n={len(er):,}")

P("\n"+"="*126); P("[2] 3단 레버리지 (사이징 유무)"); P("="*126)
ea_ns, er_ns, _ = dip(O, H, Lo, C, rv, med, BAR, *KS, *WS, EPS, SLIP, FEE, warm, N, 0, 0)
P(f"  {'구간':>12s} {'명목':>6s} " + " ".join(f"{lab:>26s}" for lab in ("사이징 없음", "변동성목표 사이징")))
for lb, nom in (("무레버리지", 1), ("중간", 2), ("중간+", 3), ("공격", 5), ("최대공격", 8), ("극단", 12)):
    row = []
    for use in (None, sz):
        r = stat(ea, er, use, nom) if use is not None else stat(ea_ns, er_ns, None, nom)
        row.append("           파산           " if r is None
                   else f"연{r['ann']:+8.1f}% 월{r['mon']:+6.2f}% MDD{r['mdd']:6.1f}%")
    P(f"  {lb:>12s} {nom:5d}x " + " ".join(f"{x:>26s}" for x in row))
P(f"\n  사이징 승수: 중앙 {np.median(sz):.2f} 평균 {sz.mean():.2f} 90% {np.percentile(sz,90):.2f}")

P("\n"+"="*126); P("[3] 깊이 × 봉 길이 고원 (무레버, 사이징 없음)"); P("="*126)
P(f"  {'깊이(분rv×)':>12s} " + " ".join(f"{str(b)+'분':>14s}" for b in (240, 480, 720, 1440, 2880)))
for mult in (5, 8, 10, 12, 15, 20, 28, 40):
    row = []
    for bm in (240, 480, 720, 1440, 2880):
        ks = (mult, mult*1.4, mult*2.0)
        ea2, er2, _ = dip(O, H, Lo, C, rv, med, bm, *ks, *WS, EPS, SLIP, FEE, warm, N, 0, 0)
        r = stat(ea2, er2)
        row.append("     파산     " if r is None else f"{r['ann']:+7.1f}%/{r['wr']:4.1f}")
    P(f"  {mult:11d} " + " ".join(f"{x:>14s}" for x in row))
P("  (셀 = 연수익% / 승률%)")

P("\n"+"="*126); P("[4] 실행 가정 고원 (rv×20)"); P("="*126)
P(f"  {'관통':>7s} " + " ".join(f"{'슬립'+str(s)+'bp':>14s}" for s in (0, 10, 20, 30, 50)))
for eps in (0, 5, 10, 20, 40):
    row = []
    for slip in (0, 10, 20, 30, 50):
        ea2, er2, _ = dip(O, H, Lo, C, rv, med, BAR, *KS, *WS, float(eps), float(slip), FEE, warm, N, 1, 0)
        r = stat(ea2, er2)
        row.append("     파산     " if r is None else f"{r['ann']:+8.1f}%/{r['n']/Y:4.0f}")
    P(f"  {eps:5d}bp " + " ".join(f"{x:>14s}" for x in row))
P("  (셀 = 연수익% / 연거래)")

P("\n"+"="*126); P("[5] 이상치 내성 · 연도별"); P("="*126)
o = np.argsort(-er)
for k in (0, 5, 10, 20, 50):
    keep = np.ones(len(er), bool); keep[o[:k]] = False
    r = stat(ea[keep], er[keep])
    P(f"  상위 {k:2d}건 제외: 거래당 {er[keep].mean()*1e4:+6.1f}bp  "
      + ("파산" if r is None else f"연 {r['ann']:+7.1f}%  승률 {r['wr']:5.2f}%"))
P("")
m = base["mser"]
out = []
for yr in range(2020, 2027):
    sub = m[m.index.year == yr]
    out.append(f"{yr}  n/a" if len(sub) == 0 else f"{yr} {((1+sub).prod()-1)*100:+7.0f}%")
P("  연도별(무레버): " + " | ".join(out))
P(f"  월 승률 {base['mwr']:.1f}%, 거래월 {int((m!=0).sum())}/{len(m)}")

P("\n"+"="*126); P("[6] 포트폴리오 — z12 + z4스윙 + ETH숏"); P("="*126)
SG = build_sigs(S)
nm = list(SG.keys())
Zb = {n: exp_z(np.asarray(SG[n], float)) for n in nm}
ml = min(min(len(Zb[n]) for n in nm), N//L2.STEP)
SIG = np.nanmean(np.vstack([Zb[n][:ml] for n in nm if n.endswith("4320")]), axis=0)
E = L2.D["ETHUSDT"]
SGE = build_sigs("ETHUSDT")
ZE = {n: exp_z(np.asarray(SGE[n], float)) for n in SGE}
mlE = min(min(len(ZE[n]) for n in ZE), len(E["c"])//L2.STEP)
SIGE = np.nanmean(np.vstack([ZE[n][:mlE] for n in ZE if n.endswith("4320")]), axis=0)


@numba.njit(cache=True)
def barrier(o, h, l, c, sg, th, bar, maxh, step, start, n_end, side):
    K = 40000
    ea = np.zeros(K, np.int64); er = np.zeros(K); k = 0; t = start
    while t < n_end-maxh-2:
        i = t//step
        if i >= len(sg) or not np.isfinite(sg[i]): t += step; continue
        if side == 1 and sg[i] < th: t += step; continue
        if side == -1 and sg[i] > th: t += step; continue
        f = t+1; px = o[f]; e = -1; res = 0.0; end = min(f+maxh, n_end-1)
        if side == 1:
            U = px*(1+bar); D = px*(1-bar)
            for j in range(f, end+1):
                if l[j] <= D: e = j; res = -bar; break
                if h[j] >= U: e = j; res = bar; break
            if e < 0: e = end; res = c[e]/px-1
        else:
            U = px*(1-bar); D = px*(1+bar)
            for j in range(f, end+1):
                if h[j] >= D: e = j; res = -bar; break
                if l[j] <= U: e = j; res = bar; break
            if e < 0: e = end; res = px/c[e]-1
        ea[k] = f; er[k] = res-10.0/1e4; k += 1
        t = e+1
        if k >= K-2: break
    return ea[:k], er[:k]


A = barrier(O, H, Lo, C, SIG, float(np.quantile(SIG[np.isfinite(SIG)], 0.90)),
            0.04, 43200, L2.STEP, warm, N, 1)
D_ = barrier(E["o"], E["h"], E["l"], E["c"], SIGE,
             float(np.quantile(SIGE[np.isfinite(SIGE)], 0.10)), 0.04, 43200, L2.STEP,
             warm, len(E["c"]), -1)
mZ = pd.Series(er, index=pd.DatetimeIndex(idx[ea])).resample("ME").sum()
mA = pd.Series(A[1], index=pd.DatetimeIndex(idx[A[0]])).resample("ME").sum()
mD = pd.Series(D_[1], index=pd.DatetimeIndex(E["idx"][D_[0]])).resample("ME").sum()
df = pd.concat([mZ, mA, mD], axis=1, keys=["z12딥", "z4스윙", "ETH숏"]).fillna(0)
P("  월별 수익 상관:")
cr = df.corr()
P("          " + " ".join(f"{c:>9s}" for c in cr.columns))
for i, r in cr.iterrows():
    P(f"  {i:>8s} " + " ".join(f"{r[c]:9.3f}" for c in cr.columns))


def pf(ws, nom):
    tot = (df*np.array(ws)).sum(axis=1)*nom
    eq = (1+tot).cumprod()
    if (eq <= 0.01).any(): return None
    yy = len(tot)/12
    ann = (eq.iloc[-1]**(1/yy)-1)*100
    return dict(ann=ann, mon=((1+ann/100)**(1/12)-1)*100,
                mdd=(eq/eq.cummax()-1).min()*100, wr=np.mean(tot > 0)*100)


P(f"\n  {'구성':>22s} " + " ".join(f"{'명목'+str(n)+'배':>17s}" for n in (1, 2, 3, 5, 8)))
for lab, ws in (("z12 단독", [1, 0, 0]), ("z4스윙 단독", [0, 1, 0]), ("ETH숏 단독", [0, 0, 1]),
                ("z12+z4 균등", [.5, .5, 0]), ("z12+ETH숏 균등", [.5, 0, .5]),
                ("셋 균등", [1/3, 1/3, 1/3]), ("z12 2 : 나머지 1:1", [.5, .25, .25])):
    row = []
    for nom in (1, 2, 3, 5, 8):
        r = pf(ws, nom)
        row.append("       파산      " if r is None else f"{r['ann']:+8.1f}%/{r['mdd']:6.1f}%")
    P(f"  {lab:>22s} " + " ".join(f"{x:>17s}" for x in row))
P("  (셀 = 연수익% / MDD%)")
