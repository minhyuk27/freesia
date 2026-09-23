"""R16 — ETH 필터 구성의 월 +87%를 조인다.

R15에서 ETH + (rv분위↑ 프리미엄↑ 직전1일수익↓) 상위70%가
무레버 +368.3%(MDD −10.4%), 8배 월 +87.14%를 냈다. 목표 월 +80%를 처음 넘겼다.

그대로 믿으면 안 되는 이유 둘:
  ①12시간봉 중간 청산 미점검 — 8배면 12.5% 역행에 청산인데 종가에만 손익 반영
  ②증거금 1.7배 과대평가 — 체결된 층만 정규화해 항상 명목 1배로 잡음

여기서 둘 다 반영하고, 플라시보·워크포워드까지 건다."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R/"research"))
import z0_20_lib as L2
from r14_dipfilter import build, EPS, SLIP, FEE, warm, BAR


def P(*a, end="\n"):
    print(" ".join(str(x) for x in a), end=end, flush=True)


Y = L2.YRS; MMR = L2.MMR
S = "ETHUSDT"
d = L2.D[S]; idx = d["idx"]; N = len(d["c"])
O, H, Lo, C, RV = d["o"], d["h"], d["l"], d["c"], d["rv"]
KS = (15.0, 21.0, 30.0); WS = (1.0, 3.0, 6.0)


@numba.njit(cache=True)
def dip2(o, h, l, c, rvv, barmin, k0, k1, k2, w0, w1, w2, eps, slip, fee,
         start, n_end, norm):
    """norm=1: 체결층만 정규화(명목 1배). norm=0: 10단위 준비 방식(미체결은 현금)."""
    K = 200000
    ea = np.zeros(K, np.int64); er = np.zeros(K); mae = np.zeros(K); nomv = np.zeros(K)
    k = 0
    nb = (n_end-start)//barmin
    wtot = w0+w1+w2
    for b in range(nb):
        t0 = start+b*barmin; t1 = t0+barmin-1
        if t1 >= n_end-1: break
        r = rvv[t0]
        if np.isnan(r) or r <= 0: continue
        px0 = o[t0]
        tot = 0.0; wsum = 0.0; ft = -1; worst = 0.0
        for li in range(3):
            kk = k0 if li == 0 else (k1 if li == 1 else k2)
            ww = w0 if li == 0 else (w1 if li == 1 else w2)
            T = px0*(1.0-r*kk); trig = T*(1.0-eps/1e4)
            got = -1
            for j in range(t0, t1+1):
                if l[j] <= trig: got = j; break
            if got >= 0:
                fill = T*(1.0+slip/1e4)
                tot += ww*(c[t1]/fill-1.0-fee/1e4); wsum += ww
                lo = fill
                for j in range(got, t1+1):
                    if l[j] < lo: lo = l[j]
                dd = lo/fill-1.0
                if dd < worst: worst = dd
                if ft < 0: ft = got
        if wsum > 0:
            if norm == 1:
                ea[k] = ft; er[k] = tot/wsum; nomv[k] = 1.0
            else:
                ea[k] = ft; er[k] = tot/wtot; nomv[k] = wsum/wtot
            mae[k] = worst; k += 1
            if k >= K-2: break
    return ea[:k], er[:k], mae[:k], nomv[:k]


def expq(v, w0=200, st=50):
    v = np.asarray(v, float); q = np.full(len(v), np.nan)
    for i in range(w0, len(v), st):
        x = v[:i]; x = x[np.isfinite(x)]
        if len(x) < 100: continue
        q[i:i+st] = np.searchsorted(np.sort(x), v[i:i+st])/len(x)
    return q


def curve(er, ea, nom=1.0, mae=None, lev_liq=True):
    """lev_liq=True면 봉중간 MAE가 청산거리를 넘을 때 증거금 전손 처리."""
    r = er.copy()
    nliq = 0
    if lev_liq and mae is not None and nom > 1:
        liq_d = 1.0/nom - MMR
        hit = (-mae) >= liq_d
        r = np.where(hit, -1.0/nom, r)
        nliq = int(hit.sum())
    m = pd.Series(nom*r, index=pd.DatetimeIndex(idx[ea])).resample("ME").sum()
    eq = (1+m).cumprod()
    if (eq <= 0.01).any(): return None, nliq
    a = (eq.iloc[-1]**(12/len(m))-1)*100
    return dict(ann=a, mon=((1+a/100)**(1/12)-1)*100,
                mdd=(eq/eq.cummax()-1).min()*100, mwr=np.mean(m > 0)*100, ser=m), nliq


F = build(S)
for NORM, tag in ((1, "정규화(체결층만) — z15 방식"), (0, "10단위 준비 — 보수적")):
    ea, er, mae, nomv = dip2(O, H, Lo, C, RV, BAR, *KS, *WS, EPS, SLIP, FEE, warm, N, NORM)
    sc = None; n = 0
    for nm, sgn in (("rv분위", 1), ("프리미엄", 1), ("직전1일수익", -1)):
        a = F.get(nm)
        if a is None: continue
        q = expq(np.asarray(a, float)[ea])
        v = q if sgn > 0 else 1.0-q
        sc = v if sc is None else np.nansum(np.vstack([sc, v]), axis=0)
        n += 1
    sc = sc/n
    m = np.isfinite(sc) & (sc >= 0.30)   # 상위 70%
    P("="*126); P(f"[{tag}] 필터 상위70% → {m.sum()}건 (연 {m.sum()/Y:.0f}), "
                  f"승률 {np.mean(er[m]>0)*100:.2f}%, 거래당 {er[m].mean()*1e4:+.1f}bp")
    P(f"  평균 실효 명목 {nomv[m].mean():.3f}배  |  봉중간 MAE 중앙 {np.percentile(-mae[m],50)*100:.2f}% "
      f"95분위 {np.percentile(-mae[m],95)*100:.2f}% 최악 {(-mae[m]).max()*100:.2f}%")
    P("="*126)
    P(f"  {'구간':>12s} {'명목':>6s} {'청산건':>6s} {'연수익%':>13s} {'월%':>9s} {'MDD%':>8s} {'월승률%':>8s}")
    for lb, nom in (("무레버리지", 1), ("중간", 2), ("중간+", 3), ("공격", 5),
                    ("최대공격", 8), ("극단", 12)):
        r, nl = curve(er[m], ea[m], nom, mae[m])
        P(f"  {lb:>12s} {nom:5d}x {nl:6d} " + ("         파산" if r is None
                                                else f"{r['ann']:13.1f} {r['mon']:9.2f} {r['mdd']:8.1f} {r['mwr']:8.1f}"))
    P("")

# 이후 분석은 보수적(10단위) 기준
ea, er, mae, nomv = dip2(O, H, Lo, C, RV, BAR, *KS, *WS, EPS, SLIP, FEE, warm, N, 0)
sc = None; n = 0
for nm, sgn in (("rv분위", 1), ("프리미엄", 1), ("직전1일수익", -1)):
    q = expq(np.asarray(F[nm], float)[ea])
    v = q if sgn > 0 else 1.0-q
    sc = v if sc is None else np.nansum(np.vstack([sc, v]), axis=0); n += 1
sc = sc/n
m = np.isfinite(sc) & (sc >= 0.30)

P("="*126); P("[3] 연도별·워크포워드 (보수적 기준, 무레버)"); P("="*126)
r, _ = curve(er[m], ea[m], 1.0, mae[m])
out = []
for yr in range(2020, 2027):
    sub = r["ser"][r["ser"].index.year == yr]
    out.append(f"{yr}  n/a" if len(sub) == 0 else f"{yr} {((1+sub).prod()-1)*100:+7.0f}%")
P("  연도별: " + " | ".join(out))
for y0 in (2021, 2022, 2023, 2024, 2025):
    mm = m & np.array([idx[a].year >= y0 for a in ea])
    if mm.sum() < 40: continue
    rr, _ = curve(er[mm], ea[mm], 1.0, mae[mm])
    P(f"  {y0}년 이후 ({mm.sum():4d}건): 승률 {np.mean(er[mm]>0)*100:5.2f}%  "
      f"거래당 {er[mm].mean()*1e4:+6.1f}bp  " + ("파산" if rr is None else f"연 {rr['ann']:+8.1f}%"))

P("\n"+"="*126); P("[4] 필터 없는 것과 비교 (같은 보수적 기준)"); P("="*126)
r0, _ = curve(er, ea, 1.0, mae)
P(f"  {'구성':>14s} {'연거래':>6s} {'승률%':>7s} {'거래당bp':>9s} {'무레버 연%':>11s} {'MDD%':>8s} "
  f"{'2025~ 연%':>10s}")
for lab, mm in (("필터 없음", np.ones(len(er), bool)), ("필터 상위70%", m)):
    rr, _ = curve(er[mm], ea[mm], 1.0, mae[mm])
    m25 = mm & np.array([idx[a].year >= 2025 for a in ea])
    r25, _ = curve(er[m25], ea[m25], 1.0, mae[m25]) if m25.sum() > 40 else (None, 0)
    P(f"  {lab:>14s} {mm.sum()/Y:6.0f} {np.mean(er[mm]>0)*100:7.2f} {er[mm].mean()*1e4:+9.1f} "
      f"{(0 if rr is None else rr['ann']):11.1f} {(0 if rr is None else rr['mdd']):8.1f} "
      + ("      파산" if r25 is None else f"{r25['ann']:10.1f}"))

P("\n"+"="*126); P("[5] 플라시보 — 필터 점수를 무작위로 (200회)"); P("="*126)
rng = np.random.default_rng(555)
real = er[m].mean()*1e4
vals = []
for it in range(200):
    p = rng.permutation(len(sc))
    m2 = np.isfinite(sc[p]) & (sc[p] >= 0.30)
    if m2.sum() < 200: continue
    vals.append(er[m2].mean()*1e4)
vals = np.array(vals)
P(f"  거래당bp 실제 {real:+7.2f} | 필터셔플 평균 {vals.mean():+6.2f} 최대 {vals.max():+6.2f} "
  f"→ 백분위 {np.mean(vals < real)*100:5.1f}%")

P("\n"+"="*126); P("[6] 비용·실행 민감도 (보수적, 명목 3배)"); P("="*126)
for eps, slip, fee in ((0, 0, 4), (5, 0, 10), (5, 10, 10), (10, 10, 15), (20, 20, 20), (40, 30, 20)):
    ea2, er2, mae2, _ = dip2(O, H, Lo, C, RV, BAR, *KS, *WS, float(eps), float(slip),
                             float(fee), warm, N, 0)
    sc2 = None; n2 = 0
    for nm, sgn in (("rv분위", 1), ("프리미엄", 1), ("직전1일수익", -1)):
        q = expq(np.asarray(F[nm], float)[ea2])
        v = q if sgn > 0 else 1.0-q
        sc2 = v if sc2 is None else np.nansum(np.vstack([sc2, v]), axis=0); n2 += 1
    sc2 = sc2/n2
    m2 = np.isfinite(sc2) & (sc2 >= 0.30)
    rr, nl = curve(er2[m2], ea2[m2], 3.0, mae2[m2])
    P(f"  관통{eps:3.0f}bp 슬립{slip:3.0f}bp 수수료{fee:3.0f}bp: "
      + ("파산" if rr is None else f"연 {rr['ann']:+9.1f}%  월 {rr['mon']:+6.2f}%  MDD {rr['mdd']:6.1f}%"))
