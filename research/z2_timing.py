"""z2 거래 타이밍 — 보유시간·거래간 공백·노출비율·빈도 분포.
사용: python research/z2_timing.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from z2_sim import SIG, D, SYM, STEP, MAXH, MMR
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)


@numba.njit(cache=True)
def sim_t(o, h, l, c, sig, th_hi, up, dn, maxh, step, entry_k, rv, W, eps, start):
    """진입·청산 시각과 결과를 기록(자본 무관)."""
    n = len(c)
    ei = np.zeros(400000, np.int64); ex = np.zeros(400000, np.int64)
    rr = np.zeros(400000); sg = np.zeros(400000, np.int64)
    k = 0; t = start; nskip = 0
    while t < n - maxh - W - 2:
        i = t//step
        if i >= len(sig): break
        sv = sig[i]
        if not np.isfinite(sv) or sv < th_hi: t += step; continue
        sig_t = t
        if entry_k > 0.0:
            if np.isnan(rv[t]) or rv[t] <= 0: t += step; continue
            T = c[t]*(1.0-rv[t]*entry_k)
            f = -1
            for u in range(t+1, min(t+1+W, n)):
                if l[u] <= T*(1.0-eps): f = u; break
            if f < 0: t += W; nskip += 1; continue
            px = T
        else:
            f = t+1; px = o[f]
        U = px*(1.0+up); Dn = px*(1.0-dn)
        e = -1; res = 0.0
        end = min(f+maxh, n-1)
        for j in range(f, end+1):
            if l[j] <= Dn: e = j; res = -dn; break
            if h[j] >= U: e = j; res = up; break
        if e < 0: e = end; res = c[e]/px-1.0
        ei[k] = f; ex[k] = e; rr[k] = res; sg[k] = sig_t; k += 1
        t = e+1
        if k >= 399990: break
    return ei[:k], ex[:k], rr[:k], sg[:k], nskip


for s in SYM:
    d = D[s]
    lr = pd.Series(np.log(d["c"])).diff()
    d["rv"] = lr.rolling(1440).std().shift(1).values

CFG = [
    ("지정가k0.5 상위2%",  0.02, 0.5, 0.01, 0.01),
    ("지정가k0.5 상위10%", 0.10, 0.5, 0.01, 0.01),
    ("비대칭 상위5% 2%/1%", 0.05, 0.0, 0.02, 0.01),
    ("시장가 상위2%",      0.02, 0.0, 0.01, 0.01),
    ("시장가 상위5%",      0.05, 0.0, 0.01, 0.01),
]
s = "BTCUSDT"; d = D[s]; sg_arr = SIG[s]
fin = sg_arr[np.isfinite(sg_arr)]
total_min = len(d["c"])
yrs = (d["idx"][-1]-d["idx"][0]).days/365

P(f"전체 기간 {yrs:.1f}년 = {total_min:,}분\n")
P(f"{'구성':22s} {'거래':>6s} {'연':>6s} {'월':>6s} {'주':>6s} | "
  f"{'보유중앙':>8s} {'보유평균':>8s} {'보유최대':>8s} | {'공백중앙':>8s} {'공백평균':>8s} {'공백최대':>9s} | {'노출':>6s}")
P("-"*150)
for lab, q, ek, up, dn in CFG:
    th = np.quantile(fin, 1-q)
    ei, ex, rr, sgt, nskip = sim_t(d["o"], d["h"], d["l"], d["c"], sg_arr, th, up, dn, MAXH, STEP,
                                   float(ek), np.asarray(d["rv"], float), 240, 0.0002, 300000)
    if len(ei) < 10: continue
    hold = ex - ei                      # 매수→매도 (분)
    gap = ei[1:] - ex[:-1]              # 청산 → 다음 진입 (분)
    expo = hold.sum()/total_min*100
    P(f"{lab:22s} {len(ei):6d} {len(ei)/yrs:6.0f} {len(ei)/yrs/12:6.1f} {len(ei)/yrs/52:6.1f} | "
      f"{np.median(hold):7.0f}분 {hold.mean():7.0f}분 {hold.max():7.0f}분 | "
      f"{np.median(gap):7.0f}분 {gap.mean():7.0f}분 {gap.max():8.0f}분 | {expo:5.1f}%")

P("\n" + "="*150)
P("보유·공백 분포 상세 (지정가k0.5 상위2%)")
P("="*150)
th = np.quantile(fin, 1-0.02)
ei, ex, rr, sgt, nskip = sim_t(d["o"], d["h"], d["l"], d["c"], sg_arr, th, 0.01, 0.01, MAXH, STEP,
                               0.5, np.asarray(d["rv"], float), 240, 0.0002, 300000)
hold = ex-ei; gap = ei[1:]-ex[:-1]
P(f"  거래 {len(ei)}건, 미체결 스킵 {nskip}건 (지정가가 안 닿아 진입 못한 신호)")
P(f"\n  보유시간(매수→매도) 분위:")
for p_ in (5, 10, 25, 50, 75, 90, 95, 99):
    P(f"    {p_:3d}%: {np.percentile(hold, p_):8.0f}분 ({np.percentile(hold, p_)/60:6.1f}시간)")
P(f"\n  거래 간 공백(청산→다음진입) 분위:")
for p_ in (5, 10, 25, 50, 75, 90, 95, 99):
    v = np.percentile(gap, p_)
    P(f"    {p_:3d}%: {v:8.0f}분 ({v/60:7.1f}시간 = {v/1440:5.1f}일)")
P(f"\n  공백 구간별 비중:")
for a, b, l_ in ((0, 60, "1시간 이내"), (60, 1440, "1시간~1일"), (1440, 10080, "1~7일"),
                 (10080, 43200, "7~30일"), (43200, 10**9, "30일 초과")):
    m = (gap >= a) & (gap < b)
    P(f"    {l_:12s}: {m.sum():5d}건 ({m.mean()*100:5.1f}%)")
P(f"\n  승패별 보유시간: 승 {np.median(hold[rr > 0]):.0f}분 / 패 {np.median(hold[rr < 0]):.0f}분")
P(f"  시장 노출: 전체 {hold.sum()/total_min*100:.1f}% (연평균 {hold.sum()/yrs/1440:.0f}일/년 보유)")

P("\n  연도별 거래 수:")
ts = d["idx"][ei]
df = pd.DataFrame({"y": pd.DatetimeIndex(ts).year, "hold": hold})
g = df.groupby("y")
for yr, gg in g:
    P(f"    {yr}: {len(gg):4d}건 (월 {len(gg)/12:4.1f}건) 보유중앙 {gg.hold.median():6.0f}분")

P("\n  월별 거래 분포 (거래 없는 달):")
mm = pd.Series(1, index=pd.DatetimeIndex(ts)).resample("ME").sum()
P(f"    총 {len(mm)}개월 중 거래 없는 달 {int((mm == 0).sum())}개 ({(mm == 0).mean()*100:.0f}%)")
P(f"    월 거래수 중앙 {mm.median():.0f} 평균 {mm.mean():.1f} 최대 {mm.max():.0f}")
