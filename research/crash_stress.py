"""폭락 시점 특화 실행 스트레스 — 깊은 딥은 폭락 한복판에서 체결된다.
(A) 고변동 구간에서만 추가 슬리피지 (B) 진입 체결 지연 N분 (C) 청산(종가) 슬리피지
(D) 부분 체결(명목의 일부만) (E) 페어 결합 효과
사용: python research/crash_stress.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
import exit_tier as X
from absfloor import met
from exit_tier import RAW, SYM, prep, CACHE
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
BL6 = (240, 360, 480, 720, 960, 1440)
DEP = {10.0: 1.0, 14.0: 3.0, 20.0: 6.0}
FLO = {10.0: 0.05, 14.0: 0.07, 20.0: 0.10}


@numba.njit(cache=True)
def ev_stress(o, h, l, c, st, B, x, eps, fm, ft, delay, slip_out, slip_in_hi, rvb, rv_hi):
    """delay: 조건 충족 후 delay분 뒤 시장가 체결(지정가 미체결 가정).
    slip_out: 청산 추가 슬리피지(bp). slip_in_hi: 고변동 봉에서 진입 추가 슬리피지(bp)."""
    n = len(st)
    ei = np.full(n, -1, np.int64); ex = np.full(n, -1, np.int64)
    rr = np.zeros(n); mm = np.zeros(n); k = 0
    for j in range(n):
        if np.isnan(x[j]) or x[j] <= 0: continue
        s0 = st[j]; e = s0+B
        if e > len(c): break
        O = o[s0]; T = O*(1-x[j])
        f = -1; lo_ = 1e18
        for t in range(s0, e):
            if f < 0:
                if l[t] <= T*(1-eps): f = t
                else: continue
            if l[t] < lo_: lo_ = l[t]
        if f < 0: continue
        if delay > 0:
            ft_ = f + delay
            if ft_ >= e: continue
            px_in = c[ft_]            # 지연 후 시장가
            fee_in = ft
        else:
            ft_ = f; px_in = T; fee_in = fm
        extra = slip_in_hi if (rvb[j] > rv_hi) else 0.0
        px_in = px_in*(1.0 + extra/1e4)
        px_out = c[e-1]*(1.0 - slip_out/1e4)
        ei[k] = ft_; ex[k] = e-1
        rr[k] = (px_out/px_in - 1.0) - (fee_in + ft)/1e4
        mm[k] = 1.0 - lo_/px_in
        k += 1
    return ei[:k], ex[:k], rr[:k], mm[:k]


def build_s(eps=0.0002, fm=2.0, ft=5.0, delay=0, slip_out=0.0, slip_hi=0.0, hi_q=0.8, frac=1.0):
    E = []
    for s in SYM:
        d = RAW[s]
        for B in BL6:
            if (s, B) not in CACHE: CACHE[(s, B)] = prep(s, B)
            st, rv = CACHE[(s, B)]
            rvh = np.nanquantile(rv[np.isfinite(rv)], hi_q)
            for k, wgt in DEP.items():
                xx = np.maximum(rv*k, FLO[k]); xx = np.where(np.isnan(rv), np.nan, xx)
                a_, b_, r_, m_ = ev_stress(d["o"], d["h"], d["l"], d["c"], st.astype(np.int64), B,
                                           np.asarray(xx, float), eps, fm, ft, delay, slip_out,
                                           slip_hi, np.asarray(rv, float), float(rvh))
                for a, bb, r, m in zip(a_, b_, r_, m_):
                    E.append((int(a), int(bb), float(r), float(m), d["idx"][int(a)], (s, B, k),
                              float(rv[int(a)//B]), wgt*frac))
    E.sort(key=lambda z: z[0])
    return E


def tiers(E, lab):
    P(f"  {lab}")
    r = np.array([x[2] for x in E])
    P(f"     n={len(r)} 평균 {r.mean()*1e4:+6.0f}bp 승률 {(r>0).mean():.3f}")
    for w, Nm, tag in ((1.0, 1.0, "무레버"), (3.0, 5.0, "중간"), (5.0, 8.0, "공격")):
        eq, fl, liq = X.sim(E, w, Nm); m = met(eq, fl, liq)
        if m is None: P(f"     {tag:6s} 파산"); continue
        P(f"     {tag:6s} 연 {m[0]:+7.0f}% 최악 {m[2]:6.1f}% | 24~ {m[3]:+7.0f}% | 25~ {m[4]:+7.0f}%")


P("[기준] 무스트레스")
tiers(build_s(), "")

P("\n[A] 고변동(rv 상위20%) 봉에서만 진입 추가 슬리피지")
for sh in (10, 25, 50, 100, 200):
    tiers(build_s(slip_hi=sh), f"고변동 진입 +{sh}bp")

P("\n[B] 진입 체결 지연 — 지정가 미체결로 N분 뒤 시장가")
for dl in (1, 3, 5, 15, 30):
    tiers(build_s(delay=dl), f"{dl}분 지연 후 시장가")

P("\n[C] 청산(봉 종가 시장가) 슬리피지")
for so in (5, 10, 25, 50):
    tiers(build_s(slip_out=so), f"청산 -{so}bp")

P("\n[D] 복합 최악 시나리오")
for lab, kw in (("보수: 관통5bp+고변동진입25bp+청산10bp", dict(eps=0.0005, slip_hi=25, slip_out=10, fm=3.0, ft=6.0)),
                ("비관: 관통10bp+고변동진입50bp+청산25bp", dict(eps=0.001, slip_hi=50, slip_out=25, fm=4.0, ft=7.0)),
                ("최악: 관통20bp+고변동진입100bp+청산50bp", dict(eps=0.002, slip_hi=100, slip_out=50, fm=5.0, ft=8.0)),
                ("파멸: 3분지연시장가+고변동200bp+청산50bp", dict(delay=3, slip_hi=200, slip_out=50, fm=5.0, ft=8.0))):
    tiers(build_s(**kw), lab)

P("\n[E] 부분 체결 — 원하는 명목의 일부만 잡힌다면")
for fr in (1.0, 0.7, 0.5, 0.3):
    tiers(build_s(frac=fr), f"체결률 {fr*100:.0f}%")
