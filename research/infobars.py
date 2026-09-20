"""방향1 — 정보 기반 바. 시간 바 대신 달러바·거래량바·불균형바.
가설: 정보 도착률에 맞춰 표본을 뽑으면 정규성·독립성이 개선되고 예측이 쉬워진다.
사용: python research/infobars.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
from scipy import stats
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
SYM = ("BTCUSDT", "ETHUSDT")
D = {}
for s in SYM:
    k = store.load("klines_1m", s).set_index("ts")
    D[s] = dict(o=k.open.values, h=k.high.values, l=k.low.values, c=k.close.values,
                qv=k.quote_volume.values, tb=k.taker_buy_quote_volume.values,
                nt=k.trades.values, idx=k.index)
    P(f"{s}: {len(k):,}분")


@numba.njit(cache=True)
def dollar_bars(c, o, h, l, qv, tb, nt, thr):
    """누적 거래대금이 thr을 넘을 때마다 바 확정"""
    n = len(c)
    bi = np.zeros(n, np.int64); bo = np.zeros(n); bh = np.zeros(n); bl = np.zeros(n)
    bc = np.zeros(n); bv = np.zeros(n); bt = np.zeros(n); bn = np.zeros(n)
    k = 0; acc = 0.0; st = 0
    hi = -1e18; lo = 1e18
    for i in range(n):
        acc += qv[i]
        if h[i] > hi: hi = h[i]
        if l[i] < lo: lo = l[i]
        if acc >= thr:
            bi[k] = i; bo[k] = o[st]; bh[k] = hi; bl[k] = lo; bc[k] = c[i]
            bv[k] = acc
            s1 = 0.0; s2 = 0.0
            for j in range(st, i+1):
                s1 += tb[j]; s2 += nt[j]
            bt[k] = s1; bn[k] = s2
            k += 1; acc = 0.0; st = i+1; hi = -1e18; lo = 1e18
    return bi[:k], bo[:k], bh[:k], bl[:k], bc[:k], bv[:k], bt[:k], bn[:k]


@numba.njit(cache=True)
def imbalance_bars(c, o, h, l, qv, tb, thr):
    """테이커 매수-매도 불균형 누적이 thr을 넘으면 바 확정 (정보 도착 기반)"""
    n = len(c)
    bi = np.zeros(n, np.int64); bo = np.zeros(n); bh = np.zeros(n); bl = np.zeros(n)
    bc = np.zeros(n); bv = np.zeros(n)
    k = 0; acc = 0.0; st = 0; hi = -1e18; lo = 1e18; av = 0.0
    for i in range(n):
        imb = 2.0*tb[i] - qv[i]
        acc += imb; av += qv[i]
        if h[i] > hi: hi = h[i]
        if l[i] < lo: lo = l[i]
        if abs(acc) >= thr:
            bi[k] = i; bo[k] = o[st]; bh[k] = hi; bl[k] = lo; bc[k] = c[i]; bv[k] = av
            k += 1; acc = 0.0; av = 0.0; st = i+1; hi = -1e18; lo = 1e18
    return bi[:k], bo[:k], bh[:k], bl[:k], bc[:k], bv[:k]


P("\n" + "="*110)
P("[1] 통계적 성질 비교 — 시간 바 vs 달러 바 vs 불균형 바")
P("="*110)
P(f"  {'종목':6s} {'바 종류':>16s} {'바 수':>8s} {'평균간격':>9s} {'왜도':>8s} {'첨도':>8s} {'JB정규성p':>10s} {'자기상관1':>10s} {'분산비':>8s}")
BARS = {}
for s in SYM:
    d = D[s]
    tot = d["qv"].sum()
    # 시간 바 (60분)
    tb60 = d["c"][::60]
    r_t = np.diff(np.log(tb60))
    n_target = len(tb60)
    thr_d = tot/n_target
    bi, bo, bh, bl, bc, bv, btk, bn = dollar_bars(d["c"], d["o"], d["h"], d["l"], d["qv"], d["tb"], d["nt"], thr_d)
    r_d = np.diff(np.log(bc))
    # 불균형 바 — 같은 개수 목표로 임계 탐색
    imb_tot = np.abs(2*d["tb"] - d["qv"]).sum()
    thr_i = imb_tot/n_target*0.5
    ii, io, ih, il, ic, iv = imbalance_bars(d["c"], d["o"], d["h"], d["l"], d["qv"], d["tb"], thr_i)
    r_i = np.diff(np.log(ic))
    BARS[s] = dict(time=(np.arange(0, len(d["c"]), 60)[:len(tb60)], tb60),
                   dollar=(bi, bc, bo, bh, bl, bv, btk, bn),
                   imb=(ii, ic, io, ih, il, iv))
    for nm, r_, idxs in (("시간(60분)", r_t, np.arange(0, len(d["c"]), 60)[:len(tb60)]),
                         ("달러", r_d, bi), ("불균형", r_i, ii)):
        r_ = r_[np.isfinite(r_)]
        if len(r_) < 100: continue
        gap = np.mean(np.diff(idxs)) if len(idxs) > 1 else 60
        jb = stats.jarque_bera(r_[:200000])
        ac1 = pd.Series(r_).autocorr(1)
        # 분산비 (VR(2)): 1이면 랜덤워크
        v1 = np.var(r_); v2 = np.var(r_[:len(r_)//2*2].reshape(-1, 2).sum(axis=1))
        vr = v2/(2*v1)
        P(f"  {s[:3]:6s} {nm:>16s} {len(r_):8d} {gap:8.1f}분 {stats.skew(r_):+8.2f} {stats.kurtosis(r_):8.1f} "
          f"{jb.pvalue:10.2e} {ac1:+10.4f} {vr:8.3f}")

P("\n  → 첨도가 낮고 자기상관이 0에 가까울수록 통계적으로 다루기 좋다")

P("\n" + "="*110)
P("[2] 예측 가능성 비교 — 각 바에서 '직전 바 정보 → 다음 바 수익' 관계")
P("="*110)
for s in SYM:
    d = D[s]
    P(f"\n  ── {s} ──")
    for nm in ("time", "dollar", "imb"):
        if nm == "time":
            idxs, cl = BARS[s]["time"]; vol = None; tk = None
        elif nm == "dollar":
            idxs, cl, o_, h_, l_, vol, tk, nn = BARS[s]["dollar"]
        else:
            idxs, cl, o_, h_, l_, vol = BARS[s]["imb"]; tk = None
        r = np.diff(np.log(cl))
        if len(r) < 1000: continue
        # 특징: 직전 수익, 직전 변동성, (가능하면) 불균형
        x1 = r[:-1]
        x2 = pd.Series(r).rolling(20).std().values[:-1]
        y = r[1:]
        m = np.isfinite(x1) & np.isfinite(x2) & np.isfinite(y)
        c1 = np.corrcoef(x1[m], y[m])[0, 1]
        # 극단 분위 조건부
        q = pd.Series(x1).rank(pct=True).values
        lo_sel = m & (q <= 0.01); hi_sel = m & (q >= 0.99)
        P(f"    {nm:8s} n={len(r):7d} | 직전수익 자기상관 {c1:+.4f} | "
          f"직전 하위1% 이후 {y[lo_sel].mean()*1e4:+7.2f}bp (n={int(lo_sel.sum())}) | "
          f"상위1% 이후 {y[hi_sel].mean()*1e4:+7.2f}bp")

P("\n" + "="*110)
P("[3] 달러바에서 '유동성 충격 되돌림'이 더 선명한가 (z0.6 논리 재현)")
P("="*110)
for s in SYM:
    d = D[s]
    idxs, cl, o_, h_, l_, vol, tk, nn = BARS[s]["dollar"]
    r = np.diff(np.log(cl))
    P(f"  ── {s} (달러바 {len(cl):,}개, 평균 {np.mean(np.diff(idxs)):.1f}분) ──")
    for H in (1, 3, 5, 10):
        fwd = pd.Series(cl).shift(-H).values/cl - 1.0
        fwd = fwd[:-1]
        x = r
        m = np.isfinite(x) & np.isfinite(fwd)
        q = pd.Series(x).rank(pct=True).values
        for lo_q, lab in ((0.005, "하위0.5%"), (0.01, "하위1%"), (0.05, "하위5%")):
            sel = m & (q <= lo_q)
            if sel.sum() < 100: continue
            P(f"    {H:2d}바 뒤 {lab:>8s}: n={int(sel.sum()):6d} {fwd[sel].mean()*1e4:+7.2f}bp "
              f"승률 {(fwd[sel] > 0).mean():.3f}")
