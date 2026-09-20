"""z2 — 룩어헤드 제거판. 확장창 z정규화 + 롤링 지표선택 + 적응형.
사용: python research/z2_clean.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from z2_adaptive import D, SYM, STEP, TH_MKT, TH_LIM
from freesia import store
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)


def sigs(s):
    d = D[s]; c = d["c"]; qv = d["qv"]; tb = d["tb"]; h = d["h"]; l = d["l"]; o = d["o"]
    lr = pd.Series(np.log(c)).diff()
    S = {}
    for w in (1440, 2880, 4320, 10080, 20160, 43200):
        us = pd.Series(np.where(lr > 0, lr**2, 0)).rolling(w).sum()
        ds = pd.Series(np.where(lr < 0, lr**2, 0)).rolling(w).sum()
        S[f"반변동성{w}"] = ((us-ds)/(us+ds)).values[::STEP]
        S[f"OFI{w}"] = (2*pd.Series(tb).rolling(w).sum()/pd.Series(qv).rolling(w).sum()-1).values[::STEP]
        chg = pd.Series(c).diff(w)
        path = pd.Series(np.abs(np.diff(c, prepend=c[0]))).rolling(w).sum()
        S[f"효율비{w}"] = (chg/path.replace(0, np.nan)).values[::STEP]
    for w in (1440, 4320):
        uw = pd.Series(h-np.maximum(c, o)).rolling(w).sum()
        dw = pd.Series(np.minimum(c, o)-l).rolling(w).sum()
        S[f"꼬리{w}"] = ((uw-dw)/(uw+dw)).values[::STEP]
    return S


def exp_z(v, warm=20000, step=2000):
    """확장창 z점수 — 시점 t까지의 평균·표준편차만 사용"""
    n = len(v); z = np.full(n, np.nan)
    for i in range(warm, n, step):
        x = v[:i]; x = x[np.isfinite(x)]
        if len(x) < 1000: continue
        mu = x.mean(); sd = x.std()
        if sd > 0:
            j = min(i+step, n)
            z[i:j] = (v[i:j]-mu)/sd
    return z


def evq(sig, y, qs=(0.05, 0.10, 0.20), lab="", show=True):
    n = min(len(sig), len(y)); sig = sig[:n]; y = y[:n]
    dec = (y != 0) & np.isfinite(sig)
    if dec.sum() < 2000: return None
    q = pd.Series(np.where(dec, sig, np.nan)).rank(pct=True).values
    best = 0; row = f"    {lab:34s}"
    for qq in qs:
        hi = dec & (q >= 1-qq); lo = dec & (q <= qq)
        if hi.sum() < 100 or lo.sum() < 100: continue
        ah = (y[hi] == 1).mean()*100; al = (y[lo] == -1).mean()*100
        both = (ah*hi.sum()+al*lo.sum())/(hi.sum()+lo.sum())
        row += f" | {qq*100:.0f}% 위{ah:5.2f}/아래{al:5.2f}/합{both:5.2f}%"
        best = max(best, both)
    if show: P(row + (f"  ★{best:.2f}%" if best > TH_LIM else ""))
    return best


P("\n[1] 확장창 z정규화 결합 (룩어헤드 제거)")
STORE = {}
for s in SYM:
    S = sigs(s); d = D[s]; y = d["y"]
    Z = {nm: exp_z(np.asarray(v, float)) for nm, v in S.items()}
    STORE[s] = (S, Z)
    P(f"\n  ══ {s} ══")
    # 개별 성과(확장창 z는 순위를 안 바꾸므로 원신호와 동일) → 롤링 선택으로 결합
    names = list(S.keys())
    n = min(min(len(Z[nm]) for nm in names), len(y))
    M = np.vstack([Z[nm][:n] for nm in names])
    yv = y[:n]
    # 롤링 선택: 최근 lookback 구간 정확도 상위 k개의 z 평균
    @numba.njit(cache=True)
    def roll_comb(M, y, lookback, k, warm):
        nsig, n = M.shape
        out = np.full(n, np.nan)
        acc = np.zeros(nsig)
        for t in range(warm, n):
            lo = t-lookback
            if lo < 0: lo = 0
            for s_ in range(nsig):
                a = 0.0; c_ = 0.0
                for j in range(lo, t):
                    if y[j] == 0: continue
                    v = M[s_, j]
                    if not np.isfinite(v): continue
                    p = 1 if v > 0 else -1
                    if p == y[j]: a += 1.0
                    c_ += 1.0
                acc[s_] = a/c_ if c_ > 30 else 0.0
            # 상위 k개 평균
            idx = np.argsort(-acc)[:k]
            sm = 0.0; cn = 0.0
            for ii in idx:
                v = M[ii, t]
                if np.isfinite(v): sm += v; cn += 1.0
            if cn > 0: out[t] = sm/cn
        return out
    for lb in (2000, 8000):
        for k in (3, 5):
            a = roll_comb(M, yv.astype(np.int8), lb, k, 21000)
            evq(a, yv, lab=f"롤링선택 최근{lb*STEP/1440:.0f}일 상위{k}개")
    # 고정 결합(전 지표 z 평균) — 선택 편향 없음
    evq(np.nanmean(M, axis=0), yv, lab="전지표 z평균(선택없음)")
    # 이론 기반 고정 3개 (사전 선택, 편향 최소)
    fixed = ["반변동성4320", "OFI1440", "효율비1440"]
    Mf = np.vstack([Z[nm][:n] for nm in fixed])
    evq(np.nanmean(Mf, axis=0), yv, lab="고정3개(반변동4320+OFI1440+효율1440)")

P("\n[2] 연도별 안정성 — 고정 3개 결합")
for s in SYM:
    S, Z = STORE[s]; d = D[s]; y = d["y"]
    fixed = ["반변동성4320", "OFI1440", "효율비1440"]
    n = min(min(len(Z[nm]) for nm in fixed), len(y))
    comb = np.nanmean(np.vstack([Z[nm][:n] for nm in fixed]), axis=0)
    ts = d["idx"][::STEP][:n]
    yv = y[:n]
    df = pd.DataFrame({"sig": comb, "y": yv}, index=ts)
    P(f"  ══ {s} ══")
    for yr, g in df.groupby(df.index.year):
        gg = g[(g.y != 0) & np.isfinite(g.sig)]
        if len(gg) < 500: continue
        q = gg.sig.rank(pct=True)
        hi = gg[q >= 0.95]; lo = gg[q <= 0.05]
        if len(hi) < 20 or len(lo) < 20: continue
        ah = (hi.y == 1).mean()*100; al = (lo.y == -1).mean()*100
        both = (ah*len(hi)+al*len(lo))/(len(hi)+len(lo))
        P(f"    {yr}: n={len(gg):6d} 상하위5% 위{ah:5.2f}/아래{al:5.2f}/합{both:5.2f}% {'★' if both > TH_MKT else ''}")
