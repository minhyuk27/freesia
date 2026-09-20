"""z2 — 최고 지표들의 긴 창 탐색 + 결합 + 적응형 선택.
사용: python research/z2_combo.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from z2_adaptive import D, SYM, STEP, ev, TH_MKT, TH_LIM
from freesia import store
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)


def mk_sigs(s):
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
    try:
        pr = store.load("premium_1m", s).set_index("ts").close.reindex(d["idx"]).values
        for w in (1440, 4320):
            S[f"프리미엄{w}"] = -pd.Series(pr).rolling(w).mean().values[::STEP]   # 부호 반전(역방향이므로)
    except Exception:
        pass
    for w in (1440, 4320):
        uw = pd.Series(h-np.maximum(c, o)).rolling(w).sum()
        dw = pd.Series(np.minimum(c, o)-l).rolling(w).sum()
        S[f"꼬리{w}"] = ((uw-dw)/(uw+dw)).values[::STEP]                          # 부호 반전
    return S


P("\n[1] 긴 창 탐색")
BEST = {}
for s in SYM:
    P(f"\n  ══ {s} ══")
    S = mk_sigs(s)
    res = []
    for nm, v in S.items():
        b = ev(nm, np.asarray(v, float), s)
        if b: res.append((b, nm))
    res.sort(reverse=True)
    BEST[s] = (S, res)
    P(f"    → 최고: " + ", ".join(f"{n}({b:.2f}%)" for b, n in res[:5]))

P("\n[2] 결합 — 상위 지표들의 z점수 평균")
for s in SYM:
    S, res = BEST[s]
    d = D[s]; y = d["y"]
    P(f"\n  ══ {s} ══")
    for k in (2, 3, 5, 8):
        names = [n for _, n in res[:k]]
        Z = []
        for n in names:
            v = np.asarray(S[n], float)
            m = np.isfinite(v)
            z = np.full(len(v), np.nan)
            mu = np.nanmean(v[m]); sd = np.nanstd(v[m])
            if sd > 0: z[m] = (v[m]-mu)/sd
            Z.append(z)
        comb = np.nanmean(np.vstack(Z), axis=0)
        ev(f"상위{k}개 결합", comb, s)

P("\n[3] 적응형 — 최근 성과 기준으로 지표 선택 (확장창, 룩어헤드 없음)")
@numba.njit(cache=True)
def adaptive(sigs, y, lookback, warm):
    nsig, n = sigs.shape
    out = np.full(n, np.nan)
    for t in range(warm, n):
        best = -1.0; bi = -1
        lo = t-lookback
        if lo < 0: lo = 0
        for s_ in range(nsig):
            acc = 0.0; cnt = 0.0
            for j in range(lo, t):
                if y[j] == 0: continue
                v = sigs[s_, j]
                if not np.isfinite(v): continue
                pred = 1 if v > 0 else -1
                if pred == y[j]: acc += 1.0
                cnt += 1.0
            if cnt > 50:
                a = acc/cnt
                if a > best: best = a; bi = s_
        if bi >= 0: out[t] = sigs[bi, t]
    return out

for s in SYM:
    S, res = BEST[s]
    d = D[s]; y = d["y"]
    names = [n for _, n in res[:8]]
    n = min(min(len(S[nm]) for nm in names), len(y))
    M = np.vstack([np.asarray(S[nm], float)[:n] for nm in names])
    # z정규화(확장창 근사: 전체 평균은 룩어헤드이므로 부호만 사용)
    Msign = np.sign(M)
    P(f"\n  ══ {s} (후보 {len(names)}개: {', '.join(names)}) ══")
    for lb in (500, 2000, 8000):
        a = adaptive(Msign, y[:n].astype(np.int8), lb, 2000)
        ev(f"적응형 최근{lb*STEP/1440:.0f}일", a, s)
