"""z2 — 고저비대칭 중심 결합·연도별 안정성·적응형. 룩어헤드 전면 차단.
사용: python research/z2_final.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from z2_adaptive import D, SYM, STEP, TH_MKT, TH_LIM
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)


def build_sigs(s):
    d = D[s]; c = d["c"]; h = d["h"]; l = d["l"]; qv = d["qv"]; tb = d["tb"]
    lr = pd.Series(np.log(c)).diff()
    cp = pd.Series(c).shift(1)
    up_r = pd.Series(np.log(h/np.maximum(c, cp)))
    dn_r = pd.Series(np.log(np.minimum(c, cp)/l))
    S = {}
    for w in (1440, 2880, 4320, 8640, 17280):
        S[f"고저비대칭{w}"] = ((up_r.rolling(w).sum()-dn_r.rolling(w).sum()) /
                           (up_r.rolling(w).sum()+dn_r.rolling(w).sum())).values[::STEP]
        ua = pd.Series(np.where(lr > 0, np.abs(lr), 0)).rolling(w).sum()
        da = pd.Series(np.where(lr < 0, np.abs(lr), 0)).rolling(w).sum()
        S[f"절대값{w}"] = ((ua-da)/(ua+da)).values[::STEP]
        us = pd.Series(np.where(lr > 0, lr**2, 0)).rolling(w).sum()
        ds = pd.Series(np.where(lr < 0, lr**2, 0)).rolling(w).sum()
        S[f"반변동성{w}"] = ((us-ds)/(us+ds)).values[::STEP]
        S[f"OFI{w}"] = (2*pd.Series(tb).rolling(w).sum()/pd.Series(qv).rolling(w).sum()-1).values[::STEP]
        chg = pd.Series(c).diff(w)
        path = pd.Series(np.abs(np.diff(c, prepend=c[0]))).rolling(w).sum()
        S[f"효율비{w}"] = (chg/path.replace(0, np.nan)).values[::STEP]
    return S


def evq(sig, y, lab, qs=(0.05, 0.10), show=True):
    n = min(len(sig), len(y)); sig = sig[:n]; y = y[:n]
    dec = (y != 0) & np.isfinite(sig)
    if dec.sum() < 2000: return None
    q = pd.Series(np.where(dec, sig, np.nan)).rank(pct=True).values
    best = 0; row = f"    {lab:36s}"
    for qq in qs:
        hi = dec & (q >= 1-qq); lo = dec & (q <= qq)
        if hi.sum() < 100 or lo.sum() < 100: continue
        ah = (y[hi] == 1).mean()*100; al = (y[lo] == -1).mean()*100
        both = (ah*hi.sum()+al*lo.sum())/(hi.sum()+lo.sum())
        row += f" | {qq*100:.0f}% 위{ah:5.2f}/아래{al:5.2f}/합{both:5.2f}%"
        best = max(best, both)
    if show: P(row + (f"  ★{best:.2f}%" if best > TH_MKT else ""))
    return best


def exp_z(v, warm=20000, step=2000):
    n = len(v); z = np.full(n, np.nan)
    for i in range(warm, n, step):
        x = v[:i]; x = x[np.isfinite(x)]
        if len(x) < 1000: continue
        mu = x.mean(); sd = x.std()
        if sd > 0:
            j = min(i+step, n); z[i:j] = (v[i:j]-mu)/sd
    return z


P("\n[1] 고저비대칭 창 길이 정밀 탐색")
ST = {}
for s in SYM:
    S = build_sigs(s); ST[s] = S; d = D[s]; y = d["y"]
    P(f"\n  ══ {s} ══")
    res = []
    for nm, v in S.items():
        b = evq(np.asarray(v, float), y, nm, show=nm.startswith("고저"))
        if b: res.append((b, nm))
    res.sort(reverse=True)
    P(f"    → 전체 최고 6개: " + ", ".join(f"{n}({b:.2f}%)" for b, n in res[:6]))
    ST[s] = (S, res)

P("\n[2] 결합 (확장창 z, 사전 고정 조합 — 선택편향 최소)")
FIXED = {
    "고저4320+고저8640": ["고저비대칭4320", "고저비대칭8640"],
    "고저4320+절대값8640": ["고저비대칭4320", "절대값8640"],
    "고저3개(2880/4320/8640)": ["고저비대칭2880", "고저비대칭4320", "고저비대칭8640"],
    "고저4320+OFI1440+효율1440": ["고저비대칭4320", "OFI1440", "효율비1440"],
    "고저4320+절대8640+OFI1440": ["고저비대칭4320", "절대값8640", "OFI1440"],
    "전고저5개": [f"고저비대칭{w}" for w in (1440, 2880, 4320, 8640, 17280)],
}
COMB = {}
for s in SYM:
    S, res = ST[s]; d = D[s]; y = d["y"]
    P(f"\n  ══ {s} ══")
    for lab, names in FIXED.items():
        n = min(min(len(S[x]) for x in names), len(y))
        M = np.vstack([exp_z(np.asarray(S[x], float))[:n] for x in names])
        c = np.nanmean(M, axis=0)
        b = evq(c, y, lab)
        COMB[(s, lab)] = c

P("\n[3] 연도별 안정성 (상하위5%)")
for s in SYM:
    d = D[s]; y = d["y"]
    P(f"\n  ══ {s} ══")
    for lab in ("고저4320+고저8640", "고저3개(2880/4320/8640)", "고저4320+절대8640+OFI1440"):
        c = COMB[(s, lab)]
        n = min(len(c), len(y))
        ts = d["idx"][::STEP][:n]
        df = pd.DataFrame({"sig": c[:n], "y": y[:n]}, index=ts)
        row = f"    {lab:28s} "
        for yr, g in df.groupby(df.index.year):
            gg = g[(g.y != 0) & np.isfinite(g.sig)]
            if len(gg) < 3000: continue
            q = gg.sig.rank(pct=True)
            hi = gg[q >= 0.95]; lo = gg[q <= 0.05]
            if len(hi) < 50 or len(lo) < 50: continue
            ah = (hi.y == 1).mean()*100; al = (lo.y == -1).mean()*100
            both = (ah*len(hi)+al*len(lo))/(len(hi)+len(lo))
            row += f"{yr}:{both:5.1f} "
        P(row)

P("\n[4] 적응형 — 최근 성과 기준 가중 (z점수 사용, 룩어헤드 없음)")
@numba.njit(cache=True)
def adapt_w(M, y, lookback, warm):
    nsig, n = M.shape
    out = np.full(n, np.nan)
    for t in range(warm, n):
        lo = t-lookback
        if lo < 0: lo = 0
        sm = 0.0; wt = 0.0
        for s_ in range(nsig):
            a = 0.0; c_ = 0.0
            for j in range(lo, t):
                if y[j] == 0: continue
                v = M[s_, j]
                if not np.isfinite(v): continue
                p = 1 if v > 0 else -1
                if p == y[j]: a += 1.0
                c_ += 1.0
            if c_ > 50:
                acc = a/c_
                w_ = acc-0.5
                if w_ < 0: w_ = 0.0
                cur = M[s_, t]
                if np.isfinite(cur):
                    sm += w_*cur; wt += w_
        if wt > 0: out[t] = sm/wt
    return out

for s in SYM:
    S, res = ST[s]; d = D[s]; y = d["y"]
    names = [f"고저비대칭{w}" for w in (1440, 2880, 4320, 8640, 17280)] + \
            [f"절대값{w}" for w in (2880, 8640)] + ["OFI1440", "효율비1440"]
    names = [x for x in names if x in S]
    n = min(min(len(S[x]) for x in names), len(y))
    M = np.vstack([exp_z(np.asarray(S[x], float))[:n] for x in names])
    P(f"\n  ══ {s} (후보 {len(names)}개) ══")
    for lb in (2000, 8000, 20000):
        a = adapt_w(M, y[:n].astype(np.int8), lb, 21000)
        evq(a, y, f"적응형가중 최근{lb*STEP/1440:.0f}일")
