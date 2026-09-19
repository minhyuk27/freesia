"""결정적 검증: 성과가 '진입 할인 x가 크다'는 산술 효과인가, 실제 되돌림 실력인가.
① 시가→종가 vs x 직접 비교  ② x 크기를 맞춘 매칭 위약  ③ x 정규화 수익(수익/x)
사용: python research/x_matched.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from lib_dip import bars
TE = pd.Timestamp("2024-01-01", tz="UTC")
rng = np.random.default_rng(251)
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
LEGS = [("BTCUSDT", 240), ("BTCUSDT", 480), ("ETHUSDT", 240), ("ETHUSDT", 480)]


@numba.njit(cache=True)
def run3(o, h, l, c, B, xarr, cond, fee):
    n = len(c) // B
    ret = np.full(n, np.nan); oc = np.full(n, np.nan); xs = np.full(n, np.nan)
    for b in range(n):
        if not cond[b] or np.isnan(xarr[b]) or xarr[b] <= 0:
            continue
        s = b * B; e = s + B
        O = o[s]; x = xarr[b]; trig = O * (1.0 - x)
        j = -1
        for t in range(s, e):
            if l[t] <= trig:
                j = t; break
        if j < 0:
            continue
        ret[b] = (c[e - 1] / trig - 1.0) * 1e4 - fee
        oc[b] = (c[e - 1] / O - 1.0) * 1e4
        xs[b] = x * 1e4
    return ret, oc, xs


C = {}
for sym, B in LEGS:
    D = bars(sym, B); s = pd.Series
    bo, bc, bh, bl = D["bo"], D["bc"], D["bh"], D["bl"]
    rngL = s((bh - bl) / bo).shift(1)
    uwL = s((bh - np.maximum(bo, bc)) / bo).shift(1)
    qmax = pd.concat([rngL.rolling(504).rank(pct=True), uwL.rolling(504).rank(pct=True)], axis=1).max(axis=1)
    C[(sym, B)] = dict(D=D, rngL=rngL, q=qmax)

P("[1] 시가→종가 vs 진입 할인 x — 상쇄되는가 (문턱 0.9, 깊이 1.25)")
rows = []
for sym, B in LEGS:
    c = C[(sym, B)]; D = c["D"]
    cd = (c["q"] >= 0.9).fillna(False).values
    xa = (c["rngL"] * 1.25).values
    ret, oc, xs = run3(D["o"], D["h"], D["l"], D["c"], B, np.asarray(xa, float), np.asarray(cd, np.bool_), 7.0)
    ok = ~np.isnan(ret); ts = D["bi"][ok]
    for nm, m in (("학습", ts < TE), ("테스트", ts >= TE)):
        if m.sum() < 15:
            continue
        rows.append((f"{sym[:3]}{B//60}h", nm, ret[ok][m].mean(), oc[ok][m].mean(), xs[ok][m].mean(),
                     oc[ok][m].mean() + xs[ok][m].mean() - 7, int(m.sum())))
P(pd.DataFrame(rows, columns=["다리", "구간", "총bp", "시가→종가bp", "x bp", "검산(시종+x-7)", "n"]).round(2).to_string(index=False))
P("  ※ 총bp = (시가→종가) + x − 7. 핵심은 '시가→종가가 −x보다 얼마나 나은가'")
P("  ※ 시가→종가 + x > 7 이면 되돌림이 선택편향을 이긴 것")

P("\n[2] x 매칭 위약 — 같은 x 크기대의 봉 중에서 무작위 선택 (산술 효과 제거)")
for sym, B in LEGS:
    c = C[(sym, B)]; D = c["D"]
    cd = (c["q"] >= 0.9).fillna(False).values
    xa = (c["rngL"] * 1.25).values
    ret, oc, xs = run3(D["o"], D["h"], D["l"], D["c"], B, np.asarray(xa, float), np.asarray(cd, np.bool_), 7.0)
    ok = ~np.isnan(ret); te = ok & (D["bi"] >= TE)
    real = ret[te].mean()
    # x 십분위별로 실제 진입 수를 세고, 같은 십분위에서 무작위 추출
    valid = ~np.isnan(xa)
    xv = pd.Series(np.where(valid, xa, np.nan))
    dec = pd.qcut(xv.rank(method="first"), 10, labels=False)
    want = pd.Series(dec[cd]).value_counts()
    pl = []
    for _ in range(200):
        cr = np.zeros(D["n"], bool)
        for dv, cnt in want.items():
            pool = np.where((dec == dv).values & valid)[0]
            if len(pool) >= cnt:
                cr[rng.choice(pool, int(cnt), replace=False)] = True
        r2, _, _ = run3(D["o"], D["h"], D["l"], D["c"], B, np.asarray(xa, float), cr, 7.0)
        m2 = ~np.isnan(r2) & (D["bi"] >= TE)
        if m2.sum() > 15:
            pl.append(r2[m2].mean())
    pl = np.array(pl)
    P(f"  {sym[:3]}{B//60}h  실제 {real:7.2f}bp | x매칭 위약 평균 {pl.mean():7.2f} 95% {np.percentile(pl,95):7.2f} 최대 {pl.max():7.2f} → 백분위 {(pl<real).mean()*100:5.1f}%")

P("\n[3] x 정규화 수익 (총수익 ÷ x) — 크기 효과 제거한 효율")
rows = []
for sym, B in LEGS:
    c = C[(sym, B)]; D = c["D"]
    xa = (c["rngL"] * 1.25).values
    for thr, tn in ((0.0, "전체"), (0.7, "상위30%"), (0.9, "상위10%")):
        cd = (c["q"] >= thr).fillna(False).values if thr > 0 else ~np.isnan(xa)
        ret, oc, xs = run3(D["o"], D["h"], D["l"], D["c"], B, np.asarray(xa, float), np.asarray(cd, np.bool_), 7.0)
        ok = ~np.isnan(ret); te = ok & (D["bi"] >= TE)
        if te.sum() < 30:
            continue
        norm = ret[te] / xs[te]
        rows.append((f"{sym[:3]}{B//60}h", tn, ret[te].mean(), xs[te].mean(), norm.mean(),
                     norm.mean() / norm.std() * np.sqrt(len(norm)), int(te.sum())))
P(pd.DataFrame(rows, columns=["다리", "조건", "총bp", "x bp", "수익/x", "t(정규화)", "n"]).round(3).to_string(index=False))
P("  ※ 수익/x 가 조건별로 비슷하면 → 크기 효과일 뿐. 상위 조건에서 더 높으면 → 실제 실력")
