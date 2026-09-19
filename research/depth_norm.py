"""꼬리 위험 축소 시도 — ① x를 변동성 정규화 ② 분할 진입(피라미딩) ③ 명목 동적 조절.
목표: 명목 배수를 올릴 수 있게 MAE 최대치를 줄인다. 사용: python research/depth_norm.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from lib_dip import bars
TE = pd.Timestamp("2024-01-01", tz="UTC")
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
LEGS = [("BTCUSDT", 240), ("BTCUSDT", 480), ("ETHUSDT", 240), ("ETHUSDT", 480)]


@numba.njit(cache=True)
def runv(o, h, l, c, B, xarr, fee):
    n = len(c) // B
    ret = np.full(n, np.nan); mae = np.full(n, np.nan); xs = np.full(n, np.nan)
    for b in range(n):
        if np.isnan(xarr[b]) or xarr[b] <= 0:
            continue
        s = b * B; e = s + B
        O = o[s]; trig = O * (1.0 - xarr[b])
        j = -1
        for t in range(s, e):
            if l[t] <= trig:
                j = t; break
        if j < 0:
            continue
        lo_ = trig
        for t in range(j, e):
            if l[t] < lo_:
                lo_ = l[t]
        ret[b] = (c[e - 1] / trig - 1.0) * 1e4 - fee
        mae[b] = (1.0 - lo_ / trig) * 1e4
        xs[b] = xarr[b] * 1e4
    return ret, mae, xs


@numba.njit(cache=True)
def runpyr(o, h, l, c, B, x1, x2, x3, fee):
    """3단 분할 진입: x1<x2<x3 각 1/3씩. 평균 진입가 기준 수익·MAE"""
    n = len(c) // B
    ret = np.full(n, np.nan); mae = np.full(n, np.nan)
    for b in range(n):
        if np.isnan(x1[b]) or x1[b] <= 0:
            continue
        s = b * B; e = s + B
        O = o[s]
        t1 = O * (1 - x1[b]); t2 = O * (1 - x2[b]); t3 = O * (1 - x3[b])
        f1 = -1; f2 = -1; f3 = -1
        for t in range(s, e):
            if f1 < 0 and l[t] <= t1:
                f1 = t
            if f2 < 0 and l[t] <= t2:
                f2 = t
            if f3 < 0 and l[t] <= t3:
                f3 = t
        if f1 < 0:
            continue
        cnt = 1.0; tot = t1; first = f1
        if f2 >= 0:
            cnt += 1; tot += t2
        if f3 >= 0:
            cnt += 1; tot += t3
        avg = tot / cnt
        lo_ = t1
        for t in range(first, e):
            if l[t] < lo_:
                lo_ = l[t]
        ret[b] = (c[e - 1] / avg - 1.0) * 1e4 - fee
        mae[b] = (1.0 - lo_ / avg) * 1e4
    return ret, mae


D = {}
for (sym, B) in LEGS:
    d = bars(sym, B); s = pd.Series
    lr = s(np.log(d["bc"])).diff()
    d["atr"] = (s((d["bh"] - d["bl"]) / d["bo"]).rolling(20).mean()).shift(1).values   # 지연
    d["rv"] = (lr.rolling(20).std()).shift(1).values
    D[(sym, B)] = d


def agg(mode, k, fee=7.0):
    ss = []
    for key in LEGS:
        d = D[key]
        if mode == "fix":
            xa = np.full(d["n"], k / 1e4)
        elif mode == "atr":
            xa = d["atr"] * k
        else:
            xa = d["rv"] * k
        ret, mae, xs = runv(d["o"], d["h"], d["l"], d["c"], d["B"], np.asarray(xa, float), fee)
        ok = ~np.isnan(ret)
        ss.append(pd.DataFrame({"r": ret[ok], "m": mae[ok], "x": xs[ok]}, index=d["bi"][ok]))
    return pd.concat(ss).sort_index()


def report(A, name):
    te = A[A.index >= TE]
    rows = []
    for N in (1, 2, 3, 4, 5):
        dist = (1 / N - 0.004) * 1e4
        liq = A.m >= dist
        pnl = np.where(liq, -1.0, N * A.r / 1e4)
        eq = np.cumprod(1 + pnl)
        ruin = bool((eq <= 0.01).any())
        yrs = (A.index[-1] - A.index[0]).days / 365
        rows.append((N, int(liq.sum()), "파산" if ruin else f"{(eq[-1]**(1/yrs)-1)*100:+.0f}%",
                     float((eq / np.maximum.accumulate(eq) - 1).min()) * 100))
    P(f"  {name:22s} n={len(A):5d} x중앙 {A.x.median() if 'x' in A else np.nan:6.0f}bp MAE최대 {A.m.max():6.0f}bp | "
      f"거래당 {A.r.mean():6.1f}bp(테스트 {te.r.mean():6.1f}) | " +
      " ".join(f"{n}배:{v}({m:.0f}%)" for n, l, v, m in rows))


P("[1] 고정 x vs 변동성 정규화 x")
for k in (200, 300, 400, 500):
    report(agg("fix", k), f"고정 {k}bp")
for k in (1.0, 1.5, 2.0, 2.5, 3.0):
    A = agg("atr", k)
    if len(A) > 200:
        report(A, f"직전ATR20 × {k}")
for k in (2.0, 3.0, 4.0, 5.0):
    A = agg("rv", k)
    if len(A) > 200:
        report(A, f"실현변동성20 × {k}")

P("\n[2] 3단 분할 진입 (피라미딩) — 꼬리 위험 축소 효과")
for base, bn in ((200, "200/350/500"), (300, "300/450/600"), (150, "150/300/450")):
    ss = []
    for key in LEGS:
        d = D[key]
        x1 = np.full(d["n"], base / 1e4)
        x2 = np.full(d["n"], (base + 150) / 1e4)
        x3 = np.full(d["n"], (base + 300) / 1e4)
        ret, mae = runpyr(d["o"], d["h"], d["l"], d["c"], d["B"], x1, x2, x3, 7.0)
        ok = ~np.isnan(ret)
        ss.append(pd.DataFrame({"r": ret[ok], "m": mae[ok], "x": np.full(ok.sum(), base)}, index=d["bi"][ok]))
    report(pd.concat(ss).sort_index(), f"분할 {bn}")

P("\n[3] ATR 정규화 상세 (최적 k 탐색, 명목 1~3배)")
rows = []
for k in (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0):
    A = agg("atr", k)
    if len(A) < 200:
        continue
    te = A[A.index >= TE]
    if len(te) < 60:
        continue
    for N in (1, 2, 3):
        dist = (1 / N - 0.004) * 1e4
        liq = A.m >= dist
        pnl = np.where(liq, -1.0, N * A.r / 1e4)
        eq = np.cumprod(1 + pnl)
        if (eq <= 0.01).any():
            rows.append((k, N, len(A), A.x.median(), A.m.max(), int(liq.sum()), "파산", np.nan, np.nan)); continue
        yrs = (A.index[-1] - A.index[0]).days / 365
        eq2 = np.cumprod(1 + pnl[(A.index >= TE)]); yrs2 = (A.index[-1] - TE).days / 365
        rows.append((k, N, len(A), A.x.median(), A.m.max(), int(liq.sum()),
                     f"{(eq[-1]**(1/yrs)-1)*100:+.0f}%", float((eq / np.maximum.accumulate(eq) - 1).min()) * 100,
                     f"{(eq2[-1]**(1/yrs2)-1)*100:+.0f}%"))
P(pd.DataFrame(rows, columns=["ATR배수", "명목", "거래수", "x중앙bp", "MAE최대bp", "청산", "전체연", "전체MDD%", "테스트연"]).round(1).to_string(index=False))
