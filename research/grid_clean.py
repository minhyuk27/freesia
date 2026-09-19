"""룩어헤드 완전 제거 상태에서 봉 안 역행 되돌림 전략 대규모 재탐색.
모든 조건 변수는 봉 b-1 종료까지 정보만 사용(shift(1) 강제).
격자: 봉 5종 × 깊이 7종 × 조건 14종 × 2방향 = 980. 위약 null-max로 판정.
사용: python research/grid_clean.py [BTCUSDT|ETHUSDT]"""
import sys, warnings, itertools
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from lib_dip import bars
TE = pd.Timestamp("2024-01-01", tz="UTC")
rng = np.random.default_rng(211)
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)


@numba.njit(cache=True)
def run2(o, h, l, c, B, xarr, cond, side, fee):
    n = len(c) // B
    ret = np.full(n, np.nan)
    for b in range(n):
        if not cond[b] or np.isnan(xarr[b]) or xarr[b] <= 0:
            continue
        s = b * B; e = s + B
        O = o[s]; x = xarr[b]
        if side == 1:
            trig = O * (1.0 - x); j = -1
            for t in range(s, e):
                if l[t] <= trig:
                    j = t; break
            if j < 0:
                continue
            ret[b] = (c[e - 1] / trig - 1.0) * 1e4 - fee
        else:
            trig = O * (1.0 + x); j = -1
            for t in range(s, e):
                if h[t] >= trig:
                    j = t; break
            if j < 0:
                continue
            ret[b] = (1.0 - c[e - 1] / trig) * 1e4 - fee
    return ret


def build(D):
    """모든 변수 shift(1) 적용 — 봉 b 시작 시점에 알 수 있는 것만"""
    s = pd.Series
    bo, bc, bh, bl, bv = D["bo"], D["bc"], D["bh"], D["bl"], D["bv"]
    bi = D["bi"]
    rng_ = s((bh - bl) / bo)
    body = s((bc - bo) / bo)
    lw = s((np.minimum(bo, bc) - bl) / bo)
    uw = s((bh - np.maximum(bo, bc)) / bo)
    lr = s(np.log(bc)).diff()
    rv24 = lr.rolling(24).std()
    rv72 = lr.rolling(72).std()
    volz = s(bv) / s(bv).rolling(168).median()
    L = lambda x: x.shift(1)                       # 지연 강제
    X = {
        "직전범위0.5": (L(rng_) * 0.5).values, "직전범위0.75": (L(rng_) * 0.75).values,
        "직전범위1.0": (L(rng_) * 1.0).values, "직전범위1.5": (L(rng_) * 1.5).values,
        "고정0.3%": np.full(D["n"], 0.003), "고정0.5%": np.full(D["n"], 0.005),
        "직전ATR3": (L(rng_.rolling(3).mean()) * 1.0).values,
    }
    rv24L = L(rv24); rv72L = L(rv72); volzL = L(volz)
    C = {
        "전체": np.ones(D["n"], bool),
        "저변동24 하위30": (rv24L < rv24L.rolling(720).quantile(.3)).fillna(False).values,
        "저변동24 하위15": (rv24L < rv24L.rolling(720).quantile(.15)).fillna(False).values,
        "고변동24 상위30": (rv24L > rv24L.rolling(720).quantile(.7)).fillna(False).values,
        "저변동72 하위30": (rv72L < rv72L.rolling(720).quantile(.3)).fillna(False).values,
        "직전봉 상승": (L(body) > 0).fillna(False).values,
        "직전봉 하락": (L(body) < 0).fillna(False).values,
        "직전 아래꼬리김": (L(lw) > L(lw).rolling(168).quantile(.7)).fillna(False).values,
        "직전 위꼬리김": (L(uw) > L(uw).rolling(168).quantile(.7)).fillna(False).values,
        "거래량 적음": (volzL < 0.8).fillna(False).values,
        "거래량 많음": (volzL > 1.3).fillna(False).values,
        "미국장 13~21": ((bi.hour >= 13) & (bi.hour < 21)),
        "아시아 0~8": ((bi.hour >= 0) & (bi.hour < 8)),
        "주중": (bi.dayofweek < 5),
    }
    return X, C


if __name__ == "__main__":
    sym = sys.argv[1] if len(sys.argv) > 1 else "ETHUSDT"
    rows = []; pool = []
    for B, bn in ((60, "1h"), (120, "2h"), (240, "4h"), (480, "8h"), (1440, "1d")):
        D = bars(sym, B)
        X, C = build(D)
        for xn, xa in X.items():
            for cn, cd in C.items():
                for side, sn in ((1, "하락후매수"), (-1, "상승후매도")):
                    ret = run2(D["o"], D["h"], D["l"], D["c"], B, np.asarray(xa, float),
                               np.asarray(cd, np.bool_), side, 7.0)
                    ok = ~np.isnan(ret); ts = D["bi"][ok]
                    if ok.sum() < 120:
                        continue
                    tr = ts < TE; te = ~tr
                    if tr.sum() < 60 or te.sum() < 40:
                        continue
                    a = ret[ok][tr]; b_ = ret[ok][te]
                    wk = te.sum() / max((ts[te][-1] - ts[te][0]).days / 7, 1)
                    rows.append((bn, xn, cn, sn, a.mean(), float((a > 0).mean()), int(tr.sum()),
                                 b_.mean(), float((b_ > 0).mean()), int(te.sum()),
                                 b_.mean() / b_.std() * np.sqrt(len(b_)) if b_.std() > 0 else 0, wk))
                    for _ in range(3):
                        cr = np.roll(cd, rng.integers(50, len(cd) - 50))
                        r2 = run2(D["o"], D["h"], D["l"], D["c"], B, np.asarray(xa, float),
                                  np.asarray(cr, np.bool_), side, 7.0)
                        m2 = ~np.isnan(r2) & (D["bi"] >= TE)
                        if m2.sum() > 40:
                            pool.append(r2[m2].mean())
    d = pd.DataFrame(rows, columns=["봉", "깊이", "조건", "방향", "학습bp", "학습승률", "학습n",
                                    "테스트bp", "테스트승률", "테스트n", "테스트t", "주당"])
    pool = np.array(pool)
    P(f"### {sym}  전략 {len(d)}종 (룩어헤드 제거)")
    P(f"위약 {len(pool)}회: 평균 {pool.mean():.2f}bp 95% {np.percentile(pool,95):.2f} 99% {np.percentile(pool,99):.2f} **최대 {pool.max():.2f}**")
    P("\n[테스트 상위 15]")
    P(d.sort_values("테스트bp", ascending=False).head(15).round(2).to_string(index=False))
    both = d[(d.학습bp > 0) & (d.테스트bp > 0)]
    P(f"\n학습·테스트 모두 양수 {len(both)}/{len(d)} (기대 25%)")
    P(f"테스트 > 위약 최대({pool.max():.2f}bp): {int((d.테스트bp > pool.max()).sum())}개")
    if len(both):
        P("\n[학습·테스트 모두 양수, 테스트 상위 10]")
        P(both.sort_values("테스트bp", ascending=False).head(10).round(2).to_string(index=False))
    d.to_csv(f"/tmp/grid_{sym}.csv", index=False)
