"""R41 — 볼록(convex) 돌파 피라미딩: 이기는 쪽에만 더 싣는다 (딥 매수의 정반대 손익 모양).

딥 매수 = 잦은 작은 이익 + 드문 큰 손실(꼬리). 돌파 피라미딩 = 잦은 작은 손실 + 드문 큰 이익.
두 모양을 섞으면 꼬리가 상쇄되어 레버리지 여력이 생길 수 있다(R42에서 결합).
규칙 (1분봉, 인과적, 롱·숏 대칭):
  σd = 직전 1440분 1분 수익 표준편차 × √1440 (일간 변동성)
  진입: 종가가 직전 N분 최고(롱)/최저(숏)를 넘고, 최근 60분 거래대금 ≥ v × 직전 1일 평균 → 다음 분 시가 시장가
  초기 손절: 진입가 ∓ ks·σd.  추가: 유리하게 ka·σd 움직일 때마다 1단위 추가(최대 U단위)
  추가 후 손절: 가장 최근 추가가 ∓ ks·σd 와 추적손절(최고/최저 ∓ kt·σd) 중 유리한 쪽
  시간 제한 T분. 손절은 봉 시가가 이미 넘었으면 시가 체결(갭 반영). 비용: 진입·청산 모두 테이커 5bp + 슬리피지 2bp.
출력: 거래당 '초기 위험 단위(R)' 수익 분포, 승률, 연도별, 하루 거래 수, 최대 연속 손실.
사용: python research/r41_breakout_pyramid.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R))
from freesia import store, costs
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
C1 = costs.TAKER + 2e-4


@numba.njit(cache=True)
def run(o, h, l, c, hiN, loN, volok, sd, ks, ka, kt, U, T, c1):
    n = len(c)
    ent = np.empty(n, np.int64); sid = np.empty(n, np.int64); ext = np.empty(n, np.int64)
    rR = np.empty(n); rP = np.empty(n); nu = np.empty(n, np.int64); risk = np.empty(n); k = 0
    t = 1
    while t < n - 2:
        side = 0
        if volok[t] and not np.isnan(hiN[t]) and not np.isnan(sd[t]) and sd[t] > 0:
            if c[t] > hiN[t]: side = 1
            elif c[t] < loN[t]: side = -1
        if side == 0:
            t += 1; continue
        s = sd[t]; e0 = o[t + 1]
        units = 1; avg = e0; last = e0; best = e0
        stop = e0 * (1 - side * ks * s)
        r0 = ks * s                                  # 초기 위험(가격 비율)
        j = t + 1; ex = -1; px = 0.0
        while j < n and j <= t + 1 + T:
            # 손절 점검 (갭 반영)
            if side > 0:
                if o[j] <= stop: px = o[j]; ex = j; break
                if l[j] <= stop: px = stop; ex = j; break
                if h[j] > best: best = h[j]
            else:
                if o[j] >= stop: px = o[j]; ex = j; break
                if h[j] >= stop: px = stop; ex = j; break
                if l[j] < best: best = l[j]
            # 추가 (봉 종가 기준, 다음 봉 시가 체결)
            if units < U and side * (c[j] / last - 1) >= ka * s and j + 1 < n:
                add = o[j + 1]
                avg = (avg * units + add) / (units + 1); units += 1; last = add
                ns = add * (1 - side * ks * s)
                stop = max(stop, ns) if side > 0 else min(stop, ns)
            tr = best * (1 - side * kt * s)
            if units > 1:
                stop = max(stop, tr) if side > 0 else min(stop, tr)
            j += 1
        if ex < 0:
            ex = min(j, n - 1); px = c[ex]
        g = side * (px / avg - 1) - 2 * c1            # 단위당 수익(명목 대비)
        ent[k] = t + 1; sid[k] = side; ext[k] = ex
        rP[k] = g * units                              # 1단위 명목 기준 총손익(단위 수 곱)
        rR[k] = rP[k] / (r0 + 2 * c1); nu[k] = units; risk[k] = r0 + 2 * c1
        k += 1
        t = ex + 1
    return ent[:k], sid[:k], ext[:k], rR[:k], rP[:k], nu[:k], risk[:k]


if __name__ == "__main__":
    out = []
    for s in ("BTCUSDT", "ETHUSDT"):
        k = store.load("klines_1m", s).set_index("ts")
        o, h, l, c = (k[x].values for x in ("open", "high", "low", "close"))
        qv = pd.Series(k.quote_volume.values)
        lr = pd.Series(np.log(c)).diff()
        sd = (lr.rolling(1440).std() * np.sqrt(1440)).values
        yr = k.index.year.values
        for N in (240, 1440, 4320):
            hiN = pd.Series(h).rolling(N).max().shift(1).values; loN = pd.Series(l).rolling(N).min().shift(1).values
            for v in (0.0, 2.0):
                volok = ((qv.rolling(60).mean() / qv.rolling(1440).mean()).values >= v) if v > 0 else np.ones(len(c), bool)
                for ks in (0.25, 0.5):
                    for ka in (0.25, 0.5):
                        for kt in (0.5, 1.0):
                            for U in (1, 4):
                                a, sd_, e, rR, rP, nu, rk = run(o, h, l, c, hiN, loN, volok, sd, ks, ka, kt, U, 4320, C1)
                                out.append(pd.DataFrame({"sym": s, "N": N, "v": v, "ks": ks, "ka": ka, "kt": kt, "U": U,
                                                         "side": sd_, "yr": yr[a], "ent": a, "ext": e, "R": rR, "ret": rP,
                                                         "units": nu, "risk": rk}))
        P(f"{s} 완료")
    E = pd.concat(out, ignore_index=True)
    E.to_parquet(R / "data" / "r41_events.parquet", index=False)
    YRS = 6.72
    keys = ["N", "v", "ks", "ka", "kt", "U"]
    g = E.groupby(keys)
    T = pd.DataFrame({"n/일": g.size() / YRS / 365, "평균R": g.R.mean(), "승률": g.R.apply(lambda x: (x > 0).mean()),
                      "R90": g.R.quantile(0.9), "R99": g.R.quantile(0.99), "최대R": g.R.max()})
    by = E.groupby(keys + ["yr"]).R.mean().unstack()
    T = T.join(by.round(2))
    T["양수연"] = (by > 0).sum(axis=1)
    P("\n[1] 평균 R 상위 25 (R = 초기 위험 대비 손익, 비용 포함)")
    P(T.sort_values("평균R", ascending=False).head(25).round(3).to_string())
    P("\n[2] 방향별 (상위 5 구성)")
    for idx in T.sort_values("평균R", ascending=False).head(5).index:
        sub = E.set_index(keys).loc[idx]
        P(f"  {idx}: 롱 {sub[sub.side>0].R.mean():+.3f}R ({(sub.side>0).sum()}) 숏 {sub[sub.side<0].R.mean():+.3f}R ({(sub.side<0).sum()})")
