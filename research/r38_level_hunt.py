"""R38 — 스톱 사냥 레벨 유동성 공급 (양방향).

아이디어: 스톱은 눈에 보이는 레벨 바로 너머에 쌓인다(Osler: 손절은 라운드넘버 바로 너머, ICT 스윕).
         레벨을 δ만큼 넘는 곳에 미리 지정가를 깔면, 스톱 연쇄가 만든 강제 흐름을 받게 된다.
         R32에서 가장 강한 새 특징이 '지정가가 스윙 저점 아래 어디 있나'(tpiv)였다 → 이것을 사건 생성기로 승격.
레벨 (전부 t 시작 시점에 이미 알려진 값):
  swing L : 직전 L분 최저/최고 (L = 60, 240, 1440, 10080)
  day/week: 직전 UTC 일·주의 최저/최고
  round   : BTC 1000·500 단위, ETH 100·50 단위 (현재가 바로 아래/위)
주문: 롱 = 레벨×(1−δ), 숏 = 레벨×(1+δ) 지정가. δ = rv60×m (m = 0.5, 1, 2) 또는 고정 0·10·25bp.
     2bp 관통해야 체결, 체결 분 청산 불가. H분(5·15·30·60) 뒤 종가 시장가 청산. 체결 후 H분 쿨다운.
비용: 메이커 2 + 테이커 5 + 청산 지연 1.2bp (R36).
사용: python research/r38_level_hunt.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R))
from freesia import store, costs
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
FEE = costs.MAKER + costs.TAKER + 1.2e-4
EPS = 2e-4


@numba.njit(cache=True)
def run(o, h, l, c, lv, dlt, side, H, eps, fee):
    n = len(c)
    ei = np.empty(n, np.int64); rr = np.empty(n); mm = np.empty(n); k = 0; t = 1
    while t < n - H - 1:
        L = lv[t]
        if np.isnan(L) or np.isnan(dlt[t]):
            t += 1; continue
        T = L * (1.0 - side * dlt[t])
        hit = (l[t] <= T * (1 - eps)) if side > 0 else (h[t] >= T * (1 + eps))
        # 이미 레벨 너머에서 시작한 분은 제외(지정가가 시장가가 되는 경우)
        if hit and ((side > 0 and o[t] > T) or (side < 0 and o[t] < T)):
            px = c[t + H]
            adv = l[t] if side > 0 else h[t]
            for j in range(t + 1, t + H + 1):
                if side > 0:
                    if l[j] < adv: adv = l[j]
                else:
                    if h[j] > adv: adv = h[j]
            if side > 0:
                rr[k] = px / T - 1 - fee; mm[k] = 1 - adv / T
            else:
                rr[k] = 1 - px / T - fee; mm[k] = adv / T - 1
            ei[k] = t; k += 1
            t += H + 1
            continue
        t += 1
    return ei[:k], rr[:k], mm[:k]


def levels(k, s):
    idx = k.index
    lo, hi, c = k.low, k.high, k.close
    Lv = {}
    for L in (60, 240, 1440, 10080):
        Lv[f"swing{L}"] = (lo.rolling(L).min().shift(1).values, hi.rolling(L).max().shift(1).values)
    for fq, nm in (("D", "day"), ("W", "week")):
        g = k.resample(fq).agg({"low": "min", "high": "max"})
        pl = g.low.shift(1).reindex(idx, method="ffill").values
        ph = g.high.shift(1).reindex(idx, method="ffill").values
        Lv[nm] = (pl, ph)
    steps = (1000.0, 500.0) if s == "BTCUSDT" else (100.0, 50.0)
    cp = c.shift(1).values
    for st in steps:
        Lv[f"round{int(st)}"] = (np.floor(cp / st) * st, np.ceil(cp / st) * st)
    return Lv


if __name__ == "__main__":
    res = []
    for s in ("BTCUSDT", "ETHUSDT"):
        k = store.load("klines_1m", s).set_index("ts")
        o, h, l, c = (k[x].values for x in ("open", "high", "low", "close"))
        yr = k.index.year.values
        rv60 = pd.Series(np.log(c)).diff().rolling(300).std().shift(1).values * np.sqrt(60)
        Lv = levels(k, s)
        for nm, (lo_, hi_) in Lv.items():
            for dname, dl in (("0", np.zeros(len(c))), ("10bp", np.full(len(c), 1e-3)), ("25bp", np.full(len(c), 2.5e-3)),
                              ("rv0.5", rv60 * 0.5), ("rv1", rv60), ("rv2", rv60 * 2)):
                for side, lv in ((1, lo_), (-1, hi_)):
                    for H in (5, 15, 30, 60):
                        a, r, m = run(o, h, l, c, np.asarray(lv, float), np.asarray(dl, float), side, H, EPS, FEE)
                        if len(a) == 0: continue
                        res.append(pd.DataFrame({"sym": s, "lvl": nm, "d": dname, "side": side, "H": H,
                                                 "yr": yr[a], "ret": r, "mae": m}))
        P(f"{s} 완료")
    E = pd.concat(res, ignore_index=True)
    E.to_parquet(R / "data" / "r38_events.parquet", index=False)
    YRS = 6.72
    g = E.groupby(["lvl", "d", "side", "H"])
    T = g.ret.agg(["size", "mean"]).assign(win=g.ret.apply(lambda x: (x > 0).mean()))
    by = E.groupby(["lvl", "d", "side", "H", "yr"]).ret.mean().unstack() * 1e4
    T = T.join(by)
    T["mean"] *= 1e4; T["perday"] = T["size"] / YRS / 365
    T["posyrs"] = (by > 0).sum(axis=1)
    T["min23"] = by[[2023, 2024, 2025, 2026]].min(axis=1)
    P("\n[1] 전체 상위 (거래당 bp, 표본 연 50건 이상) — 연도별 bp")
    top = T[T["size"] >= 50 * YRS].sort_values("mean", ascending=False).head(40)
    P(top.round(2).to_string())
    P("\n[2] 2023~2026 매년 양수 & 하루 1건 이상")
    rob = T[(T.min23 > 0) & (T.perday >= 1)].sort_values("mean", ascending=False).head(30)
    P(rob.round(2).to_string())
    P("\n[3] 숏(레벨 위 매도)만 상위 15")
    P(T[(T.index.get_level_values("side") == -1) & (T["size"] >= 50 * YRS)].sort_values("mean", ascending=False).head(15).round(2).to_string())
