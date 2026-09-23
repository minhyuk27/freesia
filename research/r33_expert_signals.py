"""R33 — 조사(R31)에서 나온 전문가 신호를 독립 고빈도 전략으로 검증 (1분봉, 비용 포함).

[1] 분기점 효과 (arXiv 2607.09426): :00/:15/:30/:45 첫 1분 수익·순공격 → 이후 5·15·30분(되돌림?) 과 4·8·12시간(지속?)
[2] 일중 모멘텀 (Shen·Urquhart·Wang 2022): 미국 개장(13:30 UTC) 첫 30분 수익 → 마지막 30분(19:30~20:00)·장 마감까지
[3] OI 사분면 (실전 파생 트레이더): 15분 가격·OI 변화 부호 조합 → 다음 60분 수익
판정: 신호 5분위(경계는 2020~2022만) 별 평균 bp, 2023~2026 평가. 테이커 왕복 10bp·메이커+테이커 7bp와 비교.
사용: python research/r33_expert_signals.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R))
from freesia import store
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)


def quint(df, sig, tgt, lab):
    tr, te = df[df.ts.dt.year <= 2022], df[df.ts.dt.year >= 2023]
    qs = np.unique(np.nanquantile(tr[sig].dropna(), [0.1, 0.3, 0.5, 0.7, 0.9]))
    f = lambda d: [d[tgt][np.digitize(d[sig], qs) == i].mean() * 1e4 for i in range(len(qs) + 1)]
    a, b = f(tr), f(te)
    P(f"  {lab:40s} 학습 " + " ".join(f"{v:+6.1f}" for v in a) + " | 평가 " + " ".join(f"{v:+6.1f}" for v in b)
      + f" | 극단차 평가 {b[-1]-b[0]:+.1f}bp")


for s in ("BTCUSDT", "ETHUSDT"):
    k = store.load("klines_1m", s).set_index("ts")
    c = k.close; o = k.open; qv = k.quote_volume; tb = k.taker_buy_quote_volume
    lc = np.log(c)
    rv = lc.diff().rolling(300).std()
    P(f"\n===== {s}")
    # [1] 분기점
    m = k.index.minute
    q = pd.DataFrame({"ts": k.index, "r0": (lc - np.log(o)).values / rv.shift(1).values,
                      "imb0": ((2 * tb - qv) / qv).values}, index=k.index)
    for h in (5, 15, 30, 240, 480, 720):
        q[f"f{h}"] = (lc.shift(-h) - lc).values          # 첫 분 종가 진입 → h분 뒤
    Q = q[m % 15 == 0].dropna()
    P("[1] 분기점 첫 1분 (10분위 경계 6개 구간)")
    for h in (5, 15, 30, 240, 480, 720):
        quint(Q, "r0", f"f{h}", f"첫분 수익(rv단위) → {h}분")
    for h in (15, 240, 720):
        quint(Q, "imb0", f"f{h}", f"첫분 순공격 → {h}분")
    O = q[m % 15 == 7].dropna()
    quint(O, "r0", "f15", "대조: 7분 위상 첫분 수익 → 15분")
    # [2] 일중 모멘텀 (미국 개장)
    d = pd.DataFrame({"c": c})
    day = d.index.normalize()
    op = c[(k.index.hour == 13) & (m == 30)]; o30 = c[(k.index.hour == 14) & (m == 0)]
    l30 = c[(k.index.hour == 19) & (m == 30)]; cl = c[(k.index.hour == 20) & (m == 0)]
    D = pd.DataFrame({"a": np.log(o30.values / op.reindex(o30.index - pd.Timedelta(minutes=30)).values)},
                     index=o30.index.normalize())
    D["last"] = pd.Series(np.log(cl.values / l30.reindex(cl.index - pd.Timedelta(minutes=30)).values), index=cl.index.normalize())
    D["rest"] = pd.Series(np.log(cl.values), index=cl.index.normalize()) - pd.Series(np.log(o30.values), index=o30.index.normalize())
    D = D.dropna(); D["ts"] = D.index
    D = D[D.index.dayofweek < 5]
    P("[2] 미국 개장 첫 30분 (평일)")
    quint(D, "a", "last", "첫30분 → 마지막30분(19:30~20:00)")
    quint(D, "a", "rest", "첫30분 → 14:00~20:00 보유")
    # [3] OI 사분면
    mt = store.load("metrics_5m", s).set_index("ts")
    mt.index = mt.index + pd.Timedelta(minutes=5)
    oi = mt.sum_open_interest.reindex(k.index, method="ffill")
    G = pd.DataFrame({"ts": k.index, "dp": (lc - lc.shift(15)).values / (rv * np.sqrt(15)).values,
                      "doi": np.log(oi / oi.shift(15)).values, "f60": (lc.shift(-60) - lc).values}, index=k.index)
    G = G[m % 15 == 0].dropna()
    G = G[G.ts >= "2022-01-01"]
    P("[3] OI 사분면 → 다음 60분 (2022~, 가격변화 크기 상위 20%만)")
    big = G[G.dp.abs() > G.dp.abs().quantile(0.8)]
    for lab, mk in (("가격↓·OI↓ (롱 청산)", (big.dp < 0) & (big.doi < 0)), ("가격↓·OI↑ (신규 숏)", (big.dp < 0) & (big.doi > 0)),
                    ("가격↑·OI↓ (숏 커버)", (big.dp > 0) & (big.doi < 0)), ("가격↑·OI↑ (신규 롱)", (big.dp > 0) & (big.doi > 0))):
        g = big[mk]
        by = g.groupby(g.ts.dt.year).f60.mean() * 1e4
        P(f"  {lab:18s} n={len(g):5d} 다음60분 {g.f60.mean()*1e4:+6.2f}bp | " + " ".join(f"{y%100}{v:+5.1f}" for y, v in by.items()))
