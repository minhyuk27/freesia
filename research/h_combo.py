"""생존 후보(캐리+추세) 결합 정밀 검증: 파라미터 고원, 비용 민감도, 연도별 워크포워드, 결합 위약, 증거금 반영.
사용: python research/h_combo.py"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from freesia import store  # noqa: E402
from freesia.bt import core  # noqa: E402

TE = pd.Timestamp("2024-01-01", tz="UTC")
OUT = []
rng = np.random.default_rng(3)


def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); OUT.append(s)


def stats(x, ppy=365):
    x = np.asarray(x, float)
    if len(x) < 5 or x.std() == 0:
        return 0, 0, 0
    eq = np.cumprod(1 + x)
    return x.mean() * ppy, x.mean() / x.std() * np.sqrt(ppy), float((eq / np.maximum.accumulate(eq) - 1).min())


def trend_sleeve(sym, tf="4h", target=0.60, cost=0.0006, cap=2.0):
    b = core.bars(sym, tf); c = b.close
    ppy = {"4h": 6 * 365, "1d": 365, "1h": 24 * 365}[tf]
    sd = np.log(c).diff().rolling(90).std()
    rv = sd * np.sqrt(ppy)
    fund = core.funding_bars(sym, b.index, tf)
    fcs = [((c.ewm(span=f).mean() - c.ewm(span=s).mean()) / (c * sd)) for f, s in ((8, 32), (16, 64), (32, 128), (64, 256))]
    fcs = [(x / x.abs().rolling(500).median()).clip(-2, 2) for x in fcs]
    ens = pd.concat(fcs, axis=1).mean(axis=1)
    pos = (ens.clip(-2, 2) * (target / rv)).clip(-cap, cap).fillna(0)
    ret = c.pct_change().shift(-1).fillna(0).values
    p = pos.values; pp = np.concatenate([[0.0], p[:-1]])
    net = pp * ret - np.abs(np.diff(pp, prepend=0.0)) * cost - pp * fund
    return pd.Series(net, index=b.index).resample("1D").sum(), pd.Series(pp, index=b.index).resample("1D").mean()


def carry_sleeve(sym, cost=0.0006, margin=1.0):
    f = store.load("funding", sym).set_index("ts").rate
    pr = store.load("premium_1m", sym).set_index("ts").close
    prem = pr.reindex(f.index, method="nearest", tolerance=pd.Timedelta("5min"))
    c = (f - prem.diff()).dropna()
    c.iloc[0] -= 2 * cost                       # 최초 진입 비용(두 다리)
    return (c / margin).resample("1D").sum()


def build(cost=0.0006, target=0.60, tf="4h", margin=1.5):
    tr = {s: trend_sleeve(s, tf, target, cost)[0] for s in ("BTCUSDT", "ETHUSDT")}
    ca = {s: carry_sleeve(s, cost, margin) for s in ("BTCUSDT", "ETHUSDT")}
    idx = None
    for v in list(tr.values()) + list(ca.values()):
        idx = v.index if idx is None else idx.intersection(v.index)
    return pd.DataFrame({"추세B": tr["BTCUSDT"].reindex(idx), "추세E": tr["ETHUSDT"].reindex(idx),
                         "캐리B": ca["BTCUSDT"].reindex(idx), "캐리E": ca["ETHUSDT"].reindex(idx)}).fillna(0)


W = {"추세B": .15, "추세E": .15, "캐리B": .35, "캐리E": .35}


def mix(D, w=W):
    return (sum(D[k] * v for k, v in w.items())).values


P("=" * 78 + "\n[1] 증거금 반영 캐리 (헤지 다리 증거금 배수로 나눔)")
for m in (1.0, 1.5, 2.0):
    D = build(margin=m)
    te = D.index >= TE
    x = mix(D)
    a1, s1, d1 = stats(x[~te]); a2, s2, d2 = stats(x[te])
    c = mix(D, {"캐리B": .5, "캐리E": .5})
    ca1 = stats(c[~te])[0]; ca2 = stats(c[te])[0]
    P(f"  증거금 {m}배: 캐리 단독 연 학습 {ca1*100:5.2f}% 테스트 {ca2*100:5.2f}% | 30/70 결합 학습 연{a1*100:5.1f}% Sh {s1:4.2f} | 테스트 연{a2*100:5.1f}% Sh {s2:4.2f} MDD {d2*100:5.1f}%")

P("\n" + "=" * 78 + "\n[2] 파라미터 고원 (변동성 목표 × 봉 주기, 증거금 1.5)")
rows = []
for tf in ("4h", "1d"):
    for t in (0.30, 0.45, 0.60, 0.80, 1.00):
        D = build(target=t, tf=tf)
        te = D.index >= TE
        x = mix(D)
        a1, s1, _ = stats(x[~te]); a2, s2, d2 = stats(x[te])
        rows.append((tf, t, a1 * 100, s1, a2 * 100, s2, d2 * 100))
P(pd.DataFrame(rows, columns=["봉", "변동성목표", "학습연%", "학습Sh", "테스트연%", "테스트Sh", "테스트MDD%"]).round(2).to_string(index=False))

P("\n" + "=" * 78 + "\n[3] 비용 민감도 (편도 bp, 4h·목표0.60·증거금1.5)")
rows = []
for c in (0.0003, 0.0006, 0.0010, 0.0020):
    D = build(cost=c)
    te = D.index >= TE
    x = mix(D)
    a1, s1, _ = stats(x[~te]); a2, s2, d2 = stats(x[te])
    rows.append((c * 1e4, a1 * 100, s1, a2 * 100, s2, d2 * 100))
P(pd.DataFrame(rows, columns=["편도bp", "학습연%", "학습Sh", "테스트연%", "테스트Sh", "테스트MDD%"]).round(2).to_string(index=False))

D = build()
te = D.index >= TE
P("\n" + "=" * 78 + "\n[4] 연도별 (30/70 결합, 증거금1.5)")
x = pd.Series(mix(D), index=D.index)
rows = []
for y in range(2020, 2027):
    g = x[x.index.year == y]
    a, s, d = stats(g.values)
    rows.append((y, a * 100, s, d * 100, float((g.resample("ME").sum() > 0).mean()), float((g.resample("W").sum() > 0).mean())))
P(pd.DataFrame(rows, columns=["연", "연%", "Sharpe", "MDD%", "월승률", "주승률"]).round(3).to_string(index=False))

P("\n[5] 앵커드 워크포워드 — 매년 직전까지만 보고 비중 선택(추세/캐리 두 후보 중 Sharpe 높은 쪽 70%)")
recs = []
for y in range(2021, 2027):
    past = x.index.year < y
    t_ = mix(D, {"추세B": .5, "추세E": .5}); c_ = mix(D, {"캐리B": .5, "캐리E": .5})
    st = stats(t_[past])[1]; sc = stats(c_[past])[1]
    w = {"추세B": .35, "추세E": .35, "캐리B": .15, "캐리E": .15} if st > sc else W
    xx = mix(D, w)[x.index.year == y]
    a, s, d = stats(xx)
    recs.append((y, "추세70" if st > sc else "캐리70", a * 100, s, d * 100))
P(pd.DataFrame(recs, columns=["연", "선택", "연%", "Sharpe", "MDD%"]).round(2).to_string(index=False))

P("\n[6] 결합 위약 — 추세 다리 위상회전 300회 (캐리는 그대로), 테스트 Sharpe 분포")
tb = D["추세B"].values; tee = D["추세E"].values
base = mix(D)
pl = []
for _ in range(300):
    s1 = rng.integers(200, len(tb) - 200); s2 = rng.integers(200, len(tb) - 200)
    y = .15 * np.roll(tb, s1) + .15 * np.roll(tee, s2) + .35 * D["캐리B"].values + .35 * D["캐리E"].values
    pl.append(stats(y[te])[1])
P(f"  위약 평균 {np.mean(pl):.2f}, 95% {np.percentile(pl,95):.2f}, 99% {np.percentile(pl,99):.2f}, 최대 {np.max(pl):.2f} | 실제 {stats(base[te])[1]:.2f}")
P(f"  (캐리 단독 테스트 Sharpe {stats(mix(D, {'캐리B': .5, '캐리E': .5})[te])[1]:.2f} — 위약이 높은 이유는 캐리가 그대로 남아서)")

P("\n[7] 전체 기간 기간별 승률 (30/70)")
for rule, nm in (("YE", "연"), ("QE", "분기"), ("ME", "월"), ("W", "주"), ("D", "일")):
    g = x.resample(rule).sum()
    P(f"  {nm}: {float((g>0).mean()):.3f} (n={len(g)})")
