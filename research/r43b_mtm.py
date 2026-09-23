"""R43b — R43 최상 구성을 매일 시가평가(mark-to-market)로 다시 평가.

R43의 equity()는 거래 청산 시점에만 자본을 찍어서, 수개월 보유 거래(F6·F9)의 보유 중 낙폭이 빠졌다(MDD 과소).
여기서는 보유 중 매일 종가로 평가하고, 레버리지 L 에서 보유 중 1시간 최악 역행 × L ≥ 1 이면 청산.
사용: python research/r43b_mtm.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R))
from freesia import store
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
SPLIT = pd.Timestamp("2023-01-01", tz="UTC"); TE = pd.Timestamp("2026-09-23", tz="UTC")
DC = {s: store.load("klines_1m", s).set_index("ts").close.resample("1D").last() for s in ("BTCUSDT", "ETHUSDT")}
T = pd.read_parquet(R / "data" / "r43_best.parquet")


def curve(x, s, L):
    days = pd.date_range(SPLIT, TE, freq="D")
    eq = pd.Series(np.nan, index=days); cap = 0.5; dc = DC[s]
    for _, r in x.sort_values("t_in").iterrows():
        if cap <= 0: break
        p_in = dc.asof(r.t_in)                                  # 근사: 진입일 직전 종가 기준 경로
        path = dc.loc[r.t_in.normalize():r.t_out.normalize()]
        side = r.side
        for d, c in path.items():
            if d in eq.index: eq[d] = cap * max(1 + L * side * (c / p_in - 1), 0.0)
        cap = 0.0 if r.mae * L >= 1 else cap * max(1 + L * r.ret, 0.0)
        if r.t_out.normalize() in eq.index: eq[r.t_out.normalize()] = cap
    eq = eq.ffill().fillna(0.5)
    return eq


for f, g in T[T.t_in >= SPLIT].groupby("fam"):
    P(f"[{f}] {g.par.iloc[0]}")
    for L in (1, 3, 5):
        eq = curve(g[g.sym == "BTCUSDT"], "BTCUSDT", L) + curve(g[g.sym == "ETHUSDT"], "ETHUSDT", L)
        yrs = (eq.index[-1] - eq.index[0]).days / 365.25
        cg = eq.iloc[-1] ** (1 / yrs) - 1 if eq.iloc[-1] > 0 else -1
        mdd = float((eq / eq.cummax() - 1).min()) * 100
        mon = eq.resample("ME").last().pct_change().dropna() * 100
        P(f"   {L}배: 연 {cg*100:+7.1f}% 월 {((1+cg)**(1/12)-1)*100 if cg>-1 else -100:+5.1f}% 월중앙 {mon.median():+5.1f}% "
          f"월 10/90 {mon.quantile(.1):+5.1f}/{mon.quantile(.9):+5.1f} MDD(일간 시가평가) {mdd:6.1f}%")
