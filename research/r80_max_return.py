"""R80 — 최대 수익 탐색: z0.31 딥(문턱·배율·명목) × R79 분산 조각(비중·레버리지) 격자.

DIP: r32 워크포워드 예측(2022~), 예측 문턱 pc ∈ {5, 10, 20bp}, 청산 지연 1.2bp, R40 엔진 sim(킬스위치 포함).
  배율 V0 고정 / V1 변동성국면^2(0.3~4) / V3 = V1 × 예측비례(0.5~3), 상한 6.  명목 (w, N) ∈ {(1,5), (2,8), (3,12), (4,16)}.
분산 조각: R79 롱+숏(하락국면 숏 포함, 매년 최근 2년 과거로 선택, K=10) 무레버 일간 수익. 비중 a ∈ {0, 0.25, 0.5}, 레버리지 Ld ∈ {1, 3, 5}.
결합: 일간 수익 = (1−a) × DIP 일간 + a × Ld × 분산 일간 (매일 재배분 근사).
지표: 월 복리, 월 중앙, 월 하위 10%, 월 ≥+100% 달 수, MDD, 최종 배수, 연도별.
선택 규칙(사전 고정): 2022~23 성과로 MDD 구간(−50/−70/−90% 이내)별 월 복리 최대 구성을 고르고 2024~26에서 확인. 전 기간 결과도 표시.
사용: python research/r80_max_return.py"""
import sys, warnings, itertools, time
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
import r40_regime_leverage as RL
import r79_longshort_book as B7
from freesia import store

IDX = pd.date_range("2022-01-01", "2026-09-25", freq="D", tz="UTC")


def dip_frame(pc):
    W = pd.read_parquet(R / "data" / "r32_wf.parquet"); W["ts"] = pd.DatetimeIndex(W.ts)
    W = W[W.pred > pc].sort_values(["f", "sym"]).reset_index(drop=True); W["ret"] = W.ret - 1.2e-4
    for s in ("BTCUSDT", "ETHUSDT"):
        k = store.load("klines_1m", s).set_index("ts").close.resample("D").last(); r = np.log(k).diff()
        v30 = r.rolling(30).std(); reg = (v30 / v30.rolling(365, min_periods=120).median()).shift(1)
        m = W.sym == s; W.loc[m, "vreg"] = reg.reindex(W.loc[m, "ts"].dt.normalize()).values
    W["vreg"] = W.vreg.fillna(1.0)
    return W


def daily_from_eq(eq):
    eq.index = pd.DatetimeIndex(eq.index); eq.index = eq.index.tz_localize("UTC") if eq.index.tz is None else eq.index.tz_convert("UTC")
    d = eq.resample("D").last().reindex(IDX).ffill().fillna(1.0)
    return d.pct_change().fillna(d.iloc[0] - 1)


def metrics(dr, a=None, b=None):
    x = dr if a is None else dr[(dr.index >= pd.Timestamp(a, tz="UTC")) & (dr.index < pd.Timestamp(b, tz="UTC"))]
    eq = (1 + x.clip(lower=-0.999)).cumprod()
    if eq.iloc[-1] <= 1e-6: return dict(mo=-100, med=np.nan, p10=np.nan, big=0, mdd=-100, fin=0, yr="파산")
    mon = eq.resample("ME").last().pct_change(); mon.iloc[0] = eq.resample("ME").last().iloc[0] - 1
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    yr = eq.resample("YE").last().pct_change(); yr.iloc[0] = eq.resample("YE").last().iloc[0] - 1
    return dict(mo=(eq.iloc[-1] ** (1 / 12 / yrs) - 1) * 100, med=mon.median() * 100, p10=mon.quantile(0.1) * 100, big=int((mon >= 1).sum()),
                mdd=(eq / eq.cummax() - 1).min() * 100, fin=eq.iloc[-1], yr=" ".join(f"{d.year%100}:{v*100:+.0f}%" for d, v in yr.items()))


if __name__ == "__main__":
    t_ = time.time()
    A = pd.read_parquet(R / "data" / "r79_trades.parquet")
    d = store.load("klines_1m", "BTCUSDT").set_index("ts").close.resample("1D").last(); reg = (np.log(d).diff(30) < 0).shift(1)
    Sx = A[A.sl.str[-1].eq("S")].copy(); bear = reg.reindex(Sx.te.dt.floor("D")).fillna(False).values.astype(bool)
    Sx = Sx[bear].assign(sl=lambda x: x.sl.str.replace("|S", "|BEAR|S", regex=False)); A2 = pd.concat([A, Sx], ignore_index=True)
    picks = {Y: B7.select(A2, Y, 10, ("L", "S"), lookback=2) for Y in range(2022, 2027)}
    T = A2[A2.sl.isin(set(sum(picks.values(), [])))].copy()
    rows = []
    for Y, pk in picks.items():
        x = T[T.sl.isin(pk) & (T.te.dt.year == Y)].copy(); x["pnl"] = x.r / len(pk) * 0.5; rows.append(x)
    X = pd.concat(rows); DIV = X.groupby(X.tx.dt.floor("D")).pnl.sum().reindex(IDX).fillna(0)
    P(f"분산 조각 일간 수익 준비: 선택 {', '.join(f'{Y}:{len(v)}' for Y, v in picks.items())} | 단독 무레버 월 {metrics(DIV)['mo']:+.2f}% MDD {metrics(DIV)['mdd']:.1f}% ({time.time()-t_:.0f}s)")
    res = []
    for pc in (0.0005, 0.001, 0.002):
        W = dip_frame(pc); RL.W = W
        V1 = np.clip(W.vreg.values ** 2.0, 0.3, 4.0); V2 = np.clip(W.pred.values / 0.001, 0.5, 3.0)
        mults = {"V0": np.ones(len(W)), "V1": V1, "V3": np.minimum(V1 * V2, 6.0)}
        for (mn, mult), (w, N) in itertools.product(mults.items(), ((1, 5), (2, 8), (3, 12), (4, 16))):
            out = RL.sim(W, w, N, mult)
            dd = daily_from_eq(out[0]) if out is not None else pd.Series(-1.0, index=IDX)
            for a, Ld in ((0, 0),) + tuple(itertools.product((0.25, 0.5), (1, 3, 5))):
                dr = (1 - a) * dd + a * Ld * DIV
                tag = f"딥 pc{pc*1e4:.0f}bp {mn} w{w}N{N} | 분산 {a:.2f}×{Ld}"
                res.append((tag, metrics(dr), metrics(dr, "2022-01-01", "2024-01-01"), metrics(dr, "2024-01-01", "2027-01-01")))
        P(f"  pc {pc*1e4:.0f}bp 완료 ({time.time()-t_:.0f}s)")
    O = pd.DataFrame([{"구성": t, **{f"전_{k}": v for k, v in m.items()}, **{f"학_{k}": v for k, v in m1.items()}, **{f"평_{k}": v for k, v in m2.items()}}
                      for t, m, m1, m2 in res])
    O.to_csv(R / "logs" / "r80.csv", index=False)
    pd.set_option("display.width", 260); pd.set_option("display.max_colwidth", 50)
    P(f"\n===== 구성 {len(O)}개")
    for lim in (-50, -70, -90):
        pick = O[O.학_mdd >= lim].sort_values("학_mo", ascending=False).head(1)
        if pick.empty: continue
        r = pick.iloc[0]
        P(f"\n  [학습(2022~23) MDD ≥ {lim}%에서 월 복리 최대] {r.구성}")
        P(f"    학습 22~23: 월 {r.학_mo:+.2f}% (중앙 {r.학_med:+.1f}%, 하위10% {r.학_p10:+.1f}%) MDD {r.학_mdd:.1f}% | 평가 24~26: 월 {r.평_mo:+.2f}% (중앙 {r.평_med:+.1f}%, 하위10% {r.평_p10:+.1f}%) MDD {r.평_mdd:.1f}% 월≥100% {r.평_big}회")
        P(f"    전 기간: 월 {r.전_mo:+.2f}% MDD {r.전_mdd:.1f}% 최종 {r.전_fin:,.0f}배 | {r.전_yr}")
    P("\n  전 기간 월 복리 상위 12 (참고, 사후 최선)")
    P(O.sort_values("전_mo", ascending=False).head(12)[["구성", "전_mo", "전_med", "전_p10", "전_big", "전_mdd", "평_mo", "평_mdd", "전_yr"]].round(2).to_string(index=False))
    P("\n  MDD 구간별 전 기간 최선 (참고)")
    for lim in (-30, -50, -70, -90):
        r = O[O.전_mdd >= lim].sort_values("전_mo", ascending=False).head(1)
        if len(r): r = r.iloc[0]; P(f"    MDD≥{lim}%: {r.구성} 월 {r.전_mo:+.2f}% (중앙 {r.전_med:+.1f}%) MDD {r.전_mdd:.1f}% | {r.전_yr}")
    (R / "logs" / "r80.summary").write_text("R80 최대 수익 탐색 완료. 세부 logs/r80.log\n")
