"""R120 — z0.33 돌파 예측 모델 v2: 돌파 거래마다 새 데이터 특징 + 거래 속성으로 기대수익을 예측해 선택·크기 조절.

배경: R113 B에서 돌파 성과를 단조롭게 가른 특징이 여럿(pnet5·pnet15·div5·share5, 평가에선 usurge·oiB60도). 지금은 pnet15 하나로만 거른다(F3).
특징(진입 분 − 1까지): R119 새 특징 24개 + 구성 번호·시작 시각(a)·진입 시각(UTC 시)·종목·변동성 국면(30일/365일, 전일까지)·최근 60·240·1440분 수익.
모델: HistGB 회귀, 목표 ret(±10% 절단), 연 단위 확장창 워크포워드(시험 연도 Y는 Y−1일까지로 학습), 2021~2026 표본 밖.
사용 방식(문턱은 학습 연도 예측 분위로 인과적 고정):
  M-F40 / M-F60: 예측 상위 60% / 40%만 거래   · M-S: 크기 × clip(1 + (예측 순위 − 0.5), 0.5, 1.5)   · 기준 = pnet15 F3(현행)
계좌(R118 엔진, 딥 = z0.31 현행): w3·N12 × L ∈ {12, 16, 20}. 선택 2022~23 → 평가 2024~26 · 2025~26.
사용: python research/r120_breakout_model_v2.py"""
import sys, warnings, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
import r118_engine as G, r119_feats as NF, r102_joint_account as J, r103_breakout_cost as K, r80_max_return as R80
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)

if __name__ == "__main__":
    t_ = time.time()
    T8 = J.add_funding(J.breakout_trades(J.TOP8)); T8["lf"] = K.last_funding(T8)
    T8 = T8[~(T8.lf > 1e-4)].reset_index(drop=True)                      # 펀딩 필터(현행과 같음)
    BK = K.events(T8, False).drop(columns="lf"); BK["side"] = 1
    ts = J.D0 + pd.to_timedelta(BK.e, unit="min"); BK["ts"] = ts; BK["yr"] = ts.dt.year.values
    X = pd.DataFrame(index=BK.index)
    X["cfg"] = T8.cfg.astype("category").cat.codes.values; X["a"] = T8.a.values; X["hr"] = ts.dt.hour.values; X["isbtc"] = (BK.s == 0).astype(float).values
    for c in NF.NEWF + ["vreg", "r60", "r240", "r1440"]: X[c] = np.nan
    for s_, sym in ((0, "BTCUSDT"), (1, "ETHUSDT")):
        m = (BK.s == s_).values
        F = NF.newfeats(sym); X.loc[m, NF.NEWF] = NF.attach(ts[m].values, F[NF.NEWF]); del F
        k = store.load("klines_1m", sym).set_index("ts").close; k.index = pd.DatetimeIndex(k.index)
        lk = np.log(k); prev = pd.DatetimeIndex(ts[m].values) - pd.Timedelta(minutes=1)
        for h in (60, 240, 1440): X.loc[m, f"r{h}"] = (lk - lk.shift(h)).reindex(prev).values
        d = k.resample("D").last(); r = np.log(d).diff(); v30 = r.rolling(30).std(); reg = (v30 / v30.rolling(365, min_periods=120).median()).shift(1)
        X.loc[m, "vreg"] = reg.reindex(prev.normalize()).values
        P(f"  {sym} 특징 결합 ({time.time()-t_:.0f}s)")
    feats = list(X.columns); y = np.clip(BK.ret.values, -0.1, 0.1)
    pred = np.full(len(BK), np.nan); qtr = {}
    for Y in range(2021, 2027):
        cut = pd.Timestamp(f"{Y}-01-01", tz="UTC")
        tr = (BK.ts < cut - pd.Timedelta(days=1)).values; te = (BK.yr == Y).values
        fu = [c for c in feats if X.loc[tr, c].notna().sum() > 500]
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=300, l2_regularization=1.0, random_state=0)
        m.fit(X.loc[tr, fu].values, y[tr]); pred[te] = m.predict(X.loc[te, fu].values)
        ptr = m.predict(X.loc[tr, fu].values[::3]); qtr[Y] = np.quantile(ptr, [0.4, 0.6])
        P(f"  {Y}: 표본 밖 IC {spearmanr(pred[te], BK.ret.values[te])[0]:+.3f} (n {te.sum():,}) ({time.time()-t_:.0f}s)")
    BK["pred"] = pred
    ok = ~np.isnan(pred)
    q = pd.Series(pred).groupby(BK.yr).rank(pct=True).values      # 정보용(같은 해 순위 — 크기 조절엔 아래 인과 순위 사용)
    # 인과 순위: 그해 이전 표본 밖 예측 분포 대비
    rk = np.full(len(BK), np.nan)
    for Y in range(2022, 2027):
        hist = pred[ok & (BK.yr.values < Y)]; mY = (BK.yr.values == Y)
        rk[mY] = np.searchsorted(np.sort(hist), pred[mY]) / max(len(hist), 1)
    P("예측 5분위(인과 순위)별 거래당 bp 2022~23 | 2024~26: " + " ".join(
        f"{BK.ret[(rk >= i/5) & (rk < (i+1)/5) & (BK.yr <= 2023)].mean()*1e4:+.0f}/{BK.ret[(rk >= i/5) & (rk < (i+1)/5) & (BK.yr >= 2024)].mean()*1e4:+.0f}" for i in range(5)))
    D = G.std_dip(); cur = G.std_breakout()
    V = {"기준 pnet15 F3": cur,
         "M-F40(상위 60%)": BK[ok & (rk >= 0.4)][G.COLS],
         "M-F60(상위 40%)": BK[ok & (rk >= 0.6)][G.COLS],
         "M-S(순위 크기)": BK[ok].assign(mult=lambda z: z.mult * np.clip(1 + (rk[ok] - 0.5), 0.5, 1.5))[G.COLS]}
    rows = []
    for lab, Bv in V.items():
        for L in (12, 16, 20):
            d, pk = G.run_multi(pd.concat([D, Bv], ignore_index=True), 3, 12, {1: L / 48})
            if d is None: rows.append(dict(돌파=lab, L=L, ok=False)); continue
            b2 = R80.metrics(d, "2025-01-01", "2026-09-23"); s2 = d["2025":"2026-09-22"]
            rows.append(dict(돌파=lab, L=L, ok=True, 거래=len(Bv), **G.summary(d), 월_25_26=b2["mo"], 샤프_25_26=s2.mean() / s2.std() * np.sqrt(365)))
        P(f"  {lab} 완료 ({time.time()-t_:.0f}s)")
    O = pd.DataFrame(rows); pd.set_option("display.width", 260)
    P("\n===== 계좌 (딥 현행 w3·N12)"); P(O.drop(columns=["연도"], errors="ignore").round(2).to_string(index=False))
    for lab, g in O[O.ok].groupby("돌파", sort=False):
        b = g.sort_values("학습월", ascending=False).iloc[0]
        P(f"  {lab} 학습 최선 L{b.L}: 24~26 월 {b.평가월:+.1f}% 샤프 {b.평가샤프:.2f} MDD {b.평가MDD:.0f}% | 25~26 월 {b.월_25_26:+.1f}% 샤프 {b.샤프_25_26:.2f} | {b.연도}")
    BK[["e", "x", "ret", "mae", "k", "mult", "s", "px", "side", "pred"]].assign(rk=rk).to_parquet(R / "data" / "r120_bk.parquet", index=False)
    P(f"완료 ({time.time()-t_:.0f}s)")
