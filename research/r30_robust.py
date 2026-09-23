"""R30 — R27 고빈도 딥 선택의 견고성 점검.

[1] 꼬리 의존: 선택 거래(예측>10bp) 중 상위 N건 제거 후 거래당 bp와 중간 구간 계좌
[2] 모델 설정 민감도: 얕은/깊은 트리, 분류기, 선형 — 같은 워크포워드
[3] 특징 묶음 제거: 차트(가격·거래량) 특징만 / 파생 지표(OI·롱숏·펀딩·프리미엄)만 / 시간만
[4] 종목·봉 길이별 분해
[5] 순열 중요도 (평가 연도에서, 거래당 bp 기준 상위 특징)
사용: python research/r30_robust.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R / "research"))
from r28_hf_port import sim
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
E = pd.read_parquet(R / "data" / "r27_events.parquet")
E["ts"] = pd.DatetimeIndex(E.ts); E["yr"] = E.ts.dt.year
E["isbtc"] = (E.sym == "BTCUSDT").astype(float)
META = {"sym", "f", "x", "ret", "mae", "ts", "yr"}
FEAT = [c for c in E.columns if c not in META]
DERIV = ["prem", "premz", "oi60", "oi1440", "tlsr", "lsr", "fund"]
TIME = ["hour", "dow"]
BASE = ["B", "k", "isbtc"]
CHART = [c for c in FEAT if c not in DERIV + TIME + BASE]
P(f"차트 특징 {len(CHART)}: {CHART}")


def walk(feats, model="gbr", seed=0):
    out = []
    for Y in range(2022, 2027):
        tr = E[E.yr < Y]; te = E[E.yr == Y]
        y = np.clip(tr.ret.values, -0.02, 0.02)
        if model == "gbr":
            m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, max_leaf_nodes=15,
                                              min_samples_leaf=400, l2_regularization=1.0, random_state=seed)
        elif model == "shallow":
            m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.04, max_leaf_nodes=7,
                                              min_samples_leaf=1500, l2_regularization=3.0, random_state=seed)
        elif model == "deep":
            m = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.03, max_leaf_nodes=31,
                                              min_samples_leaf=200, l2_regularization=0.5, random_state=seed)
        elif model == "ridge":
            m = make_pipeline(SimpleImputer(), StandardScaler(), Ridge(alpha=10.0))
        if model == "clf":
            m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.04, max_leaf_nodes=15,
                                               min_samples_leaf=400, random_state=seed)
            m.fit(tr[feats].values, (tr.ret.values > 0).astype(int))
            pr = m.predict_proba(te[feats].values)[:, 1]
        else:
            m.fit(tr[feats].values, y); pr = m.predict(te[feats].values)
        out.append(te.assign(pred=pr))
    return pd.concat(out)


def rep(S, lab, w=0.5, Nm=3.0):
    if len(S) < 50:
        P(f"  {lab:40s} 표본부족({len(S)})"); return
    yrs = (E.ts.max() - pd.Timestamp("2022-01-01", tz="UTC")).days / 365.25
    by = S.groupby("yr").ret.mean() * 1e4
    res, liq = sim(S.sort_values(["f", "sym"]).reset_index(drop=True), w, Nm)
    acc = "파산" if res is None else f"중간 연 {(res[0].iloc[-1]**(1/yrs)-1)*100:+6.0f}% MDD {res[1]*100:5.0f}%"
    P(f"  {lab:40s} n/년 {len(S)/yrs:5.0f} {S.ret.mean()*1e4:+6.2f}bp 승률{(S.ret>0).mean():.3f} | {acc} | " +
      " ".join(f"{y%100}{v:+5.1f}" for y, v in by.items()))


W = walk(FEAT)
S = W[W.pred > 0.001]
P("\n[1] 꼬리 의존 — 예측>10bp 선택에서 수익 상위 N건 제거")
for N in (0, 20, 50, 100, 200):
    rep(S.drop(S.ret.nlargest(N).index) if N else S, f"상위 {N}건 제거")

P("\n[2] 모델 설정 민감도 (회귀는 예측>10bp, 분류는 확률>0.60/0.65)")
for md in ("gbr", "shallow", "deep", "ridge"):
    Wm = walk(FEAT, md)
    rep(Wm[Wm.pred > 0.001], f"{md} 예측>10bp")
    rep(Wm[Wm.pred > 0.0005], f"{md} 예측>5bp")
Wc = walk(FEAT, "clf")
for pth in (0.60, 0.65):
    rep(Wc[Wc.pred > pth], f"분류기 확률>{pth}")
for sd in (1, 2):
    Ws = walk(FEAT, "gbr", seed=sd)
    rep(Ws[Ws.pred > 0.001], f"gbr 시드{sd} 예측>10bp")

P("\n[3] 특징 묶음")
for lab, fs in (("차트만(+B,k,종목)", CHART + BASE), ("파생지표만(+B,k,종목)", DERIV + BASE),
                ("시간만(+B,k,종목)", TIME + BASE), ("차트+시간(파생 제외)", CHART + TIME + BASE),
                ("B,k,종목만", BASE)):
    Wg = walk(fs)
    rep(Wg[Wg.pred > 0.001], f"{lab} 예측>10bp")
    rep(Wg[Wg.pred > 0.0005], f"{lab} 예측>5bp")

P("\n[4] 분해 (전 특징, 예측>10bp)")
for s_, g in S.groupby("sym"): rep(g, f"종목 {s_}")
for b_, g in S.groupby("B"): rep(g, f"봉 {b_}분")
for k_, g in S.groupby("k"): rep(g, f"깊이 rv×{k_:.0f}")

P("\n[5] 순열 중요도 — 2024~2026 평가, 특징 하나를 섞었을 때 선택 거래당 bp 하락")
imp = []
tr = E[E.yr < 2024]; te = E[E.yr >= 2024].copy()
m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, max_leaf_nodes=15,
                                  min_samples_leaf=400, l2_regularization=1.0, random_state=0)
m.fit(tr[FEAT].values, np.clip(tr.ret.values, -0.02, 0.02))
X = te[FEAT].values.copy(); base = te.ret[m.predict(X) > 0.001].mean()
rng = np.random.default_rng(0)
for j, c in enumerate(FEAT):
    Xp = X.copy(); Xp[:, j] = rng.permutation(Xp[:, j])
    sel = m.predict(Xp) > 0.001
    imp.append(((base - te.ret[sel].mean()) * 1e4, c, sel.sum()))
imp.sort(reverse=True)
P(f"  기준 {base*1e4:+.2f}bp")
for d_, c, n_ in imp[:15]:
    P(f"  {c:9s} 하락 {d_:+6.2f}bp (선택 {n_})")
