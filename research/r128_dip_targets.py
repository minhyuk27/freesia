"""R128 — 꾸준함 목표 딥 모델: 학습 목표를 평균 수익 대신 '이길 확률'·'하위 20% 분위(나쁜 경우)'로 바꾸면 주간 꾸준함이 오르는가.

기준선(R125): 딥 예측(평균) > 10bp, 크기 K1 = clip((μ̂/σ̂²)/이전 중앙값, 0.25, 4), w3·N12 + 돌파 L2 — 2025~26 양수 주 70%, 주 중앙 +2.6%, 최악 −31%, 주간 샤프 0.36.
모델(R32 특징·같은 확장창 연 단위 워크포워드 2022~26): μ̂(기존 r32_wf), σ̂²(ret², R125와 같음), p̂ = P(ret > 0) 분류기, q̂ = 하위 20% 분위 회귀(손실 quantile).
변형(사전 고정, 문턱은 그해 이전 표본 밖 예측 분포의 분위 — 인과적):
  P50 / P70: μ̂ > 10bp 이고 p̂ ≥ 이전 분포 50% / 70% 분위       Q50 / Q70: μ̂ > 10bp 이고 q̂ ≥ 이전 분포 50% / 70% 분위
  PK: 크기 K1 × clip(p̂ / 이전 중앙, 0.5, 2)                  QK: 크기 K1 × clip(1 + (q̂ 순위 − 0.5), 0.5, 1.5)
  P0: μ̂ 대신 p̂만으로 선택(p̂ ≥ 이전 분포 80% 분위), 크기 K1
선택: 2022~24 주간 샤프(동률이면 양수 주). 보고: 두 구간 양수 주·주 중앙·하위 25%·최악·주간 샤프·양수 월·월 중앙.
사용: python research/r128_dip_targets.py"""
import sys, warnings, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
import r118_engine as G, r119_dip_model_v2 as V, r125_dip_variance_sizing as R125
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
META = {"sym", "f", "x", "ret", "mae", "ts", "yr"}
HP = dict(max_iter=300, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=400, l2_regularization=1.0, random_state=0)


def prior_q(vals, yr, q):
    """그해 이전 연도 값들의 분위(첫 해는 그해 앞 200개)."""
    out = np.full(len(vals), np.nan)
    for Y in np.unique(yr):
        h = vals[yr < Y]; h = h[np.isfinite(h)]
        if len(h) < 200: h = vals[yr == Y][:200]
        out[yr == Y] = np.nanquantile(h, q)
    return out


if __name__ == "__main__":
    t_ = time.time()
    E = pd.read_parquet(R / "data" / "r32_events.parquet"); E["ts"] = pd.DatetimeIndex(E.ts); E["yr"] = E.ts.dt.year
    E["isbtc"] = (E.sym == "BTCUSDT").astype(float); E = E.replace([np.inf, -np.inf], np.nan)
    ALL = [c for c in E.columns if c not in META]
    out = {"s2": [], "p": [], "q": []}
    for Y in range(2022, 2027):
        tr = E[E.yr < Y]; te = E[E.yr == Y]; fu = [c for c in ALL if tr[c].notna().sum() > 1000]
        Xtr, Xte, y = tr[fu].values, te[fu].values, np.clip(tr.ret.values, -0.02, 0.02)
        out["s2"].append(pd.Series(np.maximum(HistGradientBoostingRegressor(**HP).fit(Xtr, y ** 2).predict(Xte), 1e-8), index=te.index))
        out["p"].append(pd.Series(HistGradientBoostingClassifier(**HP).fit(Xtr, (tr.ret.values > 0).astype(int)).predict_proba(Xte)[:, 1], index=te.index))
        out["q"].append(pd.Series(HistGradientBoostingRegressor(loss="quantile", quantile=0.2, **HP).fit(Xtr, y).predict(Xte), index=te.index))
        P(f"  모델 {Y} ({time.time()-t_:.0f}s)")
    for k_, v in out.items(): E[k_] = pd.concat(v)
    W = pd.read_parquet(R / "data" / "r32_wf.parquet"); W["ts"] = pd.DatetimeIndex(W.ts)
    W = W.merge(E[["sym", "B", "k", "f", "s2", "p", "q"]], on=["sym", "B", "k", "f"], how="left")
    from scipy.stats import spearmanr
    for c in ("pred", "p", "q"):
        P(f"  {c}: 표본 밖 순위 상관 ret {spearmanr(W[c], W.ret)[0]:+.3f} · ret>0 {spearmanr(W[c], (W.ret > 0))[0]:+.3f}")
    yrW = W.ts.dt.year.values
    thr = {f"p{int(q*100)}": prior_q(W.p.values, yrW, q) for q in (0.5, 0.7, 0.8)}
    thr.update({f"q{int(q*100)}": prior_q(W.q.values, yrW, q) for q in (0.5, 0.7)})
    W = W.assign(**{f"t_{k}": v for k, v in thr.items()})
    pmed = prior_q(W.p.values, yrW, 0.5)
    qrank = np.full(len(W), np.nan)
    for Y in np.unique(yrW):
        h = np.sort(W.q.values[(yrW < Y) & np.isfinite(W.q.values)]); m = yrW == Y
        if len(h) < 200: h = np.sort(W.q.values[m][:200])
        qrank[m] = np.searchsorted(h, W.q.values[m]) / len(h)
    W["pmed"] = pmed; W["qrank"] = qrank
    B = G.std_breakout()

    def build(Wsel, extra=None):
        """선택된 딥 → 엔진 사건, 크기 K1(× 추가 배수)."""
        D = V.dip_events(Wsel, -1.0)                                  # 이미 고른 행 그대로(pc = −1 → 전부)
        Ws = Wsel.sort_values(["f", "sym"]).reset_index(drop=True)
        yr = Ws.ts.dt.year.values; kel = Ws.pred.values / Ws.s2.values
        med = {Y: np.nanmedian((W.pred / W.s2).values[(yrW < Y) & (W.pred.values > 0.001)]) if (yrW < Y).any() else np.nanmedian(kel[yr == Y][:200]) for Y in np.unique(yr)}
        mult = np.clip(kel / pd.Series(yr).map(med).values, 0.25, 4)
        if extra is not None: mult = mult * extra(Ws)
        return D.assign(mult=np.nan_to_num(mult, nan=1.0))[G.COLS]
    base = W[W.pred > 0.001]
    VAR = {"기준선 K1": (base, None),
           "P50": (base[base.p >= base.t_p50], None), "P70": (base[base.p >= base.t_p70], None),
           "Q50": (base[base.q >= base.t_q50], None), "Q70": (base[base.q >= base.t_q70], None),
           "PK": (base, lambda Ws: np.clip(Ws.p.values / Ws.pmed.values, 0.5, 2)),
           "QK": (base, lambda Ws: np.clip(1 + (Ws.qrank.values - 0.5), 0.5, 1.5)),
           "P0(p만)": (W[(W.p >= W.t_p80) & (W.pred > 0)], None)}
    rows = []
    for nm, (Wsel, ex) in VAR.items():
        Dv = build(Wsel, ex)
        for w_, N_ in ((2, 8), (3, 12)):
            d, pk = G.run_multi(pd.concat([Dv, B], ignore_index=True), w_, N_, {1: 2 / 48})
            if d is None: rows.append(dict(변형=nm, 딥=len(Dv), w=w_, N=N_, 파산=True)); continue
            st, _ = R125.wstats(d); rows.append(dict(변형=nm, 딥=len(Dv), w=w_, N=N_, **st))
        P(f"  {nm} 완료 ({time.time()-t_:.0f}s)")
    O = pd.DataFrame(rows); pd.set_option("display.width", 270)
    P(O.round(2).to_string(index=False))
    S = O.dropna(subset=["주샤프_22_24"]); b = S.sort_values(["주샤프_22_24", "양수주_22_24"], ascending=False).iloc[0]
    P(f"\n2022~24 선택: {b.변형} w{b.w}N{b.N} → 2025~26 양수주 {b.양수주_25_26:.0f}% 중앙 {b.중앙_25_26:+.2f}% 하위25 {b.하위25_25_26:+.2f}% 최악 {b.최악_25_26:+.1f}% "
      f"주샤프 {b.주샤프_25_26:.2f} 양수월 {b.양수월_25_26:.0f}% 월중앙 {b.월중앙_25_26:+.1f}%")
    O.to_parquet(R / "data" / "r128.parquet", index=False)
    P(f"완료 ({time.time()-t_:.0f}s)")
