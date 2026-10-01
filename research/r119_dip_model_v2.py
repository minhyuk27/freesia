"""R119 — z0.31 딥 예측 모델 v2: R32 특징 + 새 데이터 특징 24개(현물·선물 흐름, 김치·코인베이스 프리미엄, 바이비트)로 재학습.

배경: R113~R116에서 새 특징 일부가 딥·돌파 성과를 단조롭게 갈랐지만 단일 필터로는 계좌 이득이 없었다(R116b). 예측 모델에 함께 넣어 기대수익 추정 자체를 개선한다.
켈리 성장 ∝ 샤프² — 거래당 엣지를 키우는 것이 레버리지 벽(N12) 아래에서 남은 지렛대다.
절차: R32 `walk`와 같은 확장창 연 단위 워크포워드(2022~2026 시험), 같은 모델·하이퍼파라미터, 목표 ret(±2% 절단).
  A = R32 전체 특징(재현 점검: 기존 r32_wf 예측과 상관), B = A + 새 특징.
평가: 연도별 표본 밖 IC(예측 vs ret), 예측 문턱별(10·15·25·35bp) 거래 수·거래당 bp(2022~23 / 2024~26 / 2025~26).
계좌(R118 엔진, 돌파 = z0.33 pnet15): 딥 문턱 pc ∈ {15, 25, 35bp} × w3·N12 × L ∈ {12, 16}. 선택 2022~23 → 평가 2024~26 · 2025~26.
사용: python research/r119_dip_model_v2.py"""
import sys, warnings, time, itertools
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
import r118_engine as G, r119_feats as NF, r102_joint_account as J, r80_max_return as R80
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
META = {"sym", "f", "x", "ret", "mae", "ts", "yr"}


def mdl():
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=400, l2_regularization=1.0, random_state=0)


def walk(E, feats):
    out = []
    for Y in range(2022, 2027):
        tr = E[E.yr < Y]; te = E[E.yr == Y]
        fu = [c for c in feats if tr[c].notna().sum() > 1000]
        m = mdl().fit(tr[fu].values, np.clip(tr.ret.values, -0.02, 0.02))
        out.append(te.assign(pred=m.predict(te[fu].values)))
    return pd.concat(out)


def dip_events(W, pc):
    """R80.dip_frame + R109.dip_with_px 와 같은 처리(예측 > pc, 청산 지연 1.2bp, V3 배율, 지정가 T 복원) — 입력 예측만 다르다."""
    W = W[W.pred > pc].sort_values(["f", "sym"]).reset_index(drop=True).copy(); W["ret"] = W.ret - 1.2e-4
    W["vreg"] = np.nan; px = np.full(len(W), np.nan)
    for s in ("BTCUSDT", "ETHUSDT"):
        k = store.load("klines_1m", s)
        c = k.set_index("ts").close.resample("D").last(); r = np.log(c).diff()
        v30 = r.rolling(30).std(); reg = (v30 / v30.rolling(365, min_periods=120).median()).shift(1)
        m = (W.sym == s).values; W.loc[m, "vreg"] = reg.reindex(W.loc[m, "ts"].dt.normalize()).values
        o, cl = k.open.values, k.close.values; rv1 = pd.Series(np.log(cl)).diff().rolling(300).std().shift(1).values
        f = W.f.values[m].astype(np.int64); B = W.B.values[m].astype(np.int64); s0 = (f // B) * B
        px[m] = o[s0] * (1 - W.k.values[m] * rv1[s0] * np.sqrt(B))
    W["vreg"] = W.vreg.fillna(1.0)
    mult = np.minimum(np.clip(W.vreg.values ** 1.5, 0.3, 4) * np.clip(W.pred.values / 0.001, 0.5, 3), 6)
    e = J.mins(W.ts)
    return pd.DataFrame(dict(e=e, x=e + np.maximum(W.x.values - W.f.values, 1), ret=W.ret.values, mae=W.mae.values, k=0, mult=mult,
                             s=(W.sym.values == "ETHUSDT").astype(int), px=px, side=1))


if __name__ == "__main__":
    t_ = time.time()
    E = pd.read_parquet(R / "data" / "r32_events.parquet"); E["ts"] = pd.DatetimeIndex(E.ts); E["yr"] = E.ts.dt.year
    E["isbtc"] = (E.sym == "BTCUSDT").astype(float); E = E.replace([np.inf, -np.inf], np.nan)
    ALL = [c for c in E.columns if c not in META]
    for c in NF.NEWF: E[c] = np.nan
    for sym in ("BTCUSDT", "ETHUSDT"):
        F = NF.newfeats(sym); m = (E.sym == sym).values
        E.loc[m, NF.NEWF] = NF.attach(E.ts[m].values, F[NF.NEWF]); del F
        P(f"  {sym} 새 특징 결합 ({time.time()-t_:.0f}s)")
    P(f"사건 {len(E):,}, 기존 특징 {len(ALL)}, 새 특징 {len(NF.NEWF)}, 새 특징 결측 {E[NF.NEWF].isna().mean().mean()*100:.1f}%")
    WS = {"A 기존": walk(E, ALL), "B 기존+새": walk(E, ALL + NF.NEWF)}
    old = pd.read_parquet(R / "data" / "r32_wf.parquet", columns=["sym", "f", "B", "k", "pred"])
    a = WS["A 기존"].merge(old, on=["sym", "f", "B", "k"], suffixes=("", "_old"))
    P(f"재현 점검: A 예측 vs 기존 r32_wf 예측 상관 {np.corrcoef(a.pred, a.pred_old)[0,1]:.3f} (n {len(a):,})")
    for lab, W in WS.items():
        ic = W.groupby("yr").apply(lambda g: spearmanr(g.pred, g.ret)[0])
        P(f"\n[{lab}] 연도별 표본 밖 IC " + " ".join(f"{y}:{v:+.3f}" for y, v in ic.items()))
        for th in (0.001, 0.0015, 0.0025, 0.0035):
            S = W[W.pred > th]; f = lambda z: f"{len(z):5d}건 {z.ret.mean()*1e4:+6.1f}bp"
            P(f"  문턱 {th*1e4:4.0f}bp: 22~23 {f(S[S.yr <= 2023])} | 24~26 {f(S[S.yr >= 2024])} | 25~26 {f(S[S.yr >= 2025])}")
    BK = G.std_breakout()
    rows = []
    for lab, W in WS.items():
        for pc in (0.0015, 0.0025, 0.0035):
            Dv = dip_events(W, pc)
            Ev = pd.concat([Dv[G.COLS], BK], ignore_index=True)
            for L in (12, 16):
                d, pk = G.run_multi(Ev, 3, 12, {1: L / 48})
                if d is None: rows.append(dict(모델=lab, pc=pc * 1e4, L=L, ok=False)); continue
                b2 = R80.metrics(d, "2025-01-01", "2026-09-23"); s2 = d["2025":"2026-09-22"]
                rows.append(dict(모델=lab, pc=pc * 1e4, L=L, ok=True, 딥수=len(Dv), **G.summary(d), 월_25_26=b2["mo"], 샤프_25_26=s2.mean() / s2.std() * np.sqrt(365)))
        P(f"  {lab} 계좌 완료 ({time.time()-t_:.0f}s)")
    O = pd.DataFrame(rows); pd.set_option("display.width", 260); pd.set_option("display.max_rows", 60)
    P("\n===== 계좌 (w3·N12, 돌파 pnet15)"); P(O.drop(columns=["연도"], errors="ignore").round(2).to_string(index=False))
    S = O[O.ok]
    for lab, g in S.groupby("모델"):
        b = g.sort_values("학습월", ascending=False).iloc[0]
        P(f"  {lab} 학습 최선 pc{b.pc:.0f} L{b.L}: 평가 24~26 월 {b.평가월:+.1f}% 샤프 {b.평가샤프:.2f} MDD {b.평가MDD:.0f}% | 25~26 월 {b.월_25_26:+.1f}% 샤프 {b.샤프_25_26:.2f} | {b.연도}")
    WS["B 기존+새"][["sym", "B", "k", "f", "x", "ret", "mae", "ts", "pred"]].to_parquet(R / "data" / "r119_wf.parquet", index=False)
    O.to_parquet(R / "data" / "r119.parquet", index=False)
    P(f"완료 ({time.time()-t_:.0f}s)")
