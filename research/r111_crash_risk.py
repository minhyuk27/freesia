"""R111 — 급락 지속 위험 모델: "이 딥 뒤에 급락이 계속되는가"를 예측해 위험한 딥만 거르면 딥 총명목 한도(12배 벽, R109)를 풀 수 있는가.

배경: 성장 천장(월 +30%)은 급락 속 딥 누적 파산 때문(R107~R109). 가격 기반 조절기(R108)는 실패 → 딥 사건 시점의 미시구조·파생 특징으로 판단.
라벨: 딥 진입가 T(r26 지정가 복원) 기준, 진입 분부터 H분 안 최저가 ≤ T × (1 − D). 주 라벨 H = 240, D = 3% (보조 H 60·D 2%).
특징: r32_events의 진입 직전(f−1) 특징 50여 개(수익·변동성·체결 흐름·OI·롱숏비·펀딩·프리미엄·VPIN·호가 기울기 등) + 종목.
모델: HistGB, 연 단위 워크포워드(시험 연도 Y는 Y 이전 사건으로 학습, 경계 1일 제외), 2021~2026 예측(전부 표본 밖).
평가: 연도별 AUC, 위험 상위 10% 적중률 vs 기저율.
적용: z0.31 선택 딥(예측 > 25bp)에서 위험 p가 문턱(직전 연도까지의 표본 밖 p 분위수 q) 초과면 건너뜀. q ∈ {없음, 0.95, 0.9, 0.8, 0.7}.
      R109 정확 시가평가 엔진(sim4), 딥 w ∈ {3, 4}, N ∈ {12, 16, 20, 24}, 돌파 L ∈ {4, 8}. 선택 2022~23 월 복리 → 평가 2024~26.
사용: python research/r111_crash_risk.py"""
import sys, warnings, time, itertools
from pathlib import Path
import numpy as np, pandas as pd, numba
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
import r102_joint_account as J, r103_breakout_cost as K, r80_max_return as R80, r107_max_growth as G7, r109_dip_exact_mtm as M
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
DROP = {"sym", "f", "x", "ret", "mae", "ts", "yr", "pred", "isbtc"}


@numba.njit(cache=True)
def fwd_min(l, f, H):
    out = np.empty(len(f))
    for i in range(len(f)):
        a = f[i]; b = min(a + H, len(l)); m = l[a]
        for j in range(a, b):
            if l[j] < m: m = l[j]
        out[i] = m
    return out


def labels(E):
    T = np.full(len(E), np.nan); m60 = np.full(len(E), np.nan); m240 = np.full(len(E), np.nan)
    for sym in ("BTCUSDT", "ETHUSDT"):
        k = store.load("klines_1m", sym); o, l, c = k.open.values, k.low.values, k.close.values
        rv1 = pd.Series(np.log(c)).diff().rolling(300).std().shift(1).values
        m = (E.sym == sym).values; f = E.f.values[m].astype(np.int64); B = E.B.values[m].astype(np.int64)
        s0 = (f // B) * B; T[m] = o[s0] * (1 - E.k.values[m] * rv1[s0] * np.sqrt(B))
        m60[m] = fwd_min(l, f, 60); m240[m] = fwd_min(l, f, 240)
    return T, 1 - m60 / T, 1 - m240 / T


if __name__ == "__main__":
    t_ = time.time()
    E = pd.read_parquet(R / "data" / "r32_events.parquet"); E["ts"] = pd.DatetimeIndex(E.ts)
    E = E.sort_values("ts").reset_index(drop=True)
    T, dd60, dd240 = labels(E)
    E["T"] = T; E["y"] = (dd240 >= 0.03).astype(float); E["y2"] = (dd60 >= 0.02).astype(float)
    E["isbtc"] = (E.sym == "BTCUSDT").astype(float)
    feats = [c for c in E.columns if c not in DROP | {"T", "y", "y2"} and E[c].dtype != object] + ["isbtc"]
    P(f"사건 {len(E):,} · 특징 {len(feats)} · 기저율 240분 −3%: {E.y.mean()*100:.2f}% · 60분 −2%: {E.y2.mean()*100:.2f}% ({time.time()-t_:.0f}s)")
    P("연도별 기저율(240분 −3%): " + " ".join(f"{y}:{v*100:.1f}%" for y, v in E.groupby(E.ts.dt.year).y.mean().items()))
    X = E[feats].astype("float32").values
    for lab in ("y", "y2"):
        E["p_" + lab] = np.nan
        for Y in range(2021, 2027):
            cut = pd.Timestamp(f"{Y}-01-01", tz="UTC")
            tr = (E.ts < cut - pd.Timedelta(days=1)).values; te = ((E.ts >= cut) & (E.ts < pd.Timestamp(f"{Y+1}-01-01", tz="UTC"))).values
            if te.sum() == 0: continue
            mdl = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
                                                 l2_regularization=1.0, random_state=0).fit(X[tr], E[lab].values[tr])
            E.loc[te, "p_" + lab] = mdl.predict_proba(X[te])[:, 1]
        g = E[E["p_" + lab].notna()]
        rows = []
        for Y, h in g.groupby(g.ts.dt.year):
            top = h[h["p_" + lab] >= h["p_" + lab].quantile(0.9)]
            rows.append(f"{Y}: AUC {roc_auc_score(h[lab], h['p_' + lab]):.3f} 상위10% {top[lab].mean()*100:.1f}% vs {h[lab].mean()*100:.1f}%")
        P(f"[{lab}] " + " | ".join(rows))
    # 특징 중요도(마지막 모델, 순열 대신 분할 기여 근사 — 상위 10개 단변량 AUC로 대신 보고)
    g = E[E.p_y.notna()]
    au = {c: roc_auc_score(g.y, g[c].fillna(g[c].median())) for c in feats if g[c].notna().mean() > 0.5}
    P("단변량 AUC(|0.5 차| 상위 10): " + ", ".join(f"{c} {v:.3f}" for c, v in sorted(au.items(), key=lambda kv: -abs(kv[1] - 0.5))[:10]))
    # z0.31 선택 딥에 붙이기
    D, _, _, W = M.dip_with_px()
    key = lambda df: df.sym.astype(str) + "|" + df.B.astype(int).astype(str) + "|" + df.k.astype(float).round(2).astype(str) + "|" + df.f.astype(np.int64).astype(str)
    pm = pd.Series(E.p_y.values, index=key(E)); pm = pm[~pm.index.duplicated()]
    D["p"] = pm.reindex(key(W)).values; D["yr"] = pd.DatetimeIndex(W.ts).year.values
    P(f"선택 딥 {len(D):,} 중 위험 p 결측 {D.p.isna().mean()*100:.1f}%")
    # 문턱: 직전 연도까지의 표본 밖 p 분위수
    Pall = E[E.p_y.notna()][["ts", "p_y"]]
    def thr(Y, q):
        h = Pall[Pall.ts.dt.year < Y].p_y
        return h.quantile(q) if len(h) else np.inf
    T8 = J.add_funding(J.breakout_trades(J.TOP8)); T8["lf"] = K.last_funding(T8)
    BK = K.events(T8, False, fmax=1e-4).drop(columns="lf")
    start = int(J.mins([pd.Timestamp("2022-01-01", tz="UTC")])[0]); Gd = G7.grid(start, "mark_1m")
    rows = []
    for q in (None, 0.95, 0.9, 0.8, 0.7):
        if q is None: Dk = D
        else:
            th = D.yr.map({Y: thr(Y, q) for Y in D.yr.unique()}); Dk = D[~(D.p > th)]
        skip = 1 - len(Dk) / len(D)
        Ek = pd.concat([Dk.drop(columns=["p", "yr"]), BK[BK.e >= start]], ignore_index=True).sort_values(["e", "k"]).reset_index(drop=True)
        for w, N, L in itertools.product((3, 4), (12, 16, 20, 24), (4, 8)):
            d, pk = M.run(Ek, Gd, w, N, L)
            if d is None:
                rows.append(dict(q=q, 생략=skip, w=w, N=N, L=L, ok=False, 파산=f"{J.D0 + pd.Timedelta(minutes=int(pk[0])):%Y-%m-%d}")); continue
            a = R80.metrics(d, "2022-01-01", "2024-01-01"); b = R80.metrics(d, "2024-01-01", "2026-09-23"); s = d["2024":"2026-09-22"]
            rows.append(dict(q=q, 생략=skip, w=w, N=N, L=L, ok=True, 학습월=a["mo"], 학습MDD=a["mdd"], 평가월=b["mo"], 평가중앙=b["med"], 평가100달=b["big"],
                             평가MDD=b["mdd"], 평가샤프=s.mean() / s.std() * np.sqrt(365), 연도=b["yr"]))
        P(f"  문턱 q={q} (딥 {skip*100:.0f}% 생략) 완료 ({time.time()-t_:.0f}s)")
    O = pd.DataFrame(rows); pd.set_option("display.width", 250); pd.set_option("display.max_rows", 100)
    S = O[O.ok]
    P(f"\n생존 {len(S)}/{len(O)} · 문턱별 생존: " + " ".join(f"{q}:{int(g.ok.sum())}/{len(g)}" for q, g in O.groupby(O.q.astype(str))))
    P("파산 날짜: " + ", ".join(f"{k} {v}" for k, v in O[~O.ok].파산.value_counts().head(6).items()))
    P("\n===== 2022~23 월 복리 상위 10 → 평가"); P(S.sort_values("학습월", ascending=False).head(10).drop(columns=["ok", "파산"], errors="ignore").round(1).to_string(index=False))
    P("\n===== 문턱별 N 최대 생존 구성(학습 최선)")
    for q, g in S.groupby(S.q.astype(str)):
        b = g.sort_values("학습월", ascending=False).iloc[0]
        P(f"  q={q}: w{b.w} N{b.N} L{b.L} 학습 {b.학습월:+.1f}% → 평가 월 {b.평가월:+.1f}% 중앙 {b.평가중앙:+.1f}% ≥100% {b.평가100달}달 MDD {b.평가MDD:.0f}%")
    E[["sym", "B", "k", "f", "ts", "y", "y2", "p_y", "p_y2"]].to_parquet(R / "data" / "r111_risk.parquet", index=False)
    O.to_parquet(R / "data" / "r111_grid.parquet", index=False)
    P(f"완료 ({time.time()-t_:.0f}s)")
