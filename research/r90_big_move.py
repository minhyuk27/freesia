"""R90 — 새 방향: ① 4시간 안에 ±2% 이상 움직일 '타이밍' 예측 → ② 먼저 움직이는 쪽으로 따라가기(양방향 스톱).

A 타이밍 모델: 15분 격자(R50·R57 특징 67개: 가격·체결·호가·파생·옵션·코인베이스), 라벨 y = 다음 240분 안에 고가/저가가 직전 종가 대비 ±2% 이상 닿음.
  HistGB 분류(시드 3 평균), 연도별 확장창 워크포워드(평가 2022~2026). 지표: 기저율, AUC, 확신 상위 5·10·20·30%의 적중률.
B 초기 추세 추종: 예측 확률이 '학습 예측의 상위 q%' 문턱 이상인 시점에 직전 종가 ±x% 양방향 스톱(240분 유효). 먼저 닿는 쪽으로 진입(시가 갭 반영).
  청산 E1 목표 +1.5%(지정가) · E2 추적(최고점 −0.75%) · E3 4시간 뒤 종가. 손절 = 진입가 −x(가격 기준). 최대 보유 4시간.
  비용: 스톱 진입·손절·시간 청산 테이커 5+2bp, 목표 지정가 2bp. 종목별 한 번에 한 포지션.
  비교: 모델 없이 모든 15분마다(q=100%) 같은 스톱 → 타이밍 예측의 가치.
x ∈ {0.3, 0.5, 0.75}%, q ∈ {5, 10, 20, 30, 100}%.
사용: python research/r90_big_move.py"""
import sys, warnings, time
from pathlib import Path
import numpy as np, pandas as pd, numba
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
from r50_direction_model import D as D0, FEAT as F0
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
CT, CM = 7e-4, 2e-4


@numba.njit(cache=True)
def label(t, h, l, c, W, thr):
    y = np.zeros(len(t), np.int8)
    for q in range(len(t)):
        i = t[q]; ref = c[i - 1]
        for j in range(i, min(len(c), i + W)):
            if h[j] >= ref * (1 + thr) or l[j] <= ref * (1 - thr): y[q] = 1; break
    return y


@numba.njit(cache=True)
def straddle_trades(t, on, o, h, l, c, x, mode, W, H):
    """on 시점마다(종목 한 포지션) 양방향 스톱. 반환 (진입 분, 방향, 순수익)."""
    n = len(c); E = np.zeros(len(t), np.int64); S = np.zeros(len(t), np.int64); Rr = np.zeros(len(t)); m = 0; busy = -1
    for q in range(len(t)):
        i = t[q]
        if not on[q] or i <= busy or i + W + H >= n: continue
        ref = c[i - 1]; up = ref * (1 + x); dn = ref * (1 - x); f = -1; sd = 0; px = 0.0
        for j in range(i, i + W):
            if h[j] >= up: f = j; sd = 1; px = max(up, o[j]); break
            if l[j] <= dn: f = j; sd = -1; px = min(dn, o[j]); break
        if f < 0: continue
        stop = px * (1 - sd * x); tgt = px * (1 + sd * 0.015); best = px; r = np.nan; j = f
        while j < f + H:
            hi = h[j]; lo = l[j]
            if sd > 0:
                if j > f and lo <= stop: r = (min(stop, o[j]) if o[j] < stop else stop) / px - 1 - 2 * CT_; break
                if mode == 0 and j > f and hi >= tgt: r = tgt / px - 1 - CT_ - CM_; break
                if mode == 1 and hi > best: best = hi; stop = max(stop, best * (1 - 0.0075))
            else:
                if j > f and hi >= stop: r = -((max(stop, o[j]) if o[j] > stop else stop) / px - 1) - 2 * CT_; break
                if mode == 0 and j > f and lo <= tgt: r = -(tgt / px - 1) - CT_ - CM_; break
                if mode == 1 and lo < best: best = lo; stop = min(stop, best * (1 + 0.0075))
            j += 1
        if np.isnan(r): j = min(j, n - 1); r = sd * (c[j] / px - 1) - 2 * CT_
        E[m] = f; S[m] = sd; Rr[m] = r; m += 1; busy = j
    return E[:m], S[:m], Rr[:m]


CT_, CM_ = CT, CM

if __name__ == "__main__":
    t_ = time.time()
    DF = pd.read_parquet(R / "data" / "r54_dfeat.parquet"); DC = [c for c in DF.columns if c not in ("sym", "t")]
    D = D0.merge(DF, on=["sym", "t"], how="left"); FE = F0 + DC
    D["hour"] = pd.DatetimeIndex(D.ts).hour; D["dow"] = pd.DatetimeIndex(D.ts).dayofweek; FE = FE + ["hour", "dow"]
    K = {s: store.load("klines_1m", s) for s in ("BTCUSDT", "ETHUSDT")}
    D["y"] = 0
    for s in K:
        m = D.sym == s; k = K[s]
        D.loc[m, "y"] = label(D.loc[m, "t"].values.astype(np.int64), k.high.values, k.low.values, k.close.values, 240, 0.02)
    P(f"시점 {len(D):,}, 특징 {len(FE)}, 기저율(4h ±2%) 전체 {D.y.mean():.3f} | 연도별 " + " ".join(f"{y%100}:{g.y.mean():.2f}" for y, g in D.groupby("yr")) + f" ({time.time()-t_:.0f}s)")
    out = []
    for Y in range(2022, 2027):
        tr = D[D.yr < Y]; te = D[D.yr == Y].copy()
        fu = [c for c in FE if tr[c].notna().sum() > 5000]
        ps, ptr = [], []
        for sd in range(3):
            mdl = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=500, l2_regularization=1.0, random_state=sd)
            mdl.fit(tr[fu].values, tr.y.values); ps.append(mdl.predict_proba(te[fu].values)[:, 1]); ptr.append(mdl.predict_proba(tr[fu].values)[:, 1])
        te["p"] = np.mean(ps, 0); pt = np.mean(ptr, 0)
        for q in (0.05, 0.10, 0.20, 0.30): te[f"th{int(q*100)}"] = np.quantile(pt, 1 - q)
        out.append(te); P(f"  {Y}: AUC {roc_auc_score(te.y, te.p):.3f} 기저 {te.y.mean():.3f} ({time.time()-t_:.0f}s)")
    W_ = pd.concat(out).sort_values(["t", "sym"]).reset_index(drop=True)
    P("\n===== A 타이밍 예측 (평가 2022~2026)")
    P(f"  AUC {roc_auc_score(W_.y, W_.p):.3f}, 기저율 {W_.y.mean():.3f}")
    for q in (5, 10, 20, 30):
        m = W_.p >= W_[f"th{q}"]; P(f"  학습 문턱 상위 {q:2d}%: 선택 비율 {m.mean():.3f}, 실제 ±2% 도달 {W_.y[m].mean():.3f} (향상 {W_.y[m].mean()/W_.y.mean():.2f}배) | 연도별 " +
                                  " ".join(f"{y%100}:{g.y[g.p >= g[f'th{q}']].mean():.2f}" for y, g in W_.groupby('yr')))
    P("\n===== B 초기 추세 추종 (양방향 스톱, 평가 2022~2026) — 거래당 순수익 bp / 승률 / 하루 거래 / 일간 샤프(명목 1)")
    days = (W_.ts.max() - W_.ts.min()).days
    for x in (0.003, 0.005, 0.0075):
        for mode, mn in ((0, "목표 1.5%"), (1, "추적 0.75%"), (2, "4시간 보유")):
            for q in (5, 10, 20, 30, 100):
                parts = []
                for s in K:
                    g = W_[W_.sym == s]; on = (g.p >= g[f"th{q}"]).values if q < 100 else np.ones(len(g), bool)
                    k = K[s]; E, S, Rr = straddle_trades(g.t.values.astype(np.int64), on, k.open.values, k.high.values, k.low.values, k.close.values, x, mode, 240, 240)
                    parts.append(pd.DataFrame({"t": pd.to_datetime(k.ts.values[E], utc=True), "r": Rr}))
                T = pd.concat(parts)
                if len(T) < 30: continue
                d = T.groupby(T.t.dt.floor("D")).r.sum() * 0.5; d = d.reindex(pd.date_range("2022-01-01", "2026-09-20", freq="D", tz="UTC")).fillna(0)
                yr = T.groupby(T.t.dt.year).r.mean() * 1e4
                P(f"  x={x*100:.2f}% {mn:9s} 상위 {q:3d}%: {T.r.mean()*1e4:+7.2f}bp 승률 {(T.r>0).mean():.3f} 하루 {len(T)/days:5.2f} 샤프 {d.mean()/d.std()*np.sqrt(365):5.2f} | " +
                  " ".join(f"{y%100}:{v:+.0f}" for y, v in yr.items()))
    W_[["sym", "t", "ts", "yr", "y", "p", "th5", "th10", "th20", "th30"]].to_parquet(R / "data" / "r90_wf.parquet", index=False)
    (R / "logs" / "r90.summary").write_text("R90 4시간 ±2% 타이밍 + 초기 추세 추종 완료. 세부 logs/r90.log\n")
