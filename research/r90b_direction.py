"""R90b — 큰 움직임(4h ±2%)이 예측된 시점에서만 '방향'을 따로 예측할 수 있는가.

라벨: 다음 240분 안에 +2%와 −2% 중 먼저 닿은 쪽(둘 다 안 닿으면 제외). 같은 분에 둘 다 = 제외.
방향 모델: y=1(큰 움직임) 표본만으로 HistGB 분류(시드 3), 연도별 확장창 워크포워드(평가 2022~2026). 특징 = R90과 같은 69개.
평가: R90 타이밍 예측 상위 5·10·20·30% 시점에서 방향 적중률(결판만), 방향 확신 상위 절반만 추가 선별.
매매: 해당 시점 다음 분 시가 시장가로 예측 방향 진입, ±2% 대칭(목표 지정가 2bp, 손절 테이커 7bp), 4시간 시간 청산(7bp).
사용: python research/r90b_direction.py"""
import sys, warnings, time
from pathlib import Path
import numpy as np, pandas as pd, numba
from sklearn.ensemble import HistGradientBoostingClassifier
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
from r50_direction_model import D as D0, FEAT as F0
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)


@numba.njit(cache=True)
def first_side(t, h, l, c, W, thr):
    y = np.zeros(len(t), np.int8)
    for q in range(len(t)):
        i = t[q]; ref = c[i - 1]
        for j in range(i, min(len(c), i + W)):
            u = h[j] >= ref * (1 + thr); d = l[j] <= ref * (1 - thr)
            if u and d: break
            if u: y[q] = 1; break
            if d: y[q] = -1; break
    return y


@numba.njit(cache=True)
def trade(t, side, o, h, l, c, thr, H):
    n = len(c); R_ = np.full(len(t), np.nan)
    for q in range(len(t)):
        i = t[q]
        if side[q] == 0 or i + H >= n: continue
        e = o[i]; s = side[q]; tp = e * (1 + s * thr); sl = e * (1 - s * thr); r = np.nan
        for j in range(i, i + H):
            if (s > 0 and l[j] <= sl) or (s < 0 and h[j] >= sl): r = -thr - 14e-4; break
            if (s > 0 and h[j] >= tp) or (s < 0 and l[j] <= tp): r = thr - 9e-4; break
        if np.isnan(r): r = s * (c[i + H - 1] / e - 1) - 14e-4
        R_[q] = r
    return R_


if __name__ == "__main__":
    t_ = time.time()
    DF = pd.read_parquet(R / "data" / "r54_dfeat.parquet"); DC = [c for c in DF.columns if c not in ("sym", "t")]
    D = D0.merge(DF, on=["sym", "t"], how="left"); FE = F0 + DC
    D["hour"] = pd.DatetimeIndex(D.ts).hour; D["dow"] = pd.DatetimeIndex(D.ts).dayofweek; FE = FE + ["hour", "dow"]
    K = {s: store.load("klines_1m", s) for s in ("BTCUSDT", "ETHUSDT")}
    D["side"] = 0
    for s in K:
        m = D.sym == s; k = K[s]
        D.loc[m, "side"] = first_side(D.loc[m, "t"].values.astype(np.int64), k.high.values, k.low.values, k.close.values, 240, 0.02)
    T90 = pd.read_parquet(R / "data" / "r90_wf.parquet")
    D = D.merge(T90[["sym", "t", "p", "th5", "th10", "th20", "th30"]], on=["sym", "t"], how="left")
    P(f"큰 움직임 표본 {(D.side != 0).sum():,} (상승 먼저 비율 {(D.side[D.side != 0] == 1).mean():.3f}) ({time.time()-t_:.0f}s)")
    out = []
    for Y in range(2022, 2027):
        tr = D[(D.yr < Y) & (D.side != 0)]; te = D[D.yr == Y].copy()
        fu = [c for c in FE if tr[c].notna().sum() > 3000]
        ps = [HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=300, l2_regularization=1.0,
                                             random_state=sd).fit(tr[fu].values, (tr.side.values == 1).astype(int)).predict_proba(te[fu].values)[:, 1] for sd in range(3)]
        te["pu"] = np.mean(ps, 0); out.append(te); P(f"  {Y} ({time.time()-t_:.0f}s)")
    W = pd.concat(out)
    W["pred"] = np.where(W.pu > 0.5, 1, -1); W["conf"] = (W.pu - 0.5).abs()
    P("\n===== 방향 적중률 (결판 표본, 평가 2022~2026)")
    dec = W[W.side != 0]
    P(f"  전체 큰 움직임: {(dec.pred == dec.side).mean():.3f} (n={len(dec):,})")
    days = (W.ts.max() - W.ts.min()).days
    for q in (5, 10, 20, 30):
        sel = W[W.p >= W[f"th{q}"]]
        for half in (False, True):
            S = sel
            if half: S = sel[sel.conf >= sel.conf.median()]
            d = S[S.side != 0]; acc = (d.pred == d.side).mean()
            parts = []
            for s in K:
                g = S[S.sym == s].sort_values("t"); k = K[s]
                # 한 번에 한 포지션: 4시간 안 겹침 제거
                keep = []; last = -10**9
                for tt in g.t.values:
                    if tt >= last + 240: keep.append(True); last = tt
                    else: keep.append(False)
                g = g[np.array(keep, bool)]
                r = trade(g.t.values.astype(np.int64), g.pred.values.astype(np.int64), k.open.values, k.high.values, k.low.values, k.close.values, 0.02, 240)
                parts.append(pd.DataFrame({"yr": g.yr.values, "r": r}))
            Tt = pd.concat(parts).dropna()
            yr = Tt.groupby("yr").r.mean() * 1e4
            P(f"  타이밍 상위 {q:2d}%{' · 방향 확신 상위 절반' if half else ''}: 방향 적중 {acc:.3f} (n={len(d):,}) | 매매 ±2%: 거래 {len(Tt)} (하루 {len(Tt)/days:.2f}) "
              f"거래당 {Tt.r.mean()*1e4:+.1f}bp 승률 {(Tt.r>0).mean():.3f} | " + " ".join(f"{y%100}:{v:+.0f}" for y, v in yr.items()))
    (R / "logs" / "r90b.summary").write_text("R90b 큰 움직임 조건부 방향 예측 완료. 세부 logs/r90b.log\n")
