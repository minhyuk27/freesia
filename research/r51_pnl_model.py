"""R51 — 방향 모델 2차: 변동성 비례 배리어 + 실현 손익 직접 학습.

R50 문제: 적중률은 올라도(±1.5% 상위 0.1%: 65.8%) 시간 청산 36~39%가 수수료만 내고 계좌를 끌어내림.
수정:
  ① 배리어 d = clip(k × σ12, 0.5%, 2%)를 격자 {0.5, 0.75, 1, 1.25, 1.5, 1.75, 2}%에 맞춤.
     σ12 = 직전 1440분 1분 수익 표준편차 × √720 (t−1 까지). k = 0.5 · 0.75 · 1.0
  ② 목표 = 실제 거래 결과(수수료·슬리피지·시간 청산 포함 가격 수익 r). 롱·숏 각각 회귀, 더 큰 쪽을 택함.
  ③ 선택(인과적): max(예측 r)가 직전 30일 자기 예측 분포 상위 q 이상 & 예측 r > 0, 한 번에 한 포지션.
계좌: R47 규칙 그대로(증거금 50% × 30배, 명목 15배, 강제청산 2.93%는 d ≤ 2%라 손절이 먼저).
위약: 학습 목표를 연도 안에서 섞음.
사용: python research/r51_pnl_model.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
from r47_x30_frame import outcome, CT, CM, NOT, MARGIN
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
GRID = np.array([0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02]); H = 720
L = pd.read_parquet(R / "data" / "r48_labels.parquet")
C = pd.read_parquet(R / "data" / "r46_cands.parquet")
C = C[C.side == 1].drop(columns=[c for c in C.columns if c.startswith("m") or c.startswith("liq") or c in ("side", "r12", "yr", "ts")])
D = L.merge(C, on=["sym", "t"], how="inner").replace([np.inf, -np.inf], np.nan).sort_values(["t", "sym"]).reset_index(drop=True)
OUT = {}
for s in ("BTCUSDT", "ETHUSDT"):
    k = store.load("klines_1m", s)
    o, h, l, c = (k[x].values for x in ("open", "high", "low", "close"))
    m = (D.sym == s).values; idx = D.loc[m, "t"].values.astype(np.int64)
    rv = pd.Series(np.log(c)).diff().rolling(1440).std().values
    D.loc[m, "sig12"] = rv[idx - 1] * np.sqrt(720)
    for d in GRID:
        for side in (1, -1):
            r, kind, tt = outcome(o, h, l, c, h, l, idx, side, d, 0.99, H, CT, CM)
            OUT[(s, d, side)] = (r, kind, tt)
    P(f"{s} 격자 결과 계산")
D["isbtc"] = (D.sym == "BTCUSDT").astype(float)
META = {"sym", "t", "ts", "yr", "y10", "y15", "tt10", "tt15"}
FEAT = [c for c in D.columns if c not in META]


def attach(kk):
    d = np.clip(kk * D.sig12.values, GRID[0], GRID[-1])
    gi = np.abs(d[:, None] - GRID[None, :]).argmin(axis=1)
    X = D.copy(); X["d"] = GRID[gi]
    for side, nm in ((1, "L"), (-1, "S")):
        r = np.full(len(X), np.nan); kd = np.zeros(len(X), np.int8); tt = np.zeros(len(X), np.int32)
        for s in ("BTCUSDT", "ETHUSDT"):
            m = (X.sym == s).values; pos = np.where(m)[0]
            for j, dv in enumerate(GRID):
                mm = gi[pos] == j
                rr, kkd, ttt = OUT[(s, dv, side)]
                r[pos[mm]] = rr[mm]; kd[pos[mm]] = kkd[mm]; tt[pos[mm]] = ttt[mm]
        X[f"r{nm}"] = r; X[f"k{nm}"] = kd; X[f"t{nm}"] = tt
    return X


def walk(X, shuffle=False, seed=0):
    rng = np.random.default_rng(seed); out = []
    for Y in range(2022, 2027):
        tr = X[X.yr < Y]; te = X[X.yr == Y].copy()
        fu = [c for c in FEAT + ["d"] if tr[c].notna().sum() > 5000]
        for nm in ("L", "S"):
            yv = np.clip(tr[f"r{nm}"].values, -0.03, 0.03)
            if shuffle:
                yv = yv.copy()
                for yy in tr.yr.unique():
                    mk = (tr.yr == yy).values; yv[mk] = rng.permutation(yv[mk])
            m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03, max_leaf_nodes=31,
                                              min_samples_leaf=1500, l2_regularization=1.0, random_state=seed)
            m.fit(tr[fu].values, yv)
            te[f"p{nm}"] = m.predict(te[fu].values)
        out.append(te)
    return pd.concat(out).sort_values(["t", "sym"]).reset_index(drop=True)


def account(W, q):
    W = W.assign(best=np.maximum(W.pL, W.pS), side=np.where(W.pL >= W.pS, 1, -1))
    thr = W.set_index("ts").best.rolling("30D").quantile(1 - q).shift(1).values
    S = W[(W.best >= thr) & (W.best > 0)]
    cap = 1.0; busy = -1; rows = []
    for r in S.itertuples():
        if r.t < busy: continue
        rr = r.rL if r.side > 0 else r.rS; kd = r.kL if r.side > 0 else r.kS; tt = r.tL if r.side > 0 else r.tS
        cap *= max(1 + NOT * rr, 1 - MARGIN); busy = r.t + tt + 1
        rows.append((r.ts, cap, kd, rr, r.yr, r.side, r.d))
    T = pd.DataFrame(rows, columns=["ts", "cap", "kind", "r", "yr", "side", "d"])
    return T


for kk in (0.5, 0.75, 1.0):
    X = attach(kk)
    P(f"\n===== 배리어 k={kk} × σ12 (d 분포 {X.d.value_counts(normalize=True).sort_index().round(2).to_dict()})")
    P(f"  무작위 롱 결과: 익절 {(X.kL==1).mean():.2f} 손절 {(X.kL==-1).mean():.2f} 시간 {(X.kL==0).mean():.2f}")
    W = walk(X); Ws = walk(X, shuffle=True)
    for q in (0.01, 0.003, 0.001):
        for lab, WW in (("실제", W), ("위약", Ws)):
            T = account(WW, q)
            if T.empty:
                P(f"  q={q*100:.1f}% {lab}: 거래 없음"); continue
            wk = (WW.ts.max() - WW.ts.min()).days / 7
            dec = T[T.kind != 0]
            eq = T.set_index("ts").cap
            yrs = (eq.index[-1] - eq.index[0]).days / 365.25
            fin = eq.iloc[-1]; mo = (fin ** (1 / max(yrs, 0.1))) ** (1 / 12) - 1 if fin > 0 else -1
            ddn = float((eq / eq.cummax() - 1).min())
            by = T.groupby("yr").apply(lambda g: (g[g.kind != 0].kind == 1).mean())
            P(f"  q={q*100:.1f}% {lab}: 주 {len(T)/wk:4.1f}회 | 익절률(결판) {(dec.kind==1).mean():.3f} 시간 {(T.kind==0).mean():.2f} "
              f"거래당 {T.r.mean()*1e4:+6.1f}bp | 월 복리 {mo*100:+6.1f}% MDD {ddn*100:5.0f}% 최종 {fin:.3g}배 | " +
              " ".join(f"{y%100}:{v:.2f}" for y, v in by.items()))
