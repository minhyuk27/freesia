"""R50 — 방향 F: 전 특징 통합 방향 모델 → 확신 양 꼬리만 매매 → R47 30배 구조로 계좌 판정.

라벨: R48 (±d 먼저 도달, 12시간). 학습은 결판 표본(±1)만, 목표 = 위 먼저(1)/아래 먼저(0).
특징: R46 후보표(R27 차트·거래량·파생 + R32 오더플로우·호가·청산 개념, t−1 분까지) + R48의 다중 척도 수익 + 추가 낙폭 특징.
모델: 부스팅 분류기, 연도별 확장창(평가 연도는 과거로만 학습).
선택(인과적): 예측 확률 p 가 직전 30일 자기 예측 분포의 상위 q → 롱, 하위 q → 숏 (주 3~6회 목표).
판정: ① 선택 표본 적중률(결판)·무승부율·연도별 ② 라벨 셔플 위약 ③ R47 simulate_entries (증거금 50% × 30배, ±d, 12시간).
사용: python research/r50_direction_model.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
from r47_x30_frame import simulate_entries, stat
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
L = pd.read_parquet(R / "data" / "r48_labels.parquet")
C = pd.read_parquet(R / "data" / "r46_cands.parquet")
C = C[C.side == 1].drop(columns=[c for c in C.columns if c.startswith("m") or c.startswith("liq") or c in ("side", "r12", "yr", "ts")])
D = L.merge(C, on=["sym", "t"], how="inner").replace([np.inf, -np.inf], np.nan).sort_values(["t", "sym"]).reset_index(drop=True)
# 추가: 7일 고저 대비 위치, 24시간 낙폭의 변동성 정규화 (t−1 까지)
for s in ("BTCUSDT", "ETHUSDT"):
    k = store.load("klines_1m", s)
    c = k.close.values; lo7 = pd.Series(k.low.values).rolling(10080).min().values; hi7 = pd.Series(k.high.values).rolling(10080).max().values
    rv = pd.Series(np.log(c)).diff().rolling(1440).std().values
    m = D.sym == s; t1 = D.loc[m, "t"].values - 1
    D.loc[m, "pos7d"] = (c[t1] - lo7[t1]) / (hi7[t1] - lo7[t1])
    D.loc[m, "r1440n"] = D.loc[m, "r1440"].values / (rv[t1] * np.sqrt(1440))
    D.loc[m, "r240n"] = D.loc[m, "r240"].values / (rv[t1] * np.sqrt(240))
D["isbtc"] = (D.sym == "BTCUSDT").astype(float)
META = {"sym", "t", "ts", "yr", "y10", "y15", "tt10", "tt15"}
FEAT = [c for c in D.columns if c not in META]
P(f"시점 {len(D):,}, 특징 {len(FEAT)}개")


def walk(y, shuffle=False, seed=0):
    rng = np.random.default_rng(seed); out = []
    for Y in range(2022, 2027):
        tr = D[(D.yr < Y) & (D[y] != 0)]; te = D[D.yr == Y]
        lab = (tr[y].values == 1).astype(int)
        if shuffle: lab = rng.permutation(lab)
        fu = [c for c in FEAT if tr[c].notna().sum() > 5000]
        m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.03, max_leaf_nodes=31,
                                           min_samples_leaf=1000, l2_regularization=1.0, random_state=seed)
        m.fit(tr[fu].values, lab)
        out.append(te.assign(p=m.predict_proba(te[fu].values)[:, 1]))
    W = pd.concat(out).sort_values(["t", "sym"]).reset_index(drop=True)
    return W


def select(W, q):
    s = W.set_index("ts").p
    hi = s.rolling("30D").quantile(1 - q).shift(1).values
    lo = s.rolling("30D").quantile(q).shift(1).values
    W = W.assign(hi=hi, lo=lo)
    L_ = W[(W.p >= W.hi)].assign(side=1); S_ = W[(W.p <= W.lo)].assign(side=-1)
    sel = pd.concat([L_, S_]).dropna(subset=["hi"]).sort_values("t")
    # 12시간 안 겹침 제거(한 번에 한 포지션과 같은 빈도)
    keep = []; busy = -1
    for r in sel.itertuples():
        if r.t < busy: continue
        keep.append(r.Index); busy = r.t + 720
    return sel.loc[keep]


if __name__ == "__main__":
    for y, d in (("y10", 0.01), ("y15", 0.015)):
        P(f"\n===== ±{d*100:.1f}% / 12시간")
        W = walk(y)
        Ws = walk(y, shuffle=True)
        wk = (W.ts.max() - W.ts.min()).days / 7
        for q in (0.01, 0.003, 0.001):
            for lab, WW in (("실제", W), ("셔플 위약", Ws)):
                S = select(WW, q)
                dec = S[S[y] != 0]
                acc = (dec[y] == dec.side).mean() if len(dec) else np.nan
                by = dec.groupby("yr").apply(lambda g: (g[y] == g.side).mean())
                P(f"  q={q*100:.1f}% {lab:5s}: 주 {len(S)/wk:4.1f}회 적중(결판) {acc:.3f} 무승부 {(S[y]==0).mean():.2f} 롱비중 {(S.side>0).mean():.2f} | " +
                  " ".join(f"{yy%100}:{v:.2f}" for yy, v in by.items()))
                if lab == "실제":
                    ent = S[["ts", "sym", "side"]].copy()
                    eq, T = simulate_entries(ent, d, 720)
                    cg, mo, mmed, mdd, fin = stat(eq)
                    P(f"      → 30배 구조 계좌: 월 복리 {mo*100:+.1f}% 월중앙 {mmed*100:+.1f}% MDD {mdd*100:.0f}% 최종 {fin:.3g}배 "
                      f"(익절 {(T.kind==1).mean():.2f} 손절 {(T.kind==-1).mean():.2f} 시간 {(T.kind==0).mean():.2f})")
        W.to_parquet(R / "data" / f"r50_wf_{y}.parquet", index=False)
