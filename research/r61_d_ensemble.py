"""R61 — D 특징 + 시드 평균 확대로 상위 0.3%(주 2회) 구간을 65%로 끌어올릴 수 있는가.

R57: D 포함 시드 3 평균에서 ±1.5% 상위 0.3% 58.6→62.8%. R60: 시드 5 평균이 꼬리를 가장 안정화.
R58: 특징을 많이 넣으면 잡음이 늘었다 → D 전체 vs 코인베이스 계열만(D-cb) 비교.
구성: {D 없음, D 전체, D-cb} × {시드 5, 시드 10}, ±1.5%/12시간, 연도별 확장창 워크포워드(2022~2026 평가).
선택: R50 양 꼬리(직전 30일 상·하위 q, 인과적), q = 0.5·0.3·0.2·0.1%. 최선 구성은 라벨 셔플 위약.
사용: python research/r61_d_ensemble.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from r50_direction_model import D as D0, FEAT as F0, select
from r47_x30_frame import simulate_entries, stat
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
DF = pd.read_parquet(R / "data" / "r54_dfeat.parquet")
DCOLS = [c for c in DF.columns if c not in ("sym", "t")]
DCB = [c for c in DCOLS if "cbp" in c]
D = D0.merge(DF, on=["sym", "t"], how="left")
Y, DB = "y15", 0.015
P(f"시점 {len(D):,} | 기존 {len(F0)} + D {len(DCOLS)} (코인베이스 계열 {DCB})")


def walk(feats, seeds, shuffle=False):
    rng = np.random.default_rng(0); out = []
    for yr in range(2022, 2027):
        tr = D[(D.yr < yr) & (D[Y] != 0)]; te = D[D.yr == yr].copy()
        lab = (tr[Y].values == 1).astype(int)
        if shuffle: lab = rng.permutation(lab)
        fu = [c for c in feats if tr[c].notna().sum() > 5000]
        ps = [HistGradientBoostingClassifier(max_iter=400, learning_rate=0.03, max_leaf_nodes=31, min_samples_leaf=1000,
                                             l2_regularization=1.0, random_state=sd).fit(tr[fu].values, lab)
              .predict_proba(te[fu].values)[:, 1] for sd in range(seeds)]
        te["p"] = np.mean(ps, axis=0); out.append(te)
    return pd.concat(out).sort_values(["t", "sym"]).reset_index(drop=True)


def report(W, lab, acct=True):
    wk = (W.ts.max() - W.ts.min()).days / 7; res = {}
    for q in (0.005, 0.003, 0.002, 0.001):
        S = select(W, q); dec = S[S[Y] != 0]
        acc = (dec[Y] == dec.side).mean() if len(dec) else np.nan
        by = dec.groupby("yr").apply(lambda g: (g[Y] == g.side).mean())
        line = (f"  {lab:14s} q={q*100:.1f}%: 주 {len(S)/wk:4.1f}회 결판 적중 {acc:.3f} (n={len(dec)}) 무결판 {(S[Y]==0).mean():.2f} | " +
                " ".join(f"{yy%100}:{v:.2f}" for yy, v in by.items()))
        if acct:
            eq, T = simulate_entries(S[["ts", "sym", "side"]], DB, 720)
            cg, mo, mmed, mdd, fin = stat(eq)
            line += f" | 계좌 월 {mo*100:+6.1f}% MDD {mdd*100:4.0f}%"
        P(line); res[q] = acc
    return res


P(f"\n===== ±{DB*100:.1f}% / 12시간")
best = (None, -1, None)
for name, feats in (("D 없음", F0), ("D 전체", F0 + DCOLS), ("D-cb", F0 + DCB)):
    for seeds in (5, 10):
        lab = f"{name} s{seeds}"
        r = report(walk(feats, seeds), lab)
        if r[0.003] > best[1]: best = (lab, r[0.003], (feats, seeds))
P(f"\n최선(상위 0.3% 기준): {best[0]} {best[1]:.3f} → 위약")
report(walk(best[2][0], best[2][1], shuffle=True), best[0] + " 위약", acct=False)

L = [f"R61 D+시드평균 (±1.5%/12h) 최선 {best[0]}: 상위 0.3% 적중 {best[1]:.3f}", "세부 logs/r61.log"]
(R / "logs" / "r61.summary").write_text("\n".join(L) + "\n")
