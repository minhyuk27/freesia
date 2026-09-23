"""R60 — 앙상블 크기·이종 모델 결합이 확신 꼬리 적중률을 올리는가.

관찰(R50b·R57): 부스팅 단일 시드 상위 0.1% 적중 61~66% → 시드 3개 확률 평균 66.7%.
비교(±1.5%, 12시간, 기존 A~E 특징, 연도별 확장창 워크포워드):
  G1/G3/G5/G10 = 부스팅 시드 1·3·5·10개 평균 (행 부트스트랩 없이 시드만 다름)
  B5 = 부스팅 5개, 각각 학습 표본 80% 부분표본(배깅)
  E5 = ExtraTrees(무작위 분할 트리) 1개
  B5+E = 배깅 부스팅 5개 + ExtraTrees 평균
선택: R50 양 꼬리(q=0.3%, 0.1%) → 결판 적중, 연도별, 30배 계좌.
사용: python research/r60_ensemble.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from r50_direction_model import D, FEAT, select
from r47_x30_frame import simulate_entries, stat
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
y, d = "y15", 0.015


def gb(sd):
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.03, max_leaf_nodes=31, min_samples_leaf=1000,
                                          l2_regularization=1.0, random_state=sd)


preds = {k: [] for k in ("G1", "G3", "G5", "G10", "B5", "E5", "B5+E")}
tes = []
for Y in range(2022, 2027):
    tr = D[(D.yr < Y) & (D[y] != 0)]; te = D[D.yr == Y]
    fu = [c for c in FEAT if tr[c].notna().sum() > 5000]
    Xtr, ytr, Xte = tr[fu].values, (tr[y].values == 1).astype(int), te[fu].values
    g = [gb(sd).fit(Xtr, ytr).predict_proba(Xte)[:, 1] for sd in range(10)]
    rng = np.random.default_rng(Y); b = []
    for sd in range(5):
        ix = rng.choice(len(Xtr), int(len(Xtr) * 0.8), replace=False)
        b.append(gb(100 + sd).fit(Xtr[ix], ytr[ix]).predict_proba(Xte)[:, 1])
    imp = SimpleImputer().fit(Xtr)
    et = ExtraTreesClassifier(n_estimators=300, min_samples_leaf=500, max_features=0.3, n_jobs=-1, random_state=0) \
        .fit(imp.transform(Xtr), ytr).predict_proba(imp.transform(Xte))[:, 1]
    preds["G1"].append(g[0]); preds["G3"].append(np.mean(g[:3], 0)); preds["G5"].append(np.mean(g[:5], 0))
    preds["G10"].append(np.mean(g, 0)); preds["B5"].append(np.mean(b, 0)); preds["E5"].append(et)
    preds["B5+E"].append((np.mean(b, 0) * 5 + et) / 6)
    tes.append(te[["sym", "t", "ts", "yr", y]])
    P(f"  {Y} 완료")
T0 = pd.concat(tes).reset_index(drop=True)
wk = (T0.ts.max() - T0.ts.min()).days / 7
for k, v in preds.items():
    W = T0.assign(p=np.concatenate(v)).sort_values(["t", "sym"]).reset_index(drop=True)
    for q in (0.003, 0.001):
        S = select(W, q); dec = S[S[y] != 0]
        acc = (dec[y] == dec.side).mean()
        eq, T = simulate_entries(S[["ts", "sym", "side"]], d, 720)
        cg, mo, mmed, mdd, fin = stat(eq)
        by = dec.groupby("yr").apply(lambda g_: (g_[y] == g_.side).mean())
        P(f"  {k:5s} q={q*100:.1f}%: 주 {len(S)/wk:4.1f}회 결판 적중 {acc:.3f} (n={len(dec)}) 무결판 {(S[y]==0).mean():.2f} | 계좌 월 {mo*100:+6.1f}% MDD {mdd*100:4.0f}% | "
          + " ".join(f"{yy%100}:{vv:.2f}" for yy, vv in by.items()))
