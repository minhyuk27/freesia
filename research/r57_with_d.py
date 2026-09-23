"""R57 — R50 방향 분류기 + D 특징(옵션 스큐·IV·풋콜 흐름, 코인베이스 프리미엄) 통합.

비교: 같은 절차에서 D 없음(A~E) vs D 포함. 둘 다 시드 3개 평균, 결판 표본 학습, 연도별 확장창 워크포워드(2022~2026 평가).
선택: R50 방식(양 꼬리 각각 직전 30일 상·하위 q, 인과적), 12시간 겹침 제거.
판정: 결판 적중·무결판·연도별, 라벨 셔플 위약, R47 30배 계좌(simulate_entries), D 특징 순열 중요도.
사용: python research/r57_with_d.py"""
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
D = D0.merge(DF, on=["sym", "t"], how="left")
P(f"시점 {len(D):,} | 기존 특징 {len(F0)} + D {len(DCOLS)} | D 커버(연도별 rr25): " +
  " ".join(f"{y}:{g.rr25.notna().mean():.2f}" for y, g in D.groupby("yr")))


def walk(feats, y, shuffle=False):
    rng = np.random.default_rng(0); out = []
    for Y in range(2022, 2027):
        tr = D[(D.yr < Y) & (D[y] != 0)]; te = D[D.yr == Y].copy()
        lab = (tr[y].values == 1).astype(int)
        if shuffle: lab = rng.permutation(lab)
        fu = [c for c in feats if tr[c].notna().sum() > 5000]
        ps = []
        for sd in (0, 1, 2):
            m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.03, max_leaf_nodes=31,
                                               min_samples_leaf=1000, l2_regularization=1.0, random_state=sd)
            m.fit(tr[fu].values, lab)
            ps.append(m.predict_proba(te[fu].values)[:, 1])
        te["p"] = np.mean(ps, axis=0); out.append(te)
    return pd.concat(out).sort_values(["t", "sym"]).reset_index(drop=True)


def report(W, y, d, lab, acct=True):
    wk = (W.ts.max() - W.ts.min()).days / 7
    for q in (0.01, 0.003, 0.001):
        S = select(W, q); dec = S[S[y] != 0]
        acc = (dec[y] == dec.side).mean() if len(dec) else np.nan
        by = dec.groupby("yr").apply(lambda g: (g[y] == g.side).mean())
        line = (f"  {lab:12s} q={q*100:.1f}%: 주 {len(S)/wk:4.1f}회 결판 적중 {acc:.3f} (n={len(dec)}) 무결판 {(S[y]==0).mean():.2f} | " +
                " ".join(f"{yy%100}:{v:.2f}" for yy, v in by.items()))
        if acct:
            eq, T = simulate_entries(S[["ts", "sym", "side"]], d, 720)
            cg, mo, mmed, mdd, fin = stat(eq)
            line += f" | 계좌 월 {mo*100:+6.1f}% MDD {mdd*100:4.0f}%"
        P(line)


for y, d in (("y15", 0.015), ("y10", 0.01)):
    P(f"\n===== ±{d*100:.1f}% / 12시간")
    W0 = walk(F0, y); report(W0, y, d, "D 없음")
    W1 = walk(F0 + DCOLS, y); report(W1, y, d, "D 포함")
    Ws = walk(F0 + DCOLS, y, shuffle=True); report(Ws, y, d, "D포함 위약", acct=False)
    W1.to_parquet(R / "data" / f"r57_wf_{y}.parquet", index=False)
    # D 특징 순열 중요도: 2024~ 평가, 상·하위 0.3% 선택의 결판 적중 하락
    tr = D[(D.yr < 2024) & (D[y] != 0)]; te = D[(D.yr >= 2024)]
    fu = [c for c in F0 + DCOLS if tr[c].notna().sum() > 5000]
    m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.03, max_leaf_nodes=31, min_samples_leaf=1000,
                                       l2_regularization=1.0, random_state=0).fit(tr[fu].values, (tr[y].values == 1).astype(int))
    X = te[fu].values.copy(); yy = te[y].values

    def tail_acc(p):
        hi, lo = np.quantile(p, 0.997), np.quantile(p, 0.003)
        mk = ((p >= hi) | (p <= lo)) & (yy != 0)
        return ((np.where(p >= hi, 1, -1) == yy)[mk]).mean()
    base = tail_acc(m.predict_proba(X)[:, 1]); rng = np.random.default_rng(0); imp = []
    for j, c in enumerate(fu):
        Xp = X.copy(); Xp[:, j] = rng.permutation(Xp[:, j])
        imp.append((base - tail_acc(m.predict_proba(Xp)[:, 1]), c))
    imp.sort(reverse=True)
    P(f"  순열 중요도(꼬리 0.3% 적중 하락, 기준 {base:.3f}): " + " ".join(f"{c}{'*' if c in DCOLS else ''} {v*100:+.1f}" for v, c in imp[:15]))
