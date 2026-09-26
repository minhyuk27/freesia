"""R75 — 사용자 확정 구조에서 '빈도 ↔ 승률' 경계 측정.

구조(2026-09-26 확정): ±1% 대칭 먼저 닿기, 롱/숏 방향만, 한 번에 한 포지션(신호 대기 허용), 증거금 = 자본 50%, 50배 고정.
요구: 승률 ≥ 70%(목표 80%), 빈도 하루 최소 1건·평균 1.5건 이상.
신호: R50·R57 방향 분류기(특징 52 + 옵션·코인베이스 D 15, ±1%/12시간 라벨, 결판 표본 학습), HistGB 시드 5 평균,
      연도별 확장창 워크포워드(평가 2022~2026). 15분 격자 × BTC·ETH.
선택: 양 꼬리 각각 직전 30일 예측 분위 q (인과적, R50 select) → 한 번에 한 포지션(R47 simulate_entries, 50배로 설정).
q를 바꿔 하루 거래 수 · 승률(익절 비율, 시간청산은 손익 부호) · 계좌(월 복리, MDD)를 표로.
사용: python research/r75_x50_frontier.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
import r47_x30_frame as X
from freesia import costs
from r50_direction_model import D as D0, FEAT as F0, select
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
X.LEV = 50.0; X.NOT = X.LEV * X.MARGIN; X.LIQ = 1 / X.LEV - costs.MMR_TIER1       # 50배, 명목 = 자본 × 25, 청산거리 1.6%

DF = pd.read_parquet(R / "data" / "r54_dfeat.parquet")
DC = [c for c in DF.columns if c not in ("sym", "t")]
D = D0.merge(DF, on=["sym", "t"], how="left"); FE = F0 + DC


def walk(y="y10"):
    out = []
    for Y in range(2022, 2027):
        tr = D[(D.yr < Y) & (D[y] != 0)]; te = D[D.yr == Y].copy()
        fu = [c for c in FE if tr[c].notna().sum() > 5000]
        te["p"] = np.mean([HistGradientBoostingClassifier(max_iter=400, learning_rate=0.03, max_leaf_nodes=31, min_samples_leaf=1000,
                                                          l2_regularization=1.0, random_state=sd).fit(tr[fu].values, (tr[y].values == 1).astype(int))
                           .predict_proba(te[fu].values)[:, 1] for sd in range(5)], axis=0)
        out.append(te); P(f"  {Y} 학습 {len(tr):,} → 평가 {len(te):,}")
    return pd.concat(out).sort_values(["t", "sym"]).reset_index(drop=True)


if __name__ == "__main__":
    W = walk()
    W.to_parquet(R / "data" / "r75_wf.parquet", index=False)
    days = (W.ts.max() - W.ts.min()).days
    P(f"\n===== ±1% 먼저 닿기, 50배 · 증거금 50% · 한 번에 한 포지션 (평가 2022-01~2026-09, {days}일)")
    P(f"  {'q':>6s} | {'하루 거래':>8s} | {'승률':>6s} (익절/손절/청산/시간) | {'거래당 자본':>10s} | {'월 복리':>8s} | {'MDD':>6s} | 연도별 승률")
    for q in (0.001, 0.003, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2):
        S = select(W, q)
        eq, T = X.simulate_entries(S[["ts", "sym", "side"]], 0.01, 720)
        if len(T) == 0: continue
        win = (T.r > 0).mean(); kinds = T.kind.value_counts(normalize=True)
        cg, mo, mmed, mdd, fin = X.stat(eq)
        T["yr"] = pd.DatetimeIndex(T.ts).year
        yr = T.groupby("yr").apply(lambda g: (g.r > 0).mean())
        P(f"  {q*100:5.1f}% | {len(T)/days:8.2f} | {win:.3f} ({kinds.get(1,0):.2f}/{kinds.get(-1,0):.2f}/{kinds.get(-2,0):.2f}/{kinds.get(0,0):.2f}) | "
          f"{(X.NOT*T.r).mean()*100:+9.2f}% | {mo*100:+7.1f}% | {mdd*100:5.0f}% | " + " ".join(f"{k%100}:{v:.2f}" for k, v in yr.items()))
    (R / "logs" / "r75.summary").write_text("R75 50배 ±1% 빈도-승률 경계 완료. 세부 logs/r75.log\n")
