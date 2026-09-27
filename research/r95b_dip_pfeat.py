"""R95b — z0.31 딥 예측 모델(실시간 동등 50특징, R83)에 변동성 예보 p·pr(R90 워크포워드, 인과)를 특징으로 추가.
p는 2022년부터만 존재 → 평가 2023~2026(학습은 매년 과거 전체, p 없는 해는 결측 처리). 같은 계좌(w1N4 · w1.5N6 · w2.5N10, 문턱 25bp, V3, 킬 3%/12h).
사용: python research/r95b_dip_pfeat.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
from sklearn.ensemble import HistGradientBoostingRegressor
import r83_live_parity as Z, r40_regime_leverage as RL, r80_max_return as R80, r92_vol_timing as V

if __name__ == "__main__":
    RL.MMR = 0.0065; E = Z.E; Wp = V.load_p(); parts = []
    for s, g in E.groupby("sym"):
        w = Wp[Wp.sym == s].sort_values("t")[["t", "p", "pr"]]
        parts.append(pd.merge_asof(g.sort_values("f"), w, left_on="f", right_on="t", direction="backward").drop(columns="t"))
    E = pd.concat(parts); E["hiP"] = (E.pr >= 0.95).astype(float).where(E.pr.notna())
    for tag, feats in (("기준(실시간 50)", Z.LIVE), ("+p·pr", Z.LIVE + ["p", "pr", "hiP"])):
        out = []
        for Y in range(2023, 2027):
            tr = E[E.yr < Y]; te = E[E.yr == Y]; fu = [c for c in feats if tr[c].notna().sum() > 1000]
            m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=400, l2_regularization=1.0,
                                              random_state=0).fit(tr[fu].values, np.clip(tr.ret.values, -0.02, 0.02))
            out.append(te.assign(pred=m.predict(te[fu].values)))
        W = pd.concat(out)
        S = Z.prep(W[W.pred > 0.0025]).sort_values(["f", "sym"]).reset_index(drop=True)
        P(f"\n[{tag}] 거래 {len(S):,} 거래당 {S.ret.mean()*1e4:+.1f}bp | " + " ".join(f"{y%100}:{v*1e4:+.0f}" for y, v in S.groupby(S.ts.dt.year).ret.mean().items()))
        mult = np.minimum(np.clip(S.vreg.values ** 1.5, 0.3, 4) * np.clip(S.pred.values / 0.001, 0.5, 3), 6)
        for w_, N_ in ((1, 4), (1.5, 6), (2.5, 10)):
            res = RL.sim(S, w_, N_, mult, X=0.03, H=720)
            if res is None: P(f"  w{w_}N{N_}: 파산"); continue
            d = R80.daily_from_eq(res[0]); x = d["2023-01-01":]; m = R80.metrics(d, "2023-01-01", "2027-01-01"); m2 = R80.metrics(d, "2024-01-01", "2027-01-01")
            P(f"  w{w_}N{N_}: 23~26 샤프 {x.mean()/x.std()*np.sqrt(365):.2f} 월 {m['mo']:+.1f}% MDD {m['mdd']:.0f}% | 24~26 월 {m2['mo']:+.1f}% MDD {m2['mdd']:.0f}% | {m['yr']}")
    (R / "logs" / "r95b.summary").write_text("done\n")
