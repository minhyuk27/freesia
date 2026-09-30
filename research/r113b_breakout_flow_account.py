"""R113b — z0.33 돌파 × 진입 직전 선물 테이커 순매수(R113 B): 공유 계좌(R109 정확 엔진)에서 샤프·성장이 오르는가.

R113 B에서 학습(2020~23)·평가(2024~26) 모두 단조였던 특징(사전 기준 |ρ| ≥ 0.9·같은 부호): pnet5, pnet15(+), div5(−), share5(−).
주 후보(사전 지정) = pnet15. 나머지는 민감도.
변형: 분위(학습 2020~23 돌파 거래 기준 경계) q ∈ 1..5
  F3 = q ≥ 3만, F4 = q ≥ 4만 거래 / S = 크기 × (1 + a × (q − 3)/2), a ∈ {0.5, 1.0} (음의 특징은 방향 반전).
계좌: z0.31 딥(w3·N12) + 돌파 L ∈ {4, 8, 12, 16}. 선택 2022~23 → 평가 2024~26. 기준 = 필터 없음.
사용: python research/r113b_breakout_flow_account.py"""
import sys, warnings, time
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
import r102_joint_account as J, r103_breakout_cost as K, r80_max_return as R80, r107_max_growth as G7, r109_dip_exact_mtm as M, r113_spot_perp_flow as F
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
CANDS = (("pnet15", 1), ("pnet5", 1), ("div5", -1), ("share5", -1))

if __name__ == "__main__":
    t_ = time.time()
    FE = {s: F.feats(s)[0] for s in ("BTCUSDT", "ETHUSDT")}
    T8 = J.add_funding(J.breakout_trades(J.TOP8)); T8["lf"] = K.last_funding(T8); BK = K.events(T8, False, fmax=1e-4).drop(columns="lf")
    for s_, sym in ((0, "BTCUSDT"), (1, "ETHUSDT")):
        mk = (BK.s == s_).values; ts = pd.DatetimeIndex(J.D0 + pd.to_timedelta(BK.loc[mk, "e"] - 1, unit="min"))
        for c, _ in CANDS: BK.loc[mk, c] = FE[sym][c].reindex(ts).values
    tsb = J.D0 + pd.to_timedelta(BK.e, unit="min"); trb = (tsb < F.SPLIT).values
    D, _, _, W = M.dip_with_px()
    start = int(J.mins([pd.Timestamp("2022-01-01", tz="UTC")])[0]); Gd = G7.grid(start, "mark_1m")
    def acct(Bx, lab):
        out = []
        E = pd.concat([D, Bx[Bx.e >= start][["e", "x", "ret", "mae", "k", "mult", "s", "px"]]], ignore_index=True).sort_values(["e", "k"]).reset_index(drop=True)
        for L in (4, 8, 12, 16):
            d, pk = M.run(E, Gd, 3, 12, L)
            if d is None: out.append(dict(lab=lab, L=L, ok=False)); continue
            a = R80.metrics(d, "2022-01-01", "2024-01-01"); b = R80.metrics(d, "2024-01-01", "2026-09-23"); sr = d["2024":"2026-09-22"]; sa = d["2022":"2023"]
            out.append(dict(lab=lab, L=L, ok=True, 학습월=a["mo"], 학습샤프=sa.mean() / sa.std() * np.sqrt(365), 평가월=b["mo"], 평가중앙=b["med"], 평가100달=b["big"],
                            평가MDD=b["mdd"], 평가샤프=sr.mean() / sr.std() * np.sqrt(365), 연도=b["yr"]))
        return out
    rows = acct(BK, "기준(필터 없음)")
    for c, sg in CANDS:
        v = BK[c] * sg; edges = v[trb].quantile([.2, .4, .6, .8]).values; q = np.searchsorted(edges, v.values) + 1; q = np.where(v.isna(), 3, q)
        yb = BK.ret.groupby([q, trb]).mean().unstack() * 1e4
        P(f"{c}(부호 {sg:+d}) 분위별 거래당 bp 학습: " + " ".join(f"{x:+.0f}" for x in yb[True].values) + " | 평가: " + " ".join(f"{x:+.0f}" for x in yb[False].values))
        for nm, keep, mul in (("F3", q >= 3, 1.0), ("F4", q >= 4, 1.0), ("S0.5", np.ones(len(q), bool), 1 + 0.5 * (q - 3) / 2), ("S1.0", np.ones(len(q), bool), 1 + 1.0 * (q - 3) / 2)):
            Bx = BK[keep].copy(); Bx["mult"] = Bx.mult * (mul[keep] if np.ndim(mul) else mul)
            rows += acct(Bx, f"{c} {nm}")
        P(f"  {c} 완료 ({time.time()-t_:.0f}s)")
    O = pd.DataFrame(rows); pd.set_option("display.width", 250); pd.set_option("display.max_rows", 120)
    S = O[O.ok]
    P("\n===== 구성별 학습(2022~23) 최선 L → 평가")
    for lab, g in S.groupby("lab", sort=False):
        b = g.sort_values("학습월", ascending=False).iloc[0]
        P(f"  {lab:18s} L{b.L:>2} 학습 월 {b.학습월:+5.1f}% 샤프 {b.학습샤프:.2f} → 평가 월 {b.평가월:+5.1f}% 중앙 {b.평가중앙:+5.1f}% ≥100% {b.평가100달}달 MDD {b.평가MDD:4.0f}% 샤프 {b.평가샤프:.2f} | {b.연도}")
    P(f"파산: {len(O) - len(S)}건 — " + ", ".join(f"{r.lab} L{r.L}" for r in O[~O.ok].itertuples()))
    O.to_parquet(R / "data" / "r113b.parquet", index=False)
    P(f"완료 ({time.time()-t_:.0f}s)")
