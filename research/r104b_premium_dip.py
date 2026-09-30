"""R104b — 코인베이스 프리미엄을 z0.31 딥 필터·크기 조절로: 딥 진입 직전(진입 분 −1까지) 프리미엄 상태별 거래 성과.

R104: 프리미엄 15분 예측력은 유의(t 3.4~4.7)하나 IC 0.01~0.02라 단독 거래는 비용에 진다 → 이미 거래하는 딥의 조건으로만 쓴다.
특징(진입 분 직전 종가까지): dp15·dp60·dp240, z1d, zx1d(스테이블 공통 성분 제거), cz1d(공통 성분).
평가: z0.31 딥 사건(예측 > 25bp = R81 실전 문턱, R32 워크포워드)의 거래당 수익을 특징 5분위별로 — 학습 2022~23 / 평가 2024~26.
학습에서 단조(1→5분위 순위상관 |ρ| ≥ 0.9)이고 상·하위 분위 차이가 큰 특징만 채택 → 평가에서 같은 방향인지 확인,
채택 시 크기 배율 m = 1 + a × (분위 − 3)/2 (a 0.5·1.0, 학습 방향)으로 R102 공유 계좌(w1N4·w3N12, z0.33 TOP8 펀딩≤1bp L33=1) 재실행.
사용: python research/r104b_premium_dip.py"""
import sys, warnings, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
import r104_coinbase_lead as C, r102_joint_account as J, r103_breakout_cost as K
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
FE = ("dp15", "dp60", "dp240", "z1d", "zx1d", "cz1d")

if __name__ == "__main__":
    t_ = time.time()
    DAT = {s: C.load(s + "USDT", s + "USD") for s in ("BTC", "ETH")}
    idx = DAT["BTC"][0].index.intersection(DAT["ETH"][0].index)
    PR = {s: (DAT[s][1] - DAT[s][0]).reindex(idx) for s in DAT}; common = (PR["BTC"] + PR["ETH"]) / 2
    FT = {}
    for s in PR:
        p = PR[s]; px = p - common
        z = lambda v, W: (v - v.rolling(W, min_periods=W // 2).mean()) / v.rolling(W, min_periods=W // 2).std()
        FT[s] = pd.DataFrame({"dp15": p - p.shift(15), "dp60": p - p.shift(60), "dp240": p - p.shift(240), "z1d": z(p, 1440),
                              "zx1d": z(px, 1440), "cz1d": z(common, 1440)}).shift(1)          # 진입 분 직전 종가까지
    D = J.dip_events(); D["sym"] = np.where(D.s == 1, "ETH", "BTC"); D["ts"] = J.D0 + pd.to_timedelta(D.e, unit="min")
    for f in FE:
        D[f] = np.nan
        for s in ("BTC", "ETH"):
            m = D.sym == s; D.loc[m, f] = FT[s][f].reindex(D.loc[m, "ts"]).values
    tr = D.ts < C.SPLIT
    P(f"딥 사건 {len(D):,} (학습 {tr.sum():,} / 평가 {(~tr).sum():,}), 특징 결측 {D[list(FE)].isna().mean().mean()*100:.1f}% ({time.time()-t_:.0f}s)")
    adopt = {}
    for f in FE:
        q = pd.qcut(D.loc[tr, f], 5, labels=False, duplicates="drop"); edges = D.loc[tr, f].quantile([.2, .4, .6, .8]).values
        D[f + "_q"] = np.searchsorted(edges, D[f].values) + 1
        a = D[tr].groupby(f + "_q").ret.agg(["mean", "size"]); b = D[~tr].groupby(f + "_q").ret.agg(["mean", "size"])
        rho_tr = spearmanr(a.index, a["mean"])[0]; rho_te = spearmanr(b.index, b["mean"])[0]
        P(f"  {f:6s} 학습 분위별 bp " + " ".join(f"{v*1e4:+5.1f}" for v in a["mean"]) + f" (ρ {rho_tr:+.1f}) | 평가 " +
          " ".join(f"{v*1e4:+5.1f}" for v in b["mean"]) + f" (ρ {rho_te:+.1f})")
        if abs(rho_tr) >= 0.9: adopt[f] = np.sign(rho_tr)
    P(f"학습 단조 채택: {adopt}")
    if adopt:
        T = J.add_funding(J.breakout_trades(J.TOP8)); T["lf"] = K.last_funding(T)
        BK = K.events(T, False, fmax=1e-4); start = J.mins([pd.Timestamp("2022-01-01", tz="UTC")])[0]; BK = BK[BK.e >= start]
        base = pd.concat([D[["e", "x", "ret", "mae", "k", "mult", "s", "px"]], BK.drop(columns="lf")], ignore_index=True).sort_values(["e", "k"]).reset_index(drop=True)
        for tier, (w, N) in (("무레버", (1, 4)), ("공격", (3, 12))):
            J.rep(f"기준 {tier}", base, w, N, w)
        for f, sg in adopt.items():
            for a_ in (0.5, 1.0):
                D2 = D.copy(); D2["mult"] = D2.mult * np.clip(1 + a_ * sg * (D2[f + "_q"] - 3) / 2, 0, None)
                E = pd.concat([D2[["e", "x", "ret", "mae", "k", "mult", "s", "px"]], BK.drop(columns="lf")], ignore_index=True).sort_values(["e", "k"]).reset_index(drop=True)
                for tier, (w, N) in (("무레버", (1, 4)), ("공격", (3, 12))):
                    J.rep(f"{f} a{a_} {tier}", E, w, N, w)
    P(f"완료 ({time.time()-t_:.0f}s)")
