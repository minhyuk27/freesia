"""R94 — 스트래들 매도(변동성 위험 프리미엄)를 p로 거르고 z0.31 딥과 헷지 결합.

R93 칸 채굴(49칸) 대신 사전 고정 규칙만 쓴다(선택 편향 제거):
  S0  무필터: 04시(잔존 4h) 스트래들 매도, 매일 종목별 1건
  S1  04시 매도, p 순위 pr < 0.8 (큰 움직임 예보 상위 20%면 쉼)
  S2  00~07시 매 시 매도(각 1/8 크기), pr < 0.5
  S3  00~07시 매 시 매도(각 1/8), pr < 0.8
  L8  00시(잔존 8h) 스트래들 매수, pr ≥ 0.95 (큰 움직임 예보 최상위)
  H   S3 + L8 (변동성 국면에 따라 매도↔매수 전환)
각 규칙: 일간 손익(기초 명목 1 기준), 샤프 학습 22~23 / 평가 24~26, 최악의 날, z0.31(w1N4) 일간 상관.
결합: z0.31 + k × 규칙(2022~23 변동성으로 z0.31에 맞춤), k ∈ {0.25, 0.5, 1}, 3단 레버리지 ×1/×2/×3.
사용: python research/r94_straddle_hedge.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
IDX = pd.date_range("2022-01-01", "2026-09-18", freq="D", tz="UTC")
sh = lambda x: x.mean() / x.std() * np.sqrt(365)

if __name__ == "__main__":
    A = pd.read_parquet(R / "data" / "r93_straddle.parquet").dropna(subset=["pr"])
    A["ent"] = pd.DatetimeIndex(A.ent); A["h"] = A.ent.dt.hour; A["d"] = A.ent.dt.floor("D")
    rules = {
        "S0": A[(A.h == 4)].assign(r=lambda x: x.S),
        "S1": A[(A.h == 4) & (A.pr < 0.8)].assign(r=lambda x: x.S),
        "S2": A[(A.h <= 7) & (A["T"] <= 8) & (A.pr < 0.5)].assign(r=lambda x: x.S / 8),
        "S3": A[(A.h <= 7) & (A["T"] <= 8) & (A.pr < 0.8)].assign(r=lambda x: x.S / 8),
        "L8": A[(A.h == 0) & (A["T"] == 8) & (A.pr >= 0.95)].assign(r=lambda x: x.L),
    }
    rules["H"] = pd.concat([rules["S3"], rules["L8"]])
    D = {k: v.groupby("d").r.sum().reindex(IDX).fillna(0) for k, v in rules.items()}
    import r40_regime_leverage as RL, r80_max_return as R80
    RL.MMR = 0.0065; W = R80.dip_frame(0.0025)
    mult = np.minimum(np.clip(W.vreg.values ** 1.5, 0.3, 4) * np.clip(W.pred.values / 0.001, 0.5, 3), 6)
    dip = R80.daily_from_eq(RL.sim(W, 1, 4, mult, X=0.03, H=720)[0]); dip.index = dip.index.normalize(); dip = dip.reindex(IDX).fillna(0)
    P(f"z0.31(w1N4) 샤프 학습 {sh(dip[:'2023-12-31']):.2f} / 평가 {sh(dip['2024-01-01':]):.2f}")
    P("\n===== 규칙 단독 (기초 명목 1, 일간)")
    for k, d in D.items():
        n = len(rules[k]); tr, te = d[:"2023-12-31"], d["2024-01-01":]
        P(f"  {k}: 건수 {n:5d} 거래당 {rules[k].r.mean()*1e4:+6.1f}bp | 샤프 학습 {sh(tr):5.2f} 평가 {sh(te):5.2f} | 연수익 학습 {tr.mean()*365*100:+6.1f}% 평가 {te.mean()*365*100:+6.1f}% "
          f"| 최악의 날 {d.min()*100:+.2f}% | z0.31 상관 {np.corrcoef(d, dip)[0,1]:+.2f} (평가 {np.corrcoef(te, dip['2024-01-01':])[0,1]:+.2f}) | 연도별 샤프 " +
          " ".join(f"{y%100}:{sh(g):+.1f}" for y, g in d.groupby(d.index.year)))
    P("\n===== z0.31 + k × 규칙 (규칙 변동성을 2022~23 z0.31에 맞춤), 평가 2024~26")
    for k_ in ("S1", "S3", "H"):
        f = D[k_] * dip[:"2023-12-31"].std() / D[k_][:"2023-12-31"].std()
        for w in (0.0, 0.25, 0.5, 1.0):
            r = dip + w * f; tr, te = r[:"2023-12-31"], r["2024-01-01":]; out = []
            for L in (1, 2, 3):
                eq = (1 + (te * L).clip(lower=-0.99)).cumprod(); yrs = len(te) / 365.25
                out.append(f"×{L} 월 {(eq.iloc[-1]**(1/12/yrs)-1)*100:+.1f}% MDD {(eq/eq.cummax()-1).min()*100:.0f}%")
            P(f"  z0.31 + {w:.2f}×{k_}: 샤프 학습 {sh(tr):.2f} 평가 {sh(te):.2f} | " + " | ".join(out))
    (R / "logs" / "r94.summary").write_text("done\n")
