"""R44 — 학습 없는 단순 딥 규칙 + 변동성 국면 레버리지 = '단순하지만 공격적인' 후보.

규칙(파라미터 3개, 학습 없음): 봉 B분 시가 − k×rv×√B 에 지정가 매수, 같은 봉 종가 청산 (z0.14 계열).
  B ∈ {10, 15, 30}, k ∈ {6, 8} 사건을 전부 사용(ML 필터 없음).
사이징: 명목 = 자본 × w × clip((30일 변동성/1년 중앙)^α, 0.3, 4), α ∈ {0, 2}. 킬스위치 K1 포함.
평가: 2022-01 ~ (R27 워크포워드와 같은 구간), 청산 지연 1.2bp 반영.
사용: python research/r44_simple_dip_regime.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
import r40_regime_leverage as R40
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
E = pd.read_parquet(R / "data" / "r32_events.parquet", columns=["sym", "B", "k", "f", "x", "ret", "mae", "ts"])
E["ts"] = pd.DatetimeIndex(E.ts)
E = E[(E.ts >= "2022-01-01") & E.B.isin([10, 15, 30]) & E.k.isin([6.0, 8.0])].sort_values(["f", "sym"]).reset_index(drop=True)
E["ret"] = E.ret - 1.2e-4
for s in ("BTCUSDT", "ETHUSDT"):
    k = store.load("klines_1m", s).set_index("ts").close.resample("D").last()
    r = np.log(k).diff(); v30 = r.rolling(30).std(); vmed = v30.rolling(365, min_periods=120).median()
    m = E.sym == s
    E.loc[m, "vreg"] = (v30 / vmed).shift(1).reindex(E.loc[m, "ts"].dt.normalize()).values
E["vreg"] = E.vreg.fillna(1.0)
R40.YRS = (E.ts.max() - E.ts.min()).days / 365.25
R40.W = E
P(f"단순 딥 사건 {len(E)/R40.YRS:.0f}건/년, 거래당 {E.ret.mean()*1e4:+.1f}bp, 승률 {(E.ret>0).mean():.3f}")
by = E.groupby(E.ts.dt.year).ret.agg(["mean", "size"])
P("연도별 거래당 " + " ".join(f"{y}:{r['mean']*1e4:+.1f}bp×{int(r['size'])}" for y, r in by.iterrows()))
for w, N in ((0.5, 3), (1, 5), (2, 8), (3, 12)):
    P(f"\n── w{w} N{N}")
    R40.show("단순 딥 (국면 없음)", w, N, np.ones(len(E)))
    R40.show("단순 딥 + 변동성국면 α2", w, N, np.clip(E.vreg.values ** 2, 0.3, 4.0))
