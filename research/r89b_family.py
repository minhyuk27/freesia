"""R89b — 스테이블코인 계열 전체를 사전 고정 바구니로(선택 편향 없이): ST7·ST30·USDT7·USDC7·CMB × BTC·ETH 롱숏, 지연 1일.
비중 = 2020~23 위험 균등. 평가 2024~26, z0.31(w1N4) 덧얹기.
사용: python research/r89b_family.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
import r89_flows as F
from freesia import store
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
sh = lambda x: x.mean() / x.std() * np.sqrt(365)

if __name__ == "__main__":
    S = F.signals(); IDX = F.IDX
    px = {s: store.load("klines_1m", s).set_index("ts").close.resample("1D").last() for s in ("BTCUSDT", "ETHUSDT")}
    ret = {s: np.log(px[s]).diff().shift(-1).reindex(IDX) for s in px}
    sl = {}
    for s in px:
        tag = "B" if s == "BTCUSDT" else "E"
        for nm in ("ST7", "ST30", "USDT7", "USDC7", "CMB"):
            z = (S["ST7"] + S["NF" + tag]) / 2 if nm == "CMB" else S[nm]
            p = (z.shift(1).reindex(IDX).clip(-2, 2) / 2).fillna(0)
            sl[f"{s[:3]}_{nm}"] = (p * ret[s] - p.diff().abs().fillna(0) * 7e-4).fillna(0)
    M = pd.DataFrame(sl); tr = M[:"2023-12-31"]; w = 1 / tr.std(); w = w / w.sum(); fb = (M * w).sum(axis=1)
    P(f"스테이블 계열 바구니(10개) 샤프: 2020~23 {sh(fb[:'2023-12-31']):.2f} / 2024~26 {sh(fb['2024-01-01':]):.2f} | 연도별 " +
      " ".join(f"{y%100}:{sh(g):.2f}" for y, g in fb.groupby(fb.index.year)))
    import r40_regime_leverage as RL, r80_max_return as R80
    RL.MMR = 0.0065; W = R80.dip_frame(0.0025)
    mult = np.minimum(np.clip(W.vreg.values ** 1.5, 0.3, 4) * np.clip(W.pred.values / 0.001, 0.5, 3), 6)
    dip = R80.daily_from_eq(RL.sim(W, 1, 4, mult, X=0.03, H=720)[0]); dip.index = dip.index.normalize(); dip = dip.reindex(IDX).fillna(0)
    fb = fb * dip["2022":"2023"].std() / fb["2022":"2023"].std()
    dt, ft = dip["2024-01-01":], fb["2024-01-01":]
    P(f"z0.31 상관 {np.corrcoef(dt, ft)[0,1]:+.2f}")
    for k in (0, 0.25, 0.5, 0.75, 1.0):
        r = dt + k * ft; out = []
        for L in (1, 2, 3):
            eq = (1 + (r * L).clip(lower=-0.99)).cumprod(); yrs = (eq.index[-1] - eq.index[0]).days / 365.25
            out.append(f"×{L} 월 {(eq.iloc[-1]**(1/12/yrs)-1)*100:+.1f}% MDD {(eq/eq.cummax()-1).min()*100:.0f}%")
        P(f"  z0.31 + {k:.2f}×스테이블 바구니: 샤프 {sh(r):.2f} | " + " | ".join(out))
