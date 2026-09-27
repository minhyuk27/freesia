"""R96 — 변동성 관리(Moreira·Muir 2017)를 p로: 예보 변동성이 클 때 노출을 줄이는 방식이 기존 일간 조각의 샤프를 올리는가.
조각: R89b 스테이블 계열 10개(롱숏, 지연 1일) 바구니, BTC·ETH 단순 보유, R79 없이 선물만.
배율(날짜 D, D−1까지 정보): P  = clip(중앙(p 학습) / 평균p(D−1), 0.25, 3)
                           P2 = P²  (분산 기준)
                           RV = clip(중앙(rv 학습) / rv30(D−1), 0.25, 3)  (표준 실현변동성 관리, 비교용)
                           PRV = P·RV 기하평균
평가: 샤프 2022~23 / 2024~26 (p는 2022년부터).
사용: python research/r96_vol_managed.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
import r89_flows as F, r92_vol_timing as V
from freesia import store
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
sh = lambda x: x.mean() / x.std() * np.sqrt(365)
IDX = pd.date_range("2022-01-01", "2026-09-17", freq="D", tz="UTC")

if __name__ == "__main__":
    S = F.signals(); Wp = V.load_p()
    pd_ = {s: g.set_index("ts").p.resample("D").mean().shift(1).reindex(IDX) for s, g in Wp.groupby("sym")}
    px = {s: store.load("klines_1m", s).set_index("ts").close.resample("1D").last() for s in ("BTCUSDT", "ETHUSDT")}
    ret = {s: np.log(px[s]).diff().shift(-1).reindex(IDX) for s in px}
    rv = {s: np.log(px[s]).diff().rolling(30).std().reindex(IDX) for s in px}      # D 종가까지 → D의 포지션은 D−1 값 사용
    rv = {s: v.shift(1) for s, v in rv.items()}
    base = {}
    for s in px:
        tag = "B" if s == "BTCUSDT" else "E"
        for nm in ("ST7", "ST30", "USDT7", "USDC7", "CMB"):
            z = (S["ST7"] + S["NF" + tag]) / 2 if nm == "CMB" else S[nm]
            base[(s, nm)] = (z.shift(1).reindex(IDX).clip(-2, 2) / 2).fillna(0)
        base[(s, "HOLD")] = pd.Series(1.0, index=IDX)
    tr = slice("2022-01-01", "2023-12-31")
    res = {}
    for mode in ("없음", "P", "P2", "RV", "PRV"):
        sl = {}
        for (s, nm), pos in base.items():
            fP = (pd_[s][tr].median() / pd_[s]).clip(0.25, 3).fillna(1); fR = (rv[s][tr].median() / rv[s]).clip(0.25, 3).fillna(1)
            f = {"없음": 1, "P": fP, "P2": (fP ** 2).clip(0.25, 3), "RV": fR, "PRV": np.sqrt(fP * fR)}[mode]
            p = pos * f; sl[(s, nm)] = (p * ret[s] - p.diff().abs().fillna(0) * 7e-4).fillna(0)
        M = pd.DataFrame(sl); res[mode] = M
        st = [c for c in M.columns if c[1] != "HOLD"]; w = 1 / M[st][tr].std(); fb = (M[st] * (w / w.sum())).sum(axis=1)
        P(f"[{mode:3s}] 스테이블 바구니 샤프 학습 {sh(fb[tr]):.2f} 평가 {sh(fb['2024-01-01':]):.2f} | BTC 보유 {sh(M[('BTCUSDT','HOLD')][tr]):.2f}/{sh(M[('BTCUSDT','HOLD')]['2024-01-01':]):.2f}"
          f" ETH 보유 {sh(M[('ETHUSDT','HOLD')][tr]):.2f}/{sh(M[('ETHUSDT','HOLD')]['2024-01-01':]):.2f} | 조각별 평가 샤프 " +
          " ".join(f"{c[0][:1]}{c[1]}:{sh(M[c]['2024-01-01':]):+.2f}" for c in st))
    (R / "logs" / "r96.summary").write_text("done\n")
