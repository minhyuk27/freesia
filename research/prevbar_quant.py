"""저차원 정밀 검정: 직전 봉의 '활발도' 분위별 성과. 과적합 여지를 최소화한 단순 구조.
축1: 직전 봉 범위 분위 / 축2: 직전 봉 위꼬리 분위 / 교차. 두 종목 × 3봉크기.
사용: python research/prevbar_quant.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from lib_dip import bars
from grid_clean import run2
TE = pd.Timestamp("2024-01-01", tz="UTC")
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)

for sym in ("BTCUSDT", "ETHUSDT"):
    P("=" * 96 + f"\n### {sym}")
    for B, bn in ((240, "4h"), (480, "8h"), (1440, "1d")):
        D = bars(sym, B); s = pd.Series
        bo, bc, bh, bl = D["bo"], D["bc"], D["bh"], D["bl"]
        rngL = s((bh - bl) / bo).shift(1)
        uwL = s((bh - np.maximum(bo, bc)) / bo).shift(1)
        qr = rngL.rolling(504).rank(pct=True)      # 직전 봉 범위 분위(과거 504봉 기준)
        qu = uwL.rolling(504).rank(pct=True)
        xa = (rngL * 0.75).values
        P(f"\n── {bn}봉, 깊이 직전범위 0.75배, 비용 7bp ──")
        P("  [A] 직전 봉 범위 분위별")
        rows = []
        for a_, b2 in ((0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)):
            cd = ((qr >= a_) & (qr < b2)).fillna(False).values
            ret = run2(D["o"], D["h"], D["l"], D["c"], B, np.asarray(xa, float), np.asarray(cd, np.bool_), 1, 7.0)
            ok = ~np.isnan(ret); ts = D["bi"][ok]
            if ok.sum() < 60:
                continue
            tr = ts < TE; te = ~tr
            if tr.sum() < 30 or te.sum() < 25:
                continue
            x, y = ret[ok][tr], ret[ok][te]
            rows.append((f"{int(a_*100)}~{int(b2*100)}%", x.mean(), int(tr.sum()), y.mean(),
                         float((y > 0).mean()), int(te.sum()),
                         y.mean() / y.std() * np.sqrt(len(y)) if y.std() > 0 else 0,
                         te.sum() / max((ts[te][-1] - ts[te][0]).days / 7, 1)))
        P("  " + pd.DataFrame(rows, columns=["범위분위", "학습bp", "학습n", "테스트bp", "승률", "테스트n", "t", "주당"]).round(2).to_string(index=False).replace("\n", "\n  "))
        P("  [B] 직전 봉 위꼬리 분위별")
        rows = []
        for a_, b2 in ((0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)):
            cd = ((qu >= a_) & (qu < b2)).fillna(False).values
            ret = run2(D["o"], D["h"], D["l"], D["c"], B, np.asarray(xa, float), np.asarray(cd, np.bool_), 1, 7.0)
            ok = ~np.isnan(ret); ts = D["bi"][ok]
            if ok.sum() < 60:
                continue
            tr = ts < TE; te = ~tr
            if tr.sum() < 30 or te.sum() < 25:
                continue
            x, y = ret[ok][tr], ret[ok][te]
            rows.append((f"{int(a_*100)}~{int(b2*100)}%", x.mean(), int(tr.sum()), y.mean(),
                         float((y > 0).mean()), int(te.sum()),
                         y.mean() / y.std() * np.sqrt(len(y)) if y.std() > 0 else 0,
                         te.sum() / max((ts[te][-1] - ts[te][0]).days / 7, 1)))
        P("  " + pd.DataFrame(rows, columns=["위꼬리분위", "학습bp", "학습n", "테스트bp", "승률", "테스트n", "t", "주당"]).round(2).to_string(index=False).replace("\n", "\n  "))
