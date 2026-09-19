"""가장 단순한 가설: 조건 없이, 진입 깊이 x의 절대 크기만으로 수익이 결정되는가.
봉 시가 대비 고정 x% 하락 시 지정가 매수 → 종가 청산. 조건 일절 없음.
사용: python research/depth_only.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from lib_dip import bars
from x_matched import run3
TE = pd.Timestamp("2024-01-01", tz="UTC")
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)

for sym in ("BTCUSDT", "ETHUSDT"):
    P("=" * 98 + f"\n### {sym}")
    for B, bn in ((60, "1h"), (240, "4h"), (480, "8h"), (1440, "1d")):
        D = bars(sym, B)
        P(f"\n── {bn}봉 (조건 없음, 비용 7bp) ──")
        rows = []
        for xbp in (25, 50, 75, 100, 150, 200, 300, 400, 500, 700, 1000):
            xa = np.full(D["n"], xbp / 1e4)
            cd = np.ones(D["n"], bool); cd[:10] = False
            ret, oc, xs = run3(D["o"], D["h"], D["l"], D["c"], B, xa, cd, 7.0)
            ok = ~np.isnan(ret); ts = D["bi"][ok]
            if ok.sum() < 60:
                continue
            tr = ts < TE; te = ~tr
            if tr.sum() < 30 or te.sum() < 25:
                continue
            a, b_ = ret[ok][tr], ret[ok][te]
            rows.append((xbp, float(ok.mean()), a.mean(), int(tr.sum()), b_.mean(),
                         float((b_ > 0).mean()), int(te.sum()),
                         b_.mean() / b_.std() * np.sqrt(len(b_)) if b_.std() > 0 else 0,
                         te.sum() / max((ts[te][-1] - ts[te][0]).days / 7, 1),
                         oc[ok][te].mean()))
        P("  " + pd.DataFrame(rows, columns=["x bp", "발생률", "학습bp", "학습n", "테스트bp", "승률",
                                             "테스트n", "t", "주당", "시가→종가bp"]).round(2).to_string(index=False).replace("\n", "\n  "))
P("\n" + "=" * 98)
P("[통합] 두 종목 × 4h·8h 를 같은 x로 동시 운용")
for xbp in (100, 150, 200, 300, 400, 500):
    ss = []
    for sym in ("BTCUSDT", "ETHUSDT"):
        for B in (240, 480):
            D = bars(sym, B)
            xa = np.full(D["n"], xbp / 1e4); cd = np.ones(D["n"], bool); cd[:10] = False
            ret, oc, xs = run3(D["o"], D["h"], D["l"], D["c"], B, xa, cd, 7.0)
            ok = ~np.isnan(ret)
            ss.append(pd.Series(ret[ok], index=D["bi"][ok]))
    a_ = pd.concat(ss).sort_index()
    tr = a_[a_.index < TE]; te = a_[a_.index >= TE]
    if len(te) < 40:
        continue
    yr = te.groupby(te.index.year).mean()
    P(f"  x={xbp:4d}bp | 학습 {tr.mean():7.2f}bp(n{len(tr):5d}) | 테스트 {te.mean():7.2f}bp(n{len(te):5d}, 승률 {float((te>0).mean()):.3f}, "
      f"t {te.mean()/te.std()*np.sqrt(len(te)):5.2f}, 주{len(te)/((te.index[-1]-te.index[0]).days/7):5.2f}) | 연도별 "
      + " ".join(f"{y}:{v:+.0f}" for y, v in yr.items()))
