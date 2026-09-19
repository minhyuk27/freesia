"""감별 진단: 위꼬리가 진짜 신호인가, 단지 '직전 봉이 컸다'(고변동)의 대리인가.
교차 조건으로 분리한다. 사용: python research/wick_vs_range.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from lib_dip import bars
from grid_clean import run2
TE = pd.Timestamp("2024-01-01", tz="UTC")
rng = np.random.default_rng(233)
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)

for sym in ("BTCUSDT", "ETHUSDT"):
    P("=" * 100 + f"\n### {sym}")
    for B, bn in ((240, "4h"), (480, "8h")):
        D = bars(sym, B)
        s = pd.Series
        bo, bc, bh, bl = D["bo"], D["bc"], D["bh"], D["bl"]
        rng_ = s((bh - bl) / bo); uw = s((bh - np.maximum(bo, bc)) / bo); lw = s((np.minimum(bo, bc) - bl) / bo)
        L = lambda x: x.shift(1)
        uwL, rngL, lwL = L(uw), L(rng_), L(lw)
        q = lambda x, p: x.rolling(168).quantile(p)
        hiUW = (uwL > q(uwL, .7)); hiRNG = (rngL > q(rngL, .7)); hiLW = (lwL > q(lwL, .7))
        C = {
            "① 위꼬리 상위30%": hiUW,
            "② 직전범위 상위30%": hiRNG,
            "③ 위꼬리↑ & 범위 보통/작음": hiUW & ~hiRNG,
            "④ 범위↑ & 위꼬리 작음": hiRNG & ~hiUW,
            "⑤ 위꼬리↑ & 범위↑": hiUW & hiRNG,
            "⑥ 아래꼬리 상위30%": hiLW,
            "⑦ 아래꼬리↑ & 범위 보통": hiLW & ~hiRNG,
            "⑧ 둘 다 작음(대조)": ~hiUW & ~hiRNG,
            "⑨ 전체(대조)": pd.Series(np.ones(D["n"], bool)),
        }
        xa = (rngL * 0.75).values
        P(f"\n── {bn}봉, 깊이 직전범위 0.75배 ──")
        rows = []
        for cn, cd in C.items():
            cdv = cd.fillna(False).values if hasattr(cd, "fillna") else np.asarray(cd, bool)
            ret = run2(D["o"], D["h"], D["l"], D["c"], B, np.asarray(xa, float), np.asarray(cdv, np.bool_), 1, 7.0)
            ok = ~np.isnan(ret); ts = D["bi"][ok]
            if ok.sum() < 80:
                continue
            tr = ts < TE; te = ~tr
            if tr.sum() < 40 or te.sum() < 30:
                continue
            a = ret[ok][tr]; b_ = ret[ok][te]
            pl = []
            for _ in range(80):
                cr = np.zeros(D["n"], bool); cr[rng.choice(D["n"], int(cdv.sum()), replace=False)] = True
                r2 = run2(D["o"], D["h"], D["l"], D["c"], B, np.asarray(xa, float), cr, 1, 7.0)
                m2 = ~np.isnan(r2) & (D["bi"] >= TE)
                if m2.sum() > 25:
                    pl.append(r2[m2].mean())
            pct = float((np.array(pl) < b_.mean()).mean() * 100) if pl else np.nan
            rows.append((cn, a.mean(), int(tr.sum()), b_.mean(), float((b_ > 0).mean()), int(te.sum()),
                         b_.mean() / b_.std() * np.sqrt(len(b_)) if b_.std() > 0 else 0,
                         te.sum() / max((ts[te][-1] - ts[te][0]).days / 7, 1), pct))
        P(pd.DataFrame(rows, columns=["조건", "학습bp", "학습n", "테스트bp", "승률", "테스트n", "t", "주당", "위약%"]).round(2).to_string(index=False))
