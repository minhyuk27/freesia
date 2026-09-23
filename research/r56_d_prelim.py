"""R56 — D 특징(옵션·코인베이스) 예비 점검: 수집된 구간만으로 단독 예측력을 본다(최종 판정 아님).

R54 특징 ⋈ R48 라벨. 특징마다 5분위(경계 = 앞 절반 기간) → 뒤 절반 기간에서 상단/하단 꼬리의 '위 먼저' 비율.
가격 특징과의 독립성: 24시간 수익(r1440)과의 상관.
사용: python research/r56_d_prelim.py"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
R = Path(__file__).resolve().parents[1]
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
L = pd.read_parquet(R / "data" / "r48_labels.parquet")
F = pd.read_parquet(R / "data" / "r54_dfeat.parquet")
D = L.merge(F, on=["sym", "t"])
DF = [c for c in F.columns if c not in ("sym", "t")]
for s, g in D.groupby("sym"):
    for y in ("y10", "y15"):
        P(f"\n===== {s} {y}")
        for f in DF:
            x = g[[f, y, "ts", "r1440"]].dropna()
            if len(x) < 5000: continue
            mid = x.ts.quantile(0.5)
            tr, te = x[x.ts < mid], x[(x.ts >= mid) & (x[y] != 0)]
            for q in (0.05, 0.01):
                hi, lo = tr[f].quantile(1 - q), tr[f].quantile(q)
                a = (te[te[f] >= hi][y] == 1).mean(); b = (te[te[f] <= lo][y] == 1).mean()
                na, nb = (te[f] >= hi).sum(), (te[f] <= lo).sum()
                P(f"  {f:7s} q{q*100:3.0f}%: 상단 '위먼저' {a:.3f}(n={na}) 하단 {b:.3f}(n={nb}) | 기간 {x.ts.min():%Y-%m}~{x.ts.max():%Y-%m} r1440상관 {x[f].corr(x.r1440):+.2f}")
