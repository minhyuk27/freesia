"""결합 H(BTC+ETH 혼합, 4시간) 최악 구간과 주요 사건일 성과."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from freesia.bt import core  # noqa: E402
import screen as S  # noqa: E402
import screen2 as S2  # noqa: E402
import combo as CB  # noqa: E402

parts, bh = [], []
for sym in S.SYMS:
    x = S2.CarryCtx(sym, "4h"); x.r_price = x.C.pct_change().fillna(0).to_numpy()
    H = CB.series(x, CB.trend_exposure(x, "ALL", None))[0]
    parts.append(pd.Series(H, index=x.idx)); bh.append(pd.Series(x.r_price, index=x.idx))
h = pd.concat(parts, axis=1).dropna().mean(axis=1); b = pd.concat(bh, axis=1).dropna().mean(axis=1)
eq = (1 + h).cumprod(); dd = eq / eq.cummax() - 1
print("최대 낙폭 %.1f%%, 종료 %s" % (dd.min() * 100, dd.idxmin().date()))
# 낙폭 구간 상위 3개
ends, used = [], []
d = dd.copy()
for _ in range(3):
    t = d.idxmin(); peak = eq[:t].idxmax()
    rec = eq[t:][eq[t:] >= eq[peak]]
    print(f"낙폭 {d.min()*100:6.1f}%  고점 {peak.date()} → 저점 {t.date()} → 회복 {rec.index[0].date() if len(rec) else '미회복'}")
    d = d.copy(); d[(d.index >= peak) & (d.index <= (rec.index[0] if len(rec) else d.index[-1]))] = 0
for name, a, z in [("2020-03 코로나", "2020-03-01", "2020-03-31"), ("2021-05 급락", "2021-05-01", "2021-05-31"),
                   ("2022-05 LUNA", "2022-05-01", "2022-05-31"), ("2022-11 FTX", "2022-11-01", "2022-11-30"),
                   ("2024-08 엔캐리", "2024-08-01", "2024-08-15"), ("2025-10 급락", "2025-10-01", "2025-10-31")]:
    s = slice(a, z)
    print(f"{name}: H {((1+h[s]).prod()-1)*100:6.1f}% | 매수후보유 {((1+b[s]).prod()-1)*100:6.1f}%")
w = h.resample("1W").sum(); m = h.resample("30D").sum()
print("주간 최악 %.1f%%, 30일 최악 %.1f%%, 30일 양수 비율 %.0f%%" % (w.min() * 100, m.min() * 100, (m > 0).mean() * 100))
