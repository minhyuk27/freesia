"""결합 H 수익 분해: 미헤지 구간 가격손익(추세) vs 헤지 구간 캐리(펀딩-베이시스) vs 비용. BTC+ETH 혼합, 4시간, 자본 1.2."""
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

rows = []
for sym in S.SYMS:
    x = S2.CarryCtx(sym, "4h"); r = x.C.pct_change().fillna(0).to_numpy()
    e = CB.trend_exposure(x, "ALL", None)
    p = np.concatenate([[0.0], e[:-1]]); p2 = np.concatenate([[0.0], p[:-1]])
    hedge, hp = 1 - p, 1 - p2
    price = p * r / 1.2
    carry = hedge * (x.ret * 1.2) / 1.2
    cost = -np.abs(hedge - hp) * core.COST / 1.2
    df = pd.DataFrame({"가격(추세)": price, "캐리(헤지 구간)": carry, "비용": cost}, index=x.idx)
    rows.append(df)
d = sum(rows) / 2
d["연"] = d.index.year
t = d.groupby("연").sum() * 100
t["합계"] = t.sum(axis=1)
print((t.round(1)).to_string())
print("전체 합(%):", (d.drop(columns='연').sum() * 100).round(1).to_dict())
print("연평균(%):", (d.drop(columns='연').sum() / (len(d) / (6 * 365)) * 100).round(1).to_dict())
