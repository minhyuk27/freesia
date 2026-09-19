"""결합 H 헤지 비율을 자주 바꾸지 않는 밴드 규칙: 보유 노출과 신호 차이가 band 이상일 때만 조정."""
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


def banded(e, band):
    out = np.zeros(len(e)); cur = 0.0
    for i, v in enumerate(e):
        if abs(v - cur) >= band or (v in (0.0, 1.0) and v != cur and abs(v - cur) >= band / 2):
            cur = v
        out[i] = cur
    return out


parts = {b: [] for b in (0.0, 0.1, 0.2, 0.34, 0.5)}
for sym in S.SYMS:
    x = S2.CarryCtx(sym, "4h"); x.r_price = x.C.pct_change().fillna(0).to_numpy()
    e0 = CB.trend_exposure(x, "ALL", None)
    for b in parts:
        e = e0 if b == 0 else banded(e0, b)
        parts[b].append(pd.Series(CB.series(x, e)[0], index=x.idx))
        parts[b][-1].attrs["turn"] = core.turnover_stats(e, "4h")
ppy = 6 * 365
for b, l in parts.items():
    h = pd.concat(l, axis=1).dropna().mean(axis=1); y = h.index.year.to_numpy()
    print(f"band {b:4}: 전체 {CB.stats(h.to_numpy(), ppy, '4h')} | 2022+ {CB.stats(h.to_numpy()[y >= 2022], ppy, '4h')} | 연 회전(편도합) {np.mean([s.attrs['turn'] for s in l]):.0f}")
