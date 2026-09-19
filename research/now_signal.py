"""현재 시점의 결합 H 추세 노출(참고용, 거래 신호 아님)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import screen as S  # noqa: E402
import screen2 as S2  # noqa: E402
import combo as CB  # noqa: E402

for tf in ("4h", "1d"):
    for sym in S.SYMS:
        x = S2.CarryCtx(sym, tf)
        e = CB.trend_exposure(x, "ALL", None)
        print(tf, sym, "마지막 봉", x.idx[-1], "종가", round(float(x.C.iloc[-1]), 1), "추세 노출(0~1)", round(float(e[-1]), 2),
              "| 최근 10봉 평균", round(float(e[-10:].mean()), 2))
