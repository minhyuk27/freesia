"""결합 H의 변형(추세 신호 5종)을 연도별 워크포워드로 선택 → 표본 밖 성과 (사후 선택 편향 점검)."""
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

tf = sys.argv[1] if len(sys.argv) > 1 else "4h"
allH = {}
for sym in S.SYMS:
    x = S2.CarryCtx(sym, tf); x.r_price = x.C.pct_change().fillna(0).to_numpy()
    cols, names = [], []
    for kind, volt in [("ALL", None), ("ALL", 0.5), ("ts", None), ("ma", None), ("dc", None)]:
        e = CB.trend_exposure(x, kind, volt)
        cols.append(CB.series(x, e)[0]); names.append(f"{kind}{'' if not volt else ' vt'}")
    R = np.array(cols).T
    oos, used, picks = core.walk_forward(R, x.idx, tf, 2022, 1.5)
    ro = oos[used]; ppy = x.ppy; yrs = x.idx.year.to_numpy()[used]
    print(sym, tf, "WF 선택 OOS:", CB.stats(ro, ppy, tf), "연도별", {int(Y): round(CB.sh(ro[yrs == Y], ppy), 2) for Y in sorted(set(yrs))})
    print("   선택:", {int(k): names[v[0]] for k, v in picks.items()})
    print("   변형별 2022+ Sharpe:", {n: round(CB.sh(R[x.idx.year >= 2022, j], ppy), 2) for j, n in enumerate(names)})
    print("   변형 전체 평균(무선택) 2022+:", round(CB.sh(R[x.idx.year >= 2022].mean(1), ppy), 2))
    allH[sym] = pd.Series(oos, index=x.idx)[used]
b = pd.concat(allH, axis=1).dropna().mean(axis=1).to_numpy()
print("BTC+ETH WF 혼합 OOS:", CB.stats(b, core.PPY[tf], tf))
