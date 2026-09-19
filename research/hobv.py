"""결합 H에 OBV 신호를 넣은 변형 (무선택 앙상블: OBV 기울기 부호를 여러 창에서 평균)."""
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
res, blend = {}, {}
for sym in S.SYMS:
    x = S2.CarryCtx(sym, tf); x.r_price = x.C.pct_change().fillna(0).to_numpy()
    b = core.bars(sym, tf).reindex(x.idx)
    obv = (np.sign(x.C.diff().fillna(0)) * b.quote_volume.fillna(0)).cumsum()
    obv_sig = np.mean([np.sign((obv - obv.shift(x.nb(h))).fillna(0).to_numpy()) for h in (168, 336, 720, 1440)], axis=0)
    e_obv = np.clip(obv_sig, 0, 1)
    e_all = CB.trend_exposure(x, "ALL", None)
    e_comb = 0.5 * e_obv + 0.5 * e_all
    yrs = x.idx.year.to_numpy(); ppy = x.ppy
    for nm, e in (("ALL", e_all), ("OBV", e_obv), ("ALL+OBV", e_comb)):
        H = CB.series(x, e)[0]
        blend.setdefault(nm, []).append(pd.Series(H, index=x.idx))
        res[(sym, nm)] = (CB.stats(H, ppy, tf), CB.stats(H[yrs >= 2022], ppy, tf), round(float(e.mean()), 2),
                          {int(Y): round(float(H[yrs == Y].sum() * 100), 1) for Y in sorted(set(yrs))})
for k, v in res.items():
    print(f"{k[0][:3]} H[{k[1]:8}] 전체 Sh {v[0]['sharpe']:5} ann {v[0]['ann']:6.3f} mdd {v[0]['mdd']:7.3f} | 2022+ Sh {v[1]['sharpe']:5} ann {v[1]['ann']:6.3f} mdd {v[1]['mdd']:7.3f} | 평균노출 {v[2]} | {v[3]}")
for nm, l in blend.items():
    h = pd.concat(l, axis=1).dropna().mean(axis=1); y = h.index.year.to_numpy(); ppy = core.PPY[tf]
    print(f"BTC+ETH 혼합 H[{nm}]: 전체", CB.stats(h.to_numpy(), ppy, tf), "| 2022+", CB.stats(h.to_numpy()[y >= 2022], ppy, tf))
