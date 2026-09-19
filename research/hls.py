"""결합 H의 롱숏 확장: 추세 신호 e∈[-1,1]. 현물 1을 항상 보유하고 무기한 숏 비율 h=1-e (0~2). h>1이면 순수 숏 노출."""
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


def exposure_ls(x):
    c = x.C.to_numpy()
    ts = np.mean([np.sign((x.C / x.C.shift(x.nb(h)) - 1).fillna(0).to_numpy()) for h in CB.HS], axis=0)
    fs, ss = [x.nb(h) for h in (12, 24, 48, 96)], [x.nb(h) for h in (168, 336, 720, 1440)]
    ma = np.mean([np.sign((x.C.rolling(f).mean() - x.C.rolling(s).mean()).fillna(0).to_numpy()) for f in fs for s in ss if f < s], axis=0)
    dc = []
    for h in [48, 168, 336, 720]:
        n = x.nb(h); hh = x.C.shift(1).rolling(n).max().to_numpy(); ll = x.C.shift(1).rolling(n).min().to_numpy()
        dc.append(S.hold(c > hh, None, c < ll))
    return (ts + ma + np.mean(dc, axis=0)) / 3


def h_series(x, e, cap=1.2):
    r = x.r_price
    p = np.concatenate([[0.0], e[:-1]]); p2 = np.concatenate([[0.0], p[:-1]])
    hedge, hedge_prev = 1 - p, 1 - p2
    basis = x.ret * 1.2
    return (p * r + hedge * basis) / cap - np.abs(hedge - hedge_prev) * core.COST / cap


out = {}
for sym in S.SYMS:
    x = S2.CarryCtx(sym, "4h"); x.r_price = x.C.pct_change().fillna(0).to_numpy()
    yrs = x.idx.year.to_numpy(); ppy = x.ppy
    e_lf = CB.trend_exposure(x, "ALL", None)
    e_ls = exposure_ls(x)
    e_half = np.where(e_ls < 0, 0.5 * e_ls, e_ls)        # 숏은 절반만
    for nm, e in (("LF (롱플랫)", e_lf), ("LS (롱숏)", e_ls), ("LS 숏 절반", e_half)):
        for cap in (1.2, 1.5):
            H = h_series(x, e, cap)
            m = yrs >= 2022
            out[(sym, nm, cap)] = (CB.stats(H, ppy, "4h"), CB.stats(H[m], ppy, "4h"),
                                   {int(Y): round(float(H[yrs == Y].sum() * 100), 1) for Y in sorted(set(yrs))})
for k, v in out.items():
    print(f"{k[0][:3]} {k[1]:12} 자본{k[2]}: 전체 Sh {v[0]['sharpe']:5} ann {v[0]['ann']:6.3f} mdd {v[0]['mdd']:7.3f} | 2022+ Sh {v[1]['sharpe']:5} ann {v[1]['ann']:6.3f} mdd {v[1]['mdd']:7.3f} | 연도별% {v[2]}")
