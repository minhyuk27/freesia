"""결합 H 강건성: 4시간봉 시작 위상(0~3시간 어긋남), 신호 지연(1·2봉), 노출·회전율."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from freesia import store  # noqa: E402
from freesia.bt import core  # noqa: E402
import screen as S  # noqa: E402
import screen2 as S2  # noqa: E402
import combo as CB  # noqa: E402


class OffCtx(S2.CarryCtx):
    def __init__(self, sym, tf, off):
        k = store.load("klines_1m", sym).set_index("ts")
        b = k.resample("4h", offset=f"{off}h").agg({"open": "first", "high": "max", "low": "min", "close": "last",
                                                     "quote_volume": "sum"}).dropna(subset=["close"])
        self.sym, self.tf, self.b, self.hb, self.idx, self.C, self.ppy = sym, tf, b, 4, b.index, b.close, core.PPY["4h"]
        self.r = self.C.pct_change().fillna(0)
        f = store.load("funding", sym).set_index("ts").rate
        self.fund = f.resample("4h", offset=f"{off}h").sum().reindex(self.idx).fillna(0.0).to_numpy()
        pr = store.load("premium_1m", sym).set_index("ts").close.resample("4h", offset=f"{off}h").last().reindex(self.idx).ffill()
        self.ret = (-pr.diff().fillna(0).to_numpy() + self.fund) / 1.2
        self.r = pd.Series(np.zeros(len(self.idx)), index=self.idx)


def main():
    out = {}
    for sym in S.SYMS:
        out[sym] = {}
        for off in (0, 1, 2, 3):
            x = OffCtx(sym, "4h", off)
            x.r_price = x.C.pct_change().fillna(0).to_numpy()
            e0 = CB.trend_exposure(x, "ALL", None)
            for d in (0, 1, 2):
                e = np.concatenate([np.zeros(d), e0[: len(e0) - d]])
                H, T, C = CB.series(x, e)
                out[sym][f"위상+{off}h 지연{d}봉"] = {"H": CB.stats(H, x.ppy, "4h"), "T": CB.stats(T, x.ppy, "4h"),
                                                   "H_2022+": CB.stats(H[x.idx.year >= 2022], x.ppy, "4h")}
            out[sym][f"위상+{off}h 노출"] = {"평균노출": round(float(e0.mean()), 2), "연회전(편도합)": round(core.turnover_stats(e0, "4h"), 1)}
    Path("logs/hrobust.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
    for sym, r in out.items():
        print("==", sym)
        for k, v in r.items():
            if "H" in v:
                print(f"  {k:16} H 전체 Sh {v['H']['sharpe']:5} ann {v['H']['ann']:6.3f} mdd {v['H']['mdd']:7.3f} | 2022+ Sh {v['H_2022+']['sharpe']:5} | T Sh {v['T']['sharpe']}")
            else:
                print(f"  {k:16} {v}")


if __name__ == "__main__":
    main()
