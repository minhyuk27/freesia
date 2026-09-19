"""1분봉 계열의 결손 구간을 REST로 메운다 (Vision 파일에서 통째로 빠진 날 등).
REST로도 못 받는 구간은 남기고 리포트한다(임의 보간 금지)."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freesia import rest, store  # noqa: E402

rep = {}
for name in ("klines_1m", "mark_1m", "premium_1m"):
    for sym in ("BTCUSDT", "ETHUSDT"):
        df = store.load(name, sym)
        if df is None:
            continue
        d = df.ts.diff()
        idx = d[d > pd.Timedelta("1min")].index
        filled = still = 0
        for i in idx:
            a, b = df.ts[i - 1] + pd.Timedelta("1min"), df.ts[i] - pd.Timedelta("1min")
            new = rest.klines(name, sym, int(a.timestamp() * 1000), int(b.timestamp() * 1000))
            want = int((b - a) / pd.Timedelta("1min")) + 1
            got = 0 if new is None else len(new)
            store.merge(name, sym, new)
            filled += got
            still += want - got
        rep[f"{name}/{sym}"] = {"gap_runs": len(idx), "filled": filled, "still_missing": still}
print(json.dumps(rep, indent=1))
