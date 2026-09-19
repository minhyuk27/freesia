"""헤지(무기한 숏) 구간의 필요 증거금: 임의 시점에서 H시간 안에 마크가격이 오를 수 있는 최대폭 분포."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freesia import store  # noqa: E402

for sym in ("BTCUSDT", "ETHUSDT"):
    m = store.load("mark_1m", sym).set_index("ts")
    m = m[m.index >= "2020-01-01"]
    h = m.resample("1h").agg({"high": "max", "close": "last"}).dropna()
    hi, cl = h.high.to_numpy(), h.close.to_numpy()
    print("==", sym)
    for label, H in (("24시간", 24), ("3일", 72), ("7일", 168), ("14일", 336)):
        fut = pd.Series(hi[::-1]).rolling(H, min_periods=H).max().to_numpy()[::-1]
        up = fut[1:] / cl[:-1] - 1
        up = up[~np.isnan(up)]
        print(f"  {label:6} 최대 상승폭: 99% {np.percentile(up, 99)*100:5.1f}%  99.9% {np.percentile(up, 99.9)*100:5.1f}%  최대 {up.max()*100:5.1f}%  | 2022~: 99.9% {np.percentile(up[-int(len(up)*0.6):], 99.9)*100:5.1f}%")
