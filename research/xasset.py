"""추세 앙상블(롱플랫)이 다른 자산에서도 되는지 (참고용). Binance 공개 일봉. ⚠️ 현재 상장·생존 종목만 → 생존 편향 있음. 펀딩 미반영."""
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from freesia import vision as v  # noqa: E402
from freesia.bt import core  # noqa: E402
import combo as CB  # noqa: E402


class X:
    def __init__(self, C):
        self.C = C; self.idx = C.index; self.hb = 24; self.ppy = 365
        self.r = C.pct_change().fillna(0)

    def nb(self, h):
        return max(1, int(round(h / 24)))


def load(sym):
    fr = []
    y, m = 2020, 1
    while (y, m) <= (2026, 8):
        b = v.fetch_zip(f"https://data.binance.vision/data/futures/um/monthly/klines/{sym}/1d/{sym}-1d-{y}-{m:02d}.zip")
        if b is not None:
            fr.append(v.parse_klines(b, True))
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return pd.concat(fr).drop_duplicates("ts").set_index("ts").close if fr else None


rows = []
for sym in ["BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "LTCUSDT", "LINKUSDT", "AVAXUSDT", "DOTUSDT", "TRXUSDT", "BTCUSDT", "ETHUSDT"]:
    C = load(sym)
    if C is None or len(C) < 800:
        continue
    x = X(C)
    e = CB.trend_exposure(x, "ALL", None)
    net = core.run(e, C.to_numpy(), np.zeros(len(C)))
    bh = x.r.to_numpy(); y = C.index.year.to_numpy()
    f = lambda a, m=slice(None): (a[m].mean() / a[m].std() * np.sqrt(365)) if a[m].std() > 0 else 0  # noqa: E731
    m22 = y >= 2022
    rows.append((sym, len(C), C.index[0].date(), f(net), f(bh), f(net, m22), f(bh, m22),
                 core.metrics(net, "1d")["mdd"], core.metrics(bh, "1d")["mdd"]))
print(f"{'종목':9}{'일수':>5} 시작       | 추세 Sh  BH Sh | 2022+ 추세 BH | MDD 추세 / BH")
for r in rows:
    print(f"{r[0]:9}{r[1]:5} {r[2]} | {r[3]:6.2f} {r[4]:6.2f} | {r[5]:6.2f} {r[6]:6.2f} | {r[7]:6.2f} {r[8]:6.2f}")
alt = [r for r in rows if r[0] not in ("BTCUSDT", "ETHUSDT")]
print("알트 %d개: 추세 Sharpe 평균 %.2f (BH %.2f), 추세>BH 종목 %d개, 2022+ 추세 평균 %.2f (BH %.2f), MDD 개선 종목 %d개" % (
    len(alt), np.mean([r[3] for r in alt]), np.mean([r[4] for r in alt]), sum(r[3] > r[4] for r in alt),
    np.mean([r[5] for r in alt]), np.mean([r[6] for r in alt]), sum(r[7] > r[8] for r in alt)))
