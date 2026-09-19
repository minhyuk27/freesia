"""최신 꼬리 이어붙이기: 각 데이터셋의 마지막 시각 다음부터 현재까지 REST로 받아 합친다.
Vision 은 하루 이상 늦게 올라오므로 실시간에 가까운 데이터는 이 스크립트가 책임진다."""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freesia import rest, store  # noqa: E402

now_ms = int(time.time() * 1000)
for sym in ("BTCUSDT", "ETHUSDT"):
    for name in ("klines_1m", "mark_1m", "premium_1m"):
        df = store.load(name, sym)
        start = int(df.ts.iloc[-1].timestamp() * 1000) + 60_000
        n = store.merge(name, sym, rest.klines(name, sym, start, now_ms))
        print(name, sym, "→ 파티션 행수", n, flush=True)
    df = store.load("funding", sym)
    start = int(df.ts.iloc[-1].timestamp() * 1000) + 1000
    store.merge("funding", sym, rest.funding(sym, start, now_ms))
print("update_rest 완료", pd.Timestamp.now(tz="UTC"))
