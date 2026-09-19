"""옛 LW 변동성 돌파 정확 재현: 당일 시가+K×전일 레인지 도달 시 진입(터치 체결), 손절 시가-0.5×전일 레인지, 당일 종가 청산. K=0.6.
ATR 필터는 문서 정의가 모호해 적용하지 않고, 별도로 ATR(14)/가격 하위 분위 조건도 시험하지 않는다. 비용 왕복 12bp, 1배."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freesia.bt import core  # noqa: E402

for K in (0.4, 0.6, 0.8):
    for sym in ("BTCUSDT", "ETHUSDT"):
        b = core.bars(sym, "1d")
        rng = (b.high - b.low).shift(1)
        tgt = b.open + K * rng; stop = b.open - 0.5 * rng
        hit = (b.high >= tgt) & rng.notna()
        stopped = hit & (b.low <= stop)                         # 같은 날 손절 도달 시 손절 우선
        px_out = np.where(stopped, stop, b.close)
        g = np.where(hit, px_out / tgt - 1 - 0.0012, 0.0)
        g = pd.Series(g, index=b.index)
        y = g.index.year; ppy = 365
        sh = lambda a: round(float(a.mean() / a.std() * np.sqrt(ppy)), 2) if a.std() > 0 else 0  # noqa: E731
        tr = g[hit]
        print(f"K={K} {sym[:3]}: 거래 {int(hit.sum())}회(연 {hit.sum()/(len(b)/365):.0f}), 승률 {100*(tr>0).mean():.0f}%, 평균 {100*tr.mean():.2f}% | 전체 Sharpe {sh(g)}, 2022~ {sh(g[y>=2022])}, 연수익 2022~ {g[y>=2022].mean()*365*100:.1f}% | 연도별 {{{', '.join(f'{Y}:{g[y==Y].sum()*100:.0f}%' for Y in sorted(set(y)))}}}")
