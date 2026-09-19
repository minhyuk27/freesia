"""양방향 100배(롱+숏 동시) 구조의 실제 기댓값. 청산거리 = 1/L - 0.4%."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R))
from freesia import store

P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
for sym in ("BTCUSDT",):
    mk = store.load("mark_1m", sym).set_index("ts").close
    P(f"### {sym} 마크가격 1분봉 {len(mk):,}개")
    for L, hold in ((100, "1D"), (100, "7D"), (50, "1D"), (20, "1D")):
        dist = 1 / L - 0.004
        g = mk.resample(hold if hold != "1D" else "1D")
        first = g.first(); hi = g.max(); lo = g.min()
        up = hi / first - 1; dn = 1 - lo / first
        liq_s = up >= dist          # 숏 청산
        liq_l = dn >= dist          # 롱 청산
        both = (liq_s & liq_l)
        one = (liq_s ^ liq_l)
        none = ~(liq_s | liq_l)
        # 한쪽만 청산된 경우 살아남은 쪽의 구간 말 손익(명목 L배)
        last = g.last()
        ret = last / first - 1
        surv = np.where(liq_s & ~liq_l, ret * L, np.where(liq_l & ~liq_s, -ret * L, 0.0))
        # 자본 1 기준: 증거금 각 0.1 → 청산 시 -0.1, 생존 시 0.1*(1+surv) 반환
        pnl = np.where(both, -0.2,
              np.where(one, -0.1 + 0.1 * np.clip(1 + surv, 0, None) - 0.1,
                       0.1 * np.clip(1 + ret * L, 0, None) + 0.1 * np.clip(1 - ret * L, 0, None) - 0.2))
        P(f"\n[{L}배 · 보유 {hold}] 청산거리 {dist*100:.2f}%  (표본 {len(first)}구간)")
        P(f"  양쪽 다 청산 {both.mean():6.1%} | 한쪽만 {one.mean():6.1%} | 둘 다 생존 {none.mean():6.1%}")
        P(f"  자본 1억·각 1000만원 기준 평균 손익 {np.nanmean(pnl)*1e8/1e4:,.0f}만원, 중앙 {np.nanmedian(pnl)*1e8/1e4:,.0f}만원")
        P(f"  이익 구간 비율 {np.nanmean(pnl>0):.1%}, 최대 이익 {np.nanmax(pnl)*1e8/1e4:,.0f}만원, 최대 손실 {np.nanmin(pnl)*1e8/1e4:,.0f}만원")
