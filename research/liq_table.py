"""배율별 청산거리·청산확률·수수료 부담 표 (마크가격 1분봉 실측).

청산 판정: 마크가격 저가(롱)/고가(숏)가 격리마진 청산가에 닿으면 청산. 진입은 매 정시(1시간 간격) 가정.
호라이즌 H 안에 한 번이라도 닿을 확률 = 그 배율로 H 동안 '아무 대응 없이' 들고 있었을 때의 청산 확률.
(손절 없이 들고 있는 최악 기준. 실제 전략은 손절이 청산보다 먼저 와야 함.)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freesia import costs, store  # noqa: E402

LEV = [5, 10, 20, 25, 50, 75, 100]
HOR = {"1시간": 60, "4시간": 240, "1일": 1440, "7일": 10080}


def fut_extreme(arr, H, fn):
    """각 시점 i 의 (i+1 .. i+H) 구간 극값. 뒤쪽 H개는 NaN."""
    s = pd.Series(arr[::-1]).rolling(H, min_periods=H)
    r = (s.min() if fn == "min" else s.max()).to_numpy()[::-1]
    return np.concatenate([r[1:], [np.nan]])


out = []
for sym in ("BTCUSDT", "ETHUSDT"):
    m = store.load("mark_1m", sym).set_index("ts")
    m = m[m.index >= "2020-01-01"]
    close, low, high = m.close.to_numpy(), m.low.to_numpy(), m.high.to_numpy()
    n = len(m)
    ret1h = np.log(pd.Series(close).iloc[::60]).diff().dropna()
    out.append(f"\n## {sym}  (마크가격 1분봉 {n:,}개, {m.index[0].date()} ~ {m.index[-1].date()})")
    out.append(f"- 1시간 수익률 표준편차 {ret1h.std()*100:.3f}%, 일간 환산 {ret1h.std()*np.sqrt(24)*100:.2f}%, "
               f"연환산 {ret1h.std()*np.sqrt(24*365)*100:.0f}%")
    d = pd.Series(close).iloc[::1440].pct_change().abs().dropna()
    out.append(f"- 일간 종가 변동 |ret|: 중앙값 {d.median()*100:.2f}%, 90분위 {d.quantile(.9)*100:.2f}%, "
               f"99분위 {d.quantile(.99)*100:.2f}%, 최대 {d.max()*100:.1f}%")
    starts = np.arange(0, n, 60)
    for label, sl in (("전체 기간", slice(None)), ("최근 365일", slice(n - 365 * 1440, None))):
        out.append(f"\n### {sym} — {label}: 청산 확률 (진입 후 H 안에 청산가 도달, 손절 없이 보유)")
        out.append("| 배율 | 청산거리(롱) | 청산거리(숏) | " + " | ".join(f"롱 {h}" for h in HOR) + " | " +
                   " | ".join(f"숏 {h}" for h in HOR) + " |")
        out.append("|" + "---|" * (3 + 2 * len(HOR)))
        idx = starts[starts >= (0 if label == "전체 기간" else n - 365 * 1440)]
        for L in LEV:
            dl = costs.liquidation_distance(L)
            ds = costs.liquidation_price(1.0, L, "short") - 1
            row = [f"{L}x", f"{dl*100:.2f}%", f"{ds*100:.2f}%"]
            for side in ("long", "short"):
                for h, H in HOR.items():
                    if side == "long":
                        ext = fut_extreme(low, H, "min")[idx]
                        p = np.nanmean(ext <= close[idx] * (1 - dl))
                    else:
                        ext = fut_extreme(high, H, "max")[idx]
                        p = np.nanmean(ext >= close[idx] * (1 + ds))
                    row.append(f"{p*100:.1f}%")
            out.append("| " + " | ".join(row) + " |")
out.append("\n## 수수료 부담 (증거금 대비 왕복 1회, 슬리피지·펀딩 제외)")
out.append("| 배율 | taker 왕복 | maker 왕복 | BNB할인 taker 왕복 | 손익분기 가격변동(taker) |")
out.append("|---|---|---|---|---|")
f0, fb = costs.Fees(), costs.Fees.with_bnb()
for L in LEV:
    out.append(f"| {L}x | {L*costs.round_trip_cost(f0)*100:.1f}% | "
               f"{L*costs.round_trip_cost(f0,'maker','maker')*100:.1f}% | "
               f"{L*costs.round_trip_cost(fb)*100:.1f}% | {costs.round_trip_cost(f0)*100:.2f}% |")
txt = "\n".join(out)
Path("logs").mkdir(exist_ok=True)
Path("logs/liq_table.md").write_text(txt)
print(txt)
