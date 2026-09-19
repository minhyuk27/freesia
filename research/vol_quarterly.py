"""분기별 변동성·청산확률 (마크가격 1분봉). 최근 흐름 확인용.

지표: 1시간 σ, 일간 환산 σ, 연환산 σ, 일중 고저폭(고가/저가-1) 중앙값·90분위,
1분 σ, 배율별(10/20/50/100x) 1시간·1일 청산확률(롱·숏 평균, 손절 없이 보유, 정시 진입).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freesia import costs, store  # noqa: E402

LEV = [10, 20, 50, 100]


def fut_extreme(arr, H, fn):
    s = pd.Series(arr[::-1]).rolling(H, min_periods=H)
    r = (s.min() if fn == "min" else s.max()).to_numpy()[::-1]
    return np.concatenate([r[1:], [np.nan]])


out = []
for sym in ("BTCUSDT", "ETHUSDT"):
    m = store.load("mark_1m", sym)
    m = m[m.ts >= "2020-01-01"].reset_index(drop=True)
    close, low, high = m.close.to_numpy(), m.low.to_numpy(), m.high.to_numpy()
    n = len(m)
    # 미래 극값(청산 판정)은 전체 시계열에서 한 번만 계산
    ext = {}
    for H in (60, 1440):
        ext[("min", H)] = fut_extreme(low, H, "min")
        ext[("max", H)] = fut_extreme(high, H, "max")
    q = m.ts.dt.tz_localize(None).dt.to_period("Q").astype(str).to_numpy()
    hourly = m.set_index("ts").close.resample("1h").last()
    daily = m.set_index("ts").resample("1D").agg({"high": "max", "low": "min", "close": "last"})
    hourly.index, daily.index = hourly.index.tz_localize(None), daily.index.tz_localize(None)  # 분기 판정용
    lr1m = np.log(pd.Series(close)).diff()
    rows = []
    for qq in sorted(set(q)):
        idx = np.where(q == qq)[0]
        idx60 = idx[idx % 60 == 0]
        if len(idx) < 1440 * 30:
            continue
        hq = np.log(hourly[(hourly.index.to_period("Q").astype(str) == qq)]).diff().dropna()
        dq = daily[daily.index.to_period("Q").astype(str) == qq]
        rng = (dq.high / dq.low - 1)
        r = {"분기": qq, "일수": len(dq), "1분σ%": lr1m.iloc[idx].std() * 100,
             "1시간σ%": hq.std() * 100, "일간σ%": hq.std() * np.sqrt(24) * 100,
             "연σ%": hq.std() * np.sqrt(24 * 365) * 100,
             "일중폭 중앙%": rng.median() * 100, "일중폭 90%": rng.quantile(.9) * 100,
             "분기수익%": (close[idx[-1]] / close[idx[0]] - 1) * 100}
        for L in LEV:
            dl = costs.liquidation_distance(L)
            ds = costs.liquidation_price(1.0, L, "short") - 1
            for lab, H in (("1h", 60), ("1d", 1440)):
                a = np.nanmean(ext[("min", H)][idx60] <= close[idx60] * (1 - dl))
                b = np.nanmean(ext[("max", H)][idx60] >= close[idx60] * (1 + ds))
                r[f"{L}x {lab}청산%"] = (a + b) / 2 * 100
        rows.append(r)
    df = pd.DataFrame(rows).set_index("분기")
    out.append(f"\n## {sym} 분기별 변동성 (마지막 분기는 진행 중)\n")
    out.append(df.round(2).to_markdown() if hasattr(df, "to_markdown") else df.round(2).to_string())
txt = "\n".join(out)
Path("logs").mkdir(exist_ok=True)
Path("logs/vol_quarterly.md").write_text(txt)
print(txt)
