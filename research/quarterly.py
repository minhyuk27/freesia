"""분기물 캐시앤캐리(현물 롱 + 분기물 숏, 만기 보유) 실측. 진입 시점에 베이시스가 고정된다(만기에 선물=지수로 수렴).
일봉 종가 기준. 진입: 만기 85~100일 전 첫 거래일. 비용: 진입 17bp(현물 10+선물 5+슬리피지 2), 청산 12bp(현물 10+슬리피지 2, 선물은 결제)."""
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freesia import vision as v  # noqa: E402

LIST = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision?delimiter=/&prefix=data/futures/um/monthly/klines/{}_"
ENTRY, EXIT = 0.0017, 0.0012


def series(urls):
    fr = []
    for u in urls:
        b = v.fetch_zip(u)
        if b is not None:
            fr.append(v.parse_klines(b, ohlc_only=True))
    return pd.concat(fr).drop_duplicates("ts").set_index("ts").close if fr else None


def month_list(a, b):
    y, m = a.year, a.month
    while (y, m) <= (b.year, b.month):
        yield f"{y}-{m:02d}"
        m += 1
        if m == 13:
            y, m = y + 1, 1


rows = []
now = datetime.now(timezone.utc)
for base in ("BTC", "ETH"):
    sym = f"{base}USDT"
    spot = series([f"https://data.binance.vision/data/spot/monthly/klines/{sym}/1d/{sym}-1d-{ym}.zip"
                   for ym in month_list(datetime(2021, 1, 1), datetime(2026, 8, 1))])
    spot.index = spot.index.normalize()
    txt = requests.get(LIST.format(sym), timeout=30).text
    for d in sorted(set(re.findall(rf"{sym}_(\d{{6}})/", txt))):
        exp = datetime.strptime(d, "%y%m%d").replace(tzinfo=timezone.utc)
        if exp > now - timedelta(days=2):
            continue
        c = f"{sym}_{d}"
        fut = series([f"https://data.binance.vision/data/futures/um/monthly/klines/{c}/1d/{c}-1d-{ym}.zip"
                      for ym in month_list(exp - timedelta(days=200), exp)])
        if fut is None:
            continue
        fut.index = fut.index.normalize()
        df = pd.concat([fut.rename("F"), spot.rename("S")], axis=1, join="inner").dropna()
        df["dte"] = (exp - df.index).days
        df["basis_ann"] = (df.F / df.S - 1) * 365 / df.dte.clip(lower=1)
        ent = df[(df.dte <= 100) & (df.dte >= 85)]
        if ent.empty:
            continue
        e0 = ent.iloc[0]
        hold = int(e0.dte)
        gross = e0.F / e0.S - 1
        net = gross - ENTRY - EXIT
        mtm = ((df.loc[df.index >= ent.index[0]].F / df.loc[df.index >= ent.index[0]].S - 1) - gross)   # 베이시스 확대 시 평가손
        rows.append({"base": base, "contract": d, "entry": str(ent.index[0].date()), "dte": hold,
                     "gross%": round(gross * 100, 2), "net%": round(net * 100, 2),
                     "gross_ann%": round(gross * 365 / hold * 100, 1), "net_ann%": round(net * 365 / hold * 100, 1),
                     "worst_basis_widening%": round(float(mtm.max() * 100), 2)})
out = pd.DataFrame(rows)
print(out.to_string(index=False))
for base in ("BTC", "ETH"):
    o = out[out.base == base]
    print(base, "계약수", len(o), "순연환산 평균 %.1f%%, 중앙 %.1f%%, 최소 %.1f%%, 최대 %.1f%% | 2024+ 평균 %.1f%%" % (
        o["net_ann%"].mean(), o["net_ann%"].median(), o["net_ann%"].min(), o["net_ann%"].max(),
        o[o.entry >= "2024-01-01"]["net_ann%"].mean()))
out.to_csv("logs/quarterly_carry.csv", index=False)
