"""분기물 베이시스 일별 시계열 (BTC·ETH). 매일 만기 14일 이상 남은 가장 가까운 분기물의 연환산 베이시스와 그 다음 분기물과의 기울기.
저장: data/ext/basis_btc.parquet, basis_eth.parquet (index=UTC 일자, basis_ann, dte, next_ann)."""
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freesia import vision as v  # noqa: E402

EXT = Path(__file__).resolve().parents[1] / "data" / "ext"
LIST = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision?delimiter=/&prefix=data/futures/um/monthly/klines/{}_"


def ser(urls):
    fr = []
    for u in urls:
        b = v.fetch_zip(u)
        if b is not None:
            fr.append(v.parse_klines(b, True))
    return pd.concat(fr).drop_duplicates("ts").set_index("ts").close if fr else None


def months(a, b):
    y, m = a.year, a.month
    while (y, m) <= (b.year, b.month):
        yield y, m
        m += 1
        if m == 13:
            y, m = y + 1, 1


now = datetime.now(timezone.utc)
for base in ("BTC", "ETH"):
    sym = f"{base}USDT"
    spot = ser([f"https://data.binance.vision/data/spot/monthly/klines/{sym}/1d/{sym}-1d-{y}-{m:02d}.zip" for y, m in months(datetime(2021, 1, 1), now - timedelta(days=32))])
    # 진행 중인 달 현물은 일별 파일
    d = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
    extra = []
    while d.date() < now.date():
        b = v.fetch_zip(f"https://data.binance.vision/data/spot/daily/klines/{sym}/1d/{sym}-1d-{d.date()}.zip")
        if b is not None:
            extra.append(v.parse_klines(b, True))
        d += timedelta(days=1)
    if extra:
        spot = pd.concat([spot.reset_index(), pd.concat(extra)]).drop_duplicates("ts").set_index("ts").close
    spot.index = spot.index.normalize()
    contracts = sorted(set(re.findall(rf"{sym}_(\d{{6}})/", requests.get(LIST.format(sym), timeout=30).text)))
    rows = {}
    for cd in contracts:
        exp = datetime.strptime(cd, "%y%m%d").replace(tzinfo=timezone.utc)
        c = f"{sym}_{cd}"
        urls = [f"https://data.binance.vision/data/futures/um/monthly/klines/{c}/1d/{c}-1d-{y}-{m:02d}.zip" for y, m in months(exp - timedelta(days=200), min(exp, now))
                if (y, m) < (now.year, now.month)]
        s = ser(urls)
        cur = []
        d = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
        while d.date() < now.date() and d <= exp:
            b = v.fetch_zip(f"https://data.binance.vision/data/futures/um/daily/klines/{c}/1d/{c}-1d-{d.date()}.zip")
            if b is not None:
                cur.append(v.parse_klines(b, True))
            d += timedelta(days=1)
        if cur:
            cs = pd.concat(cur).drop_duplicates("ts").set_index("ts").close
            s = cs if s is None else pd.concat([s, cs])
        if s is None:
            continue
        s.index = s.index.normalize(); s = s[~s.index.duplicated()]
        rows[cd] = (exp, s)
    days = sorted(set().union(*[set(s.index) for _, s in rows.values()]))
    out = []
    for dt in days:
        cand = []
        for cd, (exp, s) in rows.items():
            dte = (exp - dt).days
            if dte >= 14 and dt in s.index and dt in spot.index:
                cand.append((dte, cd, s[dt] / spot[dt] - 1))
        cand.sort()
        if not cand:
            continue
        dte, cd, b0 = cand[0]
        nx = cand[1] if len(cand) > 1 else None
        out.append({"ts": dt, "basis_ann": b0 * 365 / dte, "dte": dte, "next_ann": (nx[2] * 365 / nx[0]) if nx else np.nan})
    df = pd.DataFrame(out).set_index("ts").sort_index()
    df.to_parquet(EXT / f"basis_{base.lower()}.parquet")
    print(base, len(df), df.index[0], df.index[-1], df.basis_ann.describe().round(3).to_dict(), flush=True)
