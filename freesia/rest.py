"""Binance fapi REST 조회 (갭 메우기·최신 꼬리 이어붙이기용)."""
import time

import pandas as pd
import requests

API = "https://fapi.binance.com"
KIND_PATH = {"klines_1m": "/fapi/v1/klines", "mark_1m": "/fapi/v1/markPriceKlines",
             "premium_1m": "/fapi/v1/premiumIndexKlines"}
_S = requests.Session()


def _get(path, params, retries=4):
    last = None
    for i in range(retries):
        try:
            r = _S.get(API + path, params=params, timeout=30)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.RequestException(f"http {r.status_code}")
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            last = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"REST 실패 {path} {params}: {last}")


def klines(name, sym, start_ms, end_ms):
    """[start_ms, end_ms] 구간의 **마감된** 1분봉만."""
    ohlc_only = name != "klines_1m"
    rows, t = [], start_ms
    now_ms = int(time.time() * 1000)
    while t <= end_ms:
        js = _get(KIND_PATH[name], {"symbol": sym, "interval": "1m", "startTime": t,
                                    "endTime": end_ms, "limit": 1500})
        if not js:
            break
        rows += js
        t = js[-1][0] + 60_000
        if len(js) < 1500:
            break
    if not rows:
        return None
    a = pd.DataFrame(rows).iloc[:, :11]
    a = a[a[6].astype("int64") < now_ms]  # close_time 이 지난 봉만 (진행 중 봉 제외)
    out = pd.DataFrame({"ts": pd.to_datetime(a[0].astype("int64"), unit="ms", utc=True)})
    cols = ["open", "high", "low", "close"] if ohlc_only else \
        ["open", "high", "low", "close", "volume", "quote_volume", "trades",
         "taker_buy_volume", "taker_buy_quote_volume"]
    src = [1, 2, 3, 4] if ohlc_only else [1, 2, 3, 4, 5, 7, 8, 9, 10]
    for c, i in zip(cols, src):
        out[c] = a[i].astype("float64")
    return out


def funding(sym, start_ms, end_ms):
    rows, t = [], start_ms
    while t <= end_ms:
        js = _get("/fapi/v1/fundingRate", {"symbol": sym, "startTime": t,
                                           "endTime": end_ms, "limit": 1000})
        if not js:
            break
        rows += js
        t = js[-1]["fundingTime"] + 1
        if len(js) < 1000:
            break
    if not rows:
        return None
    df = pd.DataFrame(rows)
    return pd.DataFrame({"ts": pd.to_datetime(df["fundingTime"], unit="ms", utc=True).dt.floor("s"),
                         "interval_hours": 8, "rate": df["fundingRate"].astype("float64")})
