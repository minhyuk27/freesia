"""업비트 KRW-BTC·KRW-ETH 1분봉 수집 (get_upbit.py의 1시간봉과 별개, 분 단위 미시구조용).

출처: GET /v1/candles/minutes/1, 호출당 최대 200개, to 파라미터로 과거로 페이징
(초당 10회 제한 -> 0.13초 간격, 429는 지수 백오프).
월 단위로 뒤에서부터(to=월말+1분) 페이징해 월초(lo) 이전으로 내려가면 멈추고,
그 달 범위로 잘라 저장한다. 완결 월은 파일 존재 시 건너뛴다(재실행 안전).
오래 걸리는 작업 -> 백그라운드 실행 권장.
저장: data/parquet/upbit_1m/<KRWBTC|KRWETH>/<YYYY-MM>.parquet
      (ts=candle_date_time_utc, open, high, low, close, volume, value)
사용: python scripts/get_upbit_1m.py [시작 YYYY-MM=2020-01]"""
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freesia import store  # noqa: E402

MARKETS = {"KRW-BTC": "KRWBTC", "KRW-ETH": "KRWETH"}
URL = "https://api.upbit.com/v1/candles/minutes/1"
NAME = "upbit_1m"
S = requests.Session()


def get(market, to):
    p = {"market": market, "count": 200}
    if to:
        p["to"] = to
    for i in range(6):
        r = S.get(URL, params=p, timeout=20)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:
            time.sleep(1.5 * (i + 1))
            continue
        time.sleep(1 + i)
    raise RuntimeError(f"업비트 요청 실패 {market} to={to}")


def fetch_month(market, y, m, now):
    per = pd.Period(f"{y}-{m:02d}", freq="M")
    lo = per.start_time.tz_localize("UTC")
    month_end_excl = (per + 1).start_time.tz_localize("UTC")  # 다음 달 시작 = 이번 달 배타적 상한
    hi = min(month_end_excl, now.floor("min"))
    if hi <= lo:
        return None, True
    rows, to = [], hi.strftime("%Y-%m-%dT%H:%M:%S") + "Z"
    while True:
        js = get(market, to)
        if not js:
            break
        rows += js
        oldest = pd.Timestamp(js[-1]["candle_date_time_utc"], tz="UTC")
        to = js[-1]["candle_date_time_utc"] + "Z"
        time.sleep(0.13)
        if oldest < lo:
            break
    if not rows:
        return None, True
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["candle_date_time_utc"], utc=True)
    df = df[(df.ts >= lo) & (df.ts < hi)].drop_duplicates("ts")
    out = df.rename(columns={"opening_price": "open", "high_price": "high", "low_price": "low",
                              "trade_price": "close", "candle_acc_trade_volume": "volume",
                              "candle_acc_trade_price": "value"})
    out = out[["ts", "open", "high", "low", "close", "volume", "value"]].astype(
        {c: "float64" for c in ["open", "high", "low", "close", "volume", "value"]})
    partial = hi < month_end_excl  # now가 아직 이번 달 끝에 도달하지 않음 = 진행 중인 달
    return out, partial


def main():
    start = sys.argv[1] if len(sys.argv) > 1 else "2020-01"
    now = pd.Timestamp.now("UTC")
    for market, sym in MARKETS.items():
        d = store.PQ / NAME / sym
        for per in pd.period_range(start, now.strftime("%Y-%m"), freq="M"):
            ym = str(per)
            if (d / f"{ym}.parquet").exists():
                continue
            df, partial = fetch_month(market, per.year, per.month, now)
            if df is None:
                print(f"{sym} {ym}: 데이터 없음", flush=True)
                continue
            n = store.save(NAME, sym, ym, df, partial=partial)
            print(f"{sym} {ym}: {n:,}봉{' (진행 중)' if partial else ''}", flush=True)


if __name__ == "__main__":
    main()
