"""Binance 현물(SPOT) BTCUSDT·ETHUSDT 1분봉 수집 (선물 klines_1m과 비교용).

출처: data.binance.vision/data/spot/{monthly,daily}/klines/<SYM>/1m/...
2020-01 -> 최근 완결 월까지 월별 zip, 이번 달은 일별 zip으로 채운다(진행 중 -> .partial).
2025-01-01부터 open_time이 마이크로초로 바뀐다 — freesia.vision._ms가 이미 자릿수로
자동 판별해 ms로 변환하므로 별도 처리 불필요(같은 vision.parse_klines를 그대로 재사용).
컬럼은 선물 klines_1m과 동일: ts, open, high, low, close, volume, quote_volume, trades,
taker_buy_volume, taker_buy_quote_volume.
저장: data/parquet/spot_1m/<BTCUSDT|ETHUSDT>/<YYYY-MM>.parquet
사용: python scripts/get_spot_1m.py [시작 YYYY-MM=2020-01]"""
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freesia import store, vision  # noqa: E402

SYMS = ["BTCUSDT", "ETHUSDT"]
BASE = "https://data.binance.vision/data/spot"
NAME = "spot_1m"


def url_monthly(sym, ym):
    return f"{BASE}/monthly/klines/{sym}/1m/{sym}-1m-{ym}.zip"


def url_daily(sym, ymd):
    return f"{BASE}/daily/klines/{sym}/1m/{sym}-1m-{ymd}.zip"


def main():
    start = sys.argv[1] if len(sys.argv) > 1 else "2020-01"
    now = pd.Timestamp.now("UTC")
    cur_ym = now.strftime("%Y-%m")
    last_complete = (now.replace(day=1) - timedelta(days=1)).strftime("%Y-%m")
    for sym in SYMS:
        d = store.PQ / NAME / sym
        for per in pd.period_range(start, last_complete, freq="M"):
            ym = str(per)
            if (d / f"{ym}.parquet").exists():
                continue
            body = vision.fetch_zip(url_monthly(sym, ym))
            if body is None:
                print(f"{sym} {ym}: 월 zip 없음(스킵)", flush=True)
                continue
            df = vision.parse_klines(body)
            n = store.save(NAME, sym, ym, df, partial=False)
            print(f"{sym} {ym}: {n:,}봉", flush=True)
        # 이번 달: 일별 zip으로 어제까지 채움
        yesterday = (now - timedelta(days=1)).date()
        first_of_month = now.replace(day=1).date()
        day = first_of_month
        frames, miss = [], []
        while day <= yesterday:
            body = vision.fetch_zip(url_daily(sym, day.isoformat()))
            if body is None:
                miss.append(day.isoformat())
            else:
                frames.append(vision.parse_klines(body))
            day += timedelta(days=1)
        if frames:
            df = pd.concat(frames, ignore_index=True)
            n = store.save(NAME, sym, cur_ym, df, partial=True)
            print(f"{sym} {cur_ym}: {n:,}봉 (진행 중{', miss=' + str(miss) if miss else ''})", flush=True)


if __name__ == "__main__":
    main()
