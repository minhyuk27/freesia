"""Binance 선물 강제청산 스냅샷(liquidationSnapshot) 과거 데이터 수집.

주의: data.binance.vision에는 liquidationSnapshot이 **USDⓈ-M(um)에는 존재하지 않고
코인마진(cm) 무기한물에만 존재**한다 (2026-09-30 S3 버킷 리스팅으로 확인:
data/futures/um/{daily,monthly}/ 에는 liquidationSnapshot 디렉터리 자체가 없음,
data/futures/cm/daily/liquidationSnapshot/ 에는 BTCUSD_PERP·ETHUSD_PERP 등 존재).
그래서 이 스크립트는 요청받은 BTCUSDT/ETHUSDT(USDⓈ-M) 대신 **BTCUSD_PERP·ETHUSD_PERP
(코인마진 무기한물)** 을 받는다. 코인마진 계약은 역수(inverse) 계약이라 avg_price×qty가
실제 USD 명목가치가 아니다(계약당 고정 USD 가치는 별도 승수 필요) — notional 컬럼은
과제 지시대로 avg_price×filled_qty로 계산한 값이며 코인마진 계약에서는 근사치로만 쓸 것.
데이터셋 자체가 2024-10-14를 마지막으로 더 이상 갱신되지 않는다(Binance가 discontinue).

출처: https://data.binance.vision/data/futures/cm/daily/liquidationSnapshot/<SYM>/<SYM>-liquidationSnapshot-<YYYY-MM-DD>.zip
날짜 목록은 S3 버킷 리스팅(REST) API로 얻는다(prefix listing, marker 페이징).
체크섬(.CHECKSUM) 검증은 freesia.vision.fetch_zip 재사용.
같은 ms에 여러 청산이 겹치면 store.merge가 ts로만 dedup하므로, 그룹 내 순번만큼
마이크로초를 더해 ts를 유일하게 만든다(행 수 보존).
저장: data/parquet/liq_snap/<BTCUSD_PERP|ETHUSD_PERP>/<YYYY-MM>.parquet
      (ts, side, price, avg_price, qty, notional)
사용: python scripts/get_liq_snapshot.py"""
import io
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freesia import store, vision  # noqa: E402

SYMS = ["BTCUSD_PERP", "ETHUSD_PERP"]
DL_BASE = "https://data.binance.vision/data/futures/cm/daily/liquidationSnapshot"
S3_LIST = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
NAME = "liq_snap"
_S = requests.Session()
KEY_RE = re.compile(r"-liquidationSnapshot-(\d{4}-\d{2}-\d{2})\.zip$")


def list_days(sym):
    """해당 심볼의 daily liquidationSnapshot zip이 존재하는 날짜 목록(정렬됨)."""
    prefix = f"data/futures/cm/daily/liquidationSnapshot/{sym}/"
    days, marker = [], ""
    while True:
        params = {"prefix": prefix, "max-keys": 1000}
        if marker:
            params["marker"] = marker
        r = _S.get(S3_LIST, params=params, timeout=30)
        r.raise_for_status()
        keys = re.findall(r"<Key>([^<]+)</Key>", r.text)
        for k in keys:
            m = KEY_RE.search(k)
            if m:
                days.append(m.group(1))
        truncated = "<IsTruncated>true</IsTruncated>" in r.text
        if not truncated or not keys:
            break
        marker = keys[-1]
    return sorted(set(days))


def parse_day(zbytes, day):
    with zipfile.ZipFile(io.BytesIO(zbytes)) as z:
        raw = z.read(z.namelist()[0])
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = [c.strip() for c in df.columns]
    ms = df["time"].astype("int64")
    dup_seq = ms.groupby(ms).cumcount()  # 같은 ms 내 순번 -> us 오프셋으로 유일화
    ts = pd.to_datetime(ms, unit="ms", utc=True) + pd.to_timedelta(dup_seq, unit="us")
    out = pd.DataFrame({
        "ts": ts,
        "side": df["side"].astype(str),
        "price": df["price"].astype("float64"),
        "avg_price": df["average_price"].astype("float64"),
        "qty": df["accumulated_fill_quantity"].astype("float64"),
    })
    out["notional"] = out["avg_price"] * out["qty"]
    return out


def main():
    for sym in SYMS:
        days = list_days(sym)
        if not days:
            print(f"{sym}: 사용 가능한 날짜 없음", flush=True)
            continue
        print(f"{sym}: {days[0]} ~ {days[-1]} ({len(days)}일)", flush=True)
        last_month = days[-1][:7]
        by_month = {}
        for d in days:
            by_month.setdefault(d[:7], []).append(d)
        for ym, ds in sorted(by_month.items()):
            d = store.PQ / NAME / sym
            done = d / f"{ym}.parquet"
            if done.exists() and ym != last_month:
                continue
            frames, miss = [], []
            for day in ds:
                url = f"{DL_BASE}/{sym}/{sym}-liquidationSnapshot-{day}.zip"
                try:
                    body = vision.fetch_zip(url)
                except vision.ChecksumError as e:
                    print(f"체크섬 오류 {day}: {e}", flush=True)
                    body = None
                if body is None:
                    miss.append(day)
                    continue
                frames.append(parse_day(body, day))
            if not frames:
                print(f"{sym} {ym}: 수신 실패 (miss={miss})", flush=True)
                continue
            df = pd.concat(frames, ignore_index=True)
            partial = ym == last_month  # discontinue된 데이터셋의 마지막 달은 재확인 여지 남김
            n = store.save(NAME, sym, ym, df, partial=partial)
            print(f"{sym} {ym}: {n:,}행{' (miss=' + str(miss) + ')' if miss else ''}", flush=True)


if __name__ == "__main__":
    main()
