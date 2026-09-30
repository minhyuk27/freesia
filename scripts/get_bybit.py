"""Bybit V5 공개 REST(linear BTCUSDT·ETHUSDT) 과거 데이터 수집.

출처: api.bybit.com/v5/market/{kline,open-interest,funding/history} (category=linear).
api.bybit.com이 막히면 api.bytick.com으로 자동 재시도(HOSTS 순서대로).
세 데이터셋 모두 최초 실행 시 이진 탐색(일 단위)으로 실제 시작일을 찾는다
(kline은 과거 시작점 이전 start를 주면 최초 봉으로 클램프되는 것을 이용해 1회 조회로 확정,
open-interest·funding은 하루 단위로 존재 여부를 이진 탐색).
월 단위로 저장하며 완결 월은 건너뛴다(재실행 안전). oracle1 OOM 이력 -> 월 단위 처리 유지.
저장:
  bybit_1m    : ts, open, high, low, close, volume, turnover      (1분봉)
  bybit_oi5m  : ts, oi                                            (5분 간격 미결제약정)
  bybit_funding: ts, rate                                         (펀딩비 이력)
사용: python scripts/get_bybit.py [시작 YYYY-MM=probe]"""
import sys
import time
import urllib.parse
from datetime import timedelta
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freesia import store  # noqa: E402

SYMS = ["BTCUSDT", "ETHUSDT"]
HOSTS = ["https://api.bybit.com", "https://api.bytick.com"]
S = requests.Session()
_active_host = [None]


def _get(path, params, retries=6):
    hosts = [_active_host[0]] if _active_host[0] else HOSTS
    last = None
    for host in hosts:
        for i in range(retries):
            try:
                r = S.get(host + path, params=params, timeout=20)
                if r.status_code == 429:
                    time.sleep(1.5 * (i + 1))
                    continue
                r.raise_for_status()
                js = r.json()
                if js.get("retCode") not in (0, None):
                    raise RuntimeError(f"retCode={js.get('retCode')} {js.get('retMsg')}")
                _active_host[0] = host
                return js["result"]
            except (requests.RequestException, RuntimeError) as e:
                last = e
                time.sleep(1.2 * (i + 1))
        print(f"호스트 {host} 실패, 다음 호스트 시도: {last}", flush=True)
    raise RuntimeError(f"Bybit 요청 실패 {path} {params}: {last}")


# ---------- 1분봉 ----------
def kline_page(sym, start_ms, end_ms, limit=1000):
    r = _get("/v5/market/kline", {"category": "linear", "symbol": sym, "interval": "1",
                                   "start": start_ms, "end": end_ms, "limit": limit})
    return r["list"]  # 최신순(내림차순)


def probe_kline_earliest(sym):
    r = _get("/v5/market/kline", {"category": "linear", "symbol": sym, "interval": "1",
                                   "start": 1546300800000, "limit": 1})  # 2019-01-01, 실제 이전이면 최초봉으로 클램프
    lst = r["list"]
    if not lst:
        return None
    return int(lst[-1][0])


def do_kline(sym, start_arg):
    earliest_ms = probe_kline_earliest(sym)
    if earliest_ms is None:
        print(f"{sym} bybit_1m: 데이터 없음", flush=True)
        return
    earliest = pd.Timestamp(earliest_ms, unit="ms", tz="UTC")
    print(f"{sym} bybit_1m 시작: {earliest}", flush=True)
    now = pd.Timestamp.now("UTC")
    start_per = start_arg or earliest.strftime("%Y-%m")
    d = store.PQ / "bybit_1m" / sym
    for per in pd.period_range(max(pd.Period(start_per, "M"), pd.Period(earliest.strftime("%Y-%m"), "M")),
                                now.strftime("%Y-%m"), freq="M"):
        ym = str(per)
        if (d / f"{ym}.parquet").exists():
            continue
        lo = max(per.start_time.tz_localize("UTC"), earliest)
        hi = min(per.end_time.tz_localize("UTC").floor("min") + pd.Timedelta(minutes=1), now.floor("min"))
        if hi <= lo:
            continue
        rows, t = [], int(lo.timestamp() * 1000)
        end_ms = int(hi.timestamp() * 1000) - 60_000
        while t <= end_ms:
            batch_end = min(t + 999 * 60_000, end_ms)
            lst = kline_page(sym, t, batch_end)
            if lst:
                rows += lst
            t = batch_end + 60_000  # 배치 단위로 고정 전진(중간에 빈 분이 있어도 무한루프 방지)
            time.sleep(0.12)
        if not rows:
            print(f"{sym} bybit_1m {ym}: 없음", flush=True)
            continue
        df = pd.DataFrame(rows, columns=["t", "open", "high", "low", "close", "volume", "turnover"])
        df = df.drop_duplicates("t")
        out = pd.DataFrame({"ts": pd.to_datetime(df.t.astype("int64"), unit="ms", utc=True)})
        for c in ["open", "high", "low", "close", "volume", "turnover"]:
            out[c] = df[c].astype("float64")
        partial = per == pd.Period(now.strftime("%Y-%m"), "M")
        n = store.save("bybit_1m", sym, ym, out, partial=partial)
        print(f"{sym} bybit_1m {ym}: {n:,}봉{' (진행 중)' if partial else ''}", flush=True)


# ---------- 5분 OI ----------
def probe_day_exists(check_fn, lo_day, hi_day):
    """[lo_day, hi_day) 사이에서 데이터가 시작되는 첫 날짜(이진 탐색, 일 단위)."""
    lo, hi = lo_day, hi_day
    if not check_fn(hi - timedelta(days=1)):
        return None  # hi 근방까지도 데이터 없음 (있을 수 없지만 방어)
    while (hi - lo).days > 1:
        mid = lo + (hi - lo) // 2
        if check_fn(mid):
            hi = mid
        else:
            lo = mid + timedelta(days=1)
    return hi


def oi_has_data(sym, day):
    lo_ms = int(pd.Timestamp(day, tz="UTC").timestamp() * 1000)
    hi_ms = lo_ms + 86_400_000
    r = _get("/v5/market/open-interest",
             {"category": "linear", "symbol": sym, "intervalTime": "5min",
              "startTime": lo_ms, "endTime": hi_ms, "limit": 1})
    return bool(r["list"])


def oi_page(sym, lo_ms, hi_ms, cursor=None):
    p = {"category": "linear", "symbol": sym, "intervalTime": "5min",
         "startTime": lo_ms, "endTime": hi_ms, "limit": 200}
    if cursor:
        p["cursor"] = cursor
    r = _get("/v5/market/open-interest", p)
    return r["list"], r.get("nextPageCursor") or ""


def do_oi(sym, start_arg):
    today = pd.Timestamp.now("UTC").normalize().date()
    earliest_day = probe_day_exists(lambda d: oi_has_data(sym, d),
                                     pd.Timestamp("2019-01-01").date(), today)
    if earliest_day is None:
        print(f"{sym} bybit_oi5m: 데이터 없음", flush=True)
        return
    print(f"{sym} bybit_oi5m 시작일: {earliest_day}", flush=True)
    now = pd.Timestamp.now("UTC")
    start_per = start_arg or earliest_day.strftime("%Y-%m")
    d = store.PQ / "bybit_oi5m" / sym
    for per in pd.period_range(max(pd.Period(start_per, "M"), pd.Period(earliest_day.strftime("%Y-%m"), "M")),
                                now.strftime("%Y-%m"), freq="M"):
        ym = str(per)
        if (d / f"{ym}.parquet").exists():
            continue
        lo = max(per.start_time.tz_localize("UTC"), pd.Timestamp(earliest_day, tz="UTC"))
        hi = min(per.end_time.tz_localize("UTC").floor("min") + pd.Timedelta(minutes=1), now.floor("min"))
        if hi <= lo:
            continue
        lo_ms, hi_ms = int(lo.timestamp() * 1000), int(hi.timestamp() * 1000)
        rows, cursor = [], None
        while True:
            lst, nxt = oi_page(sym, lo_ms, hi_ms, cursor)
            if not lst:
                break
            rows += lst
            if not nxt:
                break
            cursor = urllib.parse.unquote(nxt)
            time.sleep(0.12)
        if not rows:
            print(f"{sym} bybit_oi5m {ym}: 없음", flush=True)
            continue
        df = pd.DataFrame(rows).drop_duplicates("timestamp")
        out = pd.DataFrame({"ts": pd.to_datetime(df.timestamp.astype("int64"), unit="ms", utc=True),
                             "oi": df.openInterest.astype("float64")})
        partial = per == pd.Period(now.strftime("%Y-%m"), "M")
        n = store.save("bybit_oi5m", sym, ym, out, partial=partial)
        print(f"{sym} bybit_oi5m {ym}: {n:,}행{' (진행 중)' if partial else ''}", flush=True)


# ---------- 펀딩 ----------
def funding_has_data(sym, day):
    lo_ms = int(pd.Timestamp(day, tz="UTC").timestamp() * 1000)
    hi_ms = lo_ms + 86_400_000
    r = _get("/v5/market/funding/history",
             {"category": "linear", "symbol": sym, "startTime": lo_ms, "endTime": hi_ms, "limit": 2})
    return bool(r["list"])


def funding_page(sym, lo_ms, hi_ms):
    r = _get("/v5/market/funding/history",
              {"category": "linear", "symbol": sym, "startTime": lo_ms, "endTime": hi_ms, "limit": 200})
    return r["list"]


def do_funding(sym, start_arg):
    today = pd.Timestamp.now("UTC").normalize().date()
    earliest_day = probe_day_exists(lambda d: funding_has_data(sym, d),
                                     pd.Timestamp("2019-01-01").date(), today)
    if earliest_day is None:
        print(f"{sym} bybit_funding: 데이터 없음", flush=True)
        return
    print(f"{sym} bybit_funding 시작일: {earliest_day}", flush=True)
    now = pd.Timestamp.now("UTC")
    start_per = start_arg or earliest_day.strftime("%Y-%m")
    d = store.PQ / "bybit_funding" / sym
    for per in pd.period_range(max(pd.Period(start_per, "M"), pd.Period(earliest_day.strftime("%Y-%m"), "M")),
                                now.strftime("%Y-%m"), freq="M"):
        ym = str(per)
        if (d / f"{ym}.parquet").exists():
            continue
        lo = max(per.start_time.tz_localize("UTC"), pd.Timestamp(earliest_day, tz="UTC"))
        hi = min(per.end_time.tz_localize("UTC").floor("min") + pd.Timedelta(minutes=1), now.floor("min"))
        if hi <= lo:
            continue
        lo_ms, hi_ms = int(lo.timestamp() * 1000), int(hi.timestamp() * 1000)
        rows, t = [], lo_ms
        seen_ts = set()
        while t <= hi_ms:
            lst = funding_page(sym, t, hi_ms)
            if not lst:
                break
            new = [x for x in lst if x["fundingRateTimestamp"] not in seen_ts]
            if not new:
                break
            rows += new
            for x in new:
                seen_ts.add(x["fundingRateTimestamp"])
            # list는 최신순(내림)이므로 가장 오래된 항목(마지막)+1ms부터 다음 페이지
            oldest = min(int(x["fundingRateTimestamp"]) for x in new)
            if len(new) < 200:
                break
            t = oldest + 1
            time.sleep(0.12)
        if not rows:
            print(f"{sym} bybit_funding {ym}: 없음", flush=True)
            continue
        df = pd.DataFrame(rows).drop_duplicates("fundingRateTimestamp")
        out = pd.DataFrame({"ts": pd.to_datetime(df.fundingRateTimestamp.astype("int64"), unit="ms", utc=True),
                             "rate": df.fundingRate.astype("float64")})
        partial = per == pd.Period(now.strftime("%Y-%m"), "M")
        n = store.save("bybit_funding", sym, ym, out, partial=partial)
        print(f"{sym} bybit_funding {ym}: {n:,}행{' (진행 중)' if partial else ''}", flush=True)


def main():
    start_arg = sys.argv[1] if len(sys.argv) > 1 else None
    for sym in SYMS:
        do_kline(sym, start_arg)
        do_oi(sym, start_arg)
        do_funding(sym, start_arg)


if __name__ == "__main__":
    main()
