"""Binance USDⓈ-M BTC/ETH 과거 데이터 수집 (data.binance.vision → 월별 parquet).

멱등: 완결된 월 파티션은 건너뛰고, 진행 중인 달(.partial)만 매번 다시 만든다.
결손·오류는 logs/backfill_summary.json 에 남기고, 하나라도 있으면 종료코드 1.
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freesia import rest, store, vision as v  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PQ = ROOT / "data" / "parquet"
LOGS = ROOT / "logs"
SYMBOLS = {"BTCUSDT": (2019, 9), "ETHUSDT": (2019, 11)}
METRICS_START = (2021, 12)
# name -> (vision kind, parser, mode)
DATASETS = {
    "klines_1m": ("klines", lambda b: v.parse_klines(b), "kline"),
    "mark_1m": ("markPriceKlines", lambda b: v.parse_klines(b, True), "kline"),
    "premium_1m": ("premiumIndexKlines", lambda b: v.parse_klines(b, True), "kline"),
    "funding": ("fundingRate", v.parse_funding, "funding"),
    "metrics_5m": ("metrics", v.parse_metrics, "metrics"),
}


def months(start, end):
    y, m = start
    while (y, m) <= end:
        yield y, m
        m += 1
        if m == 13:
            y, m = y + 1, 1


def days_of(y, m, last_day):
    d = datetime(y, m, 1, tzinfo=timezone.utc).date()
    while d.month == m and d <= last_day:
        yield d
        d += timedelta(days=1)


save = store.save


def funding_rest(sym, y, m, last_day):
    start = pd.Timestamp(y, m, 1, tz="UTC")
    end = pd.Timestamp(y + (m == 12), m % 12 + 1, 1, tz="UTC")
    return rest.funding(sym, int(start.timestamp() * 1000), int(end.timestamp() * 1000) - 1)


def do_task(name, sym, y, m, today):
    kind, parse, mode = DATASETS[name]
    ym = f"{y}-{m:02d}"
    d = PQ / name / sym
    done = d / f"{ym}.parquet"
    cur = (y, m) == (today.year, today.month)
    if done.exists() and not cur:
        return name, sym, ym, "skip", 0, None
    last_day = today - timedelta(days=1)
    if mode == "metrics":
        frames, miss = [], []
        for day in days_of(y, m, last_day):
            b = v.fetch_zip(v.url_daily("metrics", sym, day.isoformat()))
            (frames.append(parse(b)) if b else miss.append(day.isoformat()))
        if not frames:
            return name, sym, ym, "empty", 0, miss
        n = save(name, sym, ym, pd.concat(frames), partial=cur)
        return name, sym, ym, "ok", n, miss or None
    if not cur:
        b = v.fetch_zip(v.url_monthly(kind, sym, ym))
        if b is not None:
            return name, sym, ym, "ok", save(name, sym, ym, parse(b), False), None
    # 진행 중인 달, 또는 월 파일이 아직 게시되지 않은 달
    if mode == "funding":
        df = funding_rest(sym, y, m, last_day)
        if df is None:
            return name, sym, ym, "empty", 0, ["no funding rows"]
        return name, sym, ym, "ok", save(name, sym, ym, df, cur), None
    frames, miss = [], []
    for day in days_of(y, m, last_day):
        b = v.fetch_zip(v.url_daily(kind, sym, day.isoformat()))
        (frames.append(parse(b)) if b else miss.append(day.isoformat()))
    if not frames:
        return name, sym, ym, "empty", 0, miss
    return name, sym, ym, "ok", save(name, sym, ym, pd.concat(frames), True), miss or None


def main():
    only = set(sys.argv[1:])
    today = datetime.now(timezone.utc).date()
    tasks = []
    for name in DATASETS:
        if only and name not in only:
            continue
        for sym, start in SYMBOLS.items():
            s = METRICS_START if name == "metrics_5m" else \
                (2020, 1) if name.endswith("_1m") else start
            for y, m in months(s, (today.year, today.month)):
                tasks.append((name, sym, y, m, today))
    summary = {"started": datetime.now(timezone.utc).isoformat(), "tasks": len(tasks),
               "ok": 0, "skip": 0, "problems": []}
    with ThreadPoolExecutor(6) as ex:
        futs = {ex.submit(do_task, *t): t for t in tasks}
        for i, f in enumerate(as_completed(futs), 1):
            t = futs[f]
            try:
                name, sym, ym, st, n, miss = f.result()
            except Exception as e:  # noqa: BLE001 — 실패는 반드시 기록
                summary["problems"].append({"task": [t[0], t[1], t[2], t[3]], "error": repr(e)})
                continue
            if st in ("ok", "skip"):
                summary[st] += 1
            # 어제자 Vision 파일은 게시 지연이 정상 → REST 갱신기(update_rest)가 채운다
            miss = [d for d in (miss or []) if d != (today - timedelta(days=1)).isoformat()] or None
            if st == "empty" or miss:
                summary["problems"].append({"task": [name, sym, ym], "status": st, "missing": miss})
            if i % 50 == 0:
                print(f"[{i}/{len(tasks)}]", flush=True)
    LOGS.mkdir(exist_ok=True)
    summary["finished"] = datetime.now(timezone.utc).isoformat()
    (LOGS / "backfill_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k: (v_ if k != "problems" else len(v_)) for k, v_ in summary.items()}))
    sys.exit(1 if summary["problems"] else 0)


if __name__ == "__main__":
    main()
