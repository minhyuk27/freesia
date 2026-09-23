"""코인베이스 현물 BTC-USD·ETH-USD 1분봉 수집 (코인베이스 프리미엄 계산용).

출처: Coinbase Exchange 공개 API GET /products/{id}/candles?granularity=60 (요청당 최대 300봉).
체결이 없던 분은 봉이 없다(거래소 사양). 월 단위 저장, 완결 월은 건너뛴다(재실행 안전).
저장: data/parquet/coinbase_1m/<BTCUSD|ETHUSD>/<YYYY-MM>.parquet  (ts, open, high, low, close, volume)
사용: python scripts/get_coinbase_1m.py [시작 YYYY-MM=2020-01]"""
import sys, time
from pathlib import Path
import pandas as pd, requests
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freesia import store

URL = "https://api.exchange.coinbase.com/products/{p}/candles"
PAIRS = {"BTC-USD": "BTCUSD", "ETH-USD": "ETHUSD"}
S = requests.Session(); S.headers["User-Agent"] = "freesia-research"


def get(p, a, b):
    for i in range(6):
        try:
            r = S.get(URL.format(p=p), params={"granularity": 60, "start": a.isoformat(), "end": b.isoformat()}, timeout=30)
            if r.status_code == 429:
                time.sleep(2 * (i + 1)); continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"실패 {p} {a}")


def main():
    start = sys.argv[1] if len(sys.argv) > 1 else "2020-01"
    now = pd.Timestamp.now("UTC").floor("min")
    for p, name in PAIRS.items():
        for per in pd.period_range(start, now.strftime("%Y-%m"), freq="M"):
            ym = str(per); d = store.PQ / "coinbase_1m" / name
            if (d / f"{ym}.parquet").exists(): continue
            lo = per.start_time.tz_localize("UTC"); hi = min(per.end_time.tz_localize("UTC").floor("min") + pd.Timedelta(minutes=1), now)
            rows = []; a = lo
            while a < hi:
                b = min(a + pd.Timedelta(minutes=299), hi - pd.Timedelta(minutes=1))
                rows += get(p, a, b)
                a = b + pd.Timedelta(minutes=1)
                time.sleep(0.12)
            df = pd.DataFrame(rows, columns=["t", "low", "high", "open", "close", "volume"])
            df["ts"] = pd.to_datetime(df.t, unit="s", utc=True)
            df = df[["ts", "open", "high", "low", "close", "volume"]]
            partial = hi < per.end_time.tz_localize("UTC")
            n = store.save("coinbase_1m", name, ym, df, partial=partial)
            print(f"{name} {ym} {n:,}봉{' (진행 중)' if partial else ''}", flush=True)


if __name__ == "__main__":
    main()
