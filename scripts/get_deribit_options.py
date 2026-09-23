"""Deribit BTC·ETH 옵션 체결 전체 수집 (옵션 스큐·내재변동성·풋콜 흐름 특징용).

출처: history.deribit.com 공개 API public/get_last_trades_by_currency_and_time (요청당 최대 1000건, 시간순).
체결마다 내재변동성(iv)·기초지수(index_price)가 함께 온다. 월 단위 저장, 완결 월은 건너뜀. trade_id로 중복 제거.
저장: data/parquet/deribit_opt/<BTC|ETH>/<YYYY-MM>.parquet
  ts, expiry(UTC 08:00 만기), strike, cp(+1 콜 / −1 풋), iv(%), price(기초자산 단위), index_price, amount, side(+1 매수 공격 / −1 매도 공격)
사용: python scripts/get_deribit_options.py [시작 YYYY-MM=2020-01]"""
import sys, time
from pathlib import Path
import pandas as pd, requests
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freesia import store

URL = "https://history.deribit.com/api/v2/public/get_last_trades_by_currency_and_time"
S = requests.Session(); S.headers["User-Agent"] = "freesia-research"


def get(cur, a, b):
    for i in range(6):
        try:
            r = S.get(URL, params={"currency": cur, "kind": "option", "start_timestamp": a, "end_timestamp": b,
                                   "count": 1000, "sorting": "asc"}, timeout=30)
            if r.status_code == 429:
                time.sleep(3 * (i + 1)); continue
            r.raise_for_status()
            return r.json()["result"]
        except (requests.RequestException, KeyError, ValueError):
            time.sleep(3 * (i + 1))
    raise RuntimeError(f"실패 {cur} {a}")


def parse(tr):
    df = pd.DataFrame(tr)
    if df.empty: return df
    p = df.instrument_name.str.split("-", expand=True)
    out = pd.DataFrame({
        "ts": pd.to_datetime(df.timestamp, unit="ms", utc=True),
        "expiry": pd.to_datetime(p[1], format="%d%b%y", utc=True) + pd.Timedelta(hours=8),
        "strike": p[2].astype(float),
        "cp": p[3].map({"C": 1, "P": -1}).astype("int8"),
        "iv": df.iv.astype("float32"),
        "price": df.price.astype("float32"),
        "index_price": df.index_price.astype("float32"),
        "amount": df.amount.astype("float32"),
        "side": df.direction.map({"buy": 1, "sell": -1}).astype("int8"),
        "tid": df.trade_id.astype(str),
    })
    return out


def main():
    start = sys.argv[1] if len(sys.argv) > 1 else "2020-01"
    now = pd.Timestamp.now("UTC")
    for cur in ("BTC", "ETH"):
        for per in pd.period_range(start, now.strftime("%Y-%m"), freq="M"):
            ym = str(per); d = store.PQ / "deribit_opt" / cur
            if (d / f"{ym}.parquet").exists(): continue
            lo = int(per.start_time.tz_localize("UTC").timestamp() * 1000)
            hi = int(min(per.end_time.tz_localize("UTC"), now).timestamp() * 1000)
            parts = []; a = lo; t0 = time.time()
            while a < hi:
                res = get(cur, a, hi)
                tr = res.get("trades", [])
                if not tr: break
                parts.append(parse(tr))
                last = tr[-1]["timestamp"]
                if not res.get("has_more"): break
                a = last if last > a else a + 1          # 같은 ms 체결은 tid 중복 제거로 처리
                time.sleep(0.08)
            if not parts:
                print(f"{cur} {ym} 없음", flush=True); continue
            df = pd.concat(parts).drop_duplicates("tid").sort_values("ts")
            partial = per.end_time.tz_localize("UTC") > now
            # store.save 는 ts 중복을 지우므로 체결 단위 보존을 위해 직접 저장
            d.mkdir(parents=True, exist_ok=True)
            tmp = d / f".{ym}.tmp"; df.drop(columns="tid").to_parquet(tmp, index=False)
            tmp.replace(d / (f"{ym}.partial.parquet" if partial else f"{ym}.parquet"))
            print(f"{cur} {ym} {len(df):,}건 {time.time()-t0:.0f}s{' (진행 중)' if partial else ''}", flush=True)


if __name__ == "__main__":
    main()
