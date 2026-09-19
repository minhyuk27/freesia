"""보조 신호용 외부 데이터 수집 (BTC·ETH 거래용. 알트코인 데이터는 받지 않는다). data/ext/*.parquet 로 저장.
모두 공개 무료 API. 각 소스는 독립적으로 실패해도 다른 소스를 막지 않고 결과를 로그에 남긴다."""
import io
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

EXT = Path(__file__).resolve().parents[1] / "data" / "ext"
EXT.mkdir(parents=True, exist_ok=True)
H = {"User-Agent": "freesia-research/0.1"}
log = {}


def get(url, **kw):
    for i in range(4):
        try:
            r = requests.get(url, headers=H, timeout=40, **kw)
            if r.status_code == 429:
                time.sleep(3 * (i + 1)); continue
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            err = e; time.sleep(2 * (i + 1))
    raise RuntimeError(f"{url}: {err}")


def save(name, df):
    df.to_parquet(EXT / f"{name}.parquet")
    log[name] = {"rows": len(df), "first": str(df.index[0]), "last": str(df.index[-1])}
    print("saved", name, log[name], flush=True)


def fng():
    js = get("https://api.alternative.me/fng/?limit=0&format=json").json()["data"]
    df = pd.DataFrame(js)
    df.index = pd.to_datetime(df.timestamp.astype("int64"), unit="s", utc=True)
    save("fng", df[["value"]].astype(float).sort_index())


def fred():
    ids = {"VIXCLS": "vix", "DTWEXBGS": "usd_broad", "DGS10": "y10", "DGS2": "y2", "NASDAQCOM": "nasdaq", "SP500": "spx",
           "WALCL": "fed_assets", "T10YIE": "breakeven10", "BAMLH0A0HYM2": "hy_spread", "DCOILWTICO": "wti", "GOLDAMGBD228NLBM": "gold"}
    fr = {}
    for sid, nm in ids.items():
        try:
            df = pd.read_csv(io.StringIO(get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}").text))
            df.columns = ["date", nm]
            df["date"] = pd.to_datetime(df["date"], utc=True)
            fr[nm] = pd.to_numeric(df.set_index("date")[nm], errors="coerce")
        except Exception as e:  # noqa: BLE001
            log[f"fred:{sid}"] = repr(e)
    save("fred", pd.concat(fr, axis=1).sort_index())


def coinmetrics():
    cand = ["PriceUSD", "CapMVRVCur", "CapRealUSD", "AdrActCnt", "AdrNewCnt", "TxCnt", "TxTfrValAdjUSD", "FeeTotUSD", "NVTAdj", "NVTAdj90",
            "SplyAct1yr", "SplyActEver", "VelCur1yr", "HashRate", "IssTotUSD", "FlowInExUSD", "FlowOutExUSD", "SplyExUSD", "CapMrktCurUSD", "ROI30d"]
    ok = {}
    for a in ("btc", "eth"):
        frames = []
        for m in cand:
            try:
                r = get("https://community-api.coinmetrics.io/v4/timeseries/asset-metrics",
                        params={"assets": a, "metrics": m, "frequency": "1d", "start_time": "2018-01-01", "page_size": 10000})
                js = r.json()
                if "data" in js and js["data"]:
                    df = pd.DataFrame(js["data"]); df["time"] = pd.to_datetime(df["time"], utc=True).dt.normalize()
                    frames.append(pd.to_numeric(df.set_index("time")[m], errors="coerce"))
                    ok.setdefault(a, []).append(m)
            except Exception as e:  # noqa: BLE001
                log[f"coinmetrics:{a}:{m}"] = str(e)[:80]
        if frames:
            save(f"cm_{a}", pd.concat(frames, axis=1).sort_index())
    log["coinmetrics_ok"] = ok


def stablecoins():
    df = pd.DataFrame(get("https://stablecoins.llama.fi/stablecoincharts/all").json())
    df.index = pd.to_datetime(df.date.astype("int64"), unit="s", utc=True)
    out = pd.DataFrame({"stable_total": df.totalCirculatingUSD.apply(lambda d: d.get("peggedUSD"))}, index=df.index)
    for sid, nm in ((1, "usdt"), (2, "usdc")):
        try:
            d = pd.DataFrame(get(f"https://stablecoins.llama.fi/stablecoincharts/all?stablecoin={sid}").json())
            d.index = pd.to_datetime(d.date.astype("int64"), unit="s", utc=True)
            out[nm] = d.totalCirculatingUSD.apply(lambda x: x.get("peggedUSD"))
        except Exception as e:  # noqa: BLE001
            log[f"llama:{nm}"] = repr(e)
    save("stablecoins", out.astype(float).sort_index())


def coinbase(prod, gran, name, start="2019-01-01"):
    """코인베이스 현물 종가 (프리미엄 신호용)."""
    step = {86400: 300 * 86400, 3600: 300 * 3600}[gran]
    t = int(datetime.fromisoformat(start).replace(tzinfo=timezone.utc).timestamp()); end = int(time.time())
    rows = []
    while t < end:
        te = min(t + step, end)
        js = get(f"https://api.exchange.coinbase.com/products/{prod}/candles",
                 params={"granularity": gran, "start": datetime.fromtimestamp(t, timezone.utc).isoformat(), "end": datetime.fromtimestamp(te, timezone.utc).isoformat()}).json()
        rows += js
        t = te; time.sleep(0.12)
    df = pd.DataFrame(rows, columns=["t", "low", "high", "open", "close", "volume"]).drop_duplicates("t")
    df.index = pd.to_datetime(df.t, unit="s", utc=True)
    save(name, df.drop(columns="t").sort_index())


def dvol():
    for cur in ("BTC", "ETH"):
        rows, end = [], int(time.time() * 1000)
        start = int(datetime(2021, 3, 1, tzinfo=timezone.utc).timestamp() * 1000)
        for _ in range(60):
            js = get("https://www.deribit.com/api/v2/public/get_volatility_index_data",
                     params={"currency": cur, "start_timestamp": start, "end_timestamp": end, "resolution": 3600}).json()["result"]
            data = js.get("data") or []
            rows += data
            cont = js.get("continuation")
            if not cont or not data:
                break
            end = cont
        if rows:
            df = pd.DataFrame(rows, columns=["t", "open", "high", "low", "close"]).drop_duplicates("t")
            df.index = pd.to_datetime(df.t, unit="ms", utc=True)
            save(f"dvol_{cur.lower()}_1h", df.drop(columns="t").sort_index())
        else:
            log[f"dvol_{cur}"] = "no rows"


if __name__ == "__main__":
    which = sys.argv[1:] or ["fng", "fred", "coinmetrics", "stablecoins", "coinbase_d", "dvol"]
    for w in which:
        try:
            if w == "coinbase_d":
                coinbase("BTC-USD", 86400, "cb_btc_1d"); coinbase("ETH-USD", 86400, "cb_eth_1d")
            elif w == "coinbase_h":
                coinbase("BTC-USD", 3600, "cb_btc_1h", "2020-01-01"); coinbase("ETH-USD", 3600, "cb_eth_1h", "2020-01-01")
            else:
                globals()[w]()
        except Exception as e:  # noqa: BLE001
            log[w] = "FAILED " + repr(e)[:200]; print("FAIL", w, log[w], flush=True)
    (EXT.parent.parent / "logs").mkdir(exist_ok=True)
    (EXT.parent.parent / "logs" / "newdata_log.json").write_text(json.dumps(log, indent=1, ensure_ascii=False))
