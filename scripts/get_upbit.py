"""업비트 원화 1시간봉(KRW-BTC, KRW-ETH) + 원달러 환율(FRED DEXKOUS 일간) 수집 — 김치 프리미엄 연구용.

업비트: /v1/candles/minutes/60, 호출당 200개, to 파라미터로 과거로 넘김(초당 10회 제한 → 0.15초 간격).
환율: FRED fredgraph CSV(DEXKOUS, 원/달러, 영업일). 연구에서는 다음 날부터 사용(당일 종가는 미국 장 마감 후 확정).
저장: data/ext/upbit_1h.parquet (ts = 봉 시작 UTC, btc, eth = 종가 KRW), data/ext/usdkrw.parquet (date, krw)
사용: python scripts/get_upbit.py"""
import time, io
from pathlib import Path
import pandas as pd, requests
OUT = Path(__file__).resolve().parents[1] / "data" / "ext"


def upbit(market, start="2020-01-01"):
    rows, to = [], None; stop = pd.Timestamp(start, tz="UTC")
    while True:
        p = {"market": market, "count": 200}
        if to: p["to"] = to
        for i in range(5):
            r = requests.get("https://api.upbit.com/v1/candles/minutes/60", params=p, timeout=20)
            if r.status_code == 200: break
            time.sleep(1 + i)
        js = r.json()
        if not js: break
        rows += js
        first = pd.Timestamp(js[-1]["candle_date_time_utc"], tz="UTC")
        if first <= stop: break
        to = js[-1]["candle_date_time_utc"] + "Z"; time.sleep(0.15)
    s = pd.Series({pd.Timestamp(x["candle_date_time_utc"], tz="UTC"): float(x["trade_price"]) for x in rows}).sort_index()
    return s[s.index >= stop]


if __name__ == "__main__":
    b = upbit("KRW-BTC"); e = upbit("KRW-ETH")
    D = pd.DataFrame({"btc": b, "eth": e}); D.index.name = "ts"
    D.reset_index().to_parquet(OUT / "upbit_1h.parquet", index=False); print("업비트", D.shape, D.index.min(), D.index.max())
    csv = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXKOUS", timeout=30).text
    f = pd.read_csv(io.StringIO(csv)); f.columns = ["date", "krw"]; f["krw"] = pd.to_numeric(f.krw, errors="coerce"); f = f.dropna()
    f["date"] = pd.to_datetime(f.date, utc=True); f = f[f.date >= "2019-12-01"]
    f.to_parquet(OUT / "usdkrw.parquet", index=False); print("환율", f.shape, f.date.min(), f.date.max())
