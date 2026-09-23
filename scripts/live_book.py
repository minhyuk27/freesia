"""실시간 최우선 호가·상위 20호가 수집 (과거 데이터가 없는 미시구조를 앞으로 쌓는다).

구독: <sym>@bookTicker (최우선 호가 모든 변경), <sym>@depth20@100ms (상위 20호가 스냅샷)
1초마다 한 행으로 압축:
  bid ask bq aq     : 초 끝 최우선 호가·잔량
  ofi               : 그 초 동안 누적 OFI (Cont·Kukanov·Stoikov 2014, 최우선 호가 변경마다 합산, 수량 단위)
  nupd              : 최우선 호가 갱신 횟수
  spr_max           : 그 초의 최대 스프레드(bp)
  d5b d5a d20b d20a : 상위 5·20 호가 누적 잔량(USDT) — 초 끝 스냅샷
저장: data/parquet/live_book/<SYM>/<YYYY-MM-DD>.parquet (정시마다 그날 파일에 합침)
운영: cron 5분마다 flock으로 단일 실행 보장. 끊기면 자동 재접속.
사용: python scripts/live_book.py"""
import asyncio, json, sys, time
from pathlib import Path
import pandas as pd
import websockets
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "parquet" / "live_book"
SYMS = ("btcusdt", "ethusdt")
URL = "wss://fstream.binance.com/stream?streams=" + "/".join(
    f"{s}@bookTicker/{s}@depth20@100ms" for s in SYMS)

st = {s: dict(bid=None, ask=None, bq=0.0, aq=0.0, ofi=0.0, nupd=0, spr=0.0, d5b=0.0, d5a=0.0, d20b=0.0, d20a=0.0)
      for s in SYMS}
rows = {s: [] for s in SYMS}


def on_ticker(s, d):
    x = st[s]
    b, bq, a, aq = float(d["b"]), float(d["B"]), float(d["a"]), float(d["A"])
    if x["bid"] is not None:
        e = 0.0
        e += bq if b >= x["bid"] else 0.0
        e -= x["bq"] if b <= x["bid"] else 0.0
        e -= aq if a <= x["ask"] else 0.0
        e += x["aq"] if a >= x["ask"] else 0.0
        x["ofi"] += e
    x.update(bid=b, bq=bq, ask=a, aq=aq)
    x["nupd"] += 1
    x["spr"] = max(x["spr"], (a - b) / ((a + b) / 2) * 1e4)


def on_depth(s, d):
    x = st[s]
    bids = [(float(p), float(q)) for p, q in d["b"]]; asks = [(float(p), float(q)) for p, q in d["a"]]
    x["d5b"] = sum(p * q for p, q in bids[:5]); x["d5a"] = sum(p * q for p, q in asks[:5])
    x["d20b"] = sum(p * q for p, q in bids); x["d20a"] = sum(p * q for p, q in asks)


def flush(force=False):
    for s in SYMS:
        if not rows[s] or (not force and len(rows[s]) < 3600): continue
        df = pd.DataFrame(rows[s]); rows[s] = []
        df["ts"] = pd.to_datetime(df.ts, unit="s", utc=True)
        for day, g in df.groupby(df.ts.dt.strftime("%Y-%m-%d")):
            d = OUT / s.upper(); d.mkdir(parents=True, exist_ok=True)
            f = d / f"{day}.parquet"
            if f.exists(): g = pd.concat([pd.read_parquet(f), g]).drop_duplicates("ts")
            tmp = d / f".{day}.tmp"; g.to_parquet(tmp, index=False); tmp.replace(f)


async def sampler():
    nxt = int(time.time()) + 1
    while True:
        await asyncio.sleep(max(0.0, nxt - time.time()))
        for s in SYMS:
            x = st[s]
            if x["bid"] is None: continue
            rows[s].append(dict(ts=nxt - 1, bid=x["bid"], ask=x["ask"], bq=x["bq"], aq=x["aq"], ofi=x["ofi"],
                                nupd=x["nupd"], spr_max=x["spr"], d5b=x["d5b"], d5a=x["d5a"], d20b=x["d20b"], d20a=x["d20a"]))
            x["ofi"] = 0.0; x["nupd"] = 0; x["spr"] = 0.0
        if nxt % 3600 == 0: flush(force=True)
        nxt += 1


async def reader():
    while True:
        try:
            async with websockets.connect(URL, ping_interval=20, max_size=2**22) as ws:
                print(f"{pd.Timestamp.utcnow()} 접속", flush=True)
                async for msg in ws:
                    m = json.loads(msg); stream = m["stream"]; s = stream.split("@")[0]
                    (on_ticker if "bookTicker" in stream else on_depth)(s, m["data"])
        except Exception as e:
            print(f"{pd.Timestamp.utcnow()} 끊김: {e!r} → 5초 후 재접속", flush=True)
            for s in SYMS: st[s]["bid"] = None          # 끊긴 구간의 OFI 연결 방지
            await asyncio.sleep(5)


async def main():
    await asyncio.gather(reader(), sampler())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        flush(force=True)
