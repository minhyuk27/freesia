"""실시간 강제청산 수집 — Bybit·OKX (바이낸스는 scripts/live_book.py의 forceOrder가 이미 수집).

※ 작성·단시간 테스트 전용. cron/데몬으로 등록하지 않는다(지시에 따름, 등록은 별도 승인 필요).
운영 시엔 live_book.py처럼 cron 5분 간격 + flock 단일 실행으로 돌릴 수 있게 설계했지만
그 등록 자체는 이 스크립트의 일이 아니다.

구독:
  Bybit  wss://stream.bybit.com/v5/public/linear, topics allLiquidation.BTCUSDT/ETHUSDT
         20초마다 {"op":"ping"} 전송. side: Bybit의 S(주문 방향)는 청산 주문 방향이므로
         Sell=롱 청산, Buy=숏 청산으로 뒤집어 side(long/short)로 저장. qty는 v(코인 수량) 그대로,
         notional_usd = price * qty (선형 계약이라 그대로 USD).
  OKX    wss://ws.okx.com:8443/ws/v5/public, channel liquidation-orders, instType SWAP,
         instFamily로 BTC-USDT/ETH-USDT(선형)·BTC-USD/ETH-USD(역수) 4건만 구독해 다른 코인은
         애초에 받지 않는다. 20초마다 평문 "ping" 전송(서버는 평문 "pong" 응답).
         REST GET /api/v5/public/instruments?instType=SWAP 로 ctVal·ctType을 한 번 조회해
         contracts(sz) -> 코인 수량으로 환산: 선형은 qty=sz*ctVal, notional=qty*bkPx;
         역수는 notional=sz*ctVal(이미 USD), qty=notional/bkPx. price는 bkPx(파산가, OKX가
         제공하는 값 — 실제 체결가가 아닐 수 있음에 주의). posSide가 long/short면 그대로 side,
         net(원웨이)이면 side(buy/sell)로 유추(매도 청산=롱, 매수 청산=숏).
저장: data/parquet/live_liq/<VENUE>_<SYM>/<YYYY-MM-DD>.parquet (10분 이내 또는 종료 시 flush)
      컬럼: ts, venue, sym, side, price, qty, notional_usd
사용(테스트): timeout 150 python scripts/live_liq.py"""
import asyncio
import json
import signal
import time
from pathlib import Path

import pandas as pd
import requests
import websockets

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "parquet" / "live_liq"

BYBIT_URL = "wss://stream.bybit.com/v5/public/linear"
BYBIT_SYMS = ["BTCUSDT", "ETHUSDT"]

OKX_URL = "wss://ws.okx.com:8443/ws/v5/public"
OKX_FAMILIES = ["BTC-USDT", "ETH-USDT", "BTC-USD", "ETH-USD"]  # linear, linear, inverse, inverse
OKX_WANT = {"BTC-USDT-SWAP", "ETH-USDT-SWAP", "BTC-USD-SWAP", "ETH-USD-SWAP"}

rows = []  # 메모리 버퍼, flush 시 비움
_ctval = {}  # instId -> (ctVal float, ctType str)


def fetch_okx_ctval():
    try:
        r = requests.get("https://www.okx.com/api/v5/public/instruments",
                          params={"instType": "SWAP"}, timeout=15)
        r.raise_for_status()
        js = r.json()
        for d in js.get("data", []):
            if d["instId"] in OKX_WANT:
                _ctval[d["instId"]] = (float(d["ctVal"]), d["ctType"])
        print(f"{pd.Timestamp.now('UTC')} OKX ctVal 조회: {_ctval}", flush=True)
    except Exception as e:
        print(f"{pd.Timestamp.now('UTC')} OKX ctVal 조회 실패(추정치 없이 진행): {e!r}", flush=True)


def add_row(venue, sym, side, price, qty, notional_usd):
    rows.append(dict(ts=time.time(), venue=venue, sym=sym, side=side,
                      price=price, qty=qty, notional_usd=notional_usd))


def on_bybit_msg(m):
    topic = m.get("topic", "")
    if not topic.startswith("allLiquidation."):
        return
    sym = topic.split(".", 1)[1]
    data = m.get("data", [])
    if isinstance(data, dict):
        data = [data]
    for d in data:
        S = d.get("S") or d.get("side")
        side = "long" if S == "Sell" else "short" if S == "Buy" else "?"
        price = float(d.get("p", d.get("price", "nan")))
        qty = float(d.get("v", d.get("size", "nan")))
        add_row("bybit", sym, side, price, qty, price * qty)


def on_okx_msg(m):
    arg = m.get("arg", {})
    if arg.get("channel") != "liquidation-orders":
        return
    for d in m.get("data", []):
        inst_id = d.get("instId")
        if inst_id not in OKX_WANT:
            continue
        for det in d.get("details", []):
            price = float(det.get("bkPx", "nan"))
            sz = float(det.get("sz", "nan"))
            pos_side = det.get("posSide", "net")
            side_field = det.get("side", "")
            if pos_side in ("long", "short"):
                side = pos_side
            else:
                side = "long" if side_field == "sell" else "short" if side_field == "buy" else "?"
            ctval, cttype = _ctval.get(inst_id, (None, None))
            if ctval is None:
                qty, notional = sz, None  # 환산 불가 -> 원시 계약수만 (qty=contracts로 표기됨에 유의)
            elif cttype == "inverse":
                notional = sz * ctval
                qty = notional / price if price and price == price else None
            else:  # linear
                qty = sz * ctval
                notional = qty * price if price == price else None
            add_row("okx", inst_id, side, price, qty, notional)


def flush(force=False):
    global rows
    if not rows or (not force and len(rows) < 1):
        return
    df = pd.DataFrame(rows)
    rows = []
    df["ts"] = pd.to_datetime(df.ts, unit="s", utc=True)
    for (venue, sym), g in df.groupby(["venue", "sym"]):
        for day, gg in g.groupby(g.ts.dt.strftime("%Y-%m-%d")):
            d = OUT / f"{venue}_{sym}"
            d.mkdir(parents=True, exist_ok=True)
            f = d / f"{day}.parquet"
            if f.exists():
                gg = pd.concat([pd.read_parquet(f), gg])
            tmp = d / f".{day}.tmp"
            gg.to_parquet(tmp, index=False)
            tmp.replace(f)
    print(f"{pd.Timestamp.now('UTC')} flush {len(df)}행", flush=True)


async def bybit_pinger(ws):
    while True:
        await asyncio.sleep(20)
        await ws.send(json.dumps({"op": "ping"}))


async def bybit_reader():
    args = [f"allLiquidation.{s}" for s in BYBIT_SYMS]
    while True:
        try:
            async with websockets.connect(BYBIT_URL, ping_interval=None, max_size=2**22) as ws:
                await ws.send(json.dumps({"op": "subscribe", "args": args}))
                print(f"{pd.Timestamp.now('UTC')} bybit 접속, 구독: {args}", flush=True)
                pinger = asyncio.create_task(bybit_pinger(ws))
                try:
                    async for msg in ws:
                        m = json.loads(msg)
                        if "op" in m:
                            print(f"{pd.Timestamp.now('UTC')} bybit ctrl: {m}", flush=True)
                            continue
                        on_bybit_msg(m)
                finally:
                    pinger.cancel()
        except Exception as e:
            print(f"{pd.Timestamp.now('UTC')} bybit 끊김: {e!r} -> 5초 후 재접속", flush=True)
            await asyncio.sleep(5)


async def okx_pinger(ws):
    while True:
        await asyncio.sleep(20)
        await ws.send("ping")


async def okx_reader():
    args = [{"channel": "liquidation-orders", "instType": "SWAP", "instFamily": f} for f in OKX_FAMILIES]
    while True:
        try:
            async with websockets.connect(OKX_URL, ping_interval=None, max_size=2**22) as ws:
                await ws.send(json.dumps({"op": "subscribe", "args": args}))
                print(f"{pd.Timestamp.now('UTC')} okx 접속, 구독: {args}", flush=True)
                pinger = asyncio.create_task(okx_pinger(ws))
                try:
                    async for msg in ws:
                        if msg == "pong":
                            continue
                        m = json.loads(msg)
                        if "event" in m:
                            print(f"{pd.Timestamp.now('UTC')} okx ctrl: {m}", flush=True)
                            continue
                        on_okx_msg(m)
                finally:
                    pinger.cancel()
        except Exception as e:
            print(f"{pd.Timestamp.now('UTC')} okx 끊김: {e!r} -> 5초 후 재접속", flush=True)
            await asyncio.sleep(5)


async def flusher():
    while True:
        await asyncio.sleep(600)
        flush(force=True)


async def main():
    fetch_okx_ctval()
    await asyncio.gather(bybit_reader(), okx_reader(), flusher())


def _term(*_):
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _term)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    finally:
        flush(force=True)
