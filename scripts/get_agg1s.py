"""aggTrades(체결 틱) → 1초 봉 오더플로우로 압축 수집. 원본 틱은 저장하지 않는다(디스크 절약).

1초 봉 컬럼: ts(초 시작, UTC), o h l c, bq sq(테이커 매수/매도 체결대금 USDT), nb ns(체결 묶음 수),
            mx(최대 단일 체결대금), lbq lsq(10만 USDT 이상 대량 체결의 매수/매도 대금). 체결 없는 초는 행이 없다.
원칙: SHA256 검증된 월 파일만 사용(freesia.vision.fetch_zip). 월 파일이 없으면 일별 파일로 채우고 그 달은 partial.
저장: data/parquet/agg1s/<SYM>/<YYYY-MM>.parquet
사용: python scripts/get_agg1s.py [시작 YYYY-MM=2023-01] [끝 YYYY-MM=지난달]"""
import io, sys, zipfile, time
from pathlib import Path
import numpy as np, pandas as pd
import pyarrow as pa, pyarrow.csv as pcsv
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freesia import store, vision

SYM = ("BTCUSDT", "ETHUSDT")
COLS = ["agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id", "transact_time", "is_buyer_maker"]
BIG = 100_000.0


def agg_batch(t, p, q, m):
    """배치 하나를 초 단위 부분집계로."""
    s = (t // 1000).astype(np.int64)
    qv = p * q
    buy = ~m                                   # is_buyer_maker=False → 테이커 매수
    df = pd.DataFrame({"s": s, "p": p, "bq": np.where(buy, qv, 0.0), "sq": np.where(buy, 0.0, qv),
                       "nb": buy.astype(np.int32), "ns": (~buy).astype(np.int32), "mx": qv,
                       "lbq": np.where(buy & (qv >= BIG), qv, 0.0), "lsq": np.where(~buy & (qv >= BIG), qv, 0.0)})
    g = df.groupby("s", sort=False)
    return pd.DataFrame({"o": g.p.first(), "h": g.p.max(), "l": g.p.min(), "c": g.p.last(),
                         "bq": g.bq.sum(), "sq": g.sq.sum(), "nb": g.nb.sum(), "ns": g.ns.sum(),
                         "mx": g.mx.max(), "lbq": g.lbq.sum(), "lsq": g.lsq.sum()})


def combine(parts):
    x = pd.concat(parts)
    g = x.groupby(level=0, sort=True)
    out = pd.DataFrame({"o": g.o.first(), "h": g.h.max(), "l": g.l.min(), "c": g.c.last(),
                        "bq": g.bq.sum(), "sq": g.sq.sum(), "nb": g.nb.sum(), "ns": g.ns.sum(),
                        "mx": g.mx.max(), "lbq": g.lbq.sum(), "lsq": g.lsq.sum()})
    out.index.name = "s"
    out = out.reset_index()
    out.insert(0, "ts", pd.to_datetime(out.pop("s"), unit="s", utc=True))
    for c_ in ("bq", "sq", "mx", "lbq", "lsq"):
        out[c_] = out[c_].astype(np.float32)
    for c_ in ("nb", "ns"):
        out[c_] = out[c_].astype(np.int32)
    return out


def parse_zip(zb):
    parts = []
    with zipfile.ZipFile(io.BytesIO(zb)) as z:
        name = z.namelist()[0]
        with z.open(name) as f:
            first = f.readline()
        has_header = not first.split(b",")[0].strip().isdigit()
        with z.open(name) as f:
            ro = pcsv.ReadOptions(block_size=64 << 20, column_names=None if has_header else COLS,
                                  autogenerate_column_names=False)
            co = pcsv.ConvertOptions(include_columns=["price", "quantity", "transact_time", "is_buyer_maker"],
                                     column_types={"price": pa.float64(), "quantity": pa.float64(),
                                                   "transact_time": pa.int64(), "is_buyer_maker": pa.bool_()})
            for b in pcsv.open_csv(f, read_options=ro, convert_options=co):
                t = b.column("transact_time").to_numpy()
                t = np.where(t >= 10**14, t // 1000, t)          # 마이크로초 방어
                parts.append(agg_batch(t, b.column("price").to_numpy(), b.column("quantity").to_numpy(),
                                       b.column("is_buyer_maker").to_numpy(zero_copy_only=False)))
    return parts


def month_range(a, b):
    p = pd.Period(a, "M")
    while p <= pd.Period(b, "M"):
        yield str(p); p += 1


def main():
    start = sys.argv[1] if len(sys.argv) > 1 else "2023-01"
    end = sys.argv[2] if len(sys.argv) > 2 else str(pd.Timestamp.utcnow().to_period("M") - 1)
    for ym in month_range(start, end):
        for s in SYM:
            d = store.PQ / "agg1s" / s
            if (d / f"{ym}.parquet").exists():
                continue
            t0 = time.time()
            zb = vision.fetch_zip(f"{vision.BASE}/monthly/aggTrades/{s}/{s}-aggTrades-{ym}.zip")
            partial = False
            if zb is not None:
                parts = parse_zip(zb); del zb
            else:                                           # 월 파일 없음 → 일별로 채움
                parts = []; partial = True
                for day in pd.date_range(ym + "-01", pd.Period(ym, "M").end_time.normalize(), freq="D"):
                    zd = vision.fetch_zip(f"{vision.BASE}/daily/aggTrades/{s}/{s}-aggTrades-{day:%Y-%m-%d}.zip")
                    if zd is not None: parts += parse_zip(zd)
                if not parts:
                    print(f"{s} {ym} 데이터 없음", flush=True); continue
            out = combine(parts)
            n = store.save("agg1s", s, ym, out, partial=partial)
            print(f"{s} {ym} {'partial ' if partial else ''}{n:,}초 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
