"""구글 트렌드 일간 검색 관심도 수집 (2020-01 ~ 현재, 전 세계, tz=UTC).

구글은 긴 기간을 요청하면 주간·월간으로 뭉개므로 180일 창을 150일 간격(30일 겹침)으로 받아
겹치는 구간의 평균 비율로 이어 붙인다(첫 창 기준 상대값). 값은 '그날 하루 전체'의 관심도라서
연구에서는 반드시 다음 날부터 사용해야 한다(당일 사용 = 룩어헤드).
주의: oracle1(클라우드 IP)은 구글이 429로 막는다 → Odroid(가정 IP)에서 실행 후 oracle1 data/ext/ 로 복사.
저장: data/ext/gtrends.parquet (date, 검색어별 열)
사용: python scripts/get_gtrends.py"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from pytrends.request import TrendReq

KW = ["bitcoin", "ethereum", "crypto", "buy bitcoin", "bitcoin price"]
OUT = Path(__file__).resolve().parents[1] / "data" / "ext" / "gtrends.parquet"


def fetch(t, kw, a, b):
    for i in range(6):
        try:
            t.build_payload([kw], timeframe=f"{a:%Y-%m-%d} {b:%Y-%m-%d}")
            df = t.interest_over_time()
            return df[kw].astype(float) if len(df) else None
        except Exception as e:
            w = 30 * (i + 1); print(f"    재시도 {kw} {a:%Y-%m-%d} ({type(e).__name__}) {w}s 대기", flush=True); time.sleep(w)
    return None


def stitch(t, kw, start, end):
    s = None; a = start
    while a < end:
        b = min(a + pd.Timedelta(days=180), end)
        x = fetch(t, kw, a, b); time.sleep(8)
        if x is None or x.sum() == 0: raise RuntimeError(f"{kw} {a:%Y-%m-%d} 수집 실패")
        if s is None: s = x
        else:
            ov = s.index.intersection(x.index); ov = ov[(s[ov] > 0) & (x[ov] > 0)]
            k = s[ov].mean() / x[ov].mean() if len(ov) else 1.0
            s = pd.concat([s, x[x.index > s.index[-1]] * k])
        print(f"  {kw} {a:%Y-%m-%d}~{b:%Y-%m-%d} 누적 {len(s)}일", flush=True)
        if b >= end: break
        a = b - pd.Timedelta(days=30)
    return s


if __name__ == "__main__":
    t = TrendReq(hl="en-US", tz=0, timeout=(10, 30))
    start, end = pd.Timestamp("2019-12-01"), pd.Timestamp.utcnow().tz_localize(None).normalize() - pd.Timedelta(days=1)
    cols = {}
    for kw in KW:
        cols[kw] = stitch(t, kw, start, end)
    D = pd.DataFrame(cols); D.index.name = "date"
    D = D / D.max() * 100
    OUT.parent.mkdir(parents=True, exist_ok=True); D.reset_index().to_parquet(OUT, index=False)
    print(f"저장 {OUT} {D.shape}, {D.index.min():%Y-%m-%d}~{D.index.max():%Y-%m-%d}")
