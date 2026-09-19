"""월별 parquet 저장소. 완결 월은 YYYY-MM.parquet, 진행 중·불완전 월은 YYYY-MM.partial.parquet."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PQ = ROOT / "data" / "parquet"


def _dir(name, sym):
    return PQ / name / sym


def load(name, sym):
    files = sorted(_dir(name, sym).glob("*.parquet"))
    if not files:
        return None
    return pd.concat([pd.read_parquet(f) for f in files]).drop_duplicates("ts") \
        .sort_values("ts").reset_index(drop=True)


def save(name, sym, ym, df, partial):
    d = _dir(name, sym)
    d.mkdir(parents=True, exist_ok=True)
    y, m = int(ym[:4]), int(ym[5:])
    lo = pd.Timestamp(y, m, 1, tz="UTC")
    hi = pd.Timestamp(y + (m == 12), m % 12 + 1, 1, tz="UTC")
    df = df[(df.ts >= lo) & (df.ts < hi)].drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    tmp = d / f".{ym}.tmp"
    df.to_parquet(tmp, index=False)
    final = d / (f"{ym}.partial.parquet" if partial else f"{ym}.parquet")
    tmp.replace(final)
    other = d / (f"{ym}.parquet" if partial else f"{ym}.partial.parquet")
    if other.exists():
        other.unlink()
    return len(df)


def merge(name, sym, new: pd.DataFrame):
    """새 행을 월별로 기존 파티션에 합친다(기존 행 우선). 완결 월은 완결 상태 유지."""
    if new is None or new.empty:
        return 0
    n = 0
    for (y, m), g in new.groupby([new.ts.dt.year, new.ts.dt.month]):
        ym = f"{y}-{m:02d}"
        d = _dir(name, sym)
        complete, part = d / f"{ym}.parquet", d / f"{ym}.partial.parquet"
        old = [pd.read_parquet(p) for p in (complete, part) if p.exists()]
        merged = pd.concat(old + [g])  # 기존 행이 앞 → drop_duplicates 가 기존을 남김
        n += save(name, sym, ym, merged, partial=not complete.exists())
    return n
