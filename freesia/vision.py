"""data.binance.vision 공개 데이터 다운로드·검증·파싱.

원칙: 체크섬(SHA256)이 확인되지 않은 파일은 절대 저장하지 않는다.
"""
import hashlib
import io
import time
import zipfile

import pandas as pd
import requests

BASE = "https://data.binance.vision/data/futures/um"
_S = requests.Session()

KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
              "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]


class ChecksumError(RuntimeError):
    pass


def _get(url: str, retries: int = 4):
    """bytes, 없으면(404) None. 그 외 오류는 재시도 후 예외."""
    last = None
    for i in range(retries):
        try:
            r = _S.get(url, timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content
        except requests.RequestException as e:
            last = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"다운로드 실패 {url}: {last}")


def fetch_zip(url: str):
    """체크섬 검증된 zip bytes. 파일 없으면 None. 체크섬 불일치·부재는 예외."""
    body = _get(url)
    if body is None:
        return None
    ck = _get(url + ".CHECKSUM")
    if ck is None:
        raise ChecksumError(f"체크섬 파일 없음: {url}")
    want = ck.decode().split()[0].lower()
    got = hashlib.sha256(body).hexdigest()
    if want != got:
        raise ChecksumError(f"SHA256 불일치 {url}: {want} != {got}")
    return body


def _read_csv(zbytes: bytes, names, header_first_token: str) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(zbytes)) as z:
        raw = z.read(z.namelist()[0])
    first = raw.split(b"\n", 1)[0].decode()
    has_header = not first.split(",")[0].strip().lstrip("-").isdigit() \
        if header_first_token == "numeric" else first.startswith(header_first_token)
    return pd.read_csv(io.BytesIO(raw), header=0 if has_header else None,
                       names=None if has_header else names)


def _ms(s: pd.Series) -> pd.Series:
    s = s.astype("int64")
    s = s.where(s < 10**14, s // 1000)  # 마이크로초 방어
    return pd.to_datetime(s, unit="ms", utc=True)


def parse_klines(zbytes: bytes, ohlc_only: bool = False) -> pd.DataFrame:
    df = _read_csv(zbytes, KLINE_COLS, "numeric")
    df.columns = KLINE_COLS
    out = pd.DataFrame({"ts": _ms(df["open_time"])})
    keep = ["open", "high", "low", "close"] if ohlc_only else \
        ["open", "high", "low", "close", "volume", "quote_volume", "trades",
         "taker_buy_volume", "taker_buy_quote_volume"]
    for c in keep:
        out[c] = df[c].astype("float64")
    return out


def parse_funding(zbytes: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(zbytes)) as z:
        raw = z.read(z.namelist()[0])
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = [c.strip() for c in df.columns]
    out = pd.DataFrame({"ts": _ms(df["calc_time"]).dt.floor("s")})
    out["interval_hours"] = df["funding_interval_hours"].astype("int64") \
        if "funding_interval_hours" in df else 8
    out["rate"] = df["last_funding_rate"].astype("float64")
    return out


def parse_metrics(zbytes: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(zbytes)) as z:
        raw = z.read(z.namelist()[0])
    df = pd.read_csv(io.BytesIO(raw))
    out = pd.DataFrame({"ts": pd.to_datetime(df["create_time"], utc=True)})
    for c in df.columns:
        if c not in ("create_time", "symbol"):
            out[c] = pd.to_numeric(df[c], errors="coerce")
    return out


def url_monthly(kind: str, sym: str, ym: str, iv: str = "1m") -> str:
    if kind == "fundingRate":
        return f"{BASE}/monthly/fundingRate/{sym}/{sym}-fundingRate-{ym}.zip"
    return f"{BASE}/monthly/{kind}/{sym}/{iv}/{sym}-{iv}-{ym}.zip"


def url_daily(kind: str, sym: str, ymd: str, iv: str = "1m") -> str:
    if kind == "metrics":
        return f"{BASE}/daily/metrics/{sym}/{sym}-metrics-{ymd}.zip"
    return f"{BASE}/daily/{kind}/{sym}/{iv}/{sym}-{iv}-{ymd}.zip"
