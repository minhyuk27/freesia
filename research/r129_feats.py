"""R129 특징 — R130/R131에서 재사용할 인과적(causal) 분 단위 특징 생성기.

핵심 규칙: 모든 값은 해당 분의 "종가 시점"까지의 데이터만 사용한다(미래 참조 금지).
  - klines_1m 은 그 분의 OHLCV(완결 캔들) 그대로 사용.
  - agg1s(1초 오더플로우)는 그 분의 0~59초까지만 분 단위로 먼저 합산해 쓴다(60배 축소 → 메모리 절약).
  - bookDepth(비정규 ~30초 스냅샷)는 그 분의 종가 시각 이전에 관측된 "마지막" 스냅샷으로 전진채움(ffill)한다.
  - premium_1m·coinbase_1m·spot_1m·다른 종목 klines_1m 도 동일 분 그리드에 ffill로 맞춘다.
분 인덱스는 klines_1m(=store.load("klines_1m", sym)) 배열의 행 위치(0-base) — R90의 t와 같은 체계.

주의(의도적 생략): "1초 실현변동성", "1초 봉 중 |수익률|>X 개수" 같은 진짜 1초-해상도 특징은
  여기(분 단위 범용 모듈)에는 넣지 않았다. agg1s 1초 원본은 전체 기간 로드 시 수 GB(심볼당)라
  R129/R130 저장소 규칙("월별로만 읽거나 필요한 분만 읽기")을 어기게 된다. 그런 특징이 필요하면
  r129_onset_anatomy.py처럼 사건 주변 ±15분 등 좁은 구간만 그때그때 월별 파일에서 읽어 쓴다.

사용 예:
  from r129_feats import minute_panel
  F = minute_panel("BTCUSDT", t_array)   # t_array: klines_1m 행 인덱스(정수) 배열
"""
import sys, glob
from pathlib import Path
import numpy as np, pandas as pd
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R))
from freesia import store

SYMS = ("BTCUSDT", "ETHUSDT")
_CB_SYM = {"BTCUSDT": "BTCUSD", "ETHUSDT": "ETHUSD"}
LOOKBACKS = (1, 5, 10)
_CACHE = {}


def _klines(sym):
    if ("k", sym) not in _CACHE:
        _CACHE["k", sym] = store.load("klines_1m", sym)
    return _CACHE["k", sym]


def _glob(name, sym):
    return sorted(glob.glob(str(R / "data/parquet" / name / sym / "*.parquet")))


def _agg_minute(sym):
    """agg1s(1초) → 분 단위 합산(bq,sq,nb,ns,lbq,lsq 합, mx 그 분의 최댓값). 2023-01~ 만 존재(이전은 NaN)."""
    key = ("agm", sym)
    if key in _CACHE:
        return _CACHE[key]
    parts = []
    for f in _glob("agg1s", sym):
        d = pd.read_parquet(f, columns=["ts", "bq", "sq", "nb", "ns", "mx", "lbq", "lsq"])
        d["ts"] = pd.DatetimeIndex(d.ts).floor("min")
        g = d.groupby("ts").agg(bq=("bq", "sum"), sq=("sq", "sum"), nb=("nb", "sum"), ns=("ns", "sum"),
                                 mx=("mx", "max"), lbq=("lbq", "sum"), lsq=("lsq", "sum"))
        parts.append(g)
    out = pd.concat(parts).sort_index() if parts else pd.DataFrame(columns=["bq", "sq", "nb", "ns", "mx", "lbq", "lsq"])
    out = out[~out.index.duplicated()]
    _CACHE[key] = out
    return out


def _bookdepth_minute(sym):
    """bookDepth(~30초 스냅샷, ±1~5% 호가 잔량 notional) → 분 그리드에 ffill(그 분 종가 시점까지 관측 가능한 마지막 값만).
    imb = (매수 측 1~5% 합 − 매도 측 1~5% 합)/합, nimb = 같은 식의 ±1%(터치 바로 위·아래)만."""
    key = ("bd", sym)
    if key in _CACHE:
        return _CACHE[key]
    files = _glob("bookdepth", sym)
    if not files:
        _CACHE[key] = None
        return None
    d = pd.concat([pd.read_parquet(f) for f in files]).drop_duplicates("ts").sort_values("ts").set_index("ts")
    bid = d[["n-1", "n-2", "n-3", "n-4", "n-5"]].sum(1)
    ask = d[["n+1", "n+2", "n+3", "n+4", "n+5"]].sum(1)
    out = pd.DataFrame({"bdimb": (bid - ask) / (bid + ask), "bdnimb": (d["n-1"] - d["n+1"]) / (d["n-1"] + d["n+1"])})
    k = _klines(sym); grid = pd.DatetimeIndex(k.ts)
    out = out.reindex(out.index.union(grid)).ffill().reindex(grid)
    out.index = grid
    _CACHE[key] = out
    return out


def _series_1m(name, sym, col, cbkey=None):
    key = (name, sym, col)
    if key in _CACHE:
        return _CACHE[key]
    d = sym if cbkey is None else cbkey
    files = _glob(name, d)
    if not files:
        _CACHE[key] = None
        return None
    s = pd.concat([pd.read_parquet(f, columns=["ts", col]) for f in files]).drop_duplicates("ts").sort_values("ts").set_index("ts")[col]
    _CACHE[key] = s
    return s


def _aligned(sym):
    """klines_1m 분 그리드에 맞춰 ffill/정렬한 보조 시계열 묶음 — 심볼당 1회만 계산해 캐시."""
    key = ("aligned", sym)
    if key in _CACHE:
        return _CACHE[key]
    k = _klines(sym); grid = pd.DatetimeIndex(k.ts); n = len(k)

    def ff(s):
        if s is None:
            return np.full(n, np.nan)
        return s.reindex(s.index.union(grid)).ffill().reindex(grid).values

    agm = _agg_minute(sym).reindex(grid)  # 합산량 성격 → ffill 하지 않음(없으면 NaN, 그 분 실제 거래가 0이어도 보통 포함되어 있음)
    bd = _bookdepth_minute(sym)
    cb = _series_1m("coinbase_1m", sym, "close", cbkey=_CB_SYM[sym])
    sp = _series_1m("spot_1m", sym, "close")
    pr = _series_1m("premium_1m", sym, "close")
    other = [s for s in SYMS if s != sym][0]
    ko = _klines(other); co = pd.Series(ko.close.values, index=pd.DatetimeIndex(ko.ts))

    out = dict(close=k.close.values, high=k.high.values, low=k.low.values,
               volume=k.volume.values, trades=k.trades.values.astype(float), tbv=k.taker_buy_volume.values,
               bq=agm.bq.values, sq=agm.sq.values, lbq=agm.lbq.values, lsq=agm.lsq.values, mx=agm.mx.values,
               bdimb=ff(bd.bdimb) if bd is not None else np.full(n, np.nan),
               bdnimb=ff(bd.bdnimb) if bd is not None else np.full(n, np.nan),
               cb=ff(cb), spot=ff(sp), prem=ff(pr), other=ff(co))
    _CACHE[key] = out
    return out


# (특징명, 부호 있는 특징인지) — 부호 있는 것만 방향(DIRECTION) 검정에 쓴다.
SIGNED = {"ret", "ofi", "lti", "tbr0", "cblead", "spotlead", "othret", "bdimb", "dbdimb", "premchg"}
UNSIGNED_ONLY = {"absret", "range", "volz", "tradez", "maxtrade"}
FEAT_NAMES = sorted(SIGNED | UNSIGNED_ONLY)


def _rolling_all(sym, lookbacks=LOOKBACKS):
    """심볼 전체 길이에 대해 룩백 L별 인과적 특징 시리즈를 한 번만 계산(평균·롤링은 전부 벡터화, 이후 .iloc[minutes]로 샘플링)."""
    key = ("roll", sym, lookbacks)
    if key in _CACHE:
        return _CACHE[key]
    A = _aligned(sym); n = len(A["close"])
    idx = pd.RangeIndex(n)
    close = pd.Series(A["close"], idx); high = pd.Series(A["high"], idx); low = pd.Series(A["low"], idx)
    vol = pd.Series(A["volume"], idx); trd = pd.Series(A["trades"], idx); tbv = pd.Series(A["tbv"], idx)
    bq = pd.Series(A["bq"], idx); sq = pd.Series(A["sq"], idx); lbq = pd.Series(A["lbq"], idx); lsq = pd.Series(A["lsq"], idx)
    mx = pd.Series(A["mx"], idx)
    bdimb = pd.Series(A["bdimb"], idx); cb = pd.Series(A["cb"], idx); spot = pd.Series(A["spot"], idx)
    prem = pd.Series(A["prem"], idx); other = pd.Series(A["other"], idx)
    eps = 1e-9
    out = {}
    for L in lookbacks:
        sumv = vol.rolling(L).sum(); sumt = trd.rolling(L).sum()
        base_m = sumv.shift(L).rolling(1440).mean(); base_s = sumv.shift(L).rolling(1440).std()
        tbase_m = sumt.shift(L).rolling(1440).mean(); tbase_s = sumt.shift(L).rolling(1440).std()
        out[f"ret_{L}"] = np.log(close / close.shift(L))
        out[f"absret_{L}"] = out[f"ret_{L}"].abs()
        out[f"range_{L}"] = (high.rolling(L).max() - low.rolling(L).min()) / close.shift(L)
        out[f"volz_{L}"] = (sumv - base_m) / (base_s + eps)
        out[f"tradez_{L}"] = (sumt - tbase_m) / (tbase_s + eps)
        out[f"tbr0_{L}"] = tbv.rolling(L).sum() / (sumv + eps) - 0.5
        sbq, ssq = bq.rolling(L).sum(), sq.rolling(L).sum()
        out[f"ofi_{L}"] = (sbq - ssq) / (sbq + ssq + eps)
        slb, sls = lbq.rolling(L).sum(), lsq.rolling(L).sum()
        out[f"lti_{L}"] = (slb - sls) / (slb + sls + eps)
        out[f"maxtrade_{L}"] = mx.rolling(L).max()
        out[f"bdimb_{L}"] = bdimb.rolling(L).mean()
        out[f"dbdimb_{L}"] = bdimb - bdimb.shift(L)
        out[f"premchg_{L}"] = prem - prem.shift(L)
        out[f"cblead_{L}"] = np.log(cb / cb.shift(L)) - out[f"ret_{L}"]
        out[f"spotlead_{L}"] = np.log(spot / spot.shift(L)) - out[f"ret_{L}"]
        out[f"othret_{L}"] = np.log(other / other.shift(L))
    F = pd.DataFrame(out)
    _CACHE[key] = F
    return F


def minute_panel(sym, minutes, lookbacks=LOOKBACKS):
    """임의의 분 인덱스(klines_1m 행 위치) 배열에 대해 1/5/10분(기본) 룩백 인과적 특징을 반환.

    각 특징은 "그 분을 포함해 과거 L분"(= [m-L+1, m]) 동안의 값이며 미래를 보지 않는다.
    열 이름: f"{특징}_{L}". 특징 목록은 FEAT_NAMES(부호 있는 것은 SIGNED, 크기만 의미있는 것은 UNSIGNED_ONLY).
    agg1s·bookDepth 기반 특징(ofi/lti/maxtrade/bdimb/dbdimb)은 2023-01 이전엔 NaN(데이터 없음)."""
    minutes = np.asarray(minutes, dtype=np.int64)
    F = _rolling_all(sym, tuple(lookbacks))
    return F.iloc[minutes].reset_index(drop=True)
