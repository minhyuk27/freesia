"""실시간 입력 수집 (바이낸스 공개 REST, 키 불필요).

- 1분봉·프리미엄 지수 1분: freesia.rest.klines 재사용 (닫힌 봉만 사용)
- 펀딩: freesia.rest.funding
- 5분 지표: /futures/data/openInterestHist(sumOpenInterest), topLongShortPositionRatio(→ sum_toptrader_long_short_ratio),
            globalLongShortAccountRatio(→ count_long_short_ratio). 최근 30일까지만 제공, 호출당 500개.
  매핑은 scripts/live_parity.py 에서 저장된 일별 파일(data.binance.vision metrics)과 수치 대조로 확인.
- 일봉 종가(변동성 국면 vreg용): /fapi/v1/klines interval=1d"""
import time
import pandas as pd
from freesia import rest

MET = {"sum_open_interest": ("/futures/data/openInterestHist", "sumOpenInterest"),
       "sum_toptrader_long_short_ratio": ("/futures/data/topLongShortPositionRatio", "longShortRatio"),
       "count_long_short_ratio": ("/futures/data/globalLongShortAccountRatio", "longShortRatio")}


def now_ms():
    return int(time.time() * 1000)


def klines_1m(sym, minutes, end_ms=None):
    end = end_ms or now_ms(); start = end - minutes * 60_000
    k = rest.klines("klines_1m", sym, start, end).set_index("ts")
    return k[k.index + pd.Timedelta("1min") <= pd.Timestamp(end, unit="ms", tz="UTC")]   # 닫힌 봉만


def premium_1m(sym, minutes, end_ms=None):
    end = end_ms or now_ms(); start = end - minutes * 60_000
    p = rest.klines("premium_1m", sym, start, end).set_index("ts")
    return p[p.index + pd.Timedelta("1min") <= pd.Timestamp(end, unit="ms", tz="UTC")].close


def funding(sym, days=3):
    end = now_ms(); f = rest.funding(sym, end - days * 86_400_000, end)
    return f.set_index("ts").rate


def metrics_5m(sym, points=500):
    cols = {}
    for name, (path, field) in MET.items():
        js = rest._get(path, {"symbol": sym, "period": "5m", "limit": points})
        s = pd.Series({pd.Timestamp(int(x["timestamp"]), unit="ms", tz="UTC"): float(x[field]) for x in js})
        cols[name] = s
    return pd.DataFrame(cols).sort_index()


def daily_close(sym, days=500):
    js = rest._get("/fapi/v1/klines", {"symbol": sym, "interval": "1d", "limit": min(days, 1500)})
    s = pd.Series({pd.Timestamp(int(x[0]), unit="ms", tz="UTC"): float(x[4]) for x in js})
    return s.iloc[:-1]                                              # 진행 중인 오늘 봉 제외
