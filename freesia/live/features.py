"""z0.31 실전용 특징 계산 — research/r27_dip_events.feats + research/r32_dip_events.extra 와 같은 식(호가 bslope·bimb 제외).

입력은 연속 1분 격자(빈 분 없음)의 데이터프레임. 결과 배열의 인덱스 i 값 = 분 i 종료 시점까지의 정보(연구 코드와 동일).
주문 판단: 분 f 동안 지정가를 살려 둘지는 분 f−1 행의 특징 + 층별 특징(el, pre, tpiv, qh, hr0)으로 결정한다.
분 번호 a = 2020-01-01 00:00 UTC 이후 분 수(1분봉이 그때부터 빈 분 없이 이어짐 → 봉 경계 = 시계 정각).
대조 검증: scripts/live_parity.py"""
import numpy as np, pandas as pd

EPOCH = pd.Timestamp("2020-01-01", tz="UTC")
BARS = (5, 10, 15, 30, 60)
KS = (3.0, 4.0, 5.0, 6.0, 8.0)
EPS = 0.0002                                                      # 지정가 2bp 관통 = 체결 (r26 EPS)


def minute_index(ts):
    return ((pd.DatetimeIndex(ts) - EPOCH) // pd.Timedelta("1min")).astype(np.int64)


def minute_features(k, pr, m, fu):
    """k: 1분봉(ts 인덱스; open high low close quote_volume taker_buy_quote_volume trades),
    pr: 프리미엄 지수 1분 종가 Series(ts), m: 5분 지표(ts = 구간 시작; sum_open_interest, sum_toptrader_long_short_ratio, count_long_short_ratio),
    fu: 펀딩 Series(ts = 정산 시각). 반환 (F dict, rv1, piv) — 전부 k 행 순서의 배열."""
    idx = k.index
    o, h, l, c = (k[x].values for x in ("open", "high", "low", "close"))
    qv = k.quote_volume.values; tb = k.taker_buy_quote_volume.values; tr = k.trades.values
    lr = pd.Series(np.log(c)).diff()
    rv1 = lr.rolling(300).std().shift(1).values
    rvn = lr.rolling(300).std().values
    F = {}
    lc = pd.Series(np.log(c))
    for w in (5, 15, 60, 240, 1440):
        F[f"ret{w}"] = (lc - lc.shift(w)).values / (rvn * np.sqrt(w))
    qs = pd.Series(qv); ts_ = pd.Series(tb); trs = pd.Series(tr)
    base = qs.rolling(1440).mean()
    F["vol5"] = (qs.rolling(5).mean() / base).values
    F["vol60"] = (qs.rolling(60).mean() / base).values
    F["trd5"] = (trs.rolling(5).mean() / trs.rolling(1440).mean()).values
    F["sell5"] = (1 - ts_.rolling(5).sum() / qs.rolling(5).sum()).values
    F["sell60"] = (1 - ts_.rolling(60).sum() / qs.rolling(60).sum()).values
    F["avgsz5"] = ((qs.rolling(5).sum() / trs.rolling(5).sum()) / (qs.rolling(1440).sum() / trs.rolling(1440).sum())).values
    for w in (60, 240, 1440):
        lo = pd.Series(l).rolling(w).min(); hi = pd.Series(h).rolling(w).max()
        F[f"pos{w}"] = ((pd.Series(c) - lo) / (hi - lo)).values
        F[f"dlo{w}"] = (np.log(pd.Series(c) / lo)).values / (rvn * np.sqrt(w))
    rvs = pd.Series(rvn)
    F["rvreg"] = (rvs / rvs.rolling(10080).mean()).values
    F["rvshort"] = (lr.rolling(15).std() / rvs).values
    F["wick5"] = ((pd.Series(c) - pd.Series(l).rolling(5).min()) /
                  (pd.Series(h).rolling(5).max() - pd.Series(l).rolling(5).min())).values
    prr = pr.reindex(idx).ffill()
    F["prem"] = prr.values
    F["premz"] = ((prr - prr.rolling(1440).mean()) / prr.rolling(1440).std()).values
    mm = m.copy(); mm.index = mm.index + pd.Timedelta(minutes=5)   # 5분 지표는 구간 종료 후 사용
    oi = mm.sum_open_interest.reindex(idx, method="ffill")
    F["oi60"] = np.log(oi / oi.shift(60)).values
    F["oi1440"] = np.log(oi / oi.shift(1440)).values
    F["tlsr"] = mm.sum_toptrader_long_short_ratio.reindex(idx, method="ffill").values
    F["lsr"] = mm.count_long_short_ratio.reindex(idx, method="ffill").values
    F["fund"] = fu.reindex(idx, method="ffill").values
    F["hour"] = idx.hour.values.astype(float)
    F["dow"] = idx.dayofweek.values.astype(float)
    # r32 extra (호가 제외)
    qv_ = pd.Series(qv); tb_ = pd.Series(tb); c_ = pd.Series(c); l_ = pd.Series(l)
    net = 2 * tb_ - qv_
    for w in (5, 30):
        F[f"cvd{w}"] = (net.rolling(w).sum() / qv_.rolling(w).sum()).values
    early = (net.rolling(30).sum() - net.rolling(5).sum()) / (qv_.rolling(30).sum() - qv_.rolling(5).sum())
    F["cvddiv"] = (net.rolling(5).sum() / qv_.rolling(5).sum() - early).values
    lr2 = np.log(c_).diff(); rv = lr2.rolling(300).std()
    F["absorb"] = ((qv_.rolling(5).mean() / qv_.rolling(1440).mean()) /
                   (np.abs(np.log(c_ / c_.shift(5))) / (rv * np.sqrt(5)) + 0.25)).values
    piv = l_.shift(30).rolling(210).min()
    F["piv"] = (np.log(c_ / piv) / (rv * np.sqrt(60))).values
    F["swept"] = (l_.rolling(30).min() < piv).astype(float).values
    F["vpin15"] = ((net.abs()).rolling(15).sum() / qv_.rolling(15).sum()).values
    F["vpin60"] = ((net.abs()).rolling(60).sum() / qv_.rolling(60).sum()).values
    sell = qv_ - tb_
    F["accel"] = (sell.rolling(5).mean() / sell.rolling(30).mean()).values
    F["oi15"] = np.log(oi / oi.shift(15)).values
    oi60 = np.log(oi / oi.shift(60)).values
    F["quad"] = (np.sign(np.log(c_ / c_.shift(60)).values) * 2 + np.sign(oi60))
    F["dprem5"] = (prr - prr.shift(5)).values
    F["dprem30"] = (prr - prr.shift(30)).values
    return F, rv1, piv.values


def level_features(a, B, k, o_s0, rv1_s0, c_fm1, piv_fm1, isbtc):
    """층(B, k) 지정가의 체결 후보 분 a 에 대한 층별 특징(연구 r32 main 과 동일). 반환 (지정가 T, dict)."""
    x = k * rv1_s0 * np.sqrt(B); T = o_s0 * (1 - x)
    return T, {"B": float(B), "k": float(k), "el": (a % B) / B, "pre": (c_fm1 / o_s0 - 1) / x,
               "tpiv": np.log(T / piv_fm1) / x, "qh": float(a % 15), "hr0": float(a % 60), "isbtc": float(isbtc)}
