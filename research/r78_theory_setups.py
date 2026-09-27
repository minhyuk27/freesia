"""R78 — 이론 근거가 있는 복합 셋업 73개: '상태(지속 시간) + 방아쇠 + 목표/손절'을 규칙화해 검증.

예(사용자): 거래량이 늘고 변동성이 낮은 상태가 24시간 이상 유지 → 이후 1시간 +0.5% 이상 상승하면 롱, +4%까지(손절 −1%).
격자: 1시간봉 종가에 판정(그 시점까지의 정보만), 진입 = 다음 1분 시가. 체결·청산은 1분봉(같은 분에 손절·목표 → 손절).
목표·손절은 진입가 대비 %, 최대 보유 H시간 뒤 종가 청산. 같은 셋업·종목 안에서는 한 번에 한 포지션.
비용: 진입 테이커 5+2bp, 손절·시간청산 5+2bp, 목표 지정가 2bp.
대조: ① 방아쇠만(상태 조건 제거) ② 무작위 시각(같은 방향·목표·손절, 5회).
통과: n ≥ 60, t ≥ 2, 2020~22·2023~24·2025~26 모두 흑자, ①·② 보다 거래당 순수익이 높음.
사용: python research/r78_theory_setups.py"""
import sys, warnings, time
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R))
from freesia import store
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
CT, CM = 7e-4, 2e-4
EXT = R / "data" / "ext"


def asof(s, idx, delay="0min"):
    s = s.copy(); s.index = s.index + pd.Timedelta(delay); s = s[~s.index.duplicated()].sort_index()
    return s.reindex(s.index.union(idx)).ffill().reindex(idx)


def daily_next(s, idx):
    s = s.copy(); s.index = pd.DatetimeIndex(s.index)
    if s.index.tz is None: s.index = s.index.tz_localize("UTC")
    s.index = s.index.normalize() + pd.Timedelta(days=1); s = s[~s.index.duplicated()].sort_index()
    return s.reindex(s.index.union(idx)).ffill().reindex(idx)


def hourly(sym, other_close):
    k = store.load("klines_1m", sym).set_index("ts")
    lr = np.log(k.close).diff()
    H = k.resample("1h").agg({"open": "first", "high": "max", "low": "min", "close": "last", "quote_volume": "sum",
                              "taker_buy_quote_volume": "sum", "trades": "sum"}).dropna()
    H.columns = ["o", "h", "l", "c", "v", "tb", "n"]
    H["rv"] = lr.resample("1h").std().reindex(H.index)
    E = H.index + pd.Timedelta("1h")                                # 봉 종가 시각
    c = H.c
    F = pd.DataFrame(index=H.index)
    F["o"], F["h"], F["l"], F["c"] = H.o, H.h, H.l, c
    F["r1"] = np.log(c).diff(); F["r4"] = np.log(c / c.shift(4)); F["r24"] = np.log(c / c.shift(24))
    v24 = H.v.rolling(24).sum(); F["v24rel"] = v24 / v24.rolling(720, min_periods=240).median()
    F["vrel1"] = H.v / H.v.rolling(720, min_periods=240).median()
    rv24 = H.rv.rolling(24).mean(); F["rvpct"] = rv24.rolling(720, min_periods=240).rank(pct=True)
    F["rvjump"] = H.rv / H.rv.rolling(24).median().shift(1)
    sd20 = c.rolling(20).std(); F["bbpct"] = (sd20 / c.rolling(20).mean()).rolling(120, min_periods=60).rank(pct=True)
    F["hi24p"] = H.h.rolling(24).max().shift(1); F["lo24p"] = H.l.rolling(24).min().shift(1)
    F["hi72p"] = H.h.rolling(72).max().shift(1); F["lo72p"] = H.l.rolling(72).min().shift(1)
    F["hi7d"] = H.h.rolling(168).max().shift(1); F["lo7d"] = H.l.rolling(168).min().shift(1)
    F["hi30d"] = H.h.rolling(720).max().shift(1); F["lo30d"] = H.l.rolling(720).min().shift(1)
    for n_ in (80, 200, 800): F[f"ema{n_}"] = c.ewm(span=n_, adjust=False).mean()
    tr = pd.concat([H.h - H.l, (H.h - c.shift()).abs(), (H.l - c.shift()).abs()], axis=1).max(axis=1)
    up = H.h.diff(); dn = -H.l.diff(); a14 = tr.ewm(alpha=1 / 14, adjust=False).mean()
    pdi = 100 * pd.Series(np.where((up > dn) & (up > 0), up, 0), index=H.index).ewm(alpha=1 / 14, adjust=False).mean() / a14
    ndi = 100 * pd.Series(np.where((dn > up) & (dn > 0), dn, 0), index=H.index).ewm(alpha=1 / 14, adjust=False).mean() / a14
    F["adx"] = (100 * (pdi - ndi).abs() / (pdi + ndi)).ewm(alpha=1 / 14, adjust=False).mean()
    d = c.diff(); au = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean(); ad = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    F["rsi"] = 100 - 100 / (1 + au / ad)
    tp = (H.h + H.l + c) / 3; vw = (tp * H.v).rolling(24).sum() / H.v.rolling(24).sum(); F["vwz"] = (c - vw) / c.rolling(24).std()
    F["tbr"] = H.tb / H.v; F["tbr4"] = H.tb.rolling(4).sum() / H.v.rolling(4).sum()
    F["nr4"] = (H.h.rolling(4).max() - H.l.rolling(4).min())
    F["nr4min"] = F.nr4.rolling(28).min(); F["nr4hi"] = H.h.rolling(4).max(); F["nr4lo"] = H.l.rolling(4).min()
    day = H.index.floor("D"); dd = H.groupby(day).agg(h=("h", "max"), l=("l", "min"), o=("o", "first"), c=("c", "last"))
    F["pdh"] = dd.h.shift(1).reindex(day).values; F["pdl"] = dd.l.shift(1).reindex(day).values
    upd = (dd.c > dd.o).astype(int); dnd = (dd.c < dd.o).astype(int)
    F["up3"] = upd.rolling(3).sum().shift(1).reindex(day).values; F["dn3"] = dnd.rolling(3).sum().shift(1).reindex(day).values
    F["dch20"] = dd.h.rolling(20).max().shift(1).reindex(day).values; F["dcl20"] = dd.l.rolling(20).min().shift(1).reindex(day).values
    F["newday"] = (H.index.hour == 23)                              # 23시 봉 종가 = 일 마감
    # 파생
    m5 = store.load("metrics_5m", sym).set_index("ts")
    oi = asof(m5.sum_open_interest_value, E, "5min").values; oi = pd.Series(oi, index=H.index)
    F["oi4"] = np.log(oi / oi.shift(4)); F["oi24"] = np.log(oi / oi.shift(24))
    top = pd.Series(asof(m5.sum_toptrader_long_short_ratio, E, "5min").values, index=H.index)
    F["toppct"] = top.rolling(720, min_periods=240).rank(pct=True)
    fu = store.load("funding", sym).set_index("ts").rate; F["fund"] = asof(fu, E).values
    pr = store.load("premium_1m", sym).set_index("ts").close.resample("1h").mean().reindex(H.index)
    F["premz"] = (pr - pr.rolling(720, min_periods=240).mean()) / pr.rolling(720, min_periods=240).std()
    cb = store.load("coinbase_1m", "BTCUSD" if sym == "BTCUSDT" else "ETHUSD").set_index("ts").close
    cbp = pd.Series(np.log(asof(cb, E - pd.Timedelta("1min")).values / c.values), index=H.index)
    F["cbpz"] = (cbp - cbp.rolling(720, min_periods=240).mean()) / cbp.rolling(720, min_periods=240).std()
    dv = pd.read_parquet(EXT / f"dvol_{'btc' if sym == 'BTCUSDT' else 'eth'}_1h.parquet").close
    dva = pd.Series(asof(dv, E, "1h").values, index=H.index)
    F["dvol24"] = np.log(dva / dva.shift(24)); F["dvolpct"] = dva.rolling(720, min_periods=240).rank(pct=True)
    rvann = rv24 * np.sqrt(525600) * 100; ivrv = np.log(dva / rvann); F["ivrvpct"] = ivrv.rolling(720, min_periods=240).rank(pct=True)
    gt = pd.read_parquet(EXT / "gtrends.parquet").set_index("date")["bitcoin" if sym == "BTCUSDT" else "ethereum"]
    g = pd.Series(daily_next(gt, E).values, index=H.index); F["gtwow"] = np.log((g + 1) / (g.shift(168) + 1))
    F["fng"] = daily_next(pd.read_parquet(EXT / "fng.parquet").value, E).values
    oc = np.log(other_close.reindex(H.index)); F["or1"] = oc.diff()
    F["ethbtc"] = (np.log(c) - oc) if sym == "ETHUSDT" else np.nan
    F["ethbtc_hi7d"] = F.ethbtc.rolling(168).max().shift(1)
    F["hour"] = E.hour; F["wday"] = E.weekday
    return F


def hold(cond, k):
    return cond.astype(float).rolling(k, min_periods=k).min() == 1


def setups(F, sym):
    """(코드, 계열, 설명, 상태, 방아쇠, 방향, 목표%, 손절%, 보유시간). 목표·손절이 배열이면 신호별."""
    S = []
    add = lambda *a: S.append(a)
    T = pd.Series(True, index=F.index)
    comp = hold((F.v24rel > 1.2) & (F.rvpct < 0.3), 24)
    add("A01", "A 압축돌파", "거래량↑·저변동 24h 유지 → 1h +0.5% → 롱 +4/−1 (사용자 예)", comp, F.r1 >= 0.005, 1, 4, 1, 72)
    add("A02", "A 압축돌파", "거래량↑·저변동 24h → 1h −0.5% → 숏 +4/−1", comp, F.r1 <= -0.005, -1, 4, 1, 72)
    add("A03", "A 압축돌파", "거래량↑·저변동 24h → 1h +0.5% → 롱 +2/−1", comp, F.r1 >= 0.005, 1, 2, 1, 24)
    sq = hold(F.bbpct < 0.1, 12)
    add("A04", "A 압축돌파", "볼린저 스퀴즈 12h → 24h 고가 돌파 롱 +3/−1", sq, F.c > F.hi24p, 1, 3, 1, 48)
    add("A05", "A 압축돌파", "볼린저 스퀴즈 12h → 24h 저가 이탈 숏 +3/−1", sq, F.c < F.lo24p, -1, 3, 1, 48)
    dry = hold((F.v24rel < 0.8) & (F.rvpct < 0.3), 24)
    add("A06", "A 압축돌파", "거래 고갈·저변동 24h(와이코프 무공급) → 1h +0.5% 롱 +3/−1", dry, F.r1 >= 0.005, 1, 3, 1, 48)
    add("A07", "A 압축돌파", "거래 고갈·저변동 24h → 1h −0.5% 숏 +3/−1", dry, F.r1 <= -0.005, -1, 3, 1, 48)
    nr = (F.nr4.shift(1) <= F.nr4min.shift(1)); add("A08", "A 압축돌파", "4h 범위 28h 최소(Crabel NR) → 그 고가 돌파 롱 +2/−1", nr, F.c > F.nr4hi.shift(1), 1, 2, 1, 24)
    add("A09", "A 압축돌파", "4h 범위 28h 최소 → 그 저가 이탈 숏 +2/−1", nr, F.c < F.nr4lo.shift(1), -1, 2, 1, 24)
    rng = (F.hi72p / F.lo72p - 1) < 0.06
    spring = (F.l < F.lo72p * 0.998) & (F.c > F.lo72p)
    tgt = (F.hi72p / F.c - 1) * 100; stp = (F.c / F.l - 1) * 100 + 0.1
    add("A10", "A 압축돌파", "와이코프 스프링: 72h 박스(<6%) 저가 하회 후 박스 안 종가 → 롱, 목표 박스 상단", rng, spring, 1, tgt, stp, 72)
    upth = (F.h > F.hi72p * 1.002) & (F.c < F.hi72p)
    add("A11", "A 압축돌파", "업스러스트: 박스 상단 상회 후 박스 안 종가 → 숏, 목표 박스 하단", rng, upth, -1, (F.c / F.lo72p - 1) * 100, (F.h / F.c - 1) * 100 + 0.1, 72)
    upt = hold((F.ema80 > F.ema200) & (F.ema200 > F.ema800) & (F.adx > 25), 12); dnt = hold((F.ema80 < F.ema200) & (F.ema200 < F.ema800) & (F.adx > 25), 12)
    add("B01", "B 추세", "강한 상승추세 12h → 4h −1% 눌림 롱 +2/−1", upt, F.r4 <= -0.01, 1, 2, 1, 24)
    add("B02", "B 추세", "강한 하락추세 12h → 4h +1% 반등 숏 +2/−1", dnt, F.r4 >= 0.01, -1, 2, 1, 24)
    add("B03", "B 추세", "30일 신고가 돌파 + 1h 거래량 2배 → 롱 +3/−1.5", T, (F.c > F.hi30d) & (F.vrel1 > 2), 1, 3, 1.5, 72)
    add("B04", "B 추세", "30일 신저가 이탈 + 거래량 2배 → 숏 +3/−1.5", T, (F.c < F.lo30d) & (F.vrel1 > 2), -1, 3, 1.5, 72)
    add("B05", "B 추세", "7일 고가 돌파 + 24h 상승 + 펀딩 중앙 이하(과열 아님) → 롱 +2/−1", (F.fund <= F.fund.rolling(720, min_periods=240).median()), (F.c > F.hi7d) & (F.r24 > 0), 1, 2, 1, 48)
    add("B06", "B 추세", "7일 저가 이탈 + 펀딩 중앙 이상 → 숏 +2/−1", (F.fund >= F.fund.rolling(720, min_periods=240).median()), (F.c < F.lo7d) & (F.r24 < 0), -1, 2, 1, 48)
    add("B07", "B 추세", "3일 연속 양일 + 펀딩 ≤0.01% → 일 마감 롱 +2/−1", (F.up3 == 3) & (F.fund <= 1e-4), F.newday, 1, 2, 1, 24)
    add("B08", "B 추세", "3일 연속 음일 + 펀딩 ≥0.01% → 일 마감 숏 +2/−1", (F.dn3 == 3) & (F.fund >= 1e-4), F.newday, -1, 2, 1, 24)
    add("B09", "B 추세", "20일 돈치안 돌파(터틀) → 롱 +6/−2", T, (F.c > F.dch20) & (F.c.shift(1) <= F.dch20), 1, 6, 2, 240)
    add("B10", "B 추세", "20일 돈치안 하향 이탈 → 숏 +6/−2", T, (F.c < F.dcl20) & (F.c.shift(1) >= F.dcl20), -1, 6, 2, 240)
    add("C01", "C 평균회귀", "1h −3% 투매 + 거래량 3배 → 롱 +1.5/−1.5", T, (F.r1 <= -0.03) & (F.vrel1 >= 3), 1, 1.5, 1.5, 12)
    add("C02", "C 평균회귀", "1h +3% 급등 + 거래량 3배 → 숏 +1.5/−1.5", T, (F.r1 >= 0.03) & (F.vrel1 >= 3), -1, 1.5, 1.5, 12)
    add("C03", "C 평균회귀", "24h −8% + RSI<20 → 롱 +3/−2", T, (F.r24 <= -0.08) & (F.rsi < 20), 1, 3, 2, 48)
    add("C04", "C 평균회귀", "24h +8% + RSI>80 → 숏 +3/−2", T, (F.r24 >= 0.08) & (F.rsi > 80), -1, 3, 2, 48)
    add("C05", "C 평균회귀", "24h VWAP −3σ 이탈 → 롱 +1.5/−1.5", T, F.vwz <= -3, 1, 1.5, 1.5, 12)
    add("C06", "C 평균회귀", "24h VWAP +3σ 이탈 → 숏 +1.5/−1.5", T, F.vwz >= 3, -1, 1.5, 1.5, 12)
    neg5 = hold(F.r1 < 0, 5); pos5 = hold(F.r1 > 0, 5)
    add("C07", "C 평균회귀", "5연속 음봉 + 거래량 감소(매도 소진) → 롱 +1/−1", neg5, F.v24rel < F.v24rel.shift(5), 1, 1, 1, 12)
    add("C08", "C 평균회귀", "5연속 양봉 + 거래량 감소(매수 소진) → 숏 +1/−1", pos5, F.v24rel < F.v24rel.shift(5), -1, 1, 1, 12)
    lev = (F.oi24 >= 0.08) & (F.r24.abs() <= 0.01)
    add("D01", "D 포지셔닝", "OI 24h +8% & 가격 제자리(레버리지 축적) → 1h +0.7% 롱 +3/−1", lev, F.r1 >= 0.007, 1, 3, 1, 48)
    add("D02", "D 포지셔닝", "레버리지 축적 → 1h −0.7% 숏 +3/−1", lev, F.r1 <= -0.007, -1, 3, 1, 48)
    add("D03", "D 포지셔닝", "롱 청산 폭포(4h −4% & OI −5%) → 롱 +2/−1.5", T, (F.r4 <= -0.04) & (F.oi4 <= -0.05), 1, 2, 1.5, 24)
    add("D04", "D 포지셔닝", "숏 스퀴즈(4h +4% & OI −5%) → 숏 +2/−1.5", T, (F.r4 >= 0.04) & (F.oi4 <= -0.05), -1, 2, 1.5, 24)
    add("D05", "D 포지셔닝", "펀딩 >0.03% 24h 유지(롱 과밀) → 1h −1% 숏 +3/−1", hold(F.fund > 3e-4, 24), F.r1 <= -0.01, -1, 3, 1, 48)
    add("D06", "D 포지셔닝", "펀딩 <−0.01% 24h 유지(숏 과밀) → 1h +1% 롱 +3/−1", hold(F.fund < -1e-4, 24), F.r1 >= 0.01, 1, 3, 1, 48)
    add("D07", "D 포지셔닝", "탑트레이더 롱비율 상위10% → 4h −1% 숏 +2/−1", F.toppct > 0.9, F.r4 <= -0.01, -1, 2, 1, 24)
    add("D08", "D 포지셔닝", "탑트레이더 롱비율 하위10% → 4h +1% 롱 +2/−1", F.toppct < 0.1, F.r4 >= 0.01, 1, 2, 1, 24)
    add("D09", "D 포지셔닝", "매수체결 우위 4h 유지인데 가격 제자리(매도 흡수) → 숏 +1.5/−1", hold(F.tbr4 > 0.55, 4), F.r4.abs() < 0.003, -1, 1.5, 1, 24)
    add("D10", "D 포지셔닝", "매도체결 우위 4h 유지인데 가격 제자리(매수 흡수) → 롱 +1.5/−1", hold(F.tbr4 < 0.45, 4), F.r4.abs() < 0.003, 1, 1.5, 1, 24)
    add("E01", "E 교차", "코인베이스 프리미엄 z>2 4h 유지 + 1h 상승 → 롱 +2/−1", hold(F.cbpz > 2, 4), F.r1 > 0, 1, 2, 1, 24)
    add("E02", "E 교차", "코인베이스 디스카운트 z<−2 4h 유지 + 1h 하락 → 숏 +2/−1", hold(F.cbpz < -2, 4), F.r1 < 0, -1, 2, 1, 24)
    add("E03", "E 교차", "무기한 프리미엄 z<−2(선물 할인) + 1h 상승 → 롱 +1.5/−1", F.premz < -2, F.r1 > 0, 1, 1.5, 1, 24)
    add("E04", "E 교차", "BTC 1h +1.5%인데 ETH <+0.3% → ETH 롱 +1/−1 (추격)", pd.Series(sym == "ETHUSDT", index=F.index), (F.or1 >= 0.015) & (F.r1 < 0.003), 1, 1, 1, 6)
    add("E05", "E 교차", "BTC 1h −1.5%인데 ETH >−0.3% → ETH 숏 +1/−1", pd.Series(sym == "ETHUSDT", index=F.index), (F.or1 <= -0.015) & (F.r1 > -0.003), -1, 1, 1, 6)
    add("E06", "E 교차", "ETH/BTC 7일 고점 돌파 → ETH 롱 +3/−1.5", pd.Series(sym == "ETHUSDT", index=F.index), F.ethbtc > F.ethbtc_hi7d, 1, 3, 1.5, 72)
    add("F01", "F 변동성", "공포 정점: DVOL 24h +20% & 24h −5% → 롱 +3/−2", T, (F.dvol24 >= np.log(1.2)) & (F.r24 <= -0.05), 1, 3, 2, 48)
    ivsq = (F.dvolpct < 0.1) & (F.v24rel > 1.2)
    add("F02", "F 변동성", "내재변동성 하위10% + 거래량↑ → 1h +0.5% 롱 +3/−1", ivsq, F.r1 >= 0.005, 1, 3, 1, 48)
    add("F03", "F 변동성", "내재변동성 하위10% + 거래량↑ → 1h −0.5% 숏 +3/−1", ivsq, F.r1 <= -0.005, -1, 3, 1, 48)
    add("F04", "F 변동성", "IV/RV 상위10%(옵션 비쌈 = 불안) & 4h +1% → 숏 +1/−1", F.ivrvpct > 0.9, F.r4 >= 0.01, -1, 1, 1, 12)
    add("F05", "F 변동성", "변동성 점프(1h rv 5배) + 상승 → 추종 롱 +2/−1", T, (F.rvjump >= 5) & (F.r1 > 0), 1, 2, 1, 24)
    add("F06", "F 변동성", "변동성 점프 + 하락 → 추종 숏 +2/−1", T, (F.rvjump >= 5) & (F.r1 < 0), -1, 2, 1, 24)
    dayk = F.index.floor("D"); hr = F.index.hour
    ash = F.h.where(hr < 8).groupby(dayk).transform("max"); asl = F.l.where(hr < 8).groupby(dayk).transform("min")
    lon = (F.hour >= 9) & (F.hour <= 11)
    add("G01", "G 시간", "아시아(00~08) 고가를 런던 초반 돌파 → 롱 +1.5/−0.75", lon, F.c > ash, 1, 1.5, 0.75, 8)
    add("G02", "G 시간", "아시아 저가를 런던 초반 이탈 → 숏 +1.5/−0.75", lon, F.c < asl, -1, 1.5, 0.75, 8)
    ny = F.hour == 15
    add("G03", "G 시간", "뉴욕 첫 1시간(14~15시) +0.5% → 추종 롱 +1.5/−0.75", ny, F.r1 >= 0.005, 1, 1.5, 0.75, 6)
    add("G04", "G 시간", "뉴욕 첫 1시간 −0.5% → 추종 숏 +1.5/−0.75", ny, F.r1 <= -0.005, -1, 1.5, 0.75, 6)
    wkey = F.index.floor("D") + pd.to_timedelta((7 - F.index.weekday) % 7, unit="D")   # 토·일 → 다음 월요일
    wkh = F.h.where(F.index.weekday >= 5).groupby(wkey).transform("max"); wkl = F.l.where(F.index.weekday >= 5).groupby(wkey).transform("min")
    mon = (F.wday == 0) & (F.hour <= 12)
    add("G05", "G 시간", "주말 레인지 고가를 월요일 오전 돌파 → 롱 +2/−1", mon, F.c > wkh, 1, 2, 1, 24)
    add("G06", "G 시간", "주말 레인지 저가를 월요일 오전 이탈 → 숏 +2/−1", mon, F.c < wkl, -1, 2, 1, 24)
    add("H01", "H 관심·심리", "구글 관심 주간 +50% & 24h +5% → 관심 과열 반전 숏 +3/−2", F.gtwow >= np.log(1.5), F.r24 >= 0.05, -1, 3, 2, 72)
    add("H02", "H 관심·심리", "극단 공포(<15) → 1h +1% 롱 +3/−1.5", F.fng < 15, F.r1 >= 0.01, 1, 3, 1.5, 72)
    add("H03", "H 관심·심리", "극단 탐욕(>85) → 1h −1% 숏 +3/−1.5", F.fng > 85, F.r1 <= -0.01, -1, 3, 1.5, 72)
    add("H04", "H 관심·심리", "관심 주간 −20%인데 거래량↑(조용한 축적) → 1h +0.5% 롱 +3/−1", (F.gtwow <= np.log(0.8)) & (F.v24rel > 1.3), F.r1 >= 0.005, 1, 3, 1, 48)
    add("I01", "I 구조", "전일 고가 +0.2% 돌파 & 거래량 1.5배 → 롱 +2/−1", T, (F.c > F.pdh * 1.002) & (F.c.shift(1) < F.pdh) & (F.vrel1 > 1.5), 1, 2, 1, 24)
    add("I02", "I 구조", "전일 저가 −0.2% 이탈 & 거래량 1.5배 → 숏 +2/−1", T, (F.c < F.pdl * 0.998) & (F.c.shift(1) > F.pdl) & (F.vrel1 > 1.5), -1, 2, 1, 24)
    lo72 = F.l.rolling(72).min(); old = F.l.rolling(72).apply(lambda x: 71 - np.argmin(x), raw=True) >= 12
    retest = (F.l.rolling(12).min() <= lo72 * 1.003)
    add("I03", "I 구조", "이중바닥(72h 최저 12h+ 전, 최근 12h 재시험 유지) → 1h +0.5% 롱 +2/−1", old & retest, F.r1 >= 0.005, 1, 2, 1, 48)
    fb = (F.l.rolling(2).min() < F.lo7d.shift(2)) & (F.c > F.lo7d.shift(2))
    add("I04", "I 구조", "7일 저가 하회 후 2h 안 재탈환(실패한 이탈) → 롱 +2/−1", T, fb & ~fb.shift(1).fillna(False).astype(bool), 1, 2, 1, 24)
    fu_ = (F.h.rolling(2).max() > F.hi7d.shift(2)) & (F.c < F.hi7d.shift(2))
    add("I05", "I 구조", "7일 고가 상회 후 2h 안 반락(실패한 돌파) → 숏 +2/−1", T, fu_ & ~fu_.shift(1).fillna(False).astype(bool), -1, 2, 1, 24)
    stp_ = 1000 if sym == "BTCUSDT" else 100; rn = np.floor(F.c.shift(1) / stp_) * stp_ + stp_
    first = (F.h.rolling(24).max().shift(1) < rn)
    add("I06", "I 구조", "라운드넘버(1000$/100$) 24h 만에 첫 돌파 +0.2% → 롱 +1.5/−1", first, F.c >= rn * 1.002, 1, 1.5, 1, 24)
    j1 = hold(F.rvpct < 0.3, 24) & (F.oi24 > 0.05) & (F.fund.abs() < 1e-4)
    add("J01", "J 복합", "저변동 24h + OI 24h +5% + 펀딩 중립 → 1h +0.5% 롱 +4/−1", j1, F.r1 >= 0.005, 1, 4, 1, 72)
    add("J02", "J 복합", "저변동 24h + OI↑ + 펀딩 중립 → 1h −0.5% 숏 +4/−1", j1, F.r1 <= -0.005, -1, 4, 1, 72)
    add("J03", "J 복합", "4h −4% 투매 + 펀딩 음수 + DVOL 24h +10% → 롱 +3/−1.5", T, (F.r4 <= -0.04) & (F.fund < 0) & (F.dvol24 >= np.log(1.1)), 1, 3, 1.5, 48)
    add("J04", "J 복합", "상승추세 + 1h EMA200 터치 후 위 종가 + 매수체결 우위 → 롱 +2/−1", upt.shift(1).fillna(False).astype(bool), (F.l <= F.ema200) & (F.c > F.ema200) & (F.tbr > 0.52), 1, 2, 1, 24)
    j5 = ((F.hi24p / F.lo24p - 1) < 0.02) & (F.v24rel < 0.9) & (F.gtwow > np.log(1.2))
    add("J05", "J 복합", "24h 범위<2% + 거래 고갈 + 관심 주간 +20% → 1h +0.5% 롱 +3/−1", j5, F.r1 >= 0.005, 1, 3, 1, 48)
    add("J06", "J 복합", "같은 상태 → 1h −0.5% 숏 +3/−1", j5, F.r1 <= -0.005, -1, 3, 1, 48)
    return S


@numba.njit(cache=True)
def run_trades(ent, side, tg, st, hold_m, o, h, l, c):
    """ent: 진입 분 인덱스(정렬), 한 번에 한 포지션. 반환 (사용 여부, 순수익, 청산 분)."""
    n = len(c); m = len(ent); used = np.zeros(m, np.bool_); ret = np.zeros(m); busy = -1
    for q in range(m):
        i = ent[q]
        if i <= busy or i >= n - 1 or not (tg[q] > 0) or not (st[q] > 0): continue
        e = o[i]; s = side[q]; up = e * (1 + s * tg[q] / 100); dn = e * (1 - s * st[q] / 100)
        j = i; r = np.nan
        while j < min(n, i + hold_m[q]):
            if s > 0:
                if l[j] <= dn: r = (min(dn, o[j]) if j > i else dn) / e - 1 - 2 * CT_; break
                if h[j] >= up: r = up / e - 1 - CT_ - CM_; break
            else:
                if h[j] >= dn: r = -((max(dn, o[j]) if j > i else dn) / e - 1) - 2 * CT_; break
                if l[j] <= up: r = -(up / e - 1) - CT_ - CM_; break
            j += 1
        if np.isnan(r):
            j = min(j, n - 1); r = s * (c[j] / e - 1) - 2 * CT_
        used[q] = True; ret[q] = r; busy = j
    return used, ret


CT_, CM_ = CT, CM


def evaluate(sig, F, K, side, tg, st, H):
    """sig: 불리언(봉 종가 판정) → 다음 분 진입. 반환 거래 DataFrame."""
    idx = np.where(sig.fillna(False).values)[0]
    if len(idx) == 0: return pd.DataFrame(columns=["t", "r"])
    t_entry = F.index[idx] + pd.Timedelta("1h")
    ent = K.index.searchsorted(t_entry)
    ok = ent < len(K) - 2; idx, ent, t_entry = idx[ok], ent[ok], t_entry[ok]
    arr = lambda x: (x.values[idx] if isinstance(x, pd.Series) else np.full(len(idx), float(x))).astype(np.float64)
    sd = arr(side if isinstance(side, pd.Series) else pd.Series(side, index=F.index)).astype(np.int64)
    used, ret = run_trades(ent.astype(np.int64), sd, arr(tg), arr(st), np.full(len(idx), H * 60, np.int64),
                           K.open.values, K.high.values, K.low.values, K.close.values)
    return pd.DataFrame({"t": t_entry[used], "r": ret[used]})


if __name__ == "__main__":
    t_ = time.time()
    cl = {s: store.load("klines_1m", s).set_index("ts").close.resample("1h").last() for s in ("BTCUSDT", "ETHUSDT")}
    rows = []; rng = np.random.default_rng(0)
    res = {}
    for sym in ("BTCUSDT", "ETHUSDT"):
        F = hourly(sym, cl["ETHUSDT" if sym == "BTCUSDT" else "BTCUSDT"]); K = store.load("klines_1m", sym).set_index("ts")
        P(f"{sym} 1시간 특징 {F.shape} ({time.time()-t_:.0f}s)")
        for code, fam, desc, state, trig, side, tg, st, H in setups(F, sym):
            state = state.fillna(False).astype(bool); trig = trig.fillna(False).astype(bool)
            main = evaluate(state & trig, F, K, side, tg, st, H)
            ctrl = evaluate(trig, F, K, side, tg, st, H) if not state.all() else main
            rnds = []
            if len(main):
                for _ in range(5):
                    pick = pd.Series(False, index=F.index); ii = rng.choice(np.where(F.c.notna())[0][800:], size=len(main), replace=False)
                    pick.iloc[ii] = True
                    tg_r = tg.median() if isinstance(tg, pd.Series) else tg; st_r = st.median() if isinstance(st, pd.Series) else st
                    rnds.append(evaluate(pick, F, K, side, tg_r, st_r, H).r.mean())
            res.setdefault(code, {"fam": fam, "desc": desc, "main": [], "ctrl": [], "rnd": []})
            res[code]["main"].append(main); res[code]["ctrl"].append(ctrl); res[code]["rnd"] += rnds
        P(f"{sym} 셋업 완료 ({time.time()-t_:.0f}s)")
    days = (cl["BTCUSDT"].index[-1] - cl["BTCUSDT"].index[0]).days
    for code, d in res.items():
        M = pd.concat(d["main"]); C = pd.concat(d["ctrl"])
        n = len(M)
        if n == 0: rows.append((code, d["fam"], d["desc"], 0) + (np.nan,) * 11); continue
        mean = M.r.mean(); t = mean / (M.r.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
        per = [M[(M.t >= pd.Timestamp(a, tz="UTC")) & (M.t < pd.Timestamp(b, tz="UTC"))].r.mean() for a, b in
               (("2020-01-01", "2023-01-01"), ("2023-01-01", "2025-01-01"), ("2025-01-01", "2027-01-01"))]
        rows.append((code, d["fam"], d["desc"], n, n / days, (M.r > 0).mean(), mean * 1e4, t, *(np.array(per) * 1e4),
                     C.r.mean() * 1e4, np.nanmean(d["rnd"]) * 1e4, len(C)))
    O = pd.DataFrame(rows, columns=["코드", "계열", "설명", "n", "하루", "승률", "순bp", "t", "20~22", "23~24", "25~26", "방아쇠만bp", "무작위bp", "방아쇠만n"])
    O["통과"] = (O.n >= 60) & (O.t >= 2) & (O[["20~22", "23~24", "25~26"]] > 0).all(axis=1) & (O.순bp > O.방아쇠만bp) & (O.순bp > O.무작위bp)
    O.to_csv(R / "logs" / "r78.csv", index=False)
    pd.set_option("display.width", 260); pd.set_option("display.max_rows", 200); pd.set_option("display.max_colwidth", 60)
    P(f"\n===== 셋업 {len(O)}개, 통과 {O.통과.sum()}개 (거래당 순수익 bp, 비용 포함)")
    P(O.drop(columns=["설명"]).round(2).to_string(index=False))
    P("\n  설명"); [P(f"  {r.코드}: {r.설명}") for r in O.itertuples()]
    (R / "logs" / "r78.summary").write_text(f"R78 이론 셋업 {len(O)}개, 통과 {O.통과.sum()}개\n")
