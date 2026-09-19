"""호가 데이터 없이 가능한 초단기 접근법 배터리 (BTC·ETH 각각). 1분봉·체결합계·프리미엄·펀딩·OI(5분)·교차자산만 사용.
결정: 5분마다(1분 종가). 결과: triple-barrier 5가지 기하. 학습 2020~2023 / 테스트 2024~2026. 비용 taker 왕복 12bp.
무작위 진입 대조군 300개를 같은 파이프라인으로 함께 평가. 사용: python research/hf_battery.py <SYMBOL>"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from freesia import store  # noqa: E402
from freesia.bt import ind  # noqa: E402
from hf_barrier import barrier, FEE, SLIP  # noqa: E402
import numba  # noqa: E402

GEOMS = {"g30": (0.003, 0.003, 60), "g55": (0.005, 0.005, 120), "g510": (0.005, 0.010, 240), "g11": (0.010, 0.010, 480), "g105": (0.010, 0.005, 480)}
COST = 2 * FEE + 2 * SLIP
STEP = 5
TFS = (5, 15, 60)


@numba.njit(cache=True)
def pick(ev, sd, stl, sts):
    out = np.zeros(len(ev), np.bool_)
    nxt = -1
    for i in range(len(ev)):
        e = ev[i]
        if e < nxt:
            continue
        out[i] = True
        nxt = e + (stl[e] if sd[i] == 1 else sts[e])
    return out


class Reg:
    def __init__(self):
        self.rules = {}
        self.bases = set()
        self.cat = {}

    def add(self, cat, base, var, L, S):
        self.bases.add(f"{cat}/{base}")
        self.cat[f"{cat}/{base}"] = cat
        self.rules[f"{cat}/{base} | {var}"] = (np.asarray(L, bool), np.asarray(S, bool))

    def both(self, cat, base, var, L, S):     # 추종/역추종 한 쌍
        self.add(cat, base, var + " 추종", L, S)
        self.add(cat, base, var + " 역추종", S, L)


def sh(a, k, fill=False):
    a = np.asarray(a)
    return np.concatenate([np.full(k, fill, dtype=a.dtype), a[:-k]]) if k > 0 else a


def build(sym, other):
    k = store.load("klines_1m", sym).set_index("ts"); k = k[k.index >= "2020-01-01"]
    ko = store.load("klines_1m", other).set_index("ts").reindex(k.index)
    n = (len(k) // 60) * 60
    k, ko = k.iloc[:n], ko.iloc[:n]
    o1, h1, l1, c1 = (k[x].to_numpy(np.float64) for x in ("open", "high", "low", "close"))
    v1, tb1, tr1 = k.quote_volume.to_numpy(np.float64), k.taker_buy_quote_volume.to_numpy(np.float64), k.trades.to_numpy(np.float64)
    co1, vo1, tbo1 = ko.close.to_numpy(np.float64), ko.quote_volume.to_numpy(np.float64), ko.taker_buy_quote_volume.to_numpy(np.float64)
    pr = store.load("premium_1m", sym).set_index("ts").close.reindex(k.index).to_numpy(np.float64)
    fr = store.load("funding", sym).set_index("ts").rate.reindex(k.index, method="ffill").to_numpy(np.float64)
    met = store.load("metrics_5m", sym)
    oi = met.set_index("ts")["sum_open_interest_value"].reindex(k.index, method="ffill", limit=10).to_numpy(np.float64) if met is not None else np.full(n, np.nan)
    idx = k.index
    dec = np.arange(1500, n - 1450)
    dec = dec[(dec + 1) % STEP == 0]
    return dict(k=k, n=n, o1=o1, h1=h1, l1=l1, c1=c1, v1=v1, tb1=tb1, tr1=tr1, co1=co1, vo1=vo1, tbo1=tbo1, pr=pr, fr=fr, oi=oi, idx=idx, dec=dec)


def rules(D):
    dec, n, idx = D["dec"], D["n"], D["idx"]
    R = Reg()
    o1, h1, l1, c1, v1, tb1, tr1 = D["o1"], D["h1"], D["l1"], D["c1"], D["v1"], D["tb1"], D["tr1"]
    KB = {m: np.clip((dec + 1) // m - 1, 0, None) for m in TFS + (240, 1440)}
    NEW = {m: ((dec + 1) % m == 0) for m in TFS + (240, 1440)}
    tod = (dec + 1) % 1440                                  # 결정 시각의 UTC 자정 이후 분
    dow = np.asarray(idx.dayofweek)[dec]

    def bars(m):
        nn = (n // m) * m
        r = lambda a: a[:nn].reshape(-1, m)  # noqa: E731
        return r(o1)[:, 0], r(h1).max(1), r(l1).min(1), r(c1)[:, -1], r(v1).sum(1), r(tb1).sum(1), r(tr1).sum(1)

    B = {m: bars(m) for m in TFS + (240, 1440)}

    def S(a, m):
        return np.asarray(a)[np.minimum(KB[m], len(a) - 1)]

    def E(ev, m):                                       # 봉 이벤트 → 결정 시점(봉 마감 시)
        return NEW[m] & S(ev, m)

    def x1(a):                                           # 1분 배열 → 결정 시점
        return np.asarray(a)[dec]

    def ps(a):
        return pd.Series(a)

    # ================= A. 캔들 패턴 (봉 5/15/60분) =================
    for m in TFS:
        o, h, l, c, v, tb, tr = B[m]
        po, pc, ph, pl = sh(o, 1, np.nan), sh(c, 1, np.nan), sh(h, 1, np.nan), sh(l, 1, np.nan)
        body, rng = np.abs(c - o), np.maximum(h - l, 1e-12)
        up, dn = c > o, c < o
        pup, pdn = sh(up, 1), sh(dn, 1)
        lw, uw = np.minimum(o, c) - l, h - np.maximum(o, c)
        P = {}
        P["장악형"] = (pdn & up & (c >= po) & (o <= pc), pup & dn & (c <= po) & (o >= pc))
        P["해머·유성형"] = ((lw >= 2 * body) & (uw <= body) & (body > 0), (uw >= 2 * body) & (lw <= body) & (body > 0))
        P["도지 후 방향"] = (sh(body <= 0.1 * rng, 1) & up, sh(body <= 0.1 * rng, 1) & dn)
        st_dn = sh(pdn, 1) & sh(body <= 0.3 * sh(rng, 1), 1)
        P["샛별·석별형"] = (sh(pdn, 2) & sh(body <= 0.3 * rng, 1) & up & (c > (sh(o, 2) + sh(c, 2)) / 2), sh(pup, 2) & sh(body <= 0.3 * rng, 1) & dn & (c < (sh(o, 2) + sh(c, 2)) / 2))
        P["적삼병·흑삼병"] = (up & pup & sh(up, 2) & (c > pc) & (pc > sh(c, 2)), dn & pdn & sh(dn, 2) & (c < pc) & (pc < sh(c, 2)))
        ins = (h <= ph) & (l >= pl)
        P["인사이드바 돌파"] = (sh(ins, 1) & (c > ph), sh(ins, 1) & (c < pl))
        P["아웃사이드바 방향"] = ((h > ph) & (l < pl) & up, (h > ph) & (l < pl) & dn)
        P["마루보즈"] = ((body >= 0.9 * rng) & up, (body >= 0.9 * rng) & dn)
        tol = 0.0005 * c
        P["트위저 바닥·천장"] = (pdn & up & (np.abs(l - pl) <= tol), pup & dn & (np.abs(h - ph) <= tol))
        P["하라미"] = (pdn & (np.abs(pc - po) > 0.6 * (ph - pl)) & up & (c < po) & (o > pc), pup & (np.abs(pc - po) > 0.6 * (ph - pl)) & dn & (c > po) & (o < pc))
        P["관통·먹구름형"] = (pdn & up & (o < pc) & (c > (po + pc) / 2) & (c < po), pup & dn & (o > pc) & (c < (po + pc) / 2) & (c > po))
        lo20, hi20 = ps(l).rolling(20).min().to_numpy(), ps(h).rolling(20).max().to_numpy()
        P["극단 핀바"] = ((lw >= 2 * body) & (l <= lo20), (uw >= 2 * body) & (h >= hi20))
        clv = (c - l) / rng
        P["종가위치 극단(CLV)"] = ((clv >= 0.9) & up, (clv <= 0.1) & dn)
        for nm, (bu, be) in P.items():
            R.both("캔들", nm, f"{m}m", E(bu, m), E(be, m))

    # ================= B. 오실레이터·지표 (봉 5/15/60분) =================
    def flip(st):
        st = np.asarray(st, bool)
        return st & ~sh(st, 1), ~st & sh(st, 1)

    for m in TFS:
        o, h, l, c, v, tb, tr = B[m]
        cs, hs, ls, vs = ps(c), ps(h), ps(l), ps(v)
        for nn in (2, 5, 14):
            r = ind.rsi(cs, nn).to_numpy(); lo_, hi_ = (10, 90) if nn == 2 else (25, 75) if nn == 5 else (30, 70)
            R.both("오실레이터", f"RSI{nn} 과매도·과매수 진입", f"{m}m", E((r < lo_) & ~(sh(r, 1) < lo_), m), E((r > hi_) & ~(sh(r, 1) > hi_), m))
        k_, d_ = ind.stoch(hs, ls, cs, 14)
        kk = k_.to_numpy(); R.both("오실레이터", "스토캐스틱 14 극단 이탈", f"{m}m", E((kk > 20) & (sh(kk, 1) <= 20), m), E((kk < 80) & (sh(kk, 1) >= 80), m))
        z = ind.zscore_tp(hs, ls, cs, 20).to_numpy(); R.both("오실레이터", "CCI(근사) ±2 극단", f"{m}m", E((z < -2) & ~(sh(z, 1) < -2), m), E((z > 2) & ~(sh(z, 1) > 2), m))
        mf = ind.mfi(hs, ls, cs, vs, 14).to_numpy(); R.both("오실레이터", "MFI14 극단", f"{m}m", E((mf < 20) & ~(sh(mf, 1) < 20), m), E((mf > 80) & ~(sh(mf, 1) > 80), m))
        for nn in (3, 12):
            roc = (cs / cs.shift(nn) - 1); zz = (roc / roc.rolling(200).std()).to_numpy()
            R.both("오실레이터", f"ROC{nn} 극단(z2.5)", f"{m}m", E(zz < -2.5, m), E(zz > 2.5, m))
        mm, sg, hi = ind.macd(cs, 12, 26, 9)
        u, d = flip((sg < mm).to_numpy()); R.both("추세지표", "MACD 시그널 교차", f"{m}m", E(u, m), E(d, m))
        u, d = flip((mm > 0).to_numpy()); R.both("추세지표", "MACD 제로 교차", f"{m}m", E(u, m), E(d, m))
        for kb in (2.0,):
            bm, bu, bl = ind.bbands(cs, 20, kb); cbu, cbl = c > bu.to_numpy(), c < bl.to_numpy()
            R.both("변동성", "볼린저 밴드 이탈 후 재진입", f"{m}m", E(sh(cbl, 1) & (c > bl.to_numpy()), m), E(sh(cbu, 1) & (c < bu.to_numpy()), m))
            R.both("변동성", "볼린저 밴드 돌파", f"{m}m", E(cbu & ~sh(cbu, 1), m), E(cbl & ~sh(cbl, 1), m))
            w = ((bu - bl) / bm).to_numpy(); q = ps(w).rolling(500).quantile(0.1).to_numpy(); sq = sh(w < q, 1)
            R.both("변동성", "볼린저 스퀴즈 후 돌파", f"{m}m", E(sq & cbu, m), E(sq & cbl, m))
        km, ku, kl = ind.keltner(hs, ls, cs, 20, 2.0); R.both("추세지표", "켈트너 채널 돌파", f"{m}m", E((cs > ku).to_numpy() & ~sh((cs > ku).to_numpy(), 1), m), E((cs < kl).to_numpy() & ~sh((cs < kl).to_numpy(), 1), m))
        for N in (20, 55):
            hh, ll = hs.shift(1).rolling(N).max().to_numpy(), ls.shift(1).rolling(N).min().to_numpy()
            R.both("돌파", f"돈치안 {N}봉 돌파", f"{m}m", E(c > hh, m), E(c < ll, m))
        pdi, mdi, adx = ind.adx(hs, ls, cs, 14); a_ = adx.to_numpy(); u, d = flip((pdi > mdi).to_numpy())
        R.both("추세지표", "ADX>20 DI 교차", f"{m}m", E(u & (a_ > 20), m), E(d & (a_ > 20), m))
        for nn_, mult in ((10, 3), (7, 2)):
            sd_ = ind.supertrend(hs, ls, cs, nn_, mult); R.both("추세지표", f"슈퍼트렌드({nn_},{mult}) 전환", f"{m}m", E((sd_ > 0) & (sh(sd_, 1) <= 0), m), E((sd_ < 0) & (sh(sd_, 1) >= 0), m))
        pd_ = ind.psar(hs, ls, 0.02, 0.2); R.both("추세지표", "파라볼릭 SAR 전환", f"{m}m", E((pd_ > 0) & (sh(pd_, 1) < 0), m), E((pd_ < 0) & (sh(pd_, 1) > 0), m))
        ho, hc = ind.heikin_ashi(ps(o), hs, ls, cs); hb = hc > ho
        R.both("추세지표", "하이켄아시 색 전환(직전 2봉 반대)", f"{m}m", E(hb & ~sh(hb, 1) & ~sh(hb, 2), m), E(~hb & sh(hb, 1) & sh(hb, 2), m))
        if m >= 15:
            au, ad = ind.aroon(hs, ls, 25); u, d = flip((au > ad).to_numpy()); R.both("추세지표", "아룬(25) 교차", f"{m}m", E(u, m), E(d, m))
        vp, vm = ind.vortex(hs, ls, cs, 14); u, d = flip((vp > vm).to_numpy()); R.both("추세지표", "보텍스 교차", f"{m}m", E(u, m), E(d, m))
        tx = ind.trix(cs, 9).to_numpy(); R.both("추세지표", "TRIX 제로 교차", f"{m}m", E((tx > 0) & (sh(tx, 1) <= 0), m), E((tx < 0) & (sh(tx, 1) >= 0), m))
        tp_ = (hs + ls) / 2; ao = (tp_.rolling(5).mean() - tp_.rolling(34).mean()).to_numpy(); R.both("추세지표", "어썸 오실레이터 제로 교차", f"{m}m", E((ao > 0) & (sh(ao, 1) <= 0), m), E((ao < 0) & (sh(ao, 1) >= 0), m))
        cm = ind.cmf(hs, ls, cs, vs, 20).to_numpy(); R.both("거래량", "CMF20 제로 교차", f"{m}m", E((cm > 0) & (sh(cm, 1) <= 0), m), E((cm < 0) & (sh(cm, 1) >= 0), m))
        obv = (np.sign(np.diff(c, prepend=c[0])) * v).cumsum(); oe = ps(obv).ewm(span=20, adjust=False).mean().to_numpy(); u, d = flip(obv > oe)
        R.both("거래량", "OBV·EMA20 교차", f"{m}m", E(u, m), E(d, m))
        eb = (c - ps(c).ewm(span=13, adjust=False).mean().to_numpy())
        R.both("추세지표", "엘더 레이(EMA13 대비 힘) 전환", f"{m}m", E((eb > 0) & (sh(eb, 1) <= 0), m), E((eb < 0) & (sh(eb, 1) >= 0), m))
        kv = ind.kama(cs, 10, 2, 30).to_numpy(); sl_ = np.sign(np.diff(kv, prepend=np.nan)); R.both("추세지표", "KAMA 기울기 전환", f"{m}m", E((sl_ > 0) & (sh(sl_, 1) < 0), m), E((sl_ < 0) & (sh(sl_, 1) > 0), m))
        if m >= 15:
            hm = ind.hma(cs, 21).to_numpy(); sl_ = np.sign(np.diff(hm, prepend=np.nan)); R.both("추세지표", "헐 이동평균 기울기 전환", f"{m}m", E((sl_ > 0) & (sh(sl_, 1) < 0), m), E((sl_ < 0) & (sh(sl_, 1) > 0), m))
        ten, kij, a1, b1 = ind.ichimoku(hs, ls, cs, 9, 26, 52); u, d = flip((ten > kij).to_numpy()); R.both("추세지표", "일목 전환·기준선 교차", f"{m}m", E(u, m), E(d, m))
        e5, e10, e20, e40 = (cs.ewm(span=s_, adjust=False).mean().to_numpy() for s_ in (5, 10, 20, 40))
        al = (e5 > e10) & (e10 > e20) & (e20 > e40); bl = (e5 < e10) & (e10 < e20) & (e20 < e40)
        R.both("추세지표", "EMA 리본 정렬 형성", f"{m}m", E(al & ~sh(al, 1), m), E(bl & ~sh(bl, 1), m))
        xs = np.arange(20); xm = xs - xs.mean()
        sl_reg = (cs.rolling(20).apply(lambda y: (xm * (y - y.mean())).sum() / (xm ** 2).sum(), raw=True) / cs).to_numpy() if m == 60 else None
        if sl_reg is not None:
            R.both("추세지표", "선형회귀 기울기 부호 전환(20봉)", f"{m}m", E((sl_reg > 0) & (sh(sl_reg, 1) <= 0), m), E((sl_reg < 0) & (sh(sl_reg, 1) >= 0), m))
        r1, r5 = np.sign(np.diff(c, prepend=c[0])), np.sign(cs.diff(3).to_numpy())
        agree_u = (np.sign(cs.diff(1).to_numpy()) > 0) & (np.sign(cs.diff(3).to_numpy()) > 0) & (np.sign(cs.diff(12).to_numpy()) > 0)
        agree_d = (np.sign(cs.diff(1).to_numpy()) < 0) & (np.sign(cs.diff(3).to_numpy()) < 0) & (np.sign(cs.diff(12).to_numpy()) < 0)
        R.both("모멘텀", "다중 기간 모멘텀 일치(1·3·12봉)", f"{m}m", E(agree_u & ~sh(agree_u, 1), m), E(agree_d & ~sh(agree_d, 1), m))
        acc = (cs.diff(3) - cs.diff(12) / 4).to_numpy(); az = acc / ps(acc).rolling(200).std().to_numpy()
        R.both("모멘텀", "가속도(3봉 - 12봉/4) 극단", f"{m}m", E(az > 2, m), E(az < -2, m))

    # ================= C. 변동성·점프·분포 (1분 기반) =================
    r1m = np.diff(np.log(c1), prepend=np.nan)
    ar = np.abs(r1m)
    bv = pd.Series(ar * np.concatenate([[np.nan], ar[:-1]])).rolling(60).mean().to_numpy() * (np.pi / 2)
    rv = pd.Series(r1m ** 2).rolling(60).mean().to_numpy()
    ret5 = np.log(c1) - np.log(np.concatenate([[c1[0]] * 5, c1[:-5]]))
    sig5 = np.sqrt(np.maximum(bv, 1e-14) * 5)
    jz = ret5 / sig5
    for kz in (3.0, 4.0):
        R.both("점프", f"바이파워 점프(5분 z>{kz}) 후", "5m", x1(jz) > kz, x1(jz) < -kz)
    ar_ = np.abs(ret5) / np.sqrt(np.maximum(rv, 1e-14) * 5)
    skew = pd.Series(r1m).rolling(60).skew().to_numpy(); kurt = pd.Series(r1m).rolling(60).kurt().to_numpy()
    R.both("분포모멘트", "실현 왜도(60분) 극단", "1m", x1(skew) > 1.0, x1(skew) < -1.0)
    sp = pd.Series(np.where(r1m > 0, r1m ** 2, 0.0)).rolling(60).sum().to_numpy(); sn = pd.Series(np.where(r1m < 0, r1m ** 2, 0.0)).rolling(60).sum().to_numpy()
    sv = (sp - sn) / np.maximum(sp + sn, 1e-14)
    R.both("분포모멘트", "상·하방 반분산 불균형", "60분", x1(sv) > 0.6, x1(sv) < -0.6)
    pk = pd.Series(np.log(h1 / l1) ** 2 / (4 * np.log(2))).rolling(60).mean().to_numpy(); pk_chg = pk / pd.Series(pk).rolling(1440).mean().to_numpy()
    R.both("변동성", "파킨슨 변동성 급등 시 최근 방향", "60분", (x1(pk_chg) > 2.0) & (x1(ret5) > 0), (x1(pk_chg) > 2.0) & (x1(ret5) < 0))
    vv = pd.Series(np.sqrt(rv)).rolling(240).std().to_numpy() / np.sqrt(np.maximum(pd.Series(rv).rolling(240).mean().to_numpy(), 1e-14))
    R.both("변동성", "변동성의 변동성 급등 시 최근 방향", "240분", (x1(vv) > np.nanpercentile(vv, 90)) & (x1(ret5) > 0), (x1(vv) > np.nanpercentile(vv, 90)) & (x1(ret5) < 0))
    vz = (pd.Series(np.sqrt(rv)) - pd.Series(np.sqrt(rv)).rolling(1440).mean()) / pd.Series(np.sqrt(rv)).rolling(1440).std()
    hiv, lov = x1(vz.to_numpy()) > 1, x1(vz.to_numpy()) < -1
    R.add("변동성", "변동성 국면 전환(고변동 추종·저변동 역추종)", "5m", (hiv & (x1(ret5) > 0)) | (lov & (x1(ret5) < 0)), (hiv & (x1(ret5) < 0)) | (lov & (x1(ret5) > 0)))
    # ATR 급등·NR·압축
    for m in (15, 60):
        o, h, l, c, v, tb, tr = B[m]; trg = np.maximum(h - l, 1e-12)
        atr = ps(trg).rolling(14).mean().to_numpy(); ex = trg > 2 * sh(atr, 1, np.nan); up_ = c > o
        R.both("변동성", "ATR 확장 봉(TR>2×ATR) 방향", f"{m}m", E(ex & up_, m), E(ex & ~up_, m))
        nr4 = trg <= ps(trg).rolling(4).min().to_numpy(); nr7 = trg <= ps(trg).rolling(7).min().to_numpy()
        for nm, nr in (("NR4", nr4), ("NR7", nr7)):
            R.both("변동성", f"{nm} 협소 레인지 돌파", f"{m}m", E(sh(nr, 1) & (c > sh(h, 1, np.nan)), m), E(sh(nr, 1) & (c < sh(l, 1, np.nan)), m))
        comp = (ps(h).rolling(12).max() - ps(l).rolling(12).min()).to_numpy() / c
        cq = sh(comp < ps(comp).rolling(500).quantile(0.15).to_numpy(), 1)
        hh12, ll12 = ps(h).shift(1).rolling(12).max().to_numpy(), ps(l).shift(1).rolling(12).min().to_numpy()
        R.both("변동성", "레인지 압축 후 돌파(12봉)", f"{m}m", E(cq & (c > hh12), m), E(cq & (c < ll12), m))

    # ================= D. 시간·세션·달력 =================
    day_id = np.asarray(idx.floor("1D").asi8)
    for h_ in range(24):
        ev = (tod == h_ * 60)
        R.add("시간대", "정시 시각별 방향(24시간대)", f"UTC {h_:02d}시 롱", ev, np.zeros_like(ev))
        R.add("시간대", "정시 시각별 방향(24시간대)", f"UTC {h_:02d}시 숏", np.zeros_like(ev), ev)
    for hh_ in (0, 8, 16):
        for off_nm, off in (("정산 5분 전", -5), ("정산 직후", 5)):
            ev = (tod == (hh_ * 60 + off) % 1440)
            R.add("펀딩정산", "펀딩 정산 전후 방향", f"{hh_:02d}시 {off_nm} 롱", ev, np.zeros_like(ev)); R.add("펀딩정산", "펀딩 정산 전후 방향", f"{hh_:02d}시 {off_nm} 숏", np.zeros_like(ev), ev)
    ev_m = np.zeros(len(dec), bool)
    for nm, hm in (("미국장 개장(13:30)", 13 * 60 + 30), ("미국장 개장(14:30)", 14 * 60 + 30), ("런던 개장(08:00)", 8 * 60), ("아시아 개장(01:00)", 60)):
        ev = (tod == hm)
        R.add("세션", "세션 개장 직후 방향", f"{nm} 롱", ev, np.zeros_like(ev)); R.add("세션", "세션 개장 직후 방향", f"{nm} 숏", np.zeros_like(ev), ev)
    # 직전 세션 수익률 부호
    for hh_, nm in ((0, "00시(직전 8h)"), (8, "08시(직전 8h)"), (16, "16시(직전 8h)")):
        ev = (tod == hh_ * 60)
        prev8 = np.log(c1[dec]) - np.log(c1[dec - 480])
        R.both("세션", "직전 8시간 수익률 부호", nm, ev & (prev8 > 0), ev & (prev8 < 0))
    ev = (tod == 13 * 60 + 30); ovn = np.log(c1[dec]) - np.log(c1[np.maximum(dec - 990, 0)])
    R.both("세션", "미국 장 전 야간 수익률 부호(21시→13:30)", "13:30", ev & (ovn > 0), ev & (ovn < 0))
    hr_ret = np.log(c1[dec]) - np.log(c1[dec - 60]); ev = (tod % 60 == 0)
    R.both("세션", "직전 1시간 봉 방향 → 다음 시간", "정시", ev & (hr_ret > 0), ev & (hr_ret < 0))
    q15 = np.log(c1[dec]) - np.log(c1[dec - 15]); ev = (tod % 60 == 15)
    R.both("세션", "시간 첫 15분 방향 → 남은 45분", ":15", ev & (q15 > 0), ev & (q15 < 0))
    for mm_ in (0, 15, 30, 45):
        ev = (tod % 60 == mm_)
        R.add("캔들전환", "15분 캔들 전환 시점 롱(Turn-of-the-candle)", f":{mm_:02d}", ev, np.zeros_like(ev)); R.add("캔들전환", "15분 캔들 전환 시점 숏", f":{mm_:02d}", np.zeros_like(ev), ev)
    for d_ in range(7):
        ev = (dow == d_) & (tod == 0)
        R.add("요일", "요일별 방향(자정 진입)", f"요일{d_} 롱", ev, np.zeros_like(ev)); R.add("요일", "요일별 방향(자정 진입)", f"요일{d_} 숏", np.zeros_like(ev), ev)
    ev = (dow == 6) & (tod == 23 * 60)
    R.add("요일", "일요일 저녁(뉴욕 19시) 효과", "롱", ev, np.zeros_like(ev)); R.add("요일", "일요일 저녁(뉴욕 19시) 효과", "숏", np.zeros_like(ev), ev)
    # 일 시가·4시간 시가·전일 종가 이격
    o_d = np.asarray(B[1440][0]); pcl = np.asarray(B[1440][3])
    for nm, bo in (("UTC 일 시가", B[1440][0]), ("4시간 봉 시가", B[240][0])):
        m_ = 1440 if "일" in nm else 240
        base = np.asarray(bo)[np.minimum((dec + 1) // m_, len(bo) - 1) - 0]
        # 현재 진행 중 봉의 시가: 봉 인덱스 = dec//m_
        base = np.asarray(bo)[np.minimum(dec // m_, len(bo) - 1)]
        dv = np.log(c1[dec] / base) / pd.Series(np.sqrt(np.maximum(rv, 1e-14))).rolling(1).mean().to_numpy()[dec] / np.sqrt(max(m_, 1))
        for kz in (1.5, 2.5):
            R.both("가격레벨", f"{nm} 대비 이격 z>{kz}", "5m", (dv > kz) & ~(sh(dv, 1) > kz), (dv < -kz) & ~(sh(dv, 1) < -kz))
    pdh = np.asarray(B[1440][1]); pdl = np.asarray(B[1440][2]); pdc = np.asarray(B[1440][3])
    di = np.minimum(dec // 1440, len(pdh) - 1) - 1; di = np.clip(di, 0, None)
    P_H, P_L, P_C = pdh[di], pdl[di], pdc[di]
    cd = c1[dec]
    bu_, bd_ = cd > P_H, cd < P_L
    R.both("가격레벨", "전일 고·저 돌파", "5m", bu_ & ~sh(bu_, 1), bd_ & ~sh(bd_, 1))
    hi_touch = (h1[dec] > P_H) & (cd < P_H); lo_touch = (l1[dec] < P_L) & (cd > P_L)
    R.both("가격레벨", "전일 고·저 스윕 후 복귀(유동성 스윕)", "5m", lo_touch & ~sh(lo_touch, 1), hi_touch & ~sh(hi_touch, 1))
    dcl = np.log(cd / P_C) / (np.sqrt(np.maximum(pd.Series(rv).rolling(1440).mean().to_numpy()[dec], 1e-14) * 1440))
    R.both("가격레벨", "전일 종가 대비 이격 z(±2)", "5m", (dcl > 2) & ~(sh(dcl, 1) > 2), (dcl < -2) & ~(sh(dcl, 1) < -2))
    # 전주 고저(주 인덱스)
    wk = np.asarray(idx.to_period("W").asi8)[dec]
    wk_hi = pd.Series(h1).groupby(np.asarray(idx.to_period("W").asi8)).max(); wk_lo = pd.Series(l1).groupby(np.asarray(idx.to_period("W").asi8)).min()
    prevw = wk - 1; wH = wk_hi.reindex(prevw).to_numpy(); wL = wk_lo.reindex(prevw).to_numpy()
    wu, wd = cd > wH, cd < wL
    R.both("가격레벨", "전주 고·저 돌파", "5m", wu & ~sh(wu, 1), wd & ~sh(wd, 1))
    # 아시아 세션 레인지(00~08 UTC) 돌파 (08~16시)
    asia_h = pd.Series(h1).groupby(day_id).transform(lambda s: s.iloc[:480].max() if len(s) >= 480 else np.nan).to_numpy() if False else None
    dsr = pd.DataFrame({"h": h1, "l": l1, "d": day_id, "t": np.tile(np.arange(1440), n // 1440 + 1)[:n]})
    asia = dsr[dsr.t < 480].groupby("d").agg(H=("h", "max"), L=("l", "min"))
    aH = asia.H.reindex(day_id[dec]).to_numpy(); aL = asia.L.reindex(day_id[dec]).to_numpy(); sess = (tod >= 480) & (tod < 960)
    au, ad = sess & (cd > aH), sess & (cd < aL)
    R.both("세션", "아시아 레인지(00~08시) 돌파(런던 시간)", "5m", au & ~sh(au, 1), ad & ~sh(ad, 1))
    orh = dsr[dsr.t < 30].groupby("d").agg(H=("h", "max"), L=("l", "min")); oH = orh.H.reindex(day_id[dec]).to_numpy(); oL = orh.L.reindex(day_id[dec]).to_numpy()
    ou, od = (tod >= 30) & (tod < 480) & (cd > oH), (tod >= 30) & (tod < 480) & (cd < oL)
    R.both("세션", "UTC 일 첫 30분 레인지 돌파", "5m", ou & ~sh(ou, 1), od & ~sh(od, 1))
    # 라운드 넘버
    step_ = 1000.0 if np.nanmedian(c1) > 10000 else 50.0
    for stp in (step_, step_ / 2, step_ / 10):
        lvl = np.floor(cd / stp) * stp; plv = np.floor(c1[dec - 5] / stp) * stp
        cu, cdn = lvl > plv, lvl < plv
        R.both("가격레벨", f"라운드 넘버({stp:g}) 돌파", "5m", cu, cdn)
    # 클래식 피벗 (전일 H/L/C)
    Pv = (P_H + P_L + P_C) / 3; R1, S1 = 2 * Pv - P_L, 2 * Pv - P_H
    r1u, s1d = cd > R1, cd < S1
    R.both("가격레벨", "일 피벗 R1·S1 돌파", "5m", r1u & ~sh(r1u, 1), s1d & ~sh(s1d, 1))
    tch_s, tch_r = (l1[dec] <= S1) & (cd > S1), (h1[dec] >= R1) & (cd < R1)
    R.both("가격레벨", "일 피벗 S1·R1 터치 후 반등", "5m", tch_s & ~sh(tch_s, 1), tch_r & ~sh(tch_r, 1))
    # 일 VWAP
    vw = (pd.Series(c1 * v1).groupby(day_id).cumsum() / pd.Series(v1).groupby(day_id).cumsum()).to_numpy()
    dvw = (c1 / vw - 1); zv = dvw / pd.Series(dvw).rolling(1440).std().to_numpy()
    t_up, t_dn = (l1[dec] <= vw[dec]) & (cd > vw[dec]) & (c1[dec - 30] > vw[dec - 30]), (h1[dec] >= vw[dec]) & (cd < vw[dec]) & (c1[dec - 30] < vw[dec - 30])
    R.both("가격레벨", "일 VWAP 터치 후 반등(추세 방향)", "5m", t_up, t_dn)
    R.both("가격레벨", "일 VWAP 이격 z(±2)", "5m", (zv[dec] > 2) & ~(zv[dec - 5] > 2), (zv[dec] < -2) & ~(zv[dec - 5] < -2))
    # 볼륨 프로파일 POC (최근 1440분, 0.1% 구간)
    poc = np.full(len(dec), np.nan)
    for j in range(0, len(dec), 12):                     # 1시간마다 갱신
        a = dec[j]; seg_c, seg_v = c1[a - 1440:a], v1[a - 1440:a]
        b = np.floor(np.log(seg_c) / 0.001).astype(int); u, inv = np.unique(b, return_inverse=True)
        s_ = np.bincount(inv, weights=seg_v); poc[j:j + 12] = np.exp(u[s_.argmax()] * 0.001)
    dpoc = np.log(cd / poc)
    R.both("가격레벨", "POC(1일 거래량 최대 가격) 이격 z(±)", "5m", (dpoc > 0.01) & ~(sh(dpoc, 1) > 0.01), (dpoc < -0.01) & ~(sh(dpoc, 1) < -0.01))
    # 프랙탈 스윙 돌파
    for m in (15, 60):
        o, h, l, c, v, tb, tr = B[m]
        shv, shi, slv, sli, ish, isl = ind.swings(ps(h), ps(l), 3)
        bu = c > np.nan_to_num(shv, nan=np.inf); bd = c < np.nan_to_num(slv, nan=-np.inf)
        R.both("가격레벨", "프랙탈 스윙 고·저 돌파", f"{m}m", E(bu & ~sh(bu, 1), m), E(bd & ~sh(bd, 1), m))

    # ================= E. 파생 (프리미엄·펀딩·OI) =================
    pr = D["pr"]; prz = (pr - pd.Series(pr).rolling(1440).mean().to_numpy()) / pd.Series(pr).rolling(1440).std().to_numpy()
    for kz in (2.0, 3.0):
        R.both("파생", f"프리미엄(perp-현물) z>{kz} 극단", "5m", (prz[dec] > kz) & ~(prz[dec - 5] > kz), (prz[dec] < -kz) & ~(prz[dec - 5] < -kz))
    pch = pr - np.concatenate([[np.nan] * 60, pr[:-60]]); pchz = pch / pd.Series(pch).rolling(1440).std().to_numpy()
    R.both("파생", "프리미엄 60분 변화 모멘텀", "5m", (pchz[dec] > 2) & ~(pchz[dec - 5] > 2), (pchz[dec] < -2) & ~(pchz[dec - 5] < -2))
    fr = D["fr"]
    R.both("파생", "펀딩 극단(≥0.03%/-)", "5m", (fr[dec] > 0.0003) & (tod == 0), (fr[dec] < -0.0001) & (tod == 0))
    trend = np.log(c1[dec]) - np.log(c1[dec - 240])
    R.both("파생", "펀딩 과열 + 4시간 상승(혼잡 추격)", "5m", (fr[dec] > 0.0002) & (trend > 0) & (tod % 240 == 0), (fr[dec] < 0) & (trend < 0) & (tod % 240 == 0))
    oi = D["oi"]; oich = np.log(oi / np.concatenate([[np.nan] * 60, oi[:-60]])); oiz = oich / pd.Series(oich).rolling(2880, min_periods=500).std().to_numpy()
    r60 = np.log(c1) - np.log(np.concatenate([[c1[0]] * 60, c1[:-60]]))
    oi_up, oi_dn = oiz[dec] > 1.5, oiz[dec] < -1.5
    pu, pdn_ = r60[dec] > 0, r60[dec] < 0
    for nm, ev, sd in (("신규 롱 유입(OI↑·가격↑)", oi_up & pu, 1), ("신규 숏 유입(OI↑·가격↓)", oi_up & pdn_, -1), ("숏 커버링(OI↓·가격↑)", oi_dn & pu, 1), ("롱 청산(OI↓·가격↓)", oi_dn & pdn_, -1)):
        e = ev & ~sh(ev, 1)
        R.add("파생", f"OI 사분면: {nm}", "추종", e if sd == 1 else np.zeros_like(e), e if sd == -1 else np.zeros_like(e))
        R.add("파생", f"OI 사분면: {nm}", "역추종", e if sd == -1 else np.zeros_like(e), e if sd == 1 else np.zeros_like(e))
    R.both("파생", "OI 급증(1시간 z>2.5) 방향", "5m", (oiz[dec] > 2.5) & (r60[dec] > 0), (oiz[dec] > 2.5) & (r60[dec] < 0))
    tbr15 = pd.Series(tb1).rolling(15).sum().to_numpy() / pd.Series(v1).rolling(15).sum().to_numpy() - 0.5
    R.both("파생", "OI 증가 + 테이커 매수 우위", "5m", (oiz[dec] > 1) & (tbr15[dec] > 0.05), (oiz[dec] > 1) & (tbr15[dec] < -0.05))

    # ================= F. BTC↔ETH 교차 =================
    co1 = D["co1"]; ro = {m_: np.log(co1) - np.log(np.concatenate([[co1[0]] * m_, co1[:-m_]])) for m_ in (1, 5, 15, 60)}
    rs = {m_: np.log(c1) - np.log(np.concatenate([[c1[0]] * m_, c1[:-m_]])) for m_ in (1, 5, 15, 60)}
    for m_ in (5, 15, 60):
        zo = ro[m_] / pd.Series(ro[m_]).rolling(1440).std().to_numpy(); zs = rs[m_] / pd.Series(rs[m_]).rolling(1440).std().to_numpy()
        R.both("교차자산", f"상대 자산 {m_}분 급변(z2) 뒤따르기", "5m", (zo[dec] > 2) & ~(zo[dec - 5] > 2), (zo[dec] < -2) & ~(zo[dec - 5] < -2))
        gap = zo - zs
        R.both("교차자산", f"상대 자산 대비 {m_}분 후행(캐치업)", "5m", (gap[dec] > 2) & ~(gap[dec - 5] > 2), (gap[dec] < -2) & ~(gap[dec - 5] < -2))
    beta_r = pd.Series(rs[5]).rolling(1440).cov(pd.Series(ro[5])) / pd.Series(ro[5]).rolling(1440).var()
    resid = rs[5] - beta_r.to_numpy() * ro[5]; rzs = pd.Series(resid).rolling(12).sum().to_numpy(); rzz = rzs / pd.Series(rzs).rolling(1440).std().to_numpy()
    R.both("교차자산", "베타 조정 잔차 수익률(1시간) 극단 되돌림", "5m", (rzz[dec] < -2.5) & ~(rzz[dec - 5] < -2.5), (rzz[dec] > 2.5) & ~(rzz[dec - 5] > 2.5))
    corr = pd.Series(rs[1]).rolling(240).corr(pd.Series(ro[1])).to_numpy()
    lowc = (corr < np.nanpercentile(corr, 5))
    R.both("교차자산", "상관 붕괴(하위5%) 중 상대 자산 방향", "5m", lowc[dec] & (ro[15][dec] > 0), lowc[dec] & (ro[15][dec] < 0))
    tbro = pd.Series(D["tbo1"]).rolling(15).sum().to_numpy() / pd.Series(D["vo1"]).rolling(15).sum().to_numpy() - 0.5
    R.both("교차자산", "상대 자산 테이커 불균형 15분", "5m", tbro[dec] > 0.06, tbro[dec] < -0.06)
    qo5 = pd.Series(D["vo1"]).rolling(5).sum(); zvo = ((qo5 - qo5.rolling(1440).mean()) / qo5.rolling(1440).std()).to_numpy()
    R.both("교차자산", "상대 자산 거래량 급증 시 방향", "5m", (zvo[dec] > 3) & (ro[5][dec] > 0), (zvo[dec] > 3) & (ro[5][dec] < 0))

    # ================= G. 마이크로구조 대용 (호가 없이) =================
    q5 = pd.Series(v1).rolling(5).sum(); z5 = ((q5 - q5.rolling(1440).mean()) / q5.rolling(1440).std()).to_numpy()
    trn = pd.Series(tr1).rolling(5).sum(); ztr = ((trn - trn.rolling(1440).mean()) / trn.rolling(1440).std()).to_numpy()
    R.both("마이크로", "체결 건수 급증(z3) 방향", "5m", (ztr[dec] > 3) & (ret5[dec] > 0), (ztr[dec] > 3) & (ret5[dec] < 0))
    avg = pd.Series(v1).rolling(5).sum().to_numpy() / np.maximum(pd.Series(tr1).rolling(5).sum().to_numpy(), 1); az_ = (avg - pd.Series(avg).rolling(1440).mean().to_numpy()) / pd.Series(avg).rolling(1440).std().to_numpy()
    R.both("마이크로", "평균 체결 크기 급증(고래 흔적 대용) 방향", "5m", (az_[dec] > 3) & (ret5[dec] > 0), (az_[dec] > 3) & (ret5[dec] < 0))
    t5 = pd.Series(tb1).rolling(5).sum().to_numpy() / q5.to_numpy() - 0.5; t60 = pd.Series(tb1).rolling(60).sum().to_numpy() / pd.Series(v1).rolling(60).sum().to_numpy() - 0.5
    dtb = t5 - t60
    R.both("마이크로", "테이커 불균형 변화(5분 - 60분)", "5m", dtb[dec] > 0.10, dtb[dec] < -0.10)
    rng5 = pd.Series(h1).rolling(5).max().to_numpy() / pd.Series(l1).rolling(5).min().to_numpy() - 1; rz = rng5 / pd.Series(rng5).rolling(1440).mean().to_numpy()
    absorb = (z5[dec] > 2) & (rz[dec] < 0.7)
    R.both("마이크로", "흡수(거래량↑·변동폭↓) 직전 방향의 반대", "5m", absorb & (r60[dec] < 0), absorb & (r60[dec] > 0))
    dry = (pd.Series(z5).rolling(3).max().to_numpy() < -1.0)
    R.both("마이크로", "거래량 고갈 후 돌파", "5m", dry[dec - 5] & (cd > pd.Series(h1).shift(5).rolling(30).max().to_numpy()[dec]), dry[dec - 5] & (cd < pd.Series(l1).shift(5).rolling(30).min().to_numpy()[dec]))
    hi60 = pd.Series(h1).rolling(240).max().to_numpy(); lo60 = pd.Series(l1).rolling(240).min().to_numpy()
    clim_u = (z5[dec] > 3) & (h1[dec - 4:dec + 1].max() if False else True) & (pd.Series(h1).rolling(5).max().to_numpy()[dec] >= hi60[dec]) & (cd < (pd.Series(h1).rolling(5).max().to_numpy()[dec] + pd.Series(l1).rolling(5).min().to_numpy()[dec]) / 2)
    clim_d = (z5[dec] > 3) & (pd.Series(l1).rolling(5).min().to_numpy()[dec] <= lo60[dec]) & (cd > (pd.Series(h1).rolling(5).max().to_numpy()[dec] + pd.Series(l1).rolling(5).min().to_numpy()[dec]) / 2)
    R.both("마이크로", "클라이맥스 거래량 반전(와이코프)", "5m", clim_d, clim_u)
    spr_l = (pd.Series(l1).rolling(5).min().to_numpy()[dec] < pd.Series(l1).shift(5).rolling(60).min().to_numpy()[dec]) & (cd > pd.Series(l1).shift(5).rolling(60).min().to_numpy()[dec]) & (z5[dec] > 1)
    upt = (pd.Series(h1).rolling(5).max().to_numpy()[dec] > pd.Series(h1).shift(5).rolling(60).max().to_numpy()[dec]) & (cd < pd.Series(h1).shift(5).rolling(60).max().to_numpy()[dec]) & (z5[dec] > 1)
    R.both("마이크로", "스프링·업스러스트(범위 이탈 후 복귀, 와이코프)", "5m", spr_l, upt)
    imb = np.abs(t60) * 2
    R.both("마이크로", "VPIN 대용(60분 매수·매도 불균형) 극단 방향", "60분", (imb[dec] > np.nanpercentile(imb, 95)) & (t60[dec] > 0), (imb[dec] > np.nanpercentile(imb, 95)) & (t60[dec] < 0))
    kl = np.abs(ret5) / np.maximum(q5.to_numpy(), 1); klz = (kl - pd.Series(kl).rolling(1440).mean().to_numpy()) / pd.Series(kl).rolling(1440).std().to_numpy()
    R.both("마이크로", "Kyle 람다 대용(|수익률|/거래량) 급등 시 방향", "5m", (klz[dec] > 3) & (ret5[dec] > 0), (klz[dec] > 3) & (ret5[dec] < 0))
    upmin = pd.Series((r1m > 0).astype(float)).rolling(30).mean().to_numpy()
    R.both("마이크로", "상승 분(分) 비율(30분) 극단", "5m", upmin[dec] > 0.65, upmin[dec] < 0.35)
    for N in (6, 10):
        cnt = pd.Series((np.sign(r1m) > 0).astype(float)).rolling(N).sum().to_numpy(); cnd = pd.Series((np.sign(r1m) < 0).astype(float)).rolling(N).sum().to_numpy()
        R.both("마이크로", f"1분봉 {N}연속 같은 방향", "5m", (cnt[dec] == N), (cnd[dec] == N))
    ac = pd.Series(r1m).rolling(240).apply(lambda y: np.corrcoef(y[:-1], y[1:])[0, 1], raw=True).to_numpy() if False else (pd.Series(r1m).rolling(240).corr(pd.Series(r1m).shift(1))).to_numpy()
    R.both("통계", "1분 수익률 자기상관 국면(±0.05) 따라 최근 방향", "5m", ((ac[dec] > 0.05) & (ret5[dec] > 0)) | ((ac[dec] < -0.05) & (ret5[dec] < 0)), ((ac[dec] > 0.05) & (ret5[dec] < 0)) | ((ac[dec] < -0.05) & (ret5[dec] > 0)))
    v5 = pd.Series(ret5).rolling(240).var().to_numpy(); v1_ = pd.Series(r1m).rolling(240).var().to_numpy() * 5; vr = v5 / np.maximum(v1_, 1e-14)
    R.add("통계", "분산비(VR 5분/1분) 국면: VR>1.2 추종·VR<0.8 역추종", "5m", ((vr[dec] > 1.2) & (ret5[dec] > 0)) | ((vr[dec] < 0.8) & (ret5[dec] < 0)), ((vr[dec] > 1.2) & (ret5[dec] < 0)) | ((vr[dec] < 0.8) & (ret5[dec] > 0)))
    sdz = ret5 / np.sqrt(np.maximum(rv, 1e-14) * 5)
    for kz in (3.0, 4.0):
        R.both("통계", f"큰 5분 봉(|z|>{kz}) 직후", "5m", (sdz[dec] > kz) & ~(sdz[dec - 5] > kz), (sdz[dec] < -kz) & ~(sdz[dec - 5] < -kz))
    vwm = ret5 * z5; vwz = vwm / pd.Series(vwm).rolling(1440).std().to_numpy()
    R.both("통계", "거래량 가중 모멘텀(수익률×거래량 z) 극단", "5m", vwz[dec] > 2, vwz[dec] < -2)
    hz = (r60 / pd.Series(r60).rolling(2880).std().to_numpy())
    R.both("통계", "1시간 수익률 z(±2.5) 극단", "5m", (hz[dec] > 2.5) & (tod % 60 == 0), (hz[dec] < -2.5) & (tod % 60 == 0))
    # 대조군: 무작위 진입
    rng = np.random.default_rng(7)
    for i in range(300):
        p = rng.choice([0.002, 0.005, 0.02])
        R.add("대조군", "무작위 진입", f"#{i}", rng.random(len(dec)) < p, rng.random(len(dec)) < p)
    return R, dec


def main():
    sym = sys.argv[1]; other = "ETHUSDT" if sym == "BTCUSDT" else "BTCUSDT"
    t0 = time.time()
    D = build(sym, other)
    R, dec = rules(D)
    real = [r for r in R.rules if not r.startswith("대조군")]
    print(sym, "접근법", len(R.bases) - 1, "규칙", len(real), "대조군", len(R.rules) - len(real), f"{time.time()-t0:.0f}s", flush=True)
    c, h, l = D["c1"], D["h1"], D["l1"]
    ts = D["idx"][dec]; yrs = np.asarray(ts.year)
    out = []
    for gname, (tp, sl, H) in GEOMS.items():
        lab = {s: barrier(h, l, c, dec, tp, sl, H, s) for s in (1, -1)}
        stp = {s: np.ceil(lab[s][1] / STEP).astype(np.int64) for s in (1, -1)}
        for rname, (L, S_) in R.rules.items():
            ev = np.where((L | S_) & ~(L & S_))[0]
            if len(ev) < 50: continue
            sd = np.where(L[ev], 1, -1).astype(np.int8)
            m_ = pick(ev, sd, stp[1], stp[-1]); si, sdd = ev[m_], sd[m_]
            ret = np.where(sdd == 1, lab[1][2][si], lab[-1][2][si]).astype(np.float64) - COST
            win = np.where(sdd == 1, lab[1][0][si], lab[-1][0][si]) == 1
            y = yrs[si]
            row = {"rule": rname, "geom": gname, "sym": sym[:3], "n": int(len(si)), "gross_bps": round(float(1e4 * (ret + COST).mean()), 1)}
            for nm, mk in (("train", y <= 2023), ("test", y >= 2024)):
                a = ret[mk]
                row[nm] = {"n": int(mk.sum()), "bps": round(float(1e4 * a.mean()), 1) if mk.sum() else None, "win": round(float(win[mk].mean()), 3) if mk.sum() else None,
                           "t": round(float(a.mean() / a.std() * np.sqrt(len(a))), 2) if mk.sum() > 20 and a.std() > 0 else None}
            row["years"] = {int(Y): round(float(1e4 * ret[y == Y].mean()), 1) for Y in sorted(set(y)) if (y == Y).sum() >= 20}
            out.append(row)
        print(gname, len(out), f"{time.time()-t0:.0f}s", flush=True)
    Path(f"logs/hf_battery_{sym[:3]}.json").write_text(json.dumps({"bases": sorted(R.bases), "rows": out}, ensure_ascii=False))
    print("저장", len(out), "접근법", len(R.bases) - 1)


if __name__ == "__main__":
    main()
