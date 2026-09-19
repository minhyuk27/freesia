"""지표·차트 전략 대량 스크리닝 (BTC·ETH만, 1배, 비용 6bp, 펀딩, 연도별 워크포워드).
사용: python research/screen6.py <tf,...> <tag> [group,...]"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from freesia.bt import core, ind  # noqa: E402
import screen as S  # noqa: E402

hold = S.hold


def mode(sig, kind):
    return sig if kind == "LS" else np.where(sig > 0, 1.0, 0.0)


def unpack(x):
    b = x.b
    return b.open, b.high, b.low, b.close, b.volume, b.quote_volume, b.taker_buy_quote_volume


# ============ 추세추종 ============
def g_trend(x):
    o, h, l, c, v, qv, tb = unpack(x); P, N = [], []
    for f, s, g in [(12, 26, 9), (6, 13, 5), (24, 52, 18), (8, 17, 9), (20, 50, 10), (50, 100, 20), (30, 90, 30)]:
        m, sg, hist = ind.macd(c, f, s, g)
        for nm, sig in (("hist", np.sign(hist.fillna(0).to_numpy())), ("line", np.sign(m.fillna(0).to_numpy()))):
            for k in ("LS", "LF"):
                P.append(mode(sig, k)); N.append(f"추세추종/MACD({f},{s},{g}) {nm} {k}")
    for n in [7, 14, 28, 56]:
        pdi, mdi, a = ind.adx(h, l, c, n)
        for th in [15, 20, 25, 30]:
            sig = np.where(a.fillna(0) > th, np.sign((pdi - mdi).fillna(0)), 0.0)
            for k in ("LS", "LF"):
                P.append(mode(sig, k)); N.append(f"추세추종/ADX-DMI n={n} th={th} {k}")
    for n in [7, 10, 14, 21, 42]:
        for m in [2, 3, 4]:
            d = ind.supertrend(h, l, c, n, m)
            for k in ("LS", "LF"):
                P.append(mode(d, k)); N.append(f"추세추종/슈퍼트렌드 n={n} m={m} {k}")
    for a0, am in [(0.02, 0.2), (0.01, 0.1), (0.03, 0.3), (0.005, 0.05)]:
        d = ind.psar(h, l, a0, am)
        for k in ("LS", "LF"):
            P.append(mode(d, k)); N.append(f"추세추종/파라볼릭SAR af={a0}-{am} {k}")
    for t, k_, s in [(9, 26, 52), (7, 22, 44), (20, 60, 120), (12, 36, 72)]:
        ten, kij, a, b = ind.ichimoku(h, l, c, t, k_, s)
        top, bot = np.maximum(a, b), np.minimum(a, b)
        up = ((c > top) & (ten > kij)).fillna(False).to_numpy(); dn = ((c < bot) & (ten < kij)).fillna(False).to_numpy()
        P.append(hold(up, ~up)); N.append(f"추세추종/일목균형표({t},{k_},{s}) 구름+전환기준 LF")
        P.append(hold(up, dn, dn, up)); N.append(f"추세추종/일목균형표({t},{k_},{s}) LS")
        up2 = (c > top).fillna(False).to_numpy()
        P.append(hold(up2, ~up2)); N.append(f"추세추종/일목균형표({t},{k_},{s}) 구름돌파 LF")
    for n in [14, 25, 50]:
        au, ad = ind.aroon(h, l, n)
        up = ((au > 70) & (ad < 30)).to_numpy(); dn = ((ad > 70) & (au < 30)).to_numpy()
        P.append(hold(up, ~(au > ad).to_numpy())); N.append(f"추세추종/아룬 n={n} LF")
        P.append(hold(up, dn, dn, up)); N.append(f"추세추종/아룬 n={n} LS")
    for n in [14, 21, 42]:
        vp, vm = ind.vortex(h, l, c, n)
        sig = np.sign((vp - vm).fillna(0).to_numpy())
        for k in ("LS", "LF"):
            P.append(mode(sig, k)); N.append(f"추세추종/보텍스 n={n} {k}")
    for n in [9, 15, 30, 60]:
        t = ind.trix(c, n).fillna(0).to_numpy()
        for k in ("LS", "LF"):
            P.append(mode(np.sign(t), k)); N.append(f"추세추종/TRIX n={n} {k}")
    for n, f, s in [(10, 2, 30), (20, 2, 30), (10, 3, 50), (30, 2, 60)]:
        kv = ind.kama(c, n, f, s)
        sig = np.sign((c - kv).fillna(0).to_numpy()) * (np.sign(kv.diff().fillna(0).to_numpy()) >= 0)
        for k in ("LS", "LF"):
            P.append(mode(sig, k)); N.append(f"추세추종/적응형이동평균KAMA({n},{f},{s}) {k}")
    for n in [16, 32, 64, 128]:
        hm = ind.hma(c, n)
        sig = np.sign((c - hm).fillna(0).to_numpy()) * (np.sign(hm.diff().fillna(0).to_numpy()) >= 0)
        for k in ("LS", "LF"):
            P.append(mode(sig, k)); N.append(f"추세추종/헐이동평균 n={n} {k}")
    ho, hc = ind.heikin_ashi(o, h, l, c)
    bull = (hc > ho).astype(float)
    for m in [1, 2, 3, 5]:
        run = pd.Series(bull).rolling(m).sum().to_numpy() == m
        P.append(hold(run, hc < ho)); N.append(f"추세추종/하이켄아시 연속{m}양봉 LF")
    for fh, sh in [(12, 26), (20, 50), (50, 100), (10, 30), (25, 100), (50, 200)]:
        sig = np.sign((ind.ema(c, fh) - ind.ema(c, sh)).fillna(0).to_numpy())
        for k in ("LS", "LF"):
            P.append(mode(sig, k)); N.append(f"추세추종/지수이동평균 교차({fh},{sh}) {k}")
    return N, P


# ============ 모멘텀(지표형) ============
def g_momentum(x):
    o, h, l, c, v, qv, tb = unpack(x); P, N = [], []
    for n in [7, 14, 28, 56]:
        r = ind.rsi(c, n).fillna(50).to_numpy()
        for th in [50, 55, 60, 65]:
            P.append(hold(r > th, r < 50)); N.append(f"모멘텀/RSI 상승 n={n} 진입>{th} 청산<50")
    for n in [14, 28, 56]:
        k_, d_ = ind.stoch(h, l, c, n)
        kk = k_.fillna(50).to_numpy()
        for th in [70, 80]:
            P.append(hold(kk > th, kk < 50)); N.append(f"모멘텀/스토캐스틱 n={n} 진입>{th} 청산<50")
    for n in [20, 50, 100]:
        z = ind.zscore_tp(h, l, c, n).fillna(0).to_numpy()
        for th in [0.5, 1.0, 1.5]:
            P.append(hold(z > th, z < 0)); N.append(f"모멘텀/CCI근사 n={n} 진입>{th} 청산<0")
    for n in [14, 28, 56]:
        mf = ind.mfi(h, l, c, v, n).fillna(50).to_numpy()
        for th in [55, 60, 70]:
            P.append(hold(mf > th, mf < 50)); N.append(f"모멘텀/MFI n={n} 진입>{th} 청산<50")
    return N, P


# ============ 돌파 ============
def g_breakout(x):
    o, h, l, c, v, qv, tb = unpack(x); P, N = [], []
    cc = c.to_numpy()
    for n in [20, 50, 100]:
        for k in [1.5, 2.0, 3.0]:
            m, u, lo = ind.keltner(h, l, c, n, k)
            up, dn = (c > u).fillna(False).to_numpy(), (c < lo).fillna(False).to_numpy()
            mid = (c < m).fillna(False).to_numpy(); mu = (c > m).fillna(False).to_numpy()
            P.append(hold(up, mid)); N.append(f"돌파/켈트너채널 n={n} k={k} LF")
            P.append(hold(up, mid, dn, mu)); N.append(f"돌파/켈트너채널 n={n} k={k} LS")
    # 볼린저-켈트너 스퀴즈 후 방향 돌파
    for n in [20, 50]:
        for kb, kk in [(2.0, 1.5), (2.0, 1.0)]:
            bm, bu, bl = ind.bbands(c, n, kb)
            km, ku, kl = ind.keltner(h, l, c, n, kk)
            sq = ((bu < ku) & (bl > kl)).fillna(False)
            release = (sq.shift(1).fillna(False) & ~sq).to_numpy()
            mom = (c - (h.rolling(n).max() + l.rolling(n).min()) / 2 / 2 - bm / 2).fillna(0).to_numpy()
            for H in [n, 2 * n, 4 * n]:
                on_up = pd.Series((release & (mom > 0)).astype(float)).rolling(H).max().fillna(0).to_numpy()
                on_dn = pd.Series((release & (mom < 0)).astype(float)).rolling(H).max().fillna(0).to_numpy()
                P.append(on_up); N.append(f"돌파/변동성스퀴즈 n={n} KC={kk} 보유{H}봉 LF")
                P.append(np.clip(on_up - on_dn, -1, 1)); N.append(f"돌파/변동성스퀴즈 n={n} KC={kk} 보유{H}봉 LS")
    # 스윙 고저(지지·저항) 돌파
    for k in [3, 5, 10, 20]:
        shv, shi, slv, sli, _, _ = ind.swings(h, l, k)
        up = np.nan_to_num(cc > np.nan_to_num(shv, nan=np.inf)).astype(bool)
        dn = np.nan_to_num(cc < np.nan_to_num(slv, nan=-np.inf)).astype(bool)
        P.append(hold(up, dn)); N.append(f"돌파/스윙저항 돌파 k={k} LF")
        P.append(hold(up, dn, dn, up)); N.append(f"돌파/스윙지지·저항 돌파 k={k} LS")
    # 전일(UTC) 고저 돌파
    if x.tf != "1d":
        day = x.idx.floor("1D")
        dh = h.groupby(day).max(); dl = l.groupby(day).min()
        ph = pd.Series(dh.shift(1).reindex(day).to_numpy(), index=x.idx); pl = pd.Series(dl.shift(1).reindex(day).to_numpy(), index=x.idx)
        up, dn = (c > ph).fillna(False).to_numpy(), (c < pl).fillna(False).to_numpy()
        mid = ((ph + pl) / 2).to_numpy()
        P.append(hold(up, cc < mid)); N.append("돌파/전일 고가 돌파 (전일 중간값 이탈 청산) LF")
        P.append(hold(up, dn, dn, up)); N.append("돌파/전일 고저 돌파 LS")
    else:
        for n in [1, 2, 3, 5, 7]:
            ph, pl = h.shift(1).rolling(n).max(), l.shift(1).rolling(n).min()
            up, dn = (c > ph).fillna(False).to_numpy(), (c < pl).fillna(False).to_numpy()
            P.append(hold(up, dn)); N.append(f"돌파/{n}일 고가 돌파 LF")
    # 레인지 압축 후 돌파
    for n in [24, 48, 96]:
        rng = ((h.rolling(n).max() - l.rolling(n).min()) / c)
        for q in [0.2, 0.35, 0.5]:
            thr = rng.rolling(x.nb(24 * 365), min_periods=n * 4).quantile(q)
            comp = (rng < thr).fillna(False)
            hh = h.shift(1).rolling(n).max().to_numpy()
            for H in [n, 2 * n]:
                sig = pd.Series(((cc > hh) & comp.shift(1).fillna(False).to_numpy()).astype(float)).rolling(H).max().fillna(0).to_numpy()
                P.append(sig); N.append(f"돌파/레인지압축(n={n},하위{int(q*100)}%) 후 상방 돌파 보유{H}봉 LF")
    return N, P


# ============ 평균회귀 ============
def g_meanrev(x):
    o, h, l, c, v, qv, tb = unpack(x); P, N = [], []
    cc = c.to_numpy()
    filt = (c > c.rolling(x.nb(2400)).mean()).fillna(False).to_numpy()
    for n in [20, 50, 100, 200]:
        m, u, lo = ind.bbands(c, n, 2.0)
        for k in [1.5, 2.0, 2.5, 3.0]:
            mm, uu, ll = ind.bbands(c, n, k)
            lw = (c < ll).fillna(False).to_numpy(); hi = (c > uu).fillna(False).to_numpy()
            xm_up = (c > mm).fillna(False).to_numpy(); xm_dn = (c < mm).fillna(False).to_numpy()
            P.append(hold(lw, xm_up)); N.append(f"평균회귀/볼린저밴드 n={n} k={k} LF")
            P.append(hold(lw & filt, xm_up)); N.append(f"평균회귀/볼린저밴드 n={n} k={k} 상승추세필터 LF")
            P.append(hold(lw, xm_up, hi, xm_dn)); N.append(f"평균회귀/볼린저밴드 n={n} k={k} LS")
    for n in [2, 3, 5, 14]:
        r = ind.rsi(c, n).fillna(50).to_numpy()
        for th in [5, 10, 20, 30]:
            P.append(hold(r < th, r > 55)); N.append(f"평균회귀/RSI n={n} 과매도<{th} 청산>55 LF")
            P.append(hold((r < th) & filt, r > 55)); N.append(f"평균회귀/RSI n={n} 과매도<{th} 상승추세필터 LF")
            P.append(hold(r < th, r > 55, r > 100 - th, r < 45)); N.append(f"평균회귀/RSI n={n} ±{th} LS")
    for n in [14, 28, 56]:
        k_, d_ = ind.stoch(h, l, c, n)
        kk = k_.fillna(50).to_numpy()
        for th in [10, 20]:
            P.append(hold(kk < th, kk > 50)); N.append(f"평균회귀/스토캐스틱 n={n} <{th} LF")
            P.append(hold(kk < th, kk > 50, kk > 100 - th, kk < 50)); N.append(f"평균회귀/스토캐스틱 n={n} ±{th} LS")
    for n in [20, 50, 100]:
        z = ind.zscore_tp(h, l, c, n).fillna(0).to_numpy()
        for k in [1.0, 1.5, 2.0, 2.5]:
            P.append(hold(z < -k, z > 0)); N.append(f"평균회귀/CCI근사 n={n} <-{k} LF")
    for n in [24, 72, 168, 336]:
        nb = x.nb(n)
        vw = (c * v).rolling(nb).sum() / v.rolling(nb).sum()
        dev = (c / vw - 1); sd = dev.rolling(nb * 4, min_periods=nb).std()
        z = (dev / sd).fillna(0).to_numpy()
        for k in [1.0, 1.5, 2.0]:
            P.append(hold(z < -k, z > 0)); N.append(f"평균회귀/VWAP 이탈 {n}h k={k} LF")
            P.append(hold(z < -k, z > 0, z > k, z < 0)); N.append(f"평균회귀/VWAP 이탈 {n}h k={k} LS")
    for n in [20, 50, 100]:
        m = ind.ema(c, n); a = ind.atr(h, l, c, n)
        dev = ((c - m) / a).fillna(0).to_numpy()
        for k in [1.5, 2.0, 3.0]:
            P.append(hold(dev < -k, dev > 0)); N.append(f"평균회귀/ATR 이격 EMA{n} k={k} LF")
    return N, P


# ============ 역추세추적(반전·과열 역매매) ============
def g_contrarian(x):
    o, h, l, c, v, qv, tb = unpack(x); P, N = [], []
    r = c.pct_change().fillna(0)
    sd = r.rolling(x.nb(720)).std()
    down = (c < o).astype(float); up = (c > o).astype(float)
    for m in [2, 3, 4, 5, 6]:
        dn_run = (down.rolling(m).sum() == m).to_numpy(); up_run = (up.rolling(m).sum() == m).to_numpy()
        for H in [4, 12, 24, 48]:
            Hn = x.nb(H)
            lo = pd.Series(dn_run.astype(float)).rolling(Hn).max().fillna(0).to_numpy()
            hi = pd.Series(up_run.astype(float)).rolling(Hn).max().fillna(0).to_numpy()
            P.append(lo); N.append(f"역추세추적/연속 음봉 {m}개 후 매수 보유{H}h LF")
            P.append(np.clip(lo - hi, -1, 1)); N.append(f"역추세추적/연속 {m}봉 반대매매 보유{H}h LS")
    vz = ((qv - qv.rolling(x.nb(720)).mean()) / qv.rolling(x.nb(720)).std()).fillna(0)
    for nh in [1, 4, 12]:
        n = x.nb(nh)
        z = ((c / c.shift(n) - 1) / (sd * np.sqrt(n))).fillna(0)
        for k in [1.5, 2.0, 2.5]:
            for vk in [0.0, 1.0, 2.0]:
                for H in [12, 48]:
                    Hn = x.nb(H)
                    cap = pd.Series(((z < -k) & (vz > vk)).astype(float)).rolling(Hn).max().fillna(0).to_numpy()
                    exh = pd.Series(((z > k) & (vz > vk)).astype(float)).rolling(Hn).max().fillna(0).to_numpy()
                    P.append(cap); N.append(f"역추세추적/투매 매수(수익률z<-{k}, 거래량z>{vk}) {nh}h 보유{H}h LF")
                    P.append(np.clip(cap - exh, -1, 1)); N.append(f"역추세추적/투매·과열 반대매매(z±{k}, 거래량z>{vk}) {nh}h 보유{H}h LS")
    # 캔들 패턴 (하락 국면 망치형·장악형 → 매수, 상승 국면 유성형·하락장악형 → 매도)
    body = (c - o).abs(); rng = (h - l).replace(0, np.nan)
    lw = np.minimum(o, c) - l; uw = h - np.maximum(o, c)
    hammer = ((lw >= 2 * body) & (lw / rng >= 0.6) & (body > 0)).fillna(False)
    star = ((uw >= 2 * body) & (uw / rng >= 0.6) & (body > 0)).fillna(False)
    bull_eng = ((c.shift(1) < o.shift(1)) & (c > o) & (c > o.shift(1)) & (o < c.shift(1))).fillna(False)
    bear_eng = ((c.shift(1) > o.shift(1)) & (c < o) & (c < o.shift(1)) & (o > c.shift(1))).fillna(False)
    for nm in [20, 50, 100]:
        ma = c.rolling(nm).mean()
        dn_ctx, up_ctx = (c < ma).fillna(False), (c > ma).fillna(False)
        for H in [4, 12, 24, 48]:
            Hn = x.nb(H)
            b = pd.Series(((hammer | bull_eng) & dn_ctx).astype(float)).rolling(Hn).max().fillna(0).to_numpy()
            s = pd.Series(((star | bear_eng) & up_ctx).astype(float)).rolling(Hn).max().fillna(0).to_numpy()
            P.append(b); N.append(f"역추세추적/캔들 반전(망치·상승장악) MA{nm} 하락국면 보유{H}h LF")
            P.append(np.clip(b - s, -1, 1)); N.append(f"역추세추적/캔들 반전(망치·장악 양방향) MA{nm} 보유{H}h LS")
    return N, P


# ============ 거래량·자금흐름 ============
def g_flow(x):
    o, h, l, c, v, qv, tb = unpack(x); P, N = [], []
    for n in [14, 28, 56, 112]:
        cm = ind.cmf(h, l, c, v, n).fillna(0).to_numpy()
        for th in [0.0, 0.05, 0.1]:
            P.append(hold(cm > th, cm < 0)); N.append(f"거래량·자금흐름/CMF n={n} >{th} LF")
    tbr = (tb / qv.replace(0, np.nan)).fillna(0.5)
    for m in [6, 24, 72]:
        mm = x.nb(m)
        base = tbr.rolling(mm * 10).mean(); sdv = tbr.rolling(mm * 10).std()
        z = ((tbr.rolling(mm).mean() - base) / sdv).fillna(0).to_numpy()
        for k in [0.5, 1.0, 1.5]:
            P.append(hold(z > k, z < 0)); N.append(f"거래량·자금흐름/테이커 매수비중 추세 {m}h z>{k} LF")
            P.append(hold(z < -k, z > 0)); N.append(f"거래량·자금흐름/테이커 매수비중 역추세(z<-{k}) {m}h LF")
    vz = ((qv - qv.rolling(x.nb(720)).mean()) / qv.rolling(x.nb(720)).std()).fillna(0).to_numpy()
    cc, oo = c.to_numpy(), o.to_numpy()
    for k in [1.0, 2.0, 3.0]:
        for H in [4, 12, 48]:
            Hn = x.nb(H)
            up = pd.Series(((vz > k) & (cc > oo)).astype(float)).rolling(Hn).max().fillna(0).to_numpy()
            dn = pd.Series(((vz > k) & (cc < oo)).astype(float)).rolling(Hn).max().fillna(0).to_numpy()
            P.append(up); N.append(f"거래량·자금흐름/거래량 급증 양봉 z>{k} 보유{H}h LF")
            P.append(np.clip(up - dn, -1, 1)); N.append(f"거래량·자금흐름/거래량 급증 방향추종 z>{k} 보유{H}h LS")
    return N, P


# ============ 차트 패턴 ============
def g_pattern(x):
    o, h, l, c, v, qv, tb = unpack(x); P, N = [], []
    cc, hh, ll, oo = c.to_numpy(), h.to_numpy(), l.to_numpy(), o.to_numpy()
    n = len(cc)
    ema200 = ind.ema(c, 200).to_numpy()
    for k in [3, 5, 10]:
        shv, shi, slv, sli, ish, isl = ind.swings(h, l, k)
        # 이중 바닥 / 이중 천장
        for tol in [0.01, 0.02, 0.04]:
            posL, posS = np.zeros(n), np.zeros(n)
            lows, highs, armed_l, armed_s = [], [], None, None
            curL = curS = 0.0
            for i in range(n):
                if isl[i]:
                    t = i - k; lows.append((t, ll[t]))
                    if len(lows) >= 2:
                        (t1, l1), (t2, l2) = lows[-2], lows[-1]
                        if abs(l2 - l1) / l1 < tol and t2 - t1 >= 2 * k + 2:
                            armed_l = (max(hh[t1:t2 + 1]), min(l1, l2))
                if ish[i]:
                    t = i - k; highs.append((t, hh[t]))
                    if len(highs) >= 2:
                        (t1, h1), (t2, h2) = highs[-2], highs[-1]
                        if abs(h2 - h1) / h1 < tol and t2 - t1 >= 2 * k + 2:
                            armed_s = (min(ll[t1:t2 + 1]), max(h1, h2))
                if curL == 0 and armed_l is not None and cc[i] > armed_l[0]:
                    curL, stopL = 1.0, armed_l[1]; armed_l = None
                elif curL == 1.0 and cc[i] < stopL:
                    curL = 0.0
                if curS == 0 and armed_s is not None and cc[i] < armed_s[0]:
                    curS, stopS = -1.0, armed_s[1]; armed_s = None
                elif curS == -1.0 and cc[i] > stopS:
                    curS = 0.0
                posL[i], posS[i] = curL, curS
            P.append(posL); N.append(f"차트패턴/이중바닥 돌파 k={k} 허용오차{int(tol*100)}% LF")
            P.append(posL + posS); N.append(f"차트패턴/이중바닥·이중천장 k={k} 허용오차{int(tol*100)}% LS")
        # 피보나치 되돌림 진입 (상승 다리 후 눌림)
        leg_up = np.nan_to_num(sli) < np.nan_to_num(shi)
        rng = (shv - slv)
        retr = np.where(rng > 0, (shv - cc) / rng, np.nan)
        for lo_, hi_ in [(0.382, 0.618), (0.5, 0.786), (0.236, 0.5)]:
            el = leg_up & (retr >= lo_) & (retr <= hi_) & (cc > oo) & (cc > ema200)
            xl = (cc > shv) | (cc < slv)
            P.append(hold(np.nan_to_num(el).astype(bool), np.nan_to_num(xl).astype(bool))); N.append(f"차트패턴/피보나치 되돌림 {lo_}~{hi_} k={k} (상승추세 눌림 매수) LF")
    # 삼각수렴형: 고점은 낮아지고 저점은 높아지는(수렴) 상태에서 상단 돌파
    for k in [5, 10]:
        shv, shi, slv, sli, ish, isl = ind.swings(h, l, k)
        sh_prev = pd.Series(np.where(ish, shv, np.nan)).ffill().shift(1)
        # 직전 스윙 고점 대비 낮고, 직전 스윙 저점 대비 높은 → 수렴
        hv = pd.Series(np.where(ish, shv, np.nan)); lv = pd.Series(np.where(isl, slv, np.nan))
        h2 = hv.dropna(); l2 = lv.dropna()
        lower_high = pd.Series(np.nan, index=range(n)); higher_low = pd.Series(np.nan, index=range(n))
        lower_high.loc[h2.index] = (h2.diff() < 0).astype(float).to_numpy(); higher_low.loc[l2.index] = (l2.diff() > 0).astype(float).to_numpy()
        conv = ((lower_high.ffill() == 1) & (higher_low.ffill() == 1)).to_numpy()
        up = conv & (cc > np.nan_to_num(shv, nan=np.inf)); dn = conv & (cc < np.nan_to_num(slv, nan=-np.inf))
        for H in [24, 72, 168]:
            Hn = x.nb(H)
            u = pd.Series(up.astype(float)).rolling(Hn).max().fillna(0).to_numpy(); d = pd.Series(dn.astype(float)).rolling(Hn).max().fillna(0).to_numpy()
            P.append(u); N.append(f"차트패턴/삼각수렴 상방 이탈 k={k} 보유{H}h LF")
            P.append(np.clip(u - d, -1, 1)); N.append(f"차트패턴/삼각수렴 이탈 추종 k={k} 보유{H}h LS")
    return N, P


# ============ 상대가치·선행후행 (BTC·ETH 두 자산만) ============
def g_crossasset(x):
    P, N = [], []
    oc = x.other.reindex(x.idx)
    ro = oc.pct_change().fillna(0)
    rs = x.C.pct_change().fillna(0)
    for m in [1, 3, 6, 12, 24]:
        n = x.nb(m) if x.tf != "1d" else max(1, m // 24)
        sg = np.sign((oc / oc.shift(n) - 1).fillna(0).to_numpy())
        for H in [1, 4, 12]:
            Hn = max(1, x.nb(H))
            up = pd.Series((sg > 0).astype(float)).rolling(Hn).max().fillna(0).to_numpy()
            dn = pd.Series((sg < 0).astype(float)).rolling(Hn).max().fillna(0).to_numpy()
            P.append(up); N.append(f"선행후행/{x.otherName} {m}h 수익률 상승 → {x.sym[:3]} 보유{H}h LF")
            P.append(np.clip(up - dn, -1, 1)); N.append(f"선행후행/{x.otherName} {m}h 방향 추종 → {x.sym[:3]} 보유{H}h LS")
            P.append(np.clip(dn - up, -1, 1)); N.append(f"선행후행/{x.otherName} {m}h 방향 역추종 → {x.sym[:3]} 보유{H}h LS")
    # 상대강도: 이 자산이 상대적으로 강할 때 보유 / 약할 때 회피
    for h_ in [72, 168, 336, 720]:
        n = x.nb(h_)
        rel = (x.C / oc).pipe(np.log)
        sg = np.sign((rel - rel.shift(n)).fillna(0).to_numpy())
        P.append(np.where(sg > 0, 1.0, 0.0)); N.append(f"선행후행/상대강도 {h_}h 우위일 때 보유 LF")
        P.append(np.where(sg < 0, 1.0, 0.0)); N.append(f"선행후행/상대강도 {h_}h 열위일 때 보유(평균회귀) LF")
    return N, P


# ============ 레짐 전환 ============
def g_regime(x):
    o, h, l, c, v, qv, tb = unpack(x); P, N = [], []
    cc = c.to_numpy()
    sma_l = c.rolling(x.nb(1200)).mean().to_numpy()
    trend_pos = np.where(cc > sma_l, 1.0, 0.0)
    for n in [14, 28]:
        pdi, mdi, a = ind.adx(h, l, c, n)
        av = a.fillna(0).to_numpy()
        for th in [20, 25, 30]:
            m, u, lo = ind.bbands(c, 20, 2.0)
            lw = (c < lo).fillna(False).to_numpy(); mid = (c > m).fillna(False).to_numpy()
            mr = hold(lw, mid)
            P.append(np.where(av > th, trend_pos, mr)); N.append(f"레짐전환/ADX n={n} th={th} 추세장=이동평균 횡보장=볼린저회귀 LF")
            P.append(np.where(av > th, trend_pos, 0.0)); N.append(f"레짐전환/ADX n={n} th={th} 추세장만 거래 LF")
    return N, P


GROUPS = {"추세추종": g_trend, "모멘텀": g_momentum, "돌파": g_breakout, "평균회귀": g_meanrev, "역추세추적": g_contrarian,
          "거래량·자금흐름": g_flow, "차트패턴": g_pattern, "선행후행": g_crossasset, "레짐전환": g_regime}


def events(p, tf):
    d = np.abs(np.diff(np.nan_to_num(np.asarray(p, float)), prepend=0.0))
    return float((d >= 0.25).sum() / (len(p) / core.PPY[tf]))


def main():
    tfs, tag = sys.argv[1].split(","), sys.argv[2]
    groups = sys.argv[3].split(",") if len(sys.argv) > 3 else list(GROUPS)
    out, t0 = [], time.time()
    for g in groups:
        for tf in tfs:
            xs = {s: S.Ctx(s, tf) for s in S.SYMS}
            for sym, x in xs.items():
                other = [s for s in S.SYMS if s != sym][0]
                x.other, x.otherName = xs[other].C, other[:3]
                try:
                    N, P = GROUPS[g](x)
                    ev = np.array([events(p, tf) for p in P])
                    r = S.evaluate(g, x, N, P, 2022, 1.5)
                    if r:
                        idx = {n_: i for i, n_ in enumerate(N)}
                        r["ev_median_params"] = round(float(np.median(ev)), 1)
                        r["ev_full_best"] = round(float(ev[idx[r["full_best"]["name"]]]), 1)
                        pk = [idx[v[0]] for v in r["picks"].values()]
                        r["ev_picks_mean"] = round(float(ev[pk].mean()), 1)
                        # 그룹 내 전략 단위 상위 (전체 표본 Sharpe 기준 상위 8개와 해당 빈도)
                        R = np.array([x.run(np.asarray(p, float)) for p in P]).T
                        sh = np.where(R.std(0) > 0, R.mean(0) / np.where(R.std(0) > 0, R.std(0), 1) * np.sqrt(x.ppy), 0)
                        top = np.argsort(-sh)[:8]
                        oos_m = x.idx.year.to_numpy() >= 2022
                        r["top_full"] = [{"name": N[i], "sh_all": round(float(sh[i]), 2),
                                          "sh_2022+": round(float(R[oos_m, i].mean() / R[oos_m, i].std() * np.sqrt(x.ppy)) if R[oos_m, i].std() > 0 else 0, 2),
                                          "ev": round(float(ev[i]), 1)} for i in top]
                        r["group"] = g
                        out.append(r)
                        print(g, tf, sym[:3], "n", r["n_params"], "WF OOS Sh", r["oos_sharpe"], "ev", r["ev_picks_mean"], f"{time.time()-t0:.0f}s", flush=True)
                except Exception as e:  # noqa: BLE001
                    import traceback; traceback.print_exc()
                    out.append({"family": g, "sym": sym, "tf": tf, "error": repr(e)})
    Path(f"logs/screen_{tag}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
