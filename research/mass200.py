"""이론 계열별 전략 220종 생성·검증 (Binance BTC·ETH만).
계열: A 위험프리미엄 B 강제흐름 C 변동성 D 마이크로구조대용 E 추세 F 평균회귀 G 상대가치 H 행동 I 매크로 J 계절
규율: 학습 2020~2023 / 테스트 2024~, 비용 편도 6bp, 펀딩 반영, 위약(위상회전)으로 null-max 문턱 산출.
사용: python research/mass200.py
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R))
from freesia import store
from freesia.bt import core, ind
TE = pd.Timestamp("2024-01-01", tz="UTC")
PPY = 24 * 365
COST = 0.0006
rng = np.random.default_rng(101)
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)


def feat(sym):
    b = core.bars(sym, "1h")
    i = b.index
    F = pd.DataFrame(index=i)
    F["c"], F["h"], F["l"], F["o"] = b.close, b.high, b.low, b.open
    F["v"], F["n"], F["tb"] = b.quote_volume, b.trades, b.taker_buy_quote_volume
    F["r"] = np.log(F.c).diff()
    F["imb"] = 2 * F.tb / F.v.replace(0, np.nan) - 1
    f = store.load("funding", sym).set_index("ts").rate
    F["fund"] = f.reindex(i, method="ffill")
    F["fnow"] = core.funding_bars(sym, i, "1h")
    pr = store.load("premium_1m", sym).set_index("ts").close
    F["prem"] = pr.resample("1h").last().reindex(i) * 1e4
    try:
        m5 = store.load("metrics_5m", sym).set_index("ts")
        F["oi"] = m5["sum_open_interest"].resample("1h").last().reindex(i)
        F["lsr"] = m5["sum_toptrader_long_short_ratio"].resample("1h").last().reindex(i)
        F["tlsr"] = m5["sum_taker_long_short_vol_ratio"].resample("1h").last().reindex(i)
    except Exception:
        F["oi"] = F["lsr"] = F["tlsr"] = np.nan
    E = R / "data" / "ext"
    tag = "btc" if sym.startswith("BTC") else "eth"
    try:
        F["dvol"] = pd.read_parquet(E / f"dvol_{tag}_1h.parquet").close.reindex(i, method="ffill")
    except Exception:
        F["dvol"] = np.nan
    try:
        bs = pd.read_parquet(E / f"basis_{tag}.parquet")
        F["basis"] = bs.basis_ann.reindex(i, method="ffill").shift(24)
    except Exception:
        F["basis"] = np.nan
    try:
        fr = pd.read_parquet(E / "fred.parquet")
        for c in ("vix", "nasdaq", "usd_broad", "y10", "hy_spread"):
            if c in fr:
                F[c] = fr[c].reindex(i, method="ffill").shift(24)
    except Exception:
        pass
    try:
        F["fng"] = pd.read_parquet(E / "fng.parquet").value.reindex(i, method="ffill").shift(24)
    except Exception:
        F["fng"] = np.nan
    try:
        cm = pd.read_parquet(E / f"cm_{tag}.parquet")
        for c in ("CapMVRVCur", "AdrActCnt", "FlowInExUSD", "SplyExUSD"):
            if c in cm:
                F[c] = cm[c].reindex(i, method="ffill").shift(24)
    except Exception:
        pass
    try:
        F["stbl"] = pd.read_parquet(E / "stablecoins.parquet").stable_total.reindex(i, method="ffill").shift(24)
    except Exception:
        F["stbl"] = np.nan
    return b, F


def z(s, w):
    return (s - s.rolling(w).mean()) / s.rolling(w).std()


def build(F):
    """이론 계열별 신호 사전 반환: name -> (계열, 포지션 Series in [-1,1])"""
    S = {}
    c, r = F.c, F.r
    def add(nm, fam, p):
        if not isinstance(p, pd.Series):
            p = pd.Series(np.asarray(p, dtype=float).ravel(), index=F.index)
        S[nm] = (fam, p.astype(float).clip(-1, 1).fillna(0))

    # ── A. 위험 프리미엄 (펀딩·베이시스·기간구조)
    for w in (8, 24, 72, 168):
        fm = F.fund.rolling(w).mean()
        add(f"A 펀딩평균{w}h 숏", "A위험프리미엄", -np.sign(fm))
        add(f"A 펀딩평균{w}h 임계 숏", "A위험프리미엄", -(fm > fm.rolling(2000).quantile(.7)).astype(float))
        add(f"A 펀딩 z{w} 역", "A위험프리미엄", -z(F.fund, w * 10).clip(-2, 2) / 2)
    for w in (24, 168, 720):
        add(f"A 베이시스 z{w} 숏", "A위험프리미엄", -z(F.basis, w).clip(-2, 2) / 2)
        add(f"A 베이시스 수준 숏{w}", "A위험프리미엄", -(F.basis > F.basis.rolling(w).median()).astype(float))
    add("A 펀딩+베이시스 동시 숏", "A위험프리미엄", -((F.fund > 0) & (F.basis > 0)).astype(float))
    add("A 펀딩 음수 롱", "A위험프리미엄", (F.fund < 0).astype(float))
    for q in (.6, .8, .9):
        add(f"A 펀딩 상위{int(q*100)} 숏", "A위험프리미엄", -(F.fund > F.fund.rolling(2000).quantile(q)).astype(float))
    # ── B. 강제 흐름 (OI·청산·정산·포지셔닝)
    for w in (6, 24, 72):
        doi = F.oi.pct_change(w)
        add(f"B OI{w}h 급증 역", "B강제흐름", -np.sign(doi - doi.rolling(500).quantile(.9)).clip(0, 1) * np.sign(r.rolling(w).sum()))
        add(f"B OI{w}h 급감 추종", "B강제흐름", np.sign(-doi + doi.rolling(500).quantile(.1)).clip(0, 1) * np.sign(r.rolling(w).sum()))
        add(f"B OI↑가격↓ 롱{w}", "B강제흐름", ((doi > 0) & (r.rolling(w).sum() < 0)).astype(float))
        add(f"B OI↓가격↓ 롱{w}", "B강제흐름", ((doi < 0) & (r.rolling(w).sum() < 0)).astype(float))
    for w in (24, 72):
        add(f"B 롱숏비 z{w} 역", "B강제흐름", -z(F.lsr, w * 5).clip(-2, 2) / 2)
        add(f"B 테이커롱숏 z{w} 역", "B강제흐름", -z(F.tlsr, w * 5).clip(-2, 2) / 2)
    for h in (0, 8, 16):
        add(f"B 정산{h}시 직후 24h", "B강제흐름", pd.Series(np.where(F.index.hour == h, 1.0, 0.0), index=F.index).rolling(24).max())
    for k_ in (3, 5, 8):
        sd = r.rolling(720).std()
        add(f"B 급락{k_}σ 후 롱", "B강제흐름", (r < -k_ * sd).astype(float).rolling(24).max())
        add(f"B 급등{k_}σ 후 숏", "B강제흐름", -(r > k_ * sd).astype(float).rolling(24).max())
        add(f"B 급락{k_}σ+OI감소 롱", "B강제흐름", ((r < -k_ * sd) & (F.oi.pct_change(6) < 0)).astype(float).rolling(24).max())
    # ── C. 변동성
    rv = r.rolling(24).std()
    for w in (24, 72, 168):
        rvw = r.rolling(w).std()
        add(f"C 저변동{w} 롱", "C변동성", (rvw < rvw.rolling(2000).quantile(.3)).astype(float))
        add(f"C 고변동{w} 숏", "C변동성", -(rvw > rvw.rolling(2000).quantile(.7)).astype(float))
        add(f"C 변동성목표{w}", "C변동성", (0.02 / rvw).clip(0, 1))
        add(f"C 추세×변동성역{w}", "C변동성", np.sign(c - c.rolling(w).mean()) * (0.02 / rvw).clip(0, 1))
    for w in (24, 168):
        add(f"C DVOL-RV{w} 롱", "C변동성", ((F.dvol - rv * np.sqrt(PPY) * 100) > 0).astype(float))
        add(f"C DVOL z{w} 역", "C변동성", -z(F.dvol, w * 5).clip(-2, 2) / 2)
        add(f"C DVOL 노출조절{w}", "C변동성", np.sign(c - c.rolling(w).mean()) * (F.dvol.median() / F.dvol).clip(0.3, 2))
    har = (np.log(rv) * .4 + np.log(r.rolling(168).std()) * .3 + np.log(r.rolling(720).std()) * .3)
    add("C HAR 저변동 롱", "C변동성", (har < har.rolling(2000).quantile(.3)).astype(float))
    add("C HAR 추세 사이징", "C변동성", np.sign(c - c.rolling(168).mean()) * (har.rolling(2000).rank(pct=True) < .5).astype(float))
    for k_ in (4, 6):
        add(f"C 점프{k_}σ 후 롱", "C변동성", (r.abs() > k_ * r.rolling(720).std()).astype(float).rolling(12).max())
    # ── D. 마이크로구조 대용
    for w in (1, 6, 24, 72):
        add(f"D 테이커불균형{w} 추종", "D마이크로", np.sign(F.imb.rolling(w).mean()))
        add(f"D 테이커불균형{w} 역", "D마이크로", -np.sign(F.imb.rolling(w).mean()))
        add(f"D 불균형z{w} 역", "D마이크로", -z(F.imb.rolling(w).mean(), 500).clip(-2, 2) / 2)
    am = (r.abs() / F.v.replace(0, np.nan))
    for w in (24, 168):
        add(f"D Amihud{w} 고 역", "D마이크로", -(am.rolling(w).mean() > am.rolling(2000).quantile(.8)).astype(float) * np.sign(r.rolling(w).sum()))
        add(f"D Amihud{w} 저 추종", "D마이크로", (am.rolling(w).mean() < am.rolling(2000).quantile(.2)).astype(float) * np.sign(r.rolling(w).sum()))
    add("D Roll 스프레드 역", "D마이크로", -np.sign(r.rolling(24).apply(lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 3 else 0, raw=True)))
    for w in (6, 24):
        add(f"D 프리미엄z{w} 역", "D마이크로", -z(F.prem, w * 10).clip(-2, 2) / 2)
        add(f"D 프리미엄 부호{w} 역", "D마이크로", -np.sign(F.prem.rolling(w).mean()))
    add("D 체결당거래대금 급증 역", "D마이크로", -np.sign(r) * ((F.v / F.n) > (F.v / F.n).rolling(2000).quantile(.9)).astype(float))
    for w in (24, 168):
        add(f"D 거래량가중 반전{w}", "D마이크로", -np.sign((r * F.v).rolling(w).sum()))
    # ── E. 추세
    for f_, s_ in ((12, 48), (24, 96), (48, 192), (96, 384), (168, 672)):
        add(f"E EMA{f_}/{s_}", "E추세", np.sign(c.ewm(span=f_).mean() - c.ewm(span=s_).mean()))
        add(f"E EMA{f_}/{s_} 변동조절", "E추세", np.sign(c.ewm(span=f_).mean() - c.ewm(span=s_).mean()) * (0.02 / r.rolling(168).std()).clip(0, 1))
    for w in (24, 72, 168, 336, 720):
        add(f"E TSMOM{w}", "E추세", np.sign(c / c.shift(w) - 1))
        add(f"E 돌파{w}", "E추세", np.where(c > F.h.rolling(w).max().shift(1), 1.0, np.where(c < F.l.rolling(w).min().shift(1), -1.0, np.nan)))
        add(f"E 이격{w}", "E추세", ((c / c.rolling(w).mean() - 1) * 20).clip(-1, 1))
    add("E 다속도 앙상블", "E추세", pd.concat([np.sign(c / c.shift(w) - 1) for w in (24, 72, 168, 336, 720)], axis=1).mean(axis=1))
    for n_ in (10, 20, 30):
        add(f"E ADX{n_} 추세", "E추세", np.sign(c - c.rolling(n_ * 4).mean()) * (ind.adx(F.h, F.l, c, n_)[2] > 25).astype(float))
    add("E 수퍼트렌드", "E추세", pd.Series(np.asarray(ind.supertrend(F.h, F.l, c, 10, 3.0)).ravel(), index=F.index))
    # ── F. 평균회귀
    for w in (6, 24, 72, 168, 336):
        add(f"F 반전{w}", "F평균회귀", -np.sign(c / c.shift(w) - 1))
        add(f"F z역{w}", "F평균회귀", -z(c, w).clip(-2, 2) / 2)
    for n_ in (7, 14, 28):
        rs = ind.rsi(c, n_)
        add(f"F RSI{n_} 역", "F평균회귀", -((rs - 50) / 30).clip(-1, 1))
        add(f"F RSI{n_} 극단", "F평균회귀", np.where(rs < 30, 1.0, np.where(rs > 70, -1.0, 0.0)))
    for n_, k_ in ((20, 2), (50, 2), (20, 3)):
        m_, u_, l_ = ind.bbands(c, n_, k_)[0], ind.bbands(c, n_, k_)[1], ind.bbands(c, n_, k_)[2]
        add(f"F 볼린저{n_}/{k_} 역", "F평균회귀", np.where(c < l_, 1.0, np.where(c > u_, -1.0, 0.0)))
    # ── G. 상대가치
    add("G 프리미엄 회귀", "G상대가치", -z(F.prem, 240).clip(-2, 2) / 2)
    for w in (24, 168):
        add(f"G 베이시스-펀딩 괴리{w}", "G상대가치", -z(F.basis - F.fund * 3 * 365, w * 5).clip(-2, 2) / 2)
    # ── H. 행동
    for w in (24, 168):
        add(f"H 공포탐욕 역{w}", "H행동", -z(F.fng, w * 5).clip(-2, 2) / 2)
        add(f"H 공포탐욕 극단{w}", "H행동", np.where(F.fng < 25, 1.0, np.where(F.fng > 75, -1.0, 0.0)))
    add("H 라운드넘버 근접 역", "H행동", -np.sign(r) * ((c % (10 ** np.floor(np.log10(c)) / 10) / (10 ** np.floor(np.log10(c)) / 10)) < 0.02).astype(float))
    for w in (24, 72):
        add(f"H 연속봉{w} 역", "H행동", -np.sign(np.sign(r).rolling(w).sum()) * (np.sign(r).rolling(w).sum().abs() > w * .5).astype(float))
    # ── I. 매크로·온체인
    for cname in ("vix", "nasdaq", "usd_broad", "y10", "hy_spread"):
        if cname in F:
            for w in (72, 336):
                add(f"I {cname} z{w} 역", "I매크로", -z(F[cname], w).clip(-2, 2) / 2)
                add(f"I {cname} 추세{w}", "I매크로", np.sign(F[cname] / F[cname].shift(w) - 1) * (1 if cname == "nasdaq" else -1))
    for cname in ("CapMVRVCur", "AdrActCnt", "FlowInExUSD", "SplyExUSD", "stbl"):
        if cname in F:
            for w in (168, 720):
                add(f"I {cname} z{w} 역", "I매크로", -z(F[cname], w).clip(-2, 2) / 2)
    # ── J. 계절·시간
    for h in (0, 4, 8, 12, 14, 16, 20):
        add(f"J {h}시 롱", "J계절", pd.Series(np.where(F.index.hour == h, 1.0, 0.0), index=F.index))
    for d in range(7):
        add(f"J 요일{d} 롱", "J계절", pd.Series(np.where(F.index.dayofweek == d, 1.0, 0.0), index=F.index))
    add("J 주말 롱", "J계절", pd.Series(np.where(F.index.dayofweek >= 5, 1.0, 0.0), index=F.index))
    add("J 미국장 롱", "J계절", pd.Series(np.where((F.index.hour >= 13) & (F.index.hour < 21), 1.0, 0.0), index=F.index))
    add("J 아시아장 롱", "J계절", pd.Series(np.where((F.index.hour >= 0) & (F.index.hour < 8), 1.0, 0.0), index=F.index))
    return S


def metrics(net):
    m, s = net.mean(), net.std()
    eq = np.cumprod(1 + net)
    return m * PPY, (m / s * np.sqrt(PPY) if s > 0 else 0.0), float((eq / np.maximum.accumulate(eq) - 1).min())


if __name__ == "__main__":
    ALL = []
    for sym in ("BTCUSDT", "ETHUSDT"):
        b, F = feat(sym)
        S = build(F)
        ret = b.close.pct_change().shift(-1).fillna(0).values
        fund = core.funding_bars(sym, b.index, "1h")
        tr = (b.index < TE).values; te = ~tr
        P("=" * 100 + f"\n### {sym} — 전략 {len(S)}종")
        rows = []
        pool = []
        for nm, (fam, pos) in S.items():
            p = np.nan_to_num(pd.Series(pos, index=F.index).astype(float).values)
            pp = np.concatenate([[0.0], p[:-1]])
            net = pp * ret - np.abs(np.diff(pp, prepend=0.0)) * COST - pp * fund
            a1, s1, d1 = metrics(net[tr]); a2, s2, d2 = metrics(net[te])
            turn = np.abs(np.diff(pp, prepend=0.0))[te].sum() / 2
            rows.append((nm, fam, a1 * 100, s1, a2 * 100, s2, d2 * 100, turn / ((b.index[-1] - TE).days / 7)))
            for _ in range(10):
                ps = np.roll(p, rng.integers(500, len(p) - 500))
                pps = np.concatenate([[0.0], ps[:-1]])
                ns = pps * ret - np.abs(np.diff(pps, prepend=0.0)) * COST - pps * fund
                pool.append(metrics(ns[te])[1])
        d = pd.DataFrame(rows, columns=["전략", "계열", "학습연%", "학습Sh", "테스트연%", "테스트Sh", "테스트MDD%", "주당거래"])
        pool = np.array(pool)
        thr95 = np.percentile(pool, 95); thr99 = np.percentile(pool, 99); mx = pool.max()
        P(f"위약 {len(pool)}회 테스트 Sharpe 분포: 평균 {pool.mean():.2f} 95% {thr95:.2f} 99% {thr99:.2f} **최대 {mx:.2f}**")
        P(f"→ 전략 {len(S)}개를 시도했으므로 문턱은 **null-max {mx:.2f}**")
        P("\n[테스트 Sharpe 상위 20]")
        P(d.sort_values("테스트Sh", ascending=False).head(20).round(2).to_string(index=False))
        P(f"\n학습·테스트 모두 양수: {int(((d.학습Sh>0)&(d.테스트Sh>0)).sum())}/{len(d)}")
        P(f"테스트 Sharpe > 위약 95%({thr95:.2f}): {int((d.테스트Sh>thr95).sum())}개")
        P(f"테스트 Sharpe > **null-max({mx:.2f})**: {int((d.테스트Sh>mx).sum())}개")
        P("\n[계열별 평균]")
        P(d.groupby("계열")[["학습Sh", "테스트Sh"]].agg(["mean", "max", "count"]).round(2).to_string())
        d["종목"] = sym; ALL.append(d)
    A = pd.concat(ALL)
    P("\n" + "=" * 100 + "\n[두 종목 공통] 학습·테스트 모두 양수이고 양 종목에서 성립")
    piv = A.pivot_table(index="전략", columns="종목", values="테스트Sh")
    trn = A.pivot_table(index="전략", columns="종목", values="학습Sh")
    ok = piv[(piv.min(axis=1) > 0) & (trn.min(axis=1) > 0)]
    P(f"  {len(ok)} / {len(piv)}개")
    if len(ok):
        P(ok.assign(최소=ok.min(axis=1)).sort_values("최소", ascending=False).head(15).round(2).to_string())
