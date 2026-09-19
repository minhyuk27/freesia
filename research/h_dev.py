"""기준 v2(가설 카드 우선) 적용 전략 개발. 사전 등록한 가설 H1~H5만 검증한다.
공통 규율: 학습 2020~2023 / 테스트 2024~ (테스트 1회), BTC·ETH 둘 다, 비용 taker 왕복 12bp, 대조군 비교, 시도 수 기록.
사용: python research/h_dev.py  → logs/h_dev.out, logs/h_dev.summary"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from freesia import store  # noqa: E402
from freesia.bt import core  # noqa: E402

COST = 0.0006          # 편도 6bp (taker 5 + 슬리피지 1)
TRAIN_END = "2024-01-01"
OUT, SUMM, TRIES = [], [], {}
rng = np.random.default_rng(7)


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    OUT.append(s)


def tstat(x):
    x = np.asarray(x, float)
    return float(x.mean() / x.std() * np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else 0.0


def ann(net, ppy):
    m, s = np.mean(net), np.std(net)
    return m * ppy, (m / s * np.sqrt(ppy) if s > 0 else 0.0)


def split(idx):
    return idx < pd.Timestamp(TRAIN_END, tz="UTC"), idx >= pd.Timestamp(TRAIN_END, tz="UTC")


# ───────────────────────── H1. ETH/BTC 상대가치 평균회귀 (출처 F)
def h1():
    P("\n" + "=" * 78 + "\nH1. ETH/BTC 상대가치 평균회귀 — 달러중립 페어")
    b = core.bars("BTCUSDT", "1h"); e = core.bars("ETHUSDT", "1h")
    df = pd.DataFrame({"b": b.close, "e": e.close}).dropna()
    lr = np.log(df.e / df.b)
    rb = np.log(df.b).diff().shift(-1).fillna(0).values      # 다음 봉 수익
    re_ = np.log(df.e).diff().shift(-1).fillna(0).values
    tr, te = split(df.index)
    rows = []
    n_try = 0
    for w in (24, 72, 168, 336):
        m = lr.rolling(w).mean(); s = lr.rolling(w).std()
        z = ((lr - m) / s).values
        for k in (1.5, 2.0, 2.5):
            n_try += 1
            pos = np.zeros(len(z)); cur = 0.0
            for i in range(len(z)):
                if np.isnan(z[i]):
                    cur = 0.0
                elif cur == 0.0:
                    cur = -0.5 if z[i] > k else (0.5 if z[i] < -k else 0.0)
                elif abs(z[i]) < 0.5 or np.sign(z[i]) == -np.sign(cur):
                    cur = 0.0
                pos[i] = cur
            pp = np.concatenate([[0.0], pos[:-1]])
            turn = np.abs(np.diff(pp, prepend=0.0))
            net = pp * (re_ - rb) - 2 * turn * COST     # 두 다리 모두 회전
            for nm, msk in (("학습", tr), ("테스트", te)):
                a, sh = ann(net[msk], 24 * 365)
                rows.append((w, k, nm, a * 100, sh, int(turn[msk].sum() / 2)))
    d = pd.DataFrame(rows, columns=["창", "k", "구간", "연%", "Sharpe", "거래"])
    pv = d.pivot_table(index=["창", "k"], columns="구간", values=["연%", "Sharpe"]).round(2)
    P(pv.to_string())
    TRIES["H1"] = n_try
    best = d[(d.구간 == "테스트")].sort_values("Sharpe", ascending=False).head(3)
    P("테스트 상위 3:", best.to_string(index=False))
    ok = (d[d.구간 == "테스트"]["연%"] > 0).mean()
    P(f"→ 시도 {n_try}개 중 테스트 양수 비율 {ok:.0%}")
    SUMM.append(f"H1 ETH/BTC 페어: 테스트 최고 Sharpe {best.Sharpe.max():.2f}, 양수 비율 {ok:.0%}")


# ───────────────────────── H2. 추세 앙상블 + 변동성 목표 (출처 A+G)
def h2():
    P("\n" + "=" * 78 + "\nH2. 다속도 추세 앙상블 + 변동성 목표")
    res = {}
    for sym in ("BTCUSDT", "ETHUSDT"):
        b = core.bars(sym, "4h")
        c = b.close
        r = np.log(c).diff()
        fund = core.funding_bars(sym, b.index, "4h")
        ppy = 6 * 365
        rv = r.rolling(90).std() * np.sqrt(ppy)          # 15일 변동성
        fcs = []
        for f, s in ((8, 32), (16, 64), (32, 128), (64, 256)):
            raw = (c.ewm(span=f).mean() - c.ewm(span=s).mean()) / (c * r.rolling(90).std())
            fcs.append((raw / raw.abs().rolling(500).median()).clip(-2, 2))
        ens = pd.concat(fcs, axis=1).mean(axis=1)
        ret = c.pct_change().shift(-1).fillna(0).values
        tr, te = split(b.index)
        out = {}
        for nm, pos in (("단일(32,128)", fcs[2].clip(-1, 1)),
                        ("앙상블", ens.clip(-1, 1)),
                        ("앙상블+변동성목표", (ens.clip(-2, 2) * (0.60 / rv)).clip(-2, 2))):
            p = pos.fillna(0).values
            pp = np.concatenate([[0.0], p[:-1]])
            net = pp * ret - np.abs(np.diff(pp, prepend=0.0)) * COST - pp * fund
            out[nm] = [(ann(net[m], ppy)) for m in (tr, te)]
        for nm, v in out.items():
            P(f"  {sym} {nm:20s} 학습 연{v[0][0]*100:6.1f}% Sh {v[0][1]:5.2f} | 테스트 연{v[1][0]*100:6.1f}% Sh {v[1][1]:5.2f}")
        res[sym] = out
    TRIES["H2"] = 3 * 2
    v = [res[s]["앙상블+변동성목표"][1][1] for s in res]
    SUMM.append(f"H2 추세앙상블+변동성목표: 테스트 Sharpe BTC {v[0]:.2f} ETH {v[1]:.2f}")
    return res


# ───────────────────────── H3. 청산 연쇄 후 반전 (출처 B)
def h3():
    P("\n" + "=" * 78 + "\nH3. 청산 연쇄 사건 후 반전 (OI 급감 + 가격 급변)")
    n_try = 0
    allrows = []
    for sym in ("BTCUSDT", "ETHUSDT"):
        k = store.load("klines_1m", sym).set_index("ts")
        b = k.resample("5min").agg({"close": "last", "high": "max", "low": "min"}).dropna()
        oi = store.load("metrics_5m", sym).set_index("ts")["sum_open_interest"].resample("5min").last()
        df = pd.DataFrame({"c": b.close, "oi": oi}).dropna()
        lp = np.log(df.c)
        r15 = lp.diff(3)
        sd = r15.rolling(576).std()                      # 2일
        doi = df.oi.pct_change(3)
        for a in (3, 4, 5):
            for bb in (0.005, 0.01, 0.02):
                ev_long = (r15 < -a * sd) & (doi < -bb)   # 롱 청산
                ev_short = (r15 > a * sd) & (doi < -bb)   # 숏 청산
                for H in (12, 48, 144, 288):              # 1h 4h 12h 24h
                    n_try += 1
                    fwd = lp.shift(-H) - lp
                    for side, ev, sgn in (("롱청산후매수", ev_long, 1), ("숏청산후매도", ev_short, -1)):
                        idx = np.where(ev.values)[0]
                        sel = []                          # 중복 제거(보유 기간 겹침 방지)
                        nxt = -1
                        for i in idx:
                            if i >= nxt:
                                sel.append(i); nxt = i + H
                        if len(sel) < 20:
                            continue
                        pnl = sgn * fwd.values[sel] - 2 * COST
                        ts = df.index[sel]
                        for nm, m in (("학습", ts < pd.Timestamp(TRAIN_END, tz="UTC")), ("테스트", ts >= pd.Timestamp(TRAIN_END, tz="UTC"))):
                            x = pnl[m][~np.isnan(pnl[m])]
                            if len(x) < 10:
                                continue
                            allrows.append((sym, side, a, bb, H, nm, np.mean(x) * 1e4, tstat(x), len(x), float((x > 0).mean())))
    d = pd.DataFrame(allrows, columns=["종목", "방향", "a", "b", "H", "구간", "평균bp", "t", "n", "승률"])
    TRIES["H3"] = n_try
    P(f"시도(설정×보유) {n_try}개. 사건 수 20 이상만 집계.")
    for side in d.방향.unique():
        s = d[d.방향 == side]
        P(f"\n[{side}] 테스트 구간 평균 순수익 bp 상위")
        t = s[s.구간 == "테스트"].sort_values("t", ascending=False).head(6)
        P(t.round(3).to_string(index=False))
    both = []
    for (sym, side, a, bb, H), g in d.groupby(["종목", "방향", "a", "b", "H"]):
        if set(g.구간) == {"학습", "테스트"} and (g.평균bp > 0).all():
            both.append((sym, side, a, bb, H, float(g[g.구간 == "테스트"].t.iloc[0]), float(g[g.구간 == "테스트"].평균bp.iloc[0]), int(g[g.구간 == "테스트"].n.iloc[0])))
    P(f"\n학습·테스트 모두 양수인 설정: {len(both)} / {n_try*2}")
    if both:
        bd = pd.DataFrame(both, columns=["종목", "방향", "a", "b", "H", "테스트t", "테스트bp", "n"]).sort_values("테스트t", ascending=False)
        P(bd.head(12).round(3).to_string(index=False))
        # 두 종목 모두 성립하는 설정
        key = bd.groupby(["방향", "a", "b", "H"]).종목.nunique()
        P("\n두 종목 모두 성립:", int((key == 2).sum()), "개")
        if (key == 2).any():
            P(bd[bd.set_index(["방향", "a", "b", "H"]).index.isin(key[key == 2].index)].round(3).to_string(index=False))
    # 대조군: 무작위 진입 (같은 보유기간)
    P("\n[대조군] 무작위 진입 300회 × 보유 4시간, 테스트 구간 평균 bp 분포")
    k = store.load("klines_1m", "BTCUSDT").set_index("ts")
    b = k.resample("5min").close.last().dropna()
    lp = np.log(b); fwd = (lp.shift(-48) - lp).values
    te = (b.index >= pd.Timestamp(TRAIN_END, tz="UTC")).nonzero()[0]
    te = te[te < len(fwd) - 48]
    means = [np.nanmean(fwd[rng.choice(te, 120, replace=False)]) * 1e4 - 12 for _ in range(300)]
    P(f"  평균 {np.mean(means):.1f}bp, 95% {np.percentile(means,95):.1f}bp, 99% {np.percentile(means,99):.1f}bp, 최대 {np.max(means):.1f}bp")
    SUMM.append(f"H3 청산연쇄 반전: 학습·테스트 모두 양수 {len(both)}개 / 시도 {n_try*2}")
    return d


# ───────────────────────── H4. 캐리 타이밍 (출처 A)
def h4():
    P("\n" + "=" * 78 + "\nH4. 캐리 타이밍 — 펀딩 수준 조건부 (현물 롱 + 무기한 숏)")
    for sym in ("BTCUSDT", "ETHUSDT"):
        f = store.load("funding", sym).set_index("ts").rate
        pr = store.load("premium_1m", sym).set_index("ts").close     # (perp-index)/index
        prem = pr.reindex(f.index, method="nearest", tolerance=pd.Timedelta("5min"))
        carry = (f - prem.diff()).dropna()                            # 8시간당 순 캐리
        ppy = 3 * 365
        tr, te = split(carry.index)
        sig = f.reindex(carry.index).rolling(9).mean().shift(1)       # 직전 3일 평균 펀딩
        P(f"\n  {sym} 캐리 기본: 학습 연{carry[tr].mean()*ppy*100:.2f}% Sh {carry[tr].mean()/carry[tr].std()*np.sqrt(ppy):.2f} | 테스트 연{carry[te].mean()*ppy*100:.2f}% Sh {carry[te].mean()/carry[te].std()*np.sqrt(ppy):.2f}")
        rows = []
        for th in (0.0, 0.00005, 0.0001, 0.0002):                     # 8시간 펀딩 문턱 (0.01% = 연 10.95%)
            pos = (sig > th).astype(float)
            net = pos * carry - np.abs(pos.diff().fillna(0)) * 2 * COST
            for nm, m in (("학습", tr), ("테스트", te)):
                a, sh = ann(net[m], ppy)
                rows.append((th * 100, nm, a * 100, sh, float(pos[m].mean()), int(np.abs(pos.diff().fillna(0))[m].sum())))
        P(pd.DataFrame(rows, columns=["문턱%", "구간", "연%", "Sharpe", "가동률", "전환"]).round(3).to_string(index=False))
    TRIES["H4"] = 4 * 2
    SUMM.append("H4 캐리 타이밍: 표 참조")


# ───────────────────────── H5. 변동성 위험 프리미엄(DVOL) 레짐 (출처 A)
def h5():
    P("\n" + "=" * 78 + "\nH5. DVOL(내재변동성) − 실현변동성 = VRP 레짐")
    for sym, tag in (("BTCUSDT", "btc"), ("ETHUSDT", "eth")):
        dv = pd.read_parquet(R / "data" / "ext" / f"dvol_{tag}_1h.parquet").close
        b = core.bars(sym, "1h")
        r = np.log(b.close).diff()
        rv = r.rolling(168).std() * np.sqrt(24 * 365) * 100
        df = pd.DataFrame({"iv": dv, "rv": rv, "c": b.close}).dropna()
        vrp = df.iv - df.rv
        lp = np.log(df.c)
        P(f"\n  {sym} VRP 평균 {vrp.mean():.1f}%p (IV {df.iv.mean():.0f} vs RV {df.rv.mean():.0f}), 양수 비율 {(vrp>0).mean():.0%}")
        q = pd.qcut(vrp, 5, labels=False)
        rows = []
        for h, nm in ((24, "1일"), (168, "1주"), (720, "1개월")):
            fwd = (lp.shift(-h) - lp)
            for qq in range(5):
                m = (q == qq) & fwd.notna()
                x = fwd[m].values[::h] if m.sum() > h else fwd[m].values
                rows.append((nm, qq + 1, np.mean(x) * 100, tstat(x), len(x)))
        P(pd.DataFrame(rows, columns=["지평", "VRP분위", "평균%", "t", "n"]).round(2).to_string(index=False))
        # 추세 노출 조절기로서
        tr, te = split(df.index)
        ret = df.c.pct_change().shift(-1).fillna(0)
        trend = np.sign(lp - lp.rolling(24 * 14).mean()).fillna(0)
        for nm, pos in (("추세만", trend), ("추세×IV역비례", trend * (df.iv.median() / df.iv).clip(0.3, 2.0))):
            p = pos.values; pp = np.concatenate([[0.0], p[:-1]])
            net = pp * ret.values - np.abs(np.diff(pp, prepend=0.0)) * COST
            a1, s1 = ann(net[tr.values], 24 * 365); a2, s2 = ann(net[te.values], 24 * 365)
            P(f"    {nm:16s} 학습 연{a1*100:6.1f}% Sh {s1:5.2f} | 테스트 연{a2*100:6.1f}% Sh {s2:5.2f}")
    TRIES["H5"] = 5 * 3 * 2 + 2 * 2
    SUMM.append("H5 VRP: 표 참조")


if __name__ == "__main__":
    h1(); h2(); h3(); h4(); h5()
    P("\n" + "=" * 78 + "\n[시도 장부] " + ", ".join(f"{k} {v}개" for k, v in TRIES.items()) + f" — 합계 {sum(TRIES.values())}개")
    P("다중검정 문턱: 시도 " + str(sum(TRIES.values())) + "개 기준 t 문턱 약 " + f"{np.sqrt(2*np.log(max(sum(TRIES.values()),2))):.2f} (null-max 근사)")
    Path(R / "logs").mkdir(exist_ok=True)
    (R / "logs" / "h_dev.summary").write_text("\n".join(SUMM) + f"\n시도 합계 {sum(TRIES.values())}개, 자세한 표는 logs/h_dev.out")
