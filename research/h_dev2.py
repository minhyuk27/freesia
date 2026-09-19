"""H1 재검증(청산 조건 수정), H5 재검증, H2 정밀 평가(벤치마크·MDD·연도별·위약), 그리고 생존 전략 결합.
사용: python research/h_dev2.py"""
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

COST = 0.0006
TE = pd.Timestamp("2024-01-01", tz="UTC")
OUT, SUMM, TRIES = [], [], {}
rng = np.random.default_rng(11)


def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); OUT.append(s)


def stats(net, ppy):
    net = np.asarray(net, float)
    m, s = net.mean(), net.std()
    eq = np.cumprod(1 + net)
    mdd = float((eq / np.maximum.accumulate(eq) - 1).min())
    return m * ppy, (m / s * np.sqrt(ppy) if s > 0 else 0.0), mdd


# ── H1 재검증: 청산은 |z|<0.5 일 때만 (반대 극단에서는 반대 포지션 진입)
def h1():
    P("=" * 78 + "\nH1(수정). ETH/BTC 평균회귀 — 청산 조건 |z|<0.5 로 수정")
    b = core.bars("BTCUSDT", "1h"); e = core.bars("ETHUSDT", "1h")
    df = pd.DataFrame({"b": b.close, "e": e.close}).dropna()
    lr = np.log(df.e / df.b)
    rb = np.log(df.b).diff().shift(-1).fillna(0).values
    re_ = np.log(df.e).diff().shift(-1).fillna(0).values
    tr = (df.index < TE).values; te = ~tr
    rows = []; n = 0
    for w in (24, 72, 168, 336, 720):
        z = ((lr - lr.rolling(w).mean()) / lr.rolling(w).std()).values
        for k in (1.5, 2.0, 2.5, 3.0):
            n += 1
            pos = np.zeros(len(z)); cur = 0.0
            for i in range(len(z)):
                if np.isnan(z[i]):
                    cur = 0.0
                elif z[i] > k:
                    cur = -0.5
                elif z[i] < -k:
                    cur = 0.5
                elif abs(z[i]) < 0.5:
                    cur = 0.0
                pos[i] = cur
            pp = np.concatenate([[0.0], pos[:-1]])
            turn = np.abs(np.diff(pp, prepend=0.0))
            net = pp * (re_ - rb) - 2 * turn * COST
            gross = pp * (re_ - rb)
            for nm, m in (("학습", tr), ("테스트", te)):
                a, sh, dd = stats(net[m], 24 * 365)
                ga, _, _ = stats(gross[m], 24 * 365)
                rows.append((w, k, nm, a * 100, ga * 100, sh, dd * 100, int(turn[m].sum() / 2)))
    d = pd.DataFrame(rows, columns=["창", "k", "구간", "순연%", "수수료전연%", "Sharpe", "MDD%", "왕복"])
    P(d.pivot_table(index=["창", "k"], columns="구간", values=["수수료전연%", "순연%", "Sharpe"]).round(2).to_string())
    TRIES["H1"] = n
    t = d[d.구간 == "테스트"]
    P(f"→ 시도 {n}개 | 수수료 전 양수 {(t['수수료전연%']>0).mean():.0%} | 순 양수 {(t['순연%']>0).mean():.0%} | 최고 Sharpe {t.Sharpe.max():.2f}")
    SUMM.append(f"H1 ETH/BTC 페어(수정): 테스트 순 양수 {(t['순연%']>0).mean():.0%}, 최고 Sharpe {t.Sharpe.max():.2f}")


# ── H2 정밀: 벤치마크·MDD·연도별·위약
def h2():
    P("\n" + "=" * 78 + "\nH2(정밀). 추세 앙상블+변동성목표 — 매수보유 대비")
    ppy = 6 * 365
    sleeves = {}
    for sym in ("BTCUSDT", "ETHUSDT"):
        b = core.bars(sym, "4h"); c = b.close
        r = np.log(c).diff(); sd = r.rolling(90).std()
        rv = sd * np.sqrt(ppy)
        fund = core.funding_bars(sym, b.index, "4h")
        fcs = [((c.ewm(span=f).mean() - c.ewm(span=s).mean()) / (c * sd)) for f, s in ((8, 32), (16, 64), (32, 128), (64, 256))]
        fcs = [(x / x.abs().rolling(500).median()).clip(-2, 2) for x in fcs]
        ens = pd.concat(fcs, axis=1).mean(axis=1)
        pos = (ens.clip(-2, 2) * (0.60 / rv)).clip(-2, 2).fillna(0)
        ret = c.pct_change().shift(-1).fillna(0).values
        tr = (b.index < TE).values; te = ~tr
        p = pos.values; pp = np.concatenate([[0.0], p[:-1]])
        net = pp * ret - np.abs(np.diff(pp, prepend=0.0)) * COST - pp * fund
        bh = ret.copy()
        P(f"\n  {sym}")
        for nm, x in (("전략", net), ("매수보유", bh)):
            for lab, m in (("학습", tr), ("테스트", te)):
                a, sh, dd = stats(x[m], ppy)
                P(f"    {nm:6s} {lab} 연{a*100:7.1f}% Sh {sh:5.2f} MDD {dd*100:6.1f}%")
        # 연도별
        yr = b.index.year
        rows = [(y, stats(net[yr == y], ppy)[0] * 100, stats(net[yr == y], ppy)[1], stats(bh[yr == y], ppy)[0] * 100, float(np.mean(pp[yr == y]))) for y in range(2020, 2027)]
        P(pd.DataFrame(rows, columns=["연", "전략연%", "Sharpe", "매수보유연%", "평균노출"]).round(2).to_string(index=False))
        # 매수보유 대비 초과(베타 조정)
        bmask = te
        beta = np.polyfit(bh[bmask], net[bmask], 1)[0]
        alpha = (net[bmask] - beta * bh[bmask])
        a_, sh_, _ = stats(alpha, ppy)
        P(f"    테스트 베타 {beta:.2f}, 베타중립 초과 연{a_*100:.1f}% Sharpe {sh_:.2f}")
        # 위약: 예측 부호를 위상 회전
        pl = []
        for _ in range(200):
            sh_amt = rng.integers(500, len(p) - 500)
            ps = np.roll(p, sh_amt); pps = np.concatenate([[0.0], ps[:-1]])
            ns = pps * ret - np.abs(np.diff(pps, prepend=0.0)) * COST - pps * fund
            pl.append(stats(ns[te], ppy)[1])
        P(f"    위약(위상회전 200회) 테스트 Sharpe: 평균 {np.mean(pl):.2f}, 95% {np.percentile(pl,95):.2f}, 최대 {np.max(pl):.2f} | 실제 {stats(net[te],ppy)[1]:.2f}")
        sleeves[sym] = pd.Series(net, index=b.index)
    TRIES["H2"] = 3 * 2
    SUMM.append("H2 추세: 표 참조(매수보유·위약 비교)")
    return sleeves


# ── H5 재검증
def h5():
    P("\n" + "=" * 78 + "\nH5(수정). DVOL 기반 노출 조절")
    for sym, tag in (("BTCUSDT", "btc"), ("ETHUSDT", "eth")):
        dv = pd.read_parquet(R / "data" / "ext" / f"dvol_{tag}_1h.parquet").close
        b = core.bars(sym, "1h")
        lp = np.log(b.close)
        rv = lp.diff().rolling(168).std() * np.sqrt(24 * 365) * 100
        df = pd.DataFrame({"iv": dv, "rv": rv, "c": b.close}).dropna()
        ret = df.c.pct_change().shift(-1).fillna(0).values
        l = np.log(df.c)
        trend = np.sign(l - l.rolling(24 * 14).mean()).fillna(0)
        tr = (df.index < TE).values; te = ~tr
        P(f"\n  {sym}")
        for nm, pos in (("추세만", trend),
                        ("추세×IV역비례", trend * (df.iv.median() / df.iv).clip(0.3, 2.0)),
                        ("추세×실현변동성역비례", trend * (df.rv.median() / df.rv).clip(0.3, 2.0)),
                        ("추세×VRP양수만", trend * ((df.iv - df.rv) > 0).astype(float))):
            p = pos.fillna(0).values; pp = np.concatenate([[0.0], p[:-1]])
            net = pp * ret - np.abs(np.diff(pp, prepend=0.0)) * COST
            a1, s1, d1 = stats(net[tr], 24 * 365); a2, s2, d2 = stats(net[te], 24 * 365)
            P(f"    {nm:22s} 학습 연{a1*100:6.1f}% Sh {s1:5.2f} | 테스트 연{a2*100:6.1f}% Sh {s2:5.2f} MDD {d2*100:6.1f}%")
    TRIES["H5"] = 4 * 2
    SUMM.append("H5 DVOL 노출조절: 표 참조")


# ── 결합: 캐리 + 추세
def combo(sleeves):
    P("\n" + "=" * 78 + "\n[결합] 캐리 슬리브 + 추세 슬리브 상관·합성")
    carry = {}
    for sym in ("BTCUSDT", "ETHUSDT"):
        f = store.load("funding", sym).set_index("ts").rate
        pr = store.load("premium_1m", sym).set_index("ts").close
        prem = pr.reindex(f.index, method="nearest", tolerance=pd.Timedelta("5min"))
        carry[sym] = (f - prem.diff()).dropna().resample("1D").sum()
    tr_d = {s: v.resample("1D").sum() for s, v in sleeves.items()}
    idx = None
    for v in list(carry.values()) + list(tr_d.values()):
        idx = v.index if idx is None else idx.intersection(v.index)
    D = pd.DataFrame({f"캐리{s[:3]}": carry[s].reindex(idx) for s in carry} | {f"추세{s[:3]}": tr_d[s].reindex(idx) for s in tr_d}).fillna(0)
    P("\n일별 수익 상관")
    P(D.corr().round(2).to_string())
    te = idx >= TE
    P("\n구성별 (일 단위, 연환산)")
    for nm, w in (("추세만(BTC·ETH 균등)", {"추세BTC": .5, "추세ETH": .5}),
                  ("캐리만(균등)", {"캐리BTC": .5, "캐리ETH": .5}),
                  ("추세50+캐리50", {"추세BTC": .25, "추세ETH": .25, "캐리BTC": .25, "캐리ETH": .25}),
                  ("추세30+캐리70", {"추세BTC": .15, "추세ETH": .15, "캐리BTC": .35, "캐리ETH": .35})):
        x = sum(D[k] * v for k, v in w.items()).values
        a1, s1, d1 = stats(x[~te], 365); a2, s2, d2 = stats(x[te], 365)
        P(f"  {nm:22s} 학습 연{a1*100:6.1f}% Sh {s1:5.2f} MDD {d1*100:6.1f}% | 테스트 연{a2*100:6.1f}% Sh {s2:5.2f} MDD {d2*100:6.1f}%")
    # 연/분기/월/주/일 승률 (추세50+캐리50, 전체 기간)
    x = pd.Series(sum(D[k] * v for k, v in {"추세BTC": .25, "추세ETH": .25, "캐리BTC": .25, "캐리ETH": .25}.items()).values, index=idx)
    P("\n추세50+캐리50 기간별 승률(전체)")
    for rule, nm in (("YE", "연"), ("QE", "분기"), ("ME", "월"), ("W", "주"), ("D", "일")):
        g = x.resample(rule).sum()
        P(f"  {nm}: {float((g>0).mean()):.3f} (n={len(g)})")
    SUMM.append("결합: 캐리·추세 상관과 합성 성과는 표 참조")


if __name__ == "__main__":
    h1(); sl = h2(); h5(); combo(sl)
    P("\n[시도 장부] " + ", ".join(f"{k} {v}" for k, v in TRIES.items()))
    (R / "logs" / "h_dev2.summary").write_text("\n".join(SUMM))
