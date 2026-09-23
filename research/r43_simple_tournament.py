"""R43 — 단순 규칙 토너먼트: '단순하지만 공격적인' 전략을 엄격한 절차로 찾는다.

절차(착시 방지):
  - 규칙은 파라미터 1~3개. 파라미터는 2020~2022(학습)에서만 고르고, 2023-01~(평가)에서 고정 평가.
  - 비용: 진입·청산 각 테이커 5bp + 슬리피지 2bp. 레버리지 L: 거래 중 1시간 봉 최악 역행 × L ≥ 1 이면 그 종목 자본 청산.
  - 두 종목 자본 50:50 분리 운용(재조정 없음).
  - 시도한 구성 수를 함께 보고(다중검정).
가족:
  F1 변동성 돌파(Larry Williams)  F2 월요일 아시아 개장  F3 시간대 창  F4 월말월초
  F5 금요일 옵션 만기             F6 N일 신고가 돌파    F7 큰 하락일 반전  F8 주말  F9 이동평균 추세(기준선)
사용: python research/r43_simple_tournament.py"""
import sys, warnings, itertools
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R))
from freesia import store, costs
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
C = costs.TAKER + 2e-4
SPLIT = pd.Timestamp("2023-01-01", tz="UTC")
HB = {}; DB = {}
for s in ("BTCUSDT", "ETHUSDT"):
    k = store.load("klines_1m", s).set_index("ts")
    HB[s] = k.resample("1h").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    DB[s] = k.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()


def trade(h, t_in, p_in, t_out, p_out, side):
    """1시간 봉에서 보유 구간 최악 역행."""
    w = h.loc[t_in:t_out - pd.Timedelta(hours=1)] if t_out > t_in else h.loc[t_in:t_in]
    adv = (1 - w.low.min() / p_in) if side > 0 else (w.high.max() / p_in - 1)
    ret = side * (p_out / p_in - 1) - 2 * C
    return dict(t_in=t_in, t_out=t_out, ret=ret, mae=max(adv, 0.0), side=side)


def by_windows(h, starts, hours, side=1):
    out = []
    for t in starts:
        e = t + pd.Timedelta(hours=hours)
        if t not in h.index or e not in h.index: continue
        out.append(trade(h, t, h.open[t], e, h.open[e], side))
    return out


def fam(name, s, p):
    h, d = HB[s], DB[s]
    if name == "F1":                         # 변동성 돌파: k, MA 필터 n(0=없음), 방향
        k_, n_, both = p
        rng = (d.high - d.low).shift(1); ma = d.close.rolling(n_).mean().shift(1) if n_ else None
        out = []
        for day in d.index[1:-1]:
            O = d.open[day]; lv = O + k_ * rng[day]; ls = O - k_ * rng[day]
            if np.isnan(lv): continue
            hh = h.loc[day: day + pd.Timedelta(hours=23)]
            nxt = day + pd.Timedelta(days=1)
            if nxt not in h.index: continue
            up = hh.index[hh.high >= lv]; dn = hh.index[hh.low <= ls]
            okL = (not n_) or d.close.shift(1)[day] > ma[day]
            okS = both and ((not n_) or d.close.shift(1)[day] < ma[day])
            first = None
            if len(up) and okL: first = (up[0], 1, lv)
            if len(dn) and okS and (first is None or dn[0] < first[0]): first = (dn[0], -1, ls)
            if first is None: continue
            t, sd, px = first
            if (sd > 0 and h.open[t] > px) or (sd < 0 and h.open[t] < px): px = h.open[t]   # 갭이면 시가
            out.append(trade(h, t, px, nxt, h.open[nxt], sd))
        return out
    if name == "F2":                         # 일요일 23시 UTC 롱 H시간
        (H,) = p
        st = h.index[(h.index.dayofweek == 6) & (h.index.hour == 23)]
        return by_windows(h, st, H)
    if name == "F3":                         # 매일 a시부터 w시간, 방향
        a, w, sd = p
        return by_windows(h, h.index[h.index.hour == a], w, sd)
    if name == "F4":                         # 월말 n일 전 0시 ~ 월초 m일 0시
        n_, m_ = p
        me = d.index[d.index.is_month_end]
        st = [x - pd.Timedelta(days=n_ - 1) for x in me]
        return by_windows(h, st, 24 * (n_ + m_))
    if name == "F5":                         # 금요일 만기 08:00 UTC 기준 전/후 w시간, 방향
        pre, w, sd = p
        f8 = h.index[(h.index.dayofweek == 4) & (h.index.hour == 8)]
        st = [x - pd.Timedelta(hours=w) for x in f8] if pre else list(f8)
        return by_windows(h, st, w, sd)
    if name == "F6":                         # 일봉 종가가 직전 N일 최고 → 다음날 시가 롱 H일
        N_, H_ = p
        sig = d.close > d.close.rolling(N_).max().shift(1)
        out = []; last = None
        for day in d.index[sig.values]:
            t = day + pd.Timedelta(days=1)
            if last is not None and t < last: continue
            e = t + pd.Timedelta(days=H_)
            if t not in h.index or e not in h.index: continue
            out.append(trade(h, t, h.open[t], e, h.open[e], 1)); last = e
        return out
    if name == "F7":                         # 일간 수익 < −k·σ30 → 다음날 시가 롱 H일
        k_, H_ = p
        r = np.log(d.close).diff(); sg = r.rolling(30).std().shift(1)
        out = []; last = None
        for day in d.index[(r < -k_ * sg).values]:
            t = day + pd.Timedelta(days=1)
            if last is not None and t < last: continue
            e = t + pd.Timedelta(days=H_)
            if t not in h.index or e not in h.index: continue
            out.append(trade(h, t, h.open[t], e, h.open[e], 1)); last = e
        return out
    if name == "F8":                         # 금 21시 ~ 월 0시, 방향
        (sd,) = p
        st = h.index[(h.index.dayofweek == 4) & (h.index.hour == 21)]
        return by_windows(h, st, 51, sd)
    if name == "F9":                         # 일봉 종가 > MA(n) 이면 다음날 보유
        (n_,) = p
        on = (d.close > d.close.rolling(n_).mean()).shift(1).fillna(False)
        out = []; i = 0; idx = d.index
        while i < len(idx):
            if on.iloc[i]:
                j = i
                while j < len(idx) and on.iloc[j]: j += 1
                t, e = idx[i], idx[min(j, len(idx) - 1)]
                if t in h.index and e in h.index and e > t:
                    out.append(trade(h, t, h.open[t], e, h.open[e], 1))
                i = j
            else: i += 1
        return out


GRID = {
    "F1": list(itertools.product((0.3, 0.4, 0.5, 0.6, 0.7), (0, 5, 20), (False, True))),
    "F2": [(H,) for H in (6, 12, 24, 48)],
    "F3": list(itertools.product(range(0, 24, 2), (2, 4, 8), (1, -1))),
    "F4": list(itertools.product((1, 2, 3), (1, 2, 3))),
    "F5": list(itertools.product((True, False), (4, 8, 12), (1, -1))),
    "F6": list(itertools.product((20, 55, 90), (1, 3, 7, 14))),
    "F7": list(itertools.product((1.5, 2.0, 3.0), (1, 2, 3))),
    "F8": [(1,), (-1,)],
    "F9": [(n,) for n in (20, 50, 100, 200)],
}


def equity(T, L, t0, t1):
    """두 종목 각 50% 자본. 거래 순차 복리, 역행×L ≥ 1 이면 그 종목 자본 0."""
    curves = []
    for s in ("BTCUSDT", "ETHUSDT"):
        x = T[(T.sym == s) & (T.t_in >= t0) & (T.t_in < t1)].sort_values("t_in")
        cap = 0.5; pts = {t0: 0.5}
        for _, r in x.iterrows():
            if cap <= 0: break
            cap = 0.0 if r.mae * L >= 1 else cap * max(1 + L * r.ret, 0.0)
            pts[r.t_out] = cap
        curves.append(pd.Series(pts).sort_index())
    idx = curves[0].index.union(curves[1].index)
    eq = sum(c.reindex(idx).ffill().fillna(0.5) for c in curves)
    return eq


def stats(eq):
    eq = eq.resample("D").last().ffill()
    yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 0.1)
    cg = eq.iloc[-1] ** (1 / yrs) - 1 if eq.iloc[-1] > 0 else -1
    mon = (eq.resample("ME").last().pct_change().dropna()) * 100
    mdd = float((eq / eq.cummax() - 1).min()) * 100
    return cg * 100, ((1 + cg) ** (1 / 12) - 1) * 100 if cg > -1 else -100, mon.median(), mdd, eq


if __name__ == "__main__":
    ntest = sum(len(v) for v in GRID.values())
    P(f"총 구성 {ntest}개 (다중검정 주의)")
    T0 = pd.Timestamp("2020-02-01", tz="UTC"); TE = pd.Timestamp("2026-09-23", tz="UTC")
    best = []
    for f, grid in GRID.items():
        rows = []
        for p in grid:
            T = pd.DataFrame([dict(sym=s, **t) for s in ("BTCUSDT", "ETHUSDT") for t in fam(f, s, p)])
            if T.empty: continue
            cg_is = stats(equity(T, 1, T0, SPLIT))[0]
            rows.append((cg_is, p, T))
        rows.sort(key=lambda z: z[0], reverse=True)
        cg_is, p, T = rows[0]
        n_is = ((T.t_in < SPLIT)).sum() / 2.9; n_os = (T.t_in >= SPLIT).sum() / 3.7
        P(f"\n[{f}] 학습 최상 {p} (학습 1배 연 {cg_is:+.0f}%) | 거래 학습 {n_is:.0f}/년 평가 {n_os:.0f}/년, "
          f"평가 거래당 {T[T.t_in>=SPLIT].ret.mean()*1e4:+.1f}bp 승률 {(T[T.t_in>=SPLIT].ret>0).mean():.2f}")
        for L in (1, 3, 5):
            cg, mo, mmed, mdd, eq = stats(equity(T, L, SPLIT, TE))
            yr = eq.resample("YE").last() / eq.resample("YE").last().shift(1).fillna(1.0) - 1
            P(f"   평가 {L}배: 연 {cg:+7.1f}% 월 {mo:+5.1f}% 월중앙 {mmed:+5.1f}% MDD {mdd:6.1f}% | " +
              " ".join(f"{i.year%100}{v*100:+.0f}" for i, v in yr.items()))
        best.append((f, p, T))
    pd.concat([T.assign(fam=f, par=str(p)) for f, p, T in best]).to_parquet(R / "data" / "r43_best.parquet")
