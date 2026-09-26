"""R67 준비 — R66의 모든 선·규칙 거래를 '거래 한 건 = 한 행'으로 저장하고 조건 특징을 붙인다 (분기 무작위 분할 패턴 분석용).

R66 엔진(선 생성·사건·주문 시뮬레이션)을 그대로 쓰고, 합계 대신 거래 기록을 남긴다. 위약 선 거래도 같은 형식으로 저장.
특징 (접근봉 j의 직전 봉 j−1 종가까지, 격자 기준)
  hour, wday (봉 j 시작 UTC), volp (ATR의 직전 1000봉 백분위), trend ((종가−SMA200)/ATR, 역할 방향 부호), mom20 (20봉 수익/ATR, 역할 방향),
  vrel (5봉 거래대금 / 500봉 중앙값, 로그), tbr (5봉 테이커 매수 비율 − 0.5, 역할 방향), lslope (선의 5봉 변화/ATR, 역할 방향),
  dist (직전 종가와 선의 거리 / ATR)
  '역할 방향' = 지지(ori=+1)면 그대로, 저항(−1)이면 부호 반전 → "선 쪽으로 오는 힘"이 같은 부호가 되게.
출력: data/r67_trades_<격자>.parquet
사용: python research/r67_build.py <격자>"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
from r66_lines_all import (scan_series, sim, grid, lines_period, lines_indicator, lines_swing, options_levels, DLT, SYMS, P)


def load_1m(sym):
    k = store.load("klines_1m", sym).set_index("ts")
    return pd.DataFrame({"o": k.open, "h": k.high, "l": k.low, "c": k.close, "v": k.quote_volume, "tb": k.taker_buy_quote_volume})


def feats(g):
    a = g.atr
    F = pd.DataFrame(index=g.index)
    F["volp"] = a.rolling(1000, min_periods=200).rank(pct=True)
    sma = g.c.rolling(200).mean().shift(1); pc = g.c.shift(1)
    F["trend"] = np.log(pc / sma) / a
    F["mom20"] = (g.lc.shift(1) - g.lc.shift(21)) / a
    v5 = g.v.rolling(5).sum().shift(1); F["vrel"] = np.log(v5 / v5.rolling(500, min_periods=100).median())
    F["tbr"] = (g.tb.rolling(5).sum() / g.v.rolling(5).sum()).shift(1) - 0.5
    F["hour"] = g.index.hour; F["wday"] = g.index.weekday
    return F


def run_series_rec(name, Lp, g, F, base, rng, step):
    lh, ll, lc, atr = g.lh.values, g.ll.values, g.lc.values, g.atr.values
    dl = DLT * atr; B = atr
    L = np.log(np.where(Lp > 0, Lp, np.nan))
    day = g.index.floor("D"); ud = pd.Series(rng.uniform(0.5, 1.5, len(day.unique())) * rng.choice([-1, 1], len(day.unique())), index=day.unique())
    Lpl = L + ud.reindex(day).values * atr
    blo, blh, bll, blc = (np.log(base[k].values) for k in ("o", "h", "l", "c"))
    t0_all = base.index.searchsorted(g.index); out = []
    lcp = np.r_[np.nan, lc[:-1]]
    for plc, LL in ((False, L), (True, Lpl)):
        for ori in (1, -1):
            ev = scan_series(LL, lh, ll, lc, dl, B) if ori == 1 else scan_series(-LL, -ll, -lh, -lc, dl, B)
            if len(ev) == 0: continue
            j = ev[:, 0].astype(int); yL = ev[:, 3]; A = atr[j]; d = dl[j]
            t0 = t0_all[j]; wait = np.full(len(j), step); hold = 12 * wait
            Lraw = ori * yL
            fj = F.iloc[j]
            base_cols = {"s": name, "plc": plc, "ori": ori, "t": g.index[j],
                         "volp": fj.volp.values, "trend": ori * fj.trend.values, "mom20": -ori * fj.mom20.values,
                         "vrel": fj.vrel.values, "tbr": -ori * fj.tbr.values, "hour": fj.hour.values, "wday": fj.wday.values,
                         "lslope": ori * (LL[j] - LL[np.maximum(j - 5, 0)]) / A * (1 if ori == 1 else 1),
                         "dist": ori * (lcp[j] - Lraw) / A}
            for rule in ("E1", "E3", "E2"):
                if rule == "E1": o = sim(t0, wait, hold, np.full(len(j), ori), yL, yL - 0.5 * A, yL + 0.5 * A, False, blo, blh, bll, blc)
                elif rule == "E3": o = sim(t0, wait, hold, np.full(len(j), ori), yL - 0.5 * A, yL - 1.25 * A, yL, False, blo, blh, bll, blc)
                else: o = sim(t0, wait, hold, np.full(len(j), -ori), -yL + d, -yL - 0.25 * A, -yL + A, True, blo, blh, bll, blc)
                f = o[:, 0] == 1
                if f.sum() == 0: continue
                df = pd.DataFrame({k: (v[f] if isinstance(v, np.ndarray) or isinstance(v, pd.Index) else v) for k, v in base_cols.items()})
                df["rule"] = rule; df["g"] = o[f, 1].astype(np.float32); df["n"] = o[f, 2].astype(np.float32)
                out.append(df)
    return out


if __name__ == "__main__":
    tf = sys.argv[1]; step = int(pd.Timedelta(tf) / pd.Timedelta("1min")); rng = np.random.default_rng(0); t_ = time.time(); allr = []
    for sym in SYMS:
        b = load_1m(sym)
        g = b.resample(tf).agg({"o": "first", "h": "max", "l": "min", "c": "last", "v": "sum", "tb": "sum"}).dropna()
        g2 = grid(b[["o", "h", "l", "c", "v"]], tf); g = g2.join(g[["tb"]])
        F = feats(g); opt = options_levels(sym)
        S = {}; S.update(lines_period(b, g, sym, opt)); S.update(lines_indicator(g)); S.update(lines_swing(g))
        for nm, arr in S.items():
            arr = np.asarray(arr, dtype=float)
            if np.isfinite(arr).sum() < 1000: continue
            for df in run_series_rec(nm, arr, g, F, b, rng, step): allr.append(df.assign(sym=sym))
        P(f"{sym} {tf} 완료 ({time.time()-t_:.0f}s)")
    A = pd.concat(allr, ignore_index=True)
    for c in ("volp", "trend", "mom20", "vrel", "tbr", "lslope", "dist"): A[c] = A[c].astype(np.float32)
    for c in ("s", "rule", "sym"): A[c] = A[c].astype("category")
    A["tf"] = tf
    A.to_parquet(R / "data" / f"r67_trades_{tf}.parquet", index=False)
    P(f"{tf}: 거래 {len(A):,} (실제 {(~A.plc).sum():,} / 위약 {A.plc.sum():,})")
