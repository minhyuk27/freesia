"""프리미엄(베이시스) 평균회귀 전략 정식 검증 — 기준 v2.
학습 2020~2023 / 테스트 2024~. 메이커(왕복 4bp)와 테이커(왕복 12bp) 둘 다. 위약 300회. 연도별. 빈도.
사용: python research/rv_strat.py"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from freesia import store  # noqa: E402

TE = pd.Timestamp("2024-01-01", tz="UTC")
rng = np.random.default_rng(17)
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)


def tstat(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return float(x.mean() / x.std() * np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else 0.0


def trades(prem, lp, w, k, H, exit_zero=True):
    """겹치지 않는 거래만. 반환: (진입 인덱스, 부호, 수수료 전 bp)"""
    z = ((prem - prem.rolling(w).mean()) / prem.rolling(w).std()).values
    pv = prem.values; lv = lp.values
    n = len(z); out_i = []; out_s = []; out_r = []
    i = 0
    while i < n - 1:
        if np.isnan(z[i]) or abs(z[i]) < k:
            i += 1; continue
        s = -1 if z[i] > 0 else 1
        j = min(i + H, n - 1)
        if exit_zero:
            for t in range(i + 1, min(i + H, n - 1) + 1):
                if (s == -1 and z[t] <= 0) or (s == 1 and z[t] >= 0):
                    j = t; break
        out_i.append(i); out_s.append(s); out_r.append(s * (lv[j] - lv[i]) * 1e4)
        i = j + 1
    return np.array(out_i), np.array(out_s), np.array(out_r)


for sym in ("BTCUSDT", "ETHUSDT"):
    k_ = store.load("klines_1m", sym).set_index("ts")
    pr = store.load("premium_1m", sym).set_index("ts").close
    df = pd.DataFrame({"c": k_.close, "prem": pr.reindex(k_.index) * 1e4}).dropna()
    lp = np.log(df.c)
    weeks_all = (df.index[-1] - df.index[0]).days / 7
    P("=" * 92 + f"\n### {sym}")
    P(f"{'창':>5} {'k':>4} {'H':>5} | {'학습 총bp':>9} {'t':>6} {'거래':>7} {'주당':>6} | {'테스트 총bp':>10} {'t':>6} {'거래':>7} {'주당':>6} | {'메이커순':>8} {'테이커순':>8}")
    best = None
    for w in (240, 1440):
        for k in (2.0, 2.5, 3.0):
            for H in (60, 240, 720):
                i_, s_, r_ = trades(df.prem, lp, w, k, H)
                if len(i_) < 100:
                    continue
                ts = df.index[i_]
                tr = ts < TE; te = ~tr
                if te.sum() < 50 or tr.sum() < 50:
                    continue
                wk_tr = tr.sum() / ((TE - df.index[0]).days / 7)
                wk_te = te.sum() / ((df.index[-1] - TE).days / 7)
                mk = r_[te].mean() - 4; tk = r_[te].mean() - 12
                P(f"{w:>5} {k:>4} {H:>5} | {r_[tr].mean():9.2f} {tstat(r_[tr]):6.2f} {tr.sum():7d} {wk_tr:6.1f} | "
                  f"{r_[te].mean():10.2f} {tstat(r_[te]):6.2f} {te.sum():7d} {wk_te:6.1f} | {mk:8.2f} {tk:8.2f}")
                if best is None or r_[tr].mean() > best[0]:
                    best = (r_[tr].mean(), w, k, H)
    if best is None:
        continue
    _, w, k, H = best
    P(f"\n[학습 기준 최적] 창 {w} k {k} H {H}")
    i_, s_, r_ = trades(df.prem, lp, w, k, H)
    ts = df.index[i_]
    # 연도별
    rows = []
    for y in range(2020, 2027):
        m = ts.year == y
        if m.sum() < 10:
            continue
        rows.append((y, r_[m].mean(), r_[m].mean() - 4, tstat(r_[m]), int(m.sum()), m.sum() / 52,
                     float((r_[m] > 4).mean())))
    P(pd.DataFrame(rows, columns=["연", "총bp", "메이커순bp", "t", "거래", "주당", "메이커승률"]).round(2).to_string(index=False))
    # 방향별
    for nm, sv in (("숏(프리미엄 높음)", -1), ("롱(프리미엄 낮음)", 1)):
        m = (s_ == sv) & (ts >= TE)
        if m.sum() > 20:
            P(f"  테스트 {nm}: 총 {r_[m].mean():.2f}bp, 메이커순 {r_[m].mean()-4:.2f}bp, n={m.sum()}")
    # 위약: 프리미엄 시계열 위상 회전
    te_mask = ts >= TE
    real = r_[te_mask].mean()
    pl = []
    for _ in range(300):
        sh = rng.integers(10000, len(df) - 10000)
        pp = pd.Series(np.roll(df.prem.values, sh), index=df.index)
        ii, ss, rr = trades(pp, lp, w, k, H)
        if len(ii) < 50:
            continue
        tt = df.index[ii] >= TE
        if tt.sum() > 20:
            pl.append(rr[tt].mean())
    pl = np.array(pl)
    P(f"  위약 {len(pl)}회 테스트 총bp: 평균 {pl.mean():.2f}, 95% {np.percentile(pl,95):.2f}, 99% {np.percentile(pl,99):.2f}, 최대 {pl.max():.2f} | 실제 {real:.2f} → 백분위 {(pl<real).mean()*100:.1f}%")
    # 누적 (메이커 가정, 거래당 자본의 x% 사용 가정 없이 bp 누적)
    cum_tr = r_[ts < TE] - 4; cum_te = r_[te_mask] - 4
    P(f"  메이커 누적: 학습 {cum_tr.sum()/100:.1f}%p({len(cum_tr)}거래)  테스트 {cum_te.sum()/100:.1f}%p({len(cum_te)}거래) — 명목 1배 기준")
