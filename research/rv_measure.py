"""주식형 엣지(횡단면·유동성공급)가 BTC·ETH에 가능한지 측정. 전략이 아니라 '수수료 전 총 엣지(bp)'를 먼저 잰다.
기준 v2 비용 게이트: 총 엣지가 메이커 왕복 4bp(수수료 0.02%×2)·테이커 왕복 12bp를 넘는지.
측정 A 프리미엄(베이시스) 평균회귀 / B BTC-ETH 베타헤지 단기반전 / C 테이커 임시충격 되돌림 / D 펀딩 정산 전후 프리미엄
사용: python research/rv_measure.py"""
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
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)


def tstat(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return float(x.mean() / x.std() * np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else 0.0


def load(sym):
    k = store.load("klines_1m", sym).set_index("ts")
    pr = store.load("premium_1m", sym).set_index("ts").close
    df = pd.DataFrame({"c": k.close, "v": k.quote_volume, "tb": k.taker_buy_quote_volume, "n": k.trades,
                       "prem": pr.reindex(k.index)}).dropna(subset=["c"])
    return df


for sym in ("BTCUSDT", "ETHUSDT"):
    d = load(sym)
    P("=" * 86 + f"\n### {sym}  1분봉 {len(d):,}개  {d.index[0].date()} ~ {d.index[-1].date()}")
    prem = d.prem * 1e4      # bp 단위
    P(f"[프리미엄 분포 bp] 평균 {prem.mean():.2f} 표준편차 {prem.std():.2f} | 절대값 중앙 {prem.abs().median():.2f} 90% {prem.abs().quantile(.9):.2f} 99% {prem.abs().quantile(.99):.2f} 최대 {prem.abs().max():.1f}")
    lp = np.log(d.c)

    # ── A. 프리미엄 평균회귀: z>k 일 때 숏(프리미엄 하락 기대), 수수료 전 총 엣지
    P("\n[A] 프리미엄 평균회귀 — 진입 후 H분 perp 수익 (부호: 신호 방향), 수수료 전 bp")
    rows = []
    for w in (60, 240, 1440):
        z = (prem - prem.rolling(w).mean()) / prem.rolling(w).std()
        for k in (2.0, 3.0):
            sig = np.where(z > k, -1, np.where(z < -k, 1, 0))
            for H in (5, 15, 60, 240):
                fwd = (lp.shift(-H) - lp).values * 1e4
                m = sig != 0
                x = sig[m] * fwd[m]
                # 중복 제거 근사: H분 간격 샘플
                idx = np.where(m)[0]
                sel = []; nxt = -1
                for i in idx:
                    if i >= nxt:
                        sel.append(i); nxt = i + H
                xs = sig[sel] * fwd[sel]
                xs = xs[~np.isnan(xs)]
                if len(xs) < 50:
                    continue
                wk = len(xs) / ((d.index[-1] - d.index[0]).days / 7)
                rows.append((w, k, H, np.mean(xs), tstat(xs), len(xs), wk))
    a = pd.DataFrame(rows, columns=["창(분)", "k", "보유(분)", "총엣지bp", "t", "거래수", "주당거래"])
    P(a.round(2).to_string(index=False))
    best = a.sort_values("총엣지bp", ascending=False).head(1)
    P(f"  → 최대 총엣지 {float(best.총엣지bp.iloc[0]):.2f}bp (메이커 왕복 4bp, 테이커 12bp)")

    # ── A2. 프리미엄 자체의 되돌림 폭 (거래 가능성 상한)
    P("\n[A2] 프리미엄이 극단일 때 이후 H분 프리미엄 변화 bp (되돌림의 이론적 상한)")
    rows = []
    for w in (240,):
        z = (prem - prem.rolling(w).mean()) / prem.rolling(w).std()
        for k in (2.0, 3.0, 4.0):
            for H in (5, 15, 60):
                dp = (prem.shift(-H) - prem).values
                m = (z.values > k)
                rows.append((k, H, "상단", np.nanmean(dp[m]), int(m.sum())))
                m2 = (z.values < -k)
                rows.append((k, H, "하단", np.nanmean(dp[m2]), int(m2.sum())))
    P(pd.DataFrame(rows, columns=["k", "H", "쪽", "프리미엄변화bp", "n"]).round(2).to_string(index=False))

    # ── C. 테이커 임시 충격 되돌림
    P("\n[C] 테이커 불균형 극단 후 되돌림 — 5분 순매수 비율 상·하위 1%, 이후 H분 (반대 방향 진입) bp")
    imb = (2 * d.tb / d.v.replace(0, np.nan) - 1)
    imb5 = imb.rolling(5).mean()
    amih = (lp.diff().abs() / d.v.replace(0, np.nan))
    hi_illq = amih > amih.rolling(1440).quantile(0.8)
    rows = []
    for q in (0.99, 0.995):
        hi = imb5 > imb5.rolling(1440).quantile(q)
        lo = imb5 < imb5.rolling(1440).quantile(1 - q)
        for H in (5, 15, 60):
            fwd = (lp.shift(-H) - lp).values * 1e4
            for nm, mask, sgn in (("전체", np.ones(len(d), bool), 1), ("저유동만", hi_illq.values, 1)):
                m1 = hi.values & mask; m2 = lo.values & mask
                x = np.concatenate([-fwd[m1], fwd[m2]])
                x = x[~np.isnan(x)]
                if len(x) < 50:
                    continue
                rows.append((q, H, nm, np.mean(x), tstat(x), len(x)))
    P(pd.DataFrame(rows, columns=["분위", "H", "조건", "총엣지bp", "t", "n"]).round(2).to_string(index=False))

    # ── D. 펀딩 정산 전후 프리미엄
    P("\n[D] 펀딩 정산(00/08/16 UTC) 전후 프리미엄 평균 bp")
    mins = (d.index.hour % 8) * 60 + d.index.minute
    g = prem.groupby(np.where(mins > 240, mins - 480, mins)).mean()
    sel = [-60, -30, -15, -5, -1, 0, 1, 5, 15, 30, 60, 120]
    P("  " + " | ".join(f"{s:+d}분 {g.get(s, np.nan):.2f}" for s in sel))
    # 정산 직전 프리미엄 극단 → 정산 후 되돌림
    pre = (mins >= 420) | (mins <= 0)
    P(f"  정산 전 60분 프리미엄 평균 {prem[(mins>=420)].mean():.2f}bp vs 그 외 {prem[~(mins>=420)].mean():.2f}bp")

# ── B. BTC-ETH 베타헤지 단기 반전 (횡단면 대용)
P("\n" + "=" * 86 + "\n[B] BTC-ETH 베타헤지 단기 반전 (주식 단기반전 팩터의 2자산 대용), 수수료 전 bp")
b = load("BTCUSDT"); e = load("ETHUSDT")
idx = b.index.intersection(e.index)
rb = np.log(b.c.reindex(idx)).diff(); re_ = np.log(e.c.reindex(idx)).diff()
rows = []
for W in (5, 15, 60, 240):
    beta = re_.rolling(1440).cov(rb) / rb.rolling(1440).var()
    sp = (re_ - beta * rb)                      # ETH의 BTC 초과 수익
    cum = sp.rolling(W).sum()
    zz = (cum - cum.rolling(1440).mean()) / cum.rolling(1440).std()
    for k in (1.5, 2.5):
        for H in (5, 15, 60, 240):
            fwd = sp.shift(-1).rolling(H).sum().shift(-(H - 1)).values * 1e4
            sig = np.where(zz.values > k, -1, np.where(zz.values < -k, 1, 0))
            m = sig != 0
            ii = np.where(m)[0]
            sel = []; nxt = -1
            for i in ii:
                if i >= nxt:
                    sel.append(i); nxt = i + H
            x = sig[sel] * fwd[sel]
            x = x[~np.isnan(x)]
            if len(x) < 50:
                continue
            wk = len(x) / ((idx[-1] - idx[0]).days / 7)
            rows.append((W, k, H, np.mean(x), tstat(x), len(x), wk))
P(pd.DataFrame(rows, columns=["형성(분)", "k", "보유(분)", "총엣지bp", "t", "거래수", "주당거래"]).round(2).to_string(index=False))
P("\n※ 두 다리 거래이므로 비용 2배(메이커 왕복 8bp, 테이커 24bp)")
