"""대규모 특징 집합 — 차트움직임·변동성·체결강도·수급을 다양한 봉 단위로 전부.
전부 인과적(t 시점까지). 평가는 정확도가 아니라 순손익(bp) 기준.
사용: python research/bigfeat.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
SYM = ("BTCUSDT", "ETHUSDT")
WINS = (3, 5, 15, 30, 60, 240, 1440)


def build_features(s):
    k = store.load("klines_1m", s).set_index("ts")
    o, h, l, c = k.open, k.high, k.low, k.close
    qv, tb, nt = k.quote_volume, k.taker_buy_quote_volume, k.trades
    F = {}
    lr = np.log(c).diff()

    # ── 차트 움직임 ──
    for w in WINS:
        F[f"ret{w}"] = c.pct_change(w)
        hh = h.rolling(w).max(); ll = l.rolling(w).min()
        rng = (hh-ll)/c
        F[f"range{w}"] = rng
        F[f"pos{w}"] = ((c-ll)/(hh-ll).replace(0, np.nan))          # 고저 구간 내 위치
        F[f"body{w}"] = (c - o.shift(w-1))/c                         # 몸통
        F[f"upwick{w}"] = (hh - np.maximum(c, o.shift(w-1)))/c       # 위꼬리
        F[f"dnwick{w}"] = (np.minimum(c, o.shift(w-1)) - ll)/c       # 아래꼬리
        F[f"bodyrange{w}"] = F[f"body{w}"].abs()/rng.replace(0, np.nan)
    for w in (5, 15, 60):
        up = (c.diff() > 0).astype(float)
        F[f"upfrac{w}"] = up.rolling(w).mean()                        # 상승 분봉 비율
        F[f"streak{w}"] = np.sign(c.diff()).rolling(w).sum()          # 연속성

    # ── 변동성 ──
    for w in WINS:
        F[f"rv{w}"] = lr.rolling(w).std()
        # 파킨슨 (고저 기반)
        F[f"park{w}"] = np.sqrt((np.log(h/l)**2).rolling(w).mean()/(4*np.log(2)))
        # ATR 유사
        tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
        F[f"atr{w}"] = tr.rolling(w).mean()/c
    for a, b in ((5, 60), (15, 240), (60, 1440)):
        F[f"rvratio{a}_{b}"] = lr.rolling(a).std()/lr.rolling(b).std()
        F[f"parkratio{a}_{b}"] = (np.sqrt((np.log(h/l)**2).rolling(a).mean())/
                                  np.sqrt((np.log(h/l)**2).rolling(b).mean()))
    # 변동성의 변동성
    F["volofvol"] = lr.rolling(60).std().rolling(240).std()
    # 점프 비중
    F["jump60"] = (lr.abs() > 3*lr.rolling(1440).std()).rolling(60).sum()

    # ── 체결 강도 ──
    for w in WINS:
        F[f"qv{w}"] = qv.rolling(w).sum()/qv.rolling(1440).mean()/w
        F[f"nt{w}"] = nt.rolling(w).sum()/nt.rolling(1440).mean()/w
        F[f"ofi{w}"] = (2*tb.rolling(w).sum()/qv.rolling(w).sum().replace(0, np.nan) - 1.0)
        F[f"avgsz{w}"] = (qv.rolling(w).sum()/nt.rolling(w).sum().replace(0, np.nan))
        F[f"avgsz_z{w}"] = F[f"avgsz{w}"]/(qv/nt.replace(0, np.nan)).rolling(1440).mean()
    for w in (15, 60, 240):
        F[f"amihud{w}"] = (lr.abs().rolling(w).mean()/(qv.rolling(w).mean()/1e6))
        F[f"kyle{w}"] = (c.pct_change(w).abs()/np.sqrt(qv.rolling(w).sum()/1e6))
    # 효율비 (추세 강도)
    for w in (15, 60, 240):
        F[f"er{w}"] = c.diff(w).abs()/c.diff().abs().rolling(w).sum().replace(0, np.nan)

    # ── 수급 (프리미엄·펀딩·OI) ──
    pr = store.load("premium_1m", s).set_index("ts").close.reindex(c.index)
    F["prem"] = pr
    for w in (15, 60, 240):
        F[f"prem_ma{w}"] = pr.rolling(w).mean()
        F[f"prem_chg{w}"] = pr.diff(w)
    fr = store.load("funding", s).set_index("ts").rate.reindex(c.index, method="ffill", limit=600)
    F["fund"] = fr
    F["fund_ma"] = fr.rolling(1440).mean()
    m5 = store.load("metrics_5m", s).set_index("ts").reindex(c.index, method="ffill", limit=10)
    if m5 is not None and len(m5):
        oi = m5.sum_open_interest
        F["oi_chg30"] = oi.pct_change(30)
        F["oi_chg240"] = oi.pct_change(240)
        F["tt_ls"] = m5.sum_toptrader_long_short_ratio
        F["taker_ls"] = m5.sum_taker_long_short_vol_ratio
        F["cnt_ls"] = m5.count_long_short_ratio

    # ── 시간 ──
    F["hour_sin"] = np.sin(2*np.pi*c.index.hour/24)
    F["hour_cos"] = np.cos(2*np.pi*c.index.hour/24)
    F["dow"] = c.index.dayofweek.astype(float)

    X = pd.DataFrame(F, index=c.index)
    X = X.replace([np.inf, -np.inf], np.nan)
    return X, c


if __name__ == "__main__":
    for s in SYM:
        X, c = build_features(s)
        P(f"{s}: 특징 {X.shape[1]}개, 행 {len(X):,}")
        P(f"  결측 많은 특징: " + ", ".join(X.columns[X.isna().mean() > 0.1][:10]))
