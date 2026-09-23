"""R27-1 — 얕은 딥 체결 이벤트 + 체결 직전 차트 특징 테이블.

목적: 하루 여러 건 빈도(봉 5~60분 × 깊이 rv×3~6)의 딥 체결을 모두 모으고,
      체결 직전 분(f-1)까지의 정보만으로 만든 특징을 붙인다.
정책 해석(룩어헤드 없음): 지정가는 '직전 분까지의 조건'이 참일 때만 살아 있다.
      봉 안에서 첫 터치 분 f의 조건이 거짓이면 그 봉은 건너뛴다(이미 터치됐는지는 f-1에 관측 가능).
      따라서 특징은 전부 f-1 종료 시점까지의 값이다.
출력: data/r27_events.parquet
사용: python research/r27_dip_events.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
from r26_hf_twosided import run, EPS, FM, FT
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
SYM = ("BTCUSDT", "ETHUSDT")
BARS = (5, 10, 15, 30, 60)
KS = (3.0, 4.0, 5.0, 6.0, 8.0)


def feats(s):
    k = store.load("klines_1m", s).set_index("ts")
    idx = k.index
    o, h, l, c = (k[x].values for x in ("open", "high", "low", "close"))
    qv = k.quote_volume.values; tb = k.taker_buy_quote_volume.values; tr = k.trades.values
    lr = pd.Series(np.log(c)).diff()
    rv1 = lr.rolling(300).std().shift(1).values                # 봉 시가 시점 rv (지정가 깊이용)
    rvn = lr.rolling(300).std().values                         # f-1 종료 시점 rv (특징용, 인덱스 f-1로 읽음)
    F = {}
    lc = pd.Series(np.log(c))
    for w in (5, 15, 60, 240, 1440):
        F[f"ret{w}"] = (lc - lc.shift(w)).values / (rvn * np.sqrt(w))
    qs = pd.Series(qv); ts_ = pd.Series(tb); trs = pd.Series(tr)
    base = qs.rolling(1440).mean()
    F["vol5"] = (qs.rolling(5).mean() / base).values
    F["vol60"] = (qs.rolling(60).mean() / base).values
    F["trd5"] = (trs.rolling(5).mean() / trs.rolling(1440).mean()).values
    F["sell5"] = (1 - ts_.rolling(5).sum() / qs.rolling(5).sum()).values
    F["sell60"] = (1 - ts_.rolling(60).sum() / qs.rolling(60).sum()).values
    F["avgsz5"] = ((qs.rolling(5).sum() / trs.rolling(5).sum()) / (qs.rolling(1440).sum() / trs.rolling(1440).sum())).values
    for w in (60, 240, 1440):
        lo = pd.Series(l).rolling(w).min(); hi = pd.Series(h).rolling(w).max()
        F[f"pos{w}"] = ((pd.Series(c) - lo) / (hi - lo)).values
        F[f"dlo{w}"] = (np.log(pd.Series(c) / lo)).values / (rvn * np.sqrt(w))
    rvs = pd.Series(rvn)
    F["rvreg"] = (rvs / rvs.rolling(10080).mean()).values
    F["rvshort"] = (lr.rolling(15).std() / rvs).values
    F["wick5"] = ((pd.Series(c) - pd.Series(l).rolling(5).min()) /
                  (pd.Series(h).rolling(5).max() - pd.Series(l).rolling(5).min())).values
    pr = store.load("premium_1m", s).set_index("ts").close.reindex(idx).ffill()
    F["prem"] = pr.values
    F["premz"] = ((pr - pr.rolling(1440).mean()) / pr.rolling(1440).std()).values
    m = store.load("metrics_5m", s).set_index("ts")
    m.index = m.index + pd.Timedelta(minutes=5)                # 5분 지표는 구간 종료 후 사용
    oi = m.sum_open_interest.reindex(idx, method="ffill")
    F["oi60"] = np.log(oi / oi.shift(60)).values
    F["oi1440"] = np.log(oi / oi.shift(1440)).values
    F["tlsr"] = m.sum_toptrader_long_short_ratio.reindex(idx, method="ffill").values
    F["lsr"] = m.count_long_short_ratio.reindex(idx, method="ffill").values
    fu = store.load("funding", s).set_index("ts").rate
    F["fund"] = fu.reindex(idx, method="ffill").values
    F["hour"] = idx.hour.values.astype(float)
    F["dow"] = idx.dayofweek.values.astype(float)
    return dict(o=o, h=h, l=l, c=c, rv1=rv1, idx=idx), F


rows = []
for s in SYM:
    d, F = feats(s)
    P(f"{s} 특징 {len(F)}개 준비")
    for B in BARS:
        for kx in KS:
            a, b, r, m = run(d["o"], d["h"], d["l"], d["c"], d["rv1"], B, 1, kx, 0, 1.0, 0, EPS, FM, FT)
            if len(a) == 0: continue
            fm1 = a - 1
            df = pd.DataFrame({"sym": s, "B": B, "k": kx, "f": a, "x": b, "ret": r, "mae": m,
                               "ts": d["idx"][a], "el": (a % B) / B,
                               "pre": (d["c"][fm1] / d["o"][(a // B) * B] - 1) / (kx * d["rv1"][(a // B) * B] * np.sqrt(B))})
            for kk, v in F.items():
                df[kk] = v[fm1]
            rows.append(df)
            P(f"  {s} B{B} k{kx}: {len(df)}")
E = pd.concat(rows, ignore_index=True)
for c_ in E.columns:
    if E[c_].dtype == np.float64: E[c_] = E[c_].astype(np.float32)
out = R / "data" / "r27_events.parquet"
E.to_parquet(out, index=False)
P(f"저장 {out} {E.shape}")
