"""R36 — z0.31 실행 현실성을 1초 체결 데이터로 점검 (2023~).

[1] 체결 확실성: 선택된 딥 거래마다 체결 봉 안에서 지정가 T 아래로 실제 체결된 대금
    - 관통 체결대금(T×(1−2bp) 이하에서 거래된 테이커 매도 대금)이 우리 주문 크기보다 충분히 큰가
    - T 이하 거래가 몇 초 동안 이어졌나 (한 번 찍고 올라간 틱인지)
[2] 청산 지연 비용: 봉 종가에 시장가를 낸다고 할 때 실제 체결은 1~3초 뒤 → 종가 대비 1·2·3·5초 뒤 가격 차(bp)
사용: python research/r36_fill_check.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R))
from freesia import store
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
W = pd.read_parquet(R / "data" / "r32_wf.parquet")
W["ts"] = pd.DatetimeIndex(W.ts)
W = W[(W.pred > 0.001) & (W.ts >= "2023-01-01")].copy()
K = {s: store.load("klines_1m", s)[["ts", "open"]].reset_index(drop=True) for s in ("BTCUSDT", "ETHUSDT")}
rows = []
for s in ("BTCUSDT", "ETHUSDT"):
    S = W[W.sym == s]
    kts = K[s].ts
    for ym, g in S.groupby(S.ts.dt.strftime("%Y-%m")):
        f = store.PQ / "agg1s" / s / f"{ym}.parquet"
        if not f.exists(): continue
        d = pd.read_parquet(f).set_index("ts")
        for _, e in g.iterrows():
            B = int(e.B)
            fmin = kts[int(e.f)]; xend = kts[int(e.x)] + pd.Timedelta(minutes=1)
            # 지정가 T 복원: 체결 분 f 기준 ret = c_exit/T − 1 − 7bp → T = c_exit/(1+ret+7bp)
            cx = d.c.asof(xend - pd.Timedelta(seconds=1))
            T = cx / (1 + e.ret + 7e-4)
            w = d.loc[fmin: xend - pd.Timedelta(seconds=1)]
            below = w[w.l <= T * (1 - 2e-4)]
            vol_below = float(below.sq.sum())             # 관통 구간 초들의 테이커 매도 대금(상한 근사)
            nsec = len(below)
            post = [d.c.asof(xend + pd.Timedelta(seconds=k)) for k in (1, 2, 3, 5)]
            rows.append(dict(sym=s, ts=e.ts, B=B, k=e.k, T=T, nsec=nsec, vol_below=vol_below,
                             **{f"lat{k}": (p / cx - 1) * 1e4 for k, p in zip((1, 2, 3, 5), post)}))
    P(f"{s} 완료")
X = pd.DataFrame(rows)
X.to_parquet(R / "data" / "r36_fill.parquet", index=False)
P(f"\n점검 거래 {len(X):,}")
P("[1] 지정가 2bp 아래에서 거래가 이어진 초 수 분포: " +
  " ".join(f"{q:.0%}:{X.nsec.quantile(q):.0f}" for q in (0.05, 0.1, 0.25, 0.5)))
P("    그 초들의 테이커 매도 대금(USDT) 분위: " +
  " ".join(f"{q:.0%}:{X.vol_below.quantile(q):,.0f}" for q in (0.05, 0.1, 0.25, 0.5)))
for thr in (10_000, 50_000, 200_000, 1_000_000):
    P(f"    관통 매도대금 ≥ {thr:>9,} USDT 인 거래 비율 {(X.vol_below >= thr).mean():.3f}")
P("[2] 청산 지연: 봉 종가 대비 k초 뒤 가격(bp, +면 유리) — 평균 / 5% 최악")
for k in (1, 2, 3, 5):
    v = X[f"lat{k}"]
    P(f"    {k}초 뒤 평균 {v.mean():+.2f}bp  5%분위 {v.quantile(0.05):+.2f}bp  1%분위 {v.quantile(0.01):+.2f}bp")
for s, g in X.groupby("sym"):
    P(f"    {s}: 2초 지연 평균 {g.lat2.mean():+.2f}bp, 관통대금 중앙 {g.vol_below.median():,.0f}")
