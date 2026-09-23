"""R52 — R50 방향 분류기 + '결판이 날 때만' 필터로 시간 청산을 줄인다.

R50(±1.5%, 상위 0.1%)은 결판 적중 65.8%인데 시간 청산 39%가 계좌를 끌어내렸다(월 −1%).
필터: σ12(직전 1440분 1분 수익 표준편차 × √720, t−1 까지)가 직전 30일 자기 분포의 상위 절반/상위 1/3일 때만.
      → ±d까지 갈 만큼 움직이는 국면에서만 매매.
보유: 12시간(원 규칙) / 24시간 비교. 계좌 = R47 simulate_entries (증거금 50% × 30배).
사용: python research/r52_resolve_filter.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
from r47_x30_frame import simulate_entries, stat
from r50_direction_model import select
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)

for y, d in (("y15", 0.015), ("y10", 0.01)):
    W = pd.read_parquet(R / "data" / f"r50_wf_{y}.parquet")
    for s in ("BTCUSDT", "ETHUSDT"):
        c = store.load("klines_1m", s).close.values
        rv = pd.Series(np.log(c)).diff().rolling(1440).std().values * np.sqrt(720)
        m = W.sym == s
        W.loc[m, "sig12"] = rv[W.loc[m, "t"].values - 1]
    # 직전 30일(같은 종목) 분포에서의 백분위, 현재 시점은 제외(shift) — 인과적
    W["sigq"] = np.nan
    for s in ("BTCUSDT", "ETHUSDT"):
        g = W[W.sym == s].set_index("ts").sig12
        W.loc[W.sym == s, "sigq"] = g.rolling("30D").rank(pct=True).shift(1).values
    P(f"\n===== ±{d*100:.1f}% (R50 분류기)")
    for q in (0.003, 0.001):
        for vf, lab in ((0.0, "필터 없음"), (0.5, "변동성 상위 1/2"), (0.67, "변동성 상위 1/3")):
            Wf = W[W.sigq >= vf] if vf > 0 else W
            S = select(Wf, q)
            wk = (W.ts.max() - W.ts.min()).days / 7
            dec = S[S[y] != 0]
            acc = (dec[y] == dec.side).mean() if len(dec) else np.nan
            line = f"  q={q*100:.1f}% {lab:10s}: 주 {len(S)/wk:4.1f}회 적중(결판) {acc:.3f} 12h무승부 {(S[y]==0).mean():.2f} |"
            for Hh in (720, 1440):
                eq, T = simulate_entries(S[["ts", "sym", "side"]], d, Hh)
                cg, mo, mmed, mdd, fin = stat(eq)
                line += f" {Hh//60}h: 월 {mo*100:+5.1f}% MDD {mdd*100:4.0f}% 익절{(T.kind==1).mean():.2f}/손절{(T.kind==-1).mean():.2f}/시간{(T.kind==0).mean():.2f}"
            by = dec.groupby("yr").apply(lambda g: (g[y] == g.side).mean())
            P(line + " | " + " ".join(f"{yy%100}:{v:.2f}" for yy, v in by.items()))
