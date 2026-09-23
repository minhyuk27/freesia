"""R37 — z0.31에 하드 손절을 넣어 꼬리를 자르면 고레버리지(켈리 영역)가 열리는가.

배경: z0.31 사건당 +24bp·표준편차 67bp → 이론 켈리 ~55배, 월 +1,000% 이상.
      막는 것은 수익 분산이 아니라 MAE 꼬리(수 %). 손절 d로 손실을 묶으면 배율 상한이 1/(d+슬리피지)로 올라간다.
손절 체결(1초 데이터, 보수적): 진입 이후 1초 저가가 T×(1−d)에 닿은 초 s → 2초 지연 뒤 종가에 시장가 청산(테이커 5bp).
      갭으로 그 사이 더 밀린 만큼 그대로 손실. 체결 초 이전의 같은 분 움직임은 체결 초부터만 본다.
계좌: R28 방식(명목 먼저, 동시 보유 합산, 열린 포지션 전부 동시 최악 가정 청산 점검).
     손절 거래의 MAE = 실제 청산가까지의 손실(+ 체결~청산 사이 최저가 반영).
사용: python research/r37_stop_leverage.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
from r28_hf_port import sim
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
FEE = 7e-4; EPS = 2e-4; LAT = 2
STOPS = (0.004, 0.006, 0.008, 0.01, 0.015, 0.02, 0.03)
W = pd.read_parquet(R / "data" / "r32_wf.parquet")
W["ts"] = pd.DatetimeIndex(W.ts)
W = W[(W.pred > 0.0005) & (W.ts >= "2023-01-01")].copy()
K = {s: store.load("klines_1m", s).ts.reset_index(drop=True) for s in ("BTCUSDT", "ETHUSDT")}

rows = []
for s in ("BTCUSDT", "ETHUSDT"):
    S = W[W.sym == s]; kts = K[s]
    for ym, g in S.groupby(S.ts.dt.strftime("%Y-%m")):
        f = store.PQ / "agg1s" / s / f"{ym}.parquet"
        if not f.exists(): continue
        d = pd.read_parquet(f).set_index("ts")
        for i, e in g.iterrows():
            fmin = kts[int(e.f)]; xend = kts[int(e.x)] + pd.Timedelta(minutes=1)
            cx = d.c.asof(xend - pd.Timedelta(seconds=1))
            T = cx / (1 + e.ret + FEE)
            w = d.loc[fmin: xend - pd.Timedelta(seconds=1)]
            hit = np.where(w.l.values <= T * (1 - EPS))[0]
            if len(hit) == 0: continue
            w = w.iloc[hit[0]:]
            lows = w.l.values; tsw = w.index
            rec = dict(idx=i)
            for dd in STOPS:
                j = np.where(lows <= T * (1 - dd))[0]
                if len(j):
                    tx = tsw[j[0]] + pd.Timedelta(seconds=LAT)
                    px = d.c.asof(tx)
                    worst = min(lows[:j[0] + 1].min(), d.l.loc[tsw[j[0]]:tx].min())
                    rec[f"r{dd}"] = px / T - 1 - FEE
                    rec[f"m{dd}"] = max(1 - worst / T, 1 - px / T)
                else:
                    rec[f"r{dd}"] = e.ret; rec[f"m{dd}"] = e.mae
            rows.append(rec)
    P(f"{s} 완료")
X = pd.DataFrame(rows).set_index("idx")
Wj = W.join(X, how="inner")
Wj.to_parquet(R / "data" / "r37_stops.parquet")
YRS = (Wj.ts.max() - Wj.ts.min()).days / 365.25
P(f"\n거래 {len(Wj):,} (2023~, 예측>5bp), 손절 체결 = 닿은 초 +{LAT}초 종가")

for th in (0.0005, 0.001, 0.002):
    S0 = Wj[Wj.pred > th]
    P(f"\n===== 예측>{th*1e4:.0f}bp: 연 {len(S0)/YRS:.0f}건")
    P(f"  {'손절':>6s} {'거래당':>8s} {'승률':>6s} {'손절률':>6s} {'MAE최대':>7s} | 계좌 (w=거래당 명목, N=총명목 상한) 연%·월%·MDD")
    for dd in (None,) + STOPS:
        S = S0.copy()
        if dd is not None: S["ret"] = S[f"r{dd}"]; S["mae"] = S[f"m{dd}"]
        S = S.sort_values(["f", "sym"]).reset_index(drop=True)
        stopped = 0.0 if dd is None else (S0[f"r{dd}"] != S0.ret).mean()
        line = f"  {('없음' if dd is None else f'{dd*100:.1f}%'):>6s} {S.ret.mean()*1e4:+7.2f}bp {(S.ret>0).mean():6.3f} {stopped:6.3f} {S.mae.max()*100:6.1f}% |"
        best = None
        for w, Nm in ((1, 3), (2, 6), (3, 10), (5, 15), (8, 25), (12, 40), (20, 60)):
            res, liq = sim(S, w, Nm)
            if res is None:
                line += f" w{w}/N{Nm} 파산"; continue
            eq, mdd, worst = res
            cg = eq.iloc[-1] ** (1 / YRS) - 1; mo = (1 + cg) ** (1 / 12) - 1
            line += f" w{w}/N{Nm} {mo*100:+.0f}%m({mdd*100:.0f})"
        P(line)
