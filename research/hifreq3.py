"""남은 고빈도 계열 셋 — 펀딩 정산 흐름 / BTC→ETH 선행성 / 시간대·요일 구조.
전부 진입 시점 정보만 사용. 왕복 비용 10bp(테이커) 기준으로 판정.
사용: python research/hifreq3.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
SYM = ("BTCUSDT", "ETHUSDT")
K = {}
for s in SYM:
    k = store.load("klines_1m", s).set_index("ts")
    K[s] = k
ix = K["BTCUSDT"].index.intersection(K["ETHUSDT"].index)
cb = K["BTCUSDT"].close.reindex(ix).values
ce = K["ETHUSDT"].close.reindex(ix).values
ob = K["BTCUSDT"].open.reindex(ix).values
oe = K["ETHUSDT"].open.reindex(ix).values
P(f"공통 {len(ix):,}분봉")

f = store.load("funding", "BTCUSDT").set_index("ts")
P(f"펀딩 {len(f)}건, 최근 구간 {f.rate.tail(1000).mean()*1e4:+.2f}bp/8h")


def fwd(c, h):
    return pd.Series(c).shift(-h).values/c - 1.0


P("\n" + "="*95)
P("[A] 펀딩 정산 시각(00·08·16 UTC) 전후 흐름")
P("="*95)
mins = pd.Series(ix).dt
h = mins.hour.values; mi = mins.minute.values
tof = (h % 8)*60 + mi          # 직전 정산으로부터 경과 분 (0~479)
for s, c, o in (("BTC", cb, ob), ("ETH", ce, oe)):
    P(f"\n  ── {s} — 정산 경과 시간대별 30분 수익(bp), 비용 전 ──")
    F = fwd(c, 30)
    bins = [(0, 30), (30, 60), (60, 120), (120, 240), (240, 360), (360, 450), (450, 470), (470, 480)]
    row = []
    for lo, hi in bins:
        sel = (tof >= lo) & (tof < hi) & np.isfinite(F)
        row.append(f"{lo}-{hi}분 {F[sel].mean()*1e4:+5.2f}")
    P("     " + " | ".join(row))
    # 펀딩 부호별
    fr = f.rate.reindex(ix, method="ffill").values
    for lab, sel0 in (("펀딩 음수", fr < 0), ("펀딩 상위10%", fr > np.nanquantile(fr, 0.9))):
        sel = sel0 & (tof >= 450) & np.isfinite(F)
        if sel.sum() < 500: continue
        P(f"     {lab} & 정산 30분 전: n={sel.sum():6d} 30분 수익 {F[sel].mean()*1e4:+.2f}bp")

P("\n" + "="*95)
P("[B] BTC → ETH 선행성 — BTC의 최근 k분 움직임이 ETH의 다음 m분을 예측하나")
P("="*95)
P(f"  {'k분':>5s} {'m분':>5s} {'상관':>8s} {'상위10% 이후 ETH수익bp':>22s} {'하위10%':>10s} {'차이':>8s} {'비용10bp 초과':>12s}")
for k in (1, 3, 5, 15, 30):
    rb = pd.Series(cb).pct_change(k).values
    for m in (1, 3, 5, 15, 30):
        re_ = fwd(ce, m)
        ok = np.isfinite(rb) & np.isfinite(re_)
        if ok.sum() < 1000: continue
        cor = np.corrcoef(rb[ok], re_[ok])[0, 1]
        hi = np.nanquantile(rb[ok], 0.9); lo = np.nanquantile(rb[ok], 0.1)
        a = re_[ok & (rb >= hi)].mean()*1e4
        b = re_[ok & (rb <= lo)].mean()*1e4
        P(f"  {k:5d} {m:5d} {cor:+8.4f} {a:+21.2f} {b:+9.2f} {a-b:+8.2f} {'예' if abs(a-b) > 10 else '아니오':>12s}")

P("\n  역방향 ETH → BTC")
for k in (1, 5, 15):
    re2 = pd.Series(ce).pct_change(k).values
    for m in (1, 5, 15):
        rb2 = fwd(cb, m)
        ok = np.isfinite(re2) & np.isfinite(rb2)
        cor = np.corrcoef(re2[ok], rb2[ok])[0, 1]
        hi = np.nanquantile(re2[ok], 0.9); lo = np.nanquantile(re2[ok], 0.1)
        a = rb2[ok & (re2 >= hi)].mean()*1e4; b = rb2[ok & (re2 <= lo)].mean()*1e4
        P(f"  k={k} m={m}: 상관 {cor:+.4f} 상위10% {a:+.2f}bp 하위10% {b:+.2f}bp 차이 {a-b:+.2f}bp")

P("\n" + "="*95)
P("[C] 시간대·요일 구조 (1시간 보유, bp)")
P("="*95)
for s, c in (("BTC", cb), ("ETH", ce)):
    F = fwd(c, 60)
    g = pd.DataFrame({"h": h, "d": mins.dayofweek.values, "r": F}).dropna()
    hh = g.groupby("h").r.mean()*1e4
    dd = g.groupby("d").r.mean()*1e4
    P(f"  {s} 시간대(UTC): " + " ".join(f"{i:02d}h{v:+5.2f}" for i, v in hh.items()))
    P(f"  {s} 요일       : " + " ".join(f"{['월','화','수','목','금','토','일'][i]}{v:+5.2f}" for i, v in dd.items()))
    P(f"    최대-최소 시간대 차이 {hh.max()-hh.min():.2f}bp (비용 10bp), 요일 차이 {dd.max()-dd.min():.2f}bp")

P("\n" + "="*95)
P("[D] 종합 판정 — 비용 10bp를 넘는 계열이 하나라도 있나")
P("="*95)
P("  각 계열의 최대 관측 엣지를 왕복 비용과 비교한다.")
