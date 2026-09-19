"""체결가(last) 위크 vs 마크가격 — 순간 급락 시 청산 판정 기준이 얼마나 덜 움직이는가.
청산은 마크가격 기준이므로, last의 위크가 마크에 얼마나 전달되는지가 관건.
사용: python research/markwick.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R))
from freesia import store
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)

for sym in ("BTCUSDT", "ETHUSDT"):
    k = store.load("klines_1m", sym).set_index("ts")
    m = store.load("mark_1m", sym).set_index("ts")
    df = pd.DataFrame({"lc": k.close, "ll": k.low, "lh": k.high,
                       "mc": m.close, "ml": m.low, "mh": m.high}).dropna()
    P("=" * 92 + f"\n### {sym}  1분봉 {len(df):,}개")
    # 1분 내 하방 위크 크기: 직전 종가 대비 저가
    wick_l = (df.lc.shift(1) - df.ll) / df.lc.shift(1) * 1e4
    wick_m = (df.mc.shift(1) - df.ml) / df.mc.shift(1) * 1e4
    d = pd.DataFrame({"last": wick_l, "mark": wick_m}).dropna()
    P("\n[1] 1분 내 하방 이탈 폭 (직전 종가 대비 저가, bp)")
    P(f"  체결가(last): 중앙 {d['last'].median():.1f} | 99% {d['last'].quantile(.99):.1f} | 99.9% {d['last'].quantile(.999):.1f} | 최대 {d['last'].max():.0f}")
    P(f"  마크가격    : 중앙 {d['mark'].median():.1f} | 99% {d['mark'].quantile(.99):.1f} | 99.9% {d['mark'].quantile(.999):.1f} | 최대 {d['mark'].max():.0f}")
    P(f"  마크/체결 비율 중앙 {(d['mark']/d['last'].replace(0,np.nan)).median():.3f}")

    P("\n[2] 체결가가 X bp 급락한 분에서, 같은 분 마크가격은 얼마나 빠졌나")
    rows = []
    for th in (30, 60, 100, 200, 400):
        sel = d["last"] >= th
        if sel.sum() < 30:
            continue
        rows.append((th, int(sel.sum()), d.loc[sel, "mark"].median(), d.loc[sel, "mark"].mean(),
                     float((d.loc[sel, "mark"] < th).mean()), float((d.loc[sel, "mark"] < th * 0.5).mean())))
    P(pd.DataFrame(rows, columns=["체결 급락bp", "건수", "마크 중앙bp", "마크 평균bp",
                                  "마크가 더 작은 비율", "마크가 절반 미만 비율"]).round(3).to_string(index=False))

    P("\n[3] 청산 판정 차이 — 체결가로는 청산, 마크가격으로는 생존한 비율")
    for L, dist in ((100, 60), (50, 160), (20, 460), (10, 960)):
        a = (d["last"] >= dist); b = (d["mark"] >= dist)
        if a.sum() < 10:
            continue
        P(f"  {L:>3}배(청산거리 {dist:>3}bp): 체결가 기준 {int(a.sum()):>6}건 / 마크가격 기준 {int(b.sum()):>6}건 "
          f"→ 마크 기준으로 살아남는 비율 {1 - b.sum()/a.sum():.3f}")

    P("\n[4] 위크 복구 시간 — 체결가가 60bp 이상 급락 후 원위치까지")
    lc = df.lc.values; ll = df.ll.values
    cnt = 0; back = []
    for i in range(1, len(df) - 30):
        if (lc[i-1] - ll[i]) / lc[i-1] * 1e4 >= 60:
            cnt += 1
            for j in range(i, min(i + 30, len(df))):
                if lc[j] >= lc[i-1]:
                    back.append(j - i + 1); break
            else:
                back.append(99)
    back = np.array(back)
    P(f"  60bp 이상 급락 {cnt:,}회 | 1분 내 복구 {float((back<=1).mean()):.3f} | 3분 내 {float((back<=3).mean()):.3f} | "
      f"10분 내 {float((back<=10).mean()):.3f} | 30분 내 미복구 {float((back==99).mean()):.3f}")
