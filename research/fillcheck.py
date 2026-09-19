"""체결 지연·미체결 위험 정량화 — 자본 규모별. 저변동 역행 되돌림 전략 기준.
지정가 매수는 (a)가격이 지정가 아래로 뚫리면 확실 체결 (b)정확히 닿기만 하면 큐 경쟁.
측정: 침투 깊이, 트리거 이하 체류 시간·거래대금, 주문/거래대금 비율, 청산 시 시장가 충격.
사용: python research/fillcheck.py"""
import sys, warnings
from pathlib import Path
import numba, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R))
from freesia import store
TE = pd.Timestamp("2024-01-01", tz="UTC")
KRW = 1390.0
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
B = 240


@numba.njit(cache=True)
def detail(o, h, l, c, v, B, xarr, cond):
    """진입 건별: 침투깊이bp, 트리거이하 체류분, 그 구간 거래대금, 진입분 거래대금, 청산분 거래대금"""
    nb = len(c) // B
    pen = np.full(nb, np.nan); dwell = np.zeros(nb, np.int32)
    volb = np.full(nb, np.nan); vent = np.full(nb, np.nan); vexit = np.full(nb, np.nan)
    for b in range(nb):
        if not cond[b] or np.isnan(xarr[b]) or xarr[b] <= 0:
            continue
        s = b * B; e = s + B
        O = o[s]; trig = O * (1.0 - xarr[b])
        j = -1
        for t in range(s, e):
            if l[t] <= trig:
                j = t; break
        if j < 0:
            continue
        deep = 0.0; dw = 0; vv = 0.0
        for t in range(j, e):
            if l[t] <= trig:
                dw += 1
                vv += v[t]
                d = (trig - l[t]) / trig * 1e4
                if d > deep:
                    deep = d
        pen[b] = deep; dwell[b] = dw; volb[b] = vv; vent[b] = v[j]; vexit[b] = v[e - 1]
    return pen, dwell, volb, vent, vexit


for sym in ("BTCUSDT", "ETHUSDT"):
    k = store.load("klines_1m", sym).set_index("ts")
    o, h, l, c, v = k.open.values, k.high.values, k.low.values, k.close.values, k.quote_volume.values
    idx = k.index; nb = len(c) // B
    bo = o[::B][:nb]; bc = c[B - 1::B][:nb]
    bh = np.array([h[i * B:(i + 1) * B].max() for i in range(nb)])
    bl = np.array([l[i * B:(i + 1) * B].min() for i in range(nb)])
    bi = idx[::B][:nb]
    s = pd.Series
    xa = (s((bh - bl) / bo).shift(1) * 1.0).values
    rv = s(np.log(bc)).diff().rolling(24).std()
    cd = (rv < rv.rolling(720).quantile(0.3)).fillna(False).values
    pen, dwell, volb, vent, vexit = detail(o, h, l, c, v, B, xa, cd)
    ok = ~np.isnan(pen); te = ok & (bi >= TE)
    P("=" * 96 + f"\n### {sym}  진입 {int(te.sum())}건 (테스트 2024~)")

    P("\n[1] 지정가 침투 깊이 — 가격이 지정가보다 얼마나 더 아래로 내려갔나 (bp)")
    x = pen[te]
    P(f"  평균 {x.mean():.1f} | 중앙 {np.median(x):.1f} | 10% {np.percentile(x,10):.1f} | 25% {np.percentile(x,25):.1f} | 75% {np.percentile(x,75):.1f} | 최대 {x.max():.0f}")
    for th in (0, 1, 2, 5, 10, 20):
        P(f"    지정가보다 {th:>2}bp 이상 더 하락한 비율: {float((x >= th).mean()):.3f}")

    P("\n[2] 지정가 이하 체류 시간(분)과 그 구간 거래대금")
    d_ = dwell[te]; vb = volb[te]
    P(f"  체류 분: 중앙 {np.median(d_):.0f} | 25% {np.percentile(d_,25):.0f} | 75% {np.percentile(d_,75):.0f} | 1분 이하 비율 {float((d_<=1).mean()):.3f}")
    P(f"  체류 구간 거래대금(USD): 중앙 {np.median(vb):,.0f} | 10% {np.percentile(vb,10):,.0f} | 25% {np.percentile(vb,25):,.0f}")
    P(f"  진입 분 거래대금(USD): 중앙 {np.median(vent[te]):,.0f} | 10% {np.percentile(vent[te],10):,.0f}")
    P(f"  청산(종가) 분 거래대금(USD): 중앙 {np.median(vexit[te]):,.0f} | 10% {np.percentile(vexit[te],10):,.0f}")

    P("\n[3] 자본 규모별 주문/거래대금 비율 (명목 1배 = 자본 전액 진입 가정)")
    rows = []
    for won in (1e7, 3e7, 5e7, 1e8, 3e8, 1e9):
        usd = won / KRW
        r_b = usd / vb          # 체류 구간 거래대금 대비
        r_e = usd / vexit[te]   # 청산 분 거래대금 대비
        rows.append((f"{won/1e8:.2f}억", usd, np.median(r_b) * 100, np.percentile(r_b, 90) * 100,
                     float((r_b > 0.10).mean()), np.median(r_e) * 100, float((r_e > 0.10).mean())))
    P(pd.DataFrame(rows, columns=["자본", "USD", "진입 비중%(중앙)", "진입 비중%(90분위)", "진입 10%초과 비율",
                                  "청산 비중%(중앙)", "청산 10%초과 비율"]).round(3).to_string(index=False))

    P("\n[4] 체결 확실성 등급 (자본 1억 = $%s 기준)" % f"{1e8/KRW:,.0f}")
    usd = 1e8 / KRW
    sure = (pen[te] >= 2) | ((volb[te] > usd * 20) & (dwell[te] >= 2))   # 2bp 뚫었거나, 대금 20배+2분 체류
    mid = ~sure & ((pen[te] >= 0.5) | (volb[te] > usd * 5))
    risky = ~sure & ~mid
    P(f"  확실(2bp 이상 관통 또는 주문의 20배 대금 + 2분 이상 체류): {sure.mean():.3f}")
    P(f"  보통: {mid.mean():.3f}")
    P(f"  위험(큐 경쟁 필요): {risky.mean():.3f}")
    P(f"  ※ 위험 등급을 전부 미체결로 처리하면 거래 수 {int(te.sum())} → {int(te.sum()*(1-risky.mean()))}건, 주당 {te.sum()*(1-risky.mean())/((bi[-1]-TE).days/7):.2f}회")

    P("\n[5] 연도별 진입 분 거래대금 중앙(USD) — 유동성 추이")
    g = pd.Series(vent[ok], index=bi[ok]).groupby(lambda t: t.year).median()
    P("  " + " | ".join(f"{y} {val:,.0f}" for y, val in g.items()))
