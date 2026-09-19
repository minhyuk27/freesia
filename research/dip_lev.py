"""저변동 역행 되돌림 전략 + 레버리지 — 진입 후 MAE 분포, 배율별 청산·수익·낙폭, 켈리, 캐리 병행.
사용: python research/dip_lev.py"""
import sys, warnings
from pathlib import Path
import numba, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R))
from freesia import store
TE = pd.Timestamp("2024-01-01", tz="UTC")
MMR = 0.004
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
B = 240


@numba.njit(cache=True)
def detail(o, h, l, c, B, xarr, cond):
    """진입가 대비: 종가수익bp, MAE bp(최대역행), MFE bp, 보유분"""
    nb = len(c) // B
    ret = np.full(nb, np.nan); mae = np.full(nb, np.nan); mfe = np.full(nb, np.nan); hold = np.zeros(nb, np.int32)
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
        lo_ = trig; hi_ = trig
        for t in range(j, e):
            if l[t] < lo_:
                lo_ = l[t]
            if h[t] > hi_:
                hi_ = h[t]
        ret[b] = (c[e - 1] / trig - 1.0) * 1e4
        mae[b] = (1.0 - lo_ / trig) * 1e4
        mfe[b] = (hi_ / trig - 1.0) * 1e4
        hold[b] = e - 1 - j
    return ret, mae, mfe, hold


for sym in ("BTCUSDT", "ETHUSDT"):
    k = store.load("klines_1m", sym).set_index("ts")
    o, h, l, c = k.open.values, k.high.values, k.low.values, k.close.values
    idx = k.index; nb = len(c) // B
    bo = o[::B][:nb]; bc = c[B - 1::B][:nb]
    bh = np.array([h[i * B:(i + 1) * B].max() for i in range(nb)])
    bl = np.array([l[i * B:(i + 1) * B].min() for i in range(nb)])
    bi = idx[::B][:nb]
    s = pd.Series
    xa = (s((bh - bl) / bo).shift(1) * 1.0).values
    rv = s(np.log(bc)).diff().rolling(24).std()
    cd = (rv < rv.rolling(720).quantile(0.3)).fillna(False).values
    ret, mae, mfe, hold = detail(o, h, l, c, B, xa, cd)
    ok = ~np.isnan(ret); te = ok & (bi >= TE)
    P("=" * 96 + f"\n### {sym}  진입 {int(ok.sum())}건 (테스트 {int(te.sum())})")

    P("\n[1] 진입가 대비 최대 역행(MAE) 분포 — 청산 거리 결정 근거 (bp)")
    for nm, m in (("전체", ok), ("테스트", te)):
        x = mae[m]
        P(f"  {nm}: 중앙 {np.median(x):6.1f} | 75% {np.percentile(x,75):6.1f} | 90% {np.percentile(x,90):6.1f} | "
          f"99% {np.percentile(x,99):7.1f} | 99.9% {np.percentile(x,99.9):7.1f} | 최대 {x.max():7.1f}")
    P("  역행이 특정 폭을 넘은 비율(전체)")
    P("  " + " | ".join(f"{t}bp {float((mae[ok] >= t).mean()):.4f}" for t in (50, 100, 200, 300, 500, 960, 1960)))

    P("\n[2] 배율별 청산 확률과 순수익 (청산거리 = 1/L − 0.4%, 청산 시 증거금 전액 손실)")
    rows = []
    base = ret[ok] - 7
    for L in (1, 2, 3, 5, 10, 20, 50):
        d = (1.0 / L - MMR) * 1e4
        liq = mae[ok] >= d
        pnl = np.where(liq, -1.0, (ret[ok] - 7) / 1e4 * L)      # 자본 대비(증거금 전액 사용 가정)
        for nm, m in (("학습", bi[ok] < TE), ("테스트", bi[ok] >= TE)):
            x = pnl[m]
            eq = np.cumprod(1 + x)
            mdd = float((eq / np.maximum.accumulate(eq) - 1).min())
            rows.append((L, d, nm, float(liq[m].mean()), x.mean() * 1e4, eq[-1] - 1,
                         mdd, float((x > 0).mean()), int(m.sum())))
    d_ = pd.DataFrame(rows, columns=["배율", "청산거리bp", "구간", "청산률", "거래당 자본대비bp", "누적배수-1", "MDD", "승률", "n"])
    P(d_.round(4).to_string(index=False))

    P("\n[3] 켈리 (거래당 수익률 기준, 보유 2시간)")
    for nm, m in (("학습", ok & (bi < TE)), ("테스트", te)):
        r = (ret[m] - 7) / 1e4
        f = r.mean() / r.var() if r.var() > 0 else 0
        P(f"  {nm}: 평균 {r.mean()*1e4:.2f}bp 분산 {r.var():.2e} → 풀켈리 {f:.1f}배 | 1/4켈리 {f/4:.1f}배 | "
          f"최악 거래 {r.min()*1e4:.0f}bp → 풀켈리 시 {r.min()*f*100:.1f}% 손실")

    P("\n[4] 배율별 최악 시나리오")
    for L in (1, 2, 3, 5):
        d = (1.0 / L - MMR) * 1e4
        liq = mae[ok] >= d
        pnl = np.where(liq, -1.0, (ret[ok] - 7) / 1e4 * L)
        eq = np.cumprod(1 + pnl)
        lo = (pnl < 0).astype(int)
        mx = 0; cur = 0
        for z in lo:
            cur = cur + 1 if z else 0
            mx = max(mx, cur)
        P(f"  {L}배: 청산 {int(liq.sum())}건 | 최대 연속손실 {mx}회 | 최악 거래 {pnl.min()*100:.1f}% | "
          f"전체 누적 {(eq[-1]-1)*100:+.0f}% | MDD {float((eq/np.maximum.accumulate(eq)-1).min())*100:.1f}%")

    P("\n[5] 청산 없는 대안 — 손절을 명시적으로 걸 때 (배율 무관, 손절 폭별)")
    rows = []
    for sl in (50, 100, 150, 200, 300):
        stopped = mae[ok] >= sl
        r2 = np.where(stopped, -sl - 7, ret[ok] - 7)
        for nm, m in (("학습", bi[ok] < TE), ("테스트", bi[ok] >= TE)):
            rows.append((sl, nm, float(stopped[m].mean()), r2[m].mean(), float((r2[m] > 0).mean())))
    P(pd.DataFrame(rows, columns=["손절bp", "구간", "손절발동률", "거래당 순bp", "승률"]).round(3).to_string(index=False))
