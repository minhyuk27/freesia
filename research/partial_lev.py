"""부분 투입 고배율 vs 전액 투입 저배율 — 같은 명목에서 어느 쪽이 나은가.
명목 N(자본 배수), 증거금 비율 m, 배율 L=N/m, 청산거리 d=m/N-0.4%, 청산 시 손실 = m.
실제 MAE·수익 데이터로 시뮬레이션. 사용: python research/partial_lev.py"""
import sys, warnings
from pathlib import Path
import numba, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R))
from freesia import store
from dip_lev import detail
TE = pd.Timestamp("2024-01-01", tz="UTC")
MMR = 0.004; MAXL = 125.0
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
B = 240

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
    ok = ~np.isnan(ret)
    r = (ret[ok] - 7) / 1e4      # 거래당 명목 대비 수익(비용 후)
    ma = mae[ok] / 1e4           # 역행(비율)
    bts = bi[ok]
    P("=" * 104 + f"\n### {sym}  진입 {len(r)}건")
    P(f"  MAE: 중앙 {np.median(ma)*1e4:.0f}bp | 60bp 초과 {float((ma>=0.006).mean()):.3f} | 160bp 초과 {float((ma>=0.016).mean()):.3f} | "
      f"460bp 초과 {float((ma>=0.046).mean()):.3f} | 960bp 초과 {float((ma>=0.0096).mean()):.3f}")

    P("\n[1] 사용자 안: 자본의 30%를 100배 → 명목 30배, 청산거리 0.6%")
    m0, L0 = 0.30, 100.0
    d0 = 1 / L0 - MMR
    liq = ma >= d0
    pnl = np.where(liq, -m0, (m0 * L0) * r)
    eq = np.cumprod(1 + pnl)
    P(f"  청산률 {liq.mean():.3f} ({int(liq.sum())}건) | 거래당 평균 {pnl.mean()*100:+.2f}% | "
      f"자본 소진 시점 {'있음' if (eq<=0.01).any() else '없음'} | 최종 배수 {eq[-1]:.2e}")
    P(f"  청산 손실만 합계 {liq.mean()*m0*100:.2f}%/거래 vs 생존 시 기대 {(1-liq.mean())*m0*L0*r[~liq].mean()*100:+.2f}%/거래")

    P("\n[2] 명목을 같게 맞춘 비교 — 명목 N배를 만드는 (증거금 m, 배율 L) 조합")
    rows = []
    for N in (1, 3, 5, 10, 30):
        for m in (1.0, 0.5, 0.3, 0.2, 0.1, 0.05):
            L = N / m
            if L < 1 or L > MAXL:
                continue
            d = 1 / L - MMR
            if d <= 0:
                continue
            liq = ma >= d
            pnl = np.where(liq, -m, N * r)
            eq = np.cumprod(1 + pnl)
            ruin = bool((eq <= 0.01).any())
            mdd = float((eq / np.maximum.accumulate(eq) - 1).min())
            te = bts >= TE
            eq2 = np.cumprod(1 + pnl[te])
            rows.append((N, m, L, d * 1e4, liq.mean(), pnl.mean() * 100,
                         "파산" if ruin else f"{eq[-1]:.1f}배", mdd * 100,
                         "파산" if (eq2 <= 0.01).any() else f"{eq2[-1]:.2f}배"))
    d_ = pd.DataFrame(rows, columns=["명목N", "증거금m", "배율L", "청산거리bp", "청산률", "거래당%", "전체 최종", "MDD%", "테스트 최종"])
    P(d_.round(4).to_string(index=False))

    P("\n[3] 같은 명목에서 증거금을 줄이면(배율↑) 어떻게 되는가 — 명목 3배 고정")
    for m in (1.0, 0.5, 0.3, 0.15, 0.1, 0.05, 0.03):
        L = 3 / m
        if L > MAXL:
            continue
        d = 1 / L - MMR
        liq = ma >= d
        pnl = np.where(liq, -m, 3 * r)
        eq = np.cumprod(1 + pnl)
        P(f"  증거금 {m*100:5.1f}% (={L:6.1f}배) 청산거리 {d*1e4:7.1f}bp | 청산률 {liq.mean():.4f} | "
          f"거래당 {pnl.mean()*100:+.3f}% | MDD {float((eq/np.maximum.accumulate(eq)-1).min())*100:6.1f}% | 최종 {('파산' if (eq<=0.01).any() else f'{eq[-1]:.1f}배')}")

    P("\n[4] 최적 조합 탐색 (전체 기간 기하평균 성장률 기준, 파산 제외)")
    best = []
    for N in np.arange(1, 21, 1.0):
        for m in np.arange(0.05, 1.01, 0.05):
            L = N / m
            if L < 1 or L > MAXL:
                continue
            d = 1 / L - MMR
            if d <= 0:
                continue
            liq = ma >= d
            pnl = np.where(liq, -m, N * r)
            if (np.cumprod(1 + pnl) <= 0.01).any():
                continue
            g = np.mean(np.log1p(pnl))
            eq = np.cumprod(1 + pnl)
            best.append((N, m, L, d * 1e4, liq.mean(), g * 1e4, eq[-1],
                         float((eq / np.maximum.accumulate(eq) - 1).min()) * 100))
    bd = pd.DataFrame(best, columns=["명목N", "증거금m", "배율L", "청산거리bp", "청산률", "로그성장bp", "최종배수", "MDD%"])
    P(bd.sort_values("로그성장bp", ascending=False).head(8).round(3).to_string(index=False))
    P("  MDD -30% 이하로 제한하면:")
    P(bd[bd["MDD%"] > -30].sort_values("로그성장bp", ascending=False).head(5).round(3).to_string(index=False))
