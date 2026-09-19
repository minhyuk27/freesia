"""깊이 전략의 배율 상한 — 전체 기간(학습 포함) 청산 위험, 손절 대안, 최적 명목.
사용: python research/depth_lev.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from lib_dip import bars
from depth_valid import run4
TE = pd.Timestamp("2024-01-01", tz="UTC")
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
LEGS = [("BTCUSDT", 240), ("BTCUSDT", 480), ("ETHUSDT", 240), ("ETHUSDT", 480)]
D = {k: bars(k[0], k[1]) for k in LEGS}


def agg(xbp, buf=0.0, fee=7.0):
    ss = []
    for (sym, B) in LEGS:
        d = D[(sym, B)]
        ret, mae, tin, hold = run4(d["o"], d["h"], d["l"], d["c"], B, xbp / 1e4, buf, fee)
        ok = ~np.isnan(ret)
        ss.append(pd.DataFrame({"r": ret[ok], "m": mae[ok]}, index=d["bi"][ok]))
    return pd.concat(ss).sort_index()


P("[1] 전체 기간(2020~) 배율별 청산 — 학습 구간 포함")
for xbp in (300, 400, 500, 700):
    A = agg(xbp)
    P(f"  ── x={xbp}bp 전체 n={len(A)} (MAE 최대 {A.m.max():.0f}bp, 99.9% {A.m.quantile(.999):.0f}bp) ──")
    for N in (1, 2, 3, 4, 5, 6):
        dist = (1 / N - 0.004) * 1e4
        liq = A.m >= dist
        pnl = np.where(liq, -1.0, N * A.r / 1e4)
        eq = np.cumprod(1 + pnl)
        ruin = bool((eq <= 0.01).any())
        yrs = (A.index[-1] - A.index[0]).days / 365
        te = A.index >= TE
        eq2 = np.cumprod(1 + pnl[te])
        ruin2 = bool((eq2 <= 0.01).any())
        yrs2 = (A.index[-1] - TE).days / 365
        P(f"    {N}배 거리{dist:5.0f}bp 청산 {int(liq.sum()):3d}건(학습 {int(liq[~te].sum())}, 테스트 {int(liq[te].sum())}) | "
          f"전체 {'파산' if ruin else f'연 {(eq[-1]**(1/yrs)-1)*100:+8.1f}%'} MDD {float((eq/np.maximum.accumulate(eq)-1).min())*100:6.1f}% | "
          f"테스트 {'파산' if ruin2 else f'연 {(eq2[-1]**(1/yrs2)-1)*100:+8.1f}%'}")

P("\n[2] 손절 도입 — 청산 대신 명시적 손절로 배율 확대 가능한가 (x=400bp)")
A = agg(400)
for sl in (300, 500, 800, 1200, 2000, 99999):
    st = A.m >= sl
    r2 = np.where(st, -sl - 7, A.r)
    for N in (1, 3, 5, 8, 12):
        dist = (1 / N - 0.004) * 1e4
        if sl < 99999 and sl >= dist:
            continue                          # 손절이 청산보다 멀면 무의미
        liq = A.m >= dist
        pnl = np.where(liq, -1.0, np.where(st, (-sl - 7) / 1e4 * N, N * A.r / 1e4))
        eq = np.cumprod(1 + pnl)
        if (eq <= 0.01).any():
            continue
        yrs = (A.index[-1] - A.index[0]).days / 365
        te = A.index >= TE
        eq2 = np.cumprod(1 + pnl[te]); yrs2 = (A.index[-1] - TE).days / 365
        if (eq2 <= 0.01).any():
            continue
        P(f"  손절 {sl if sl<99999 else '없음':>6}bp × {N:2d}배: 손절발동 {float(st.mean()):.3f} 청산 {int(liq.sum()):2d}건 | "
          f"전체 연 {(eq[-1]**(1/yrs)-1)*100:+8.1f}% MDD {float((eq/np.maximum.accumulate(eq)-1).min())*100:6.1f}% | "
          f"테스트 연 {(eq2[-1]**(1/yrs2)-1)*100:+8.1f}%")

P("\n[3] 최적 명목 — 로그성장 기준 (x별)")
for xbp in (300, 400, 500, 700):
    A = agg(xbp)
    best = []
    for N in np.arange(0.5, 6.1, 0.25):
        dist = (1 / N - 0.004) * 1e4
        if dist <= 0:
            continue
        liq = A.m >= dist
        pnl = np.where(liq, -1.0, N * A.r / 1e4)
        if (np.cumprod(1 + pnl) <= 0.01).any():
            continue
        eq = np.cumprod(1 + pnl)
        best.append((N, np.mean(np.log1p(pnl)) * 1e4, float((eq / np.maximum.accumulate(eq) - 1).min()) * 100,
                     int(liq.sum())))
    if not best:
        continue
    b = pd.DataFrame(best, columns=["명목N", "로그성장bp", "MDD%", "청산"])
    top = b.sort_values("로그성장bp", ascending=False).head(1)
    safe = b[b["MDD%"] > -50].sort_values("로그성장bp", ascending=False).head(1)
    P(f"  x={xbp}: 최적 N={float(top.명목N.iloc[0]):.2f}(MDD {float(top['MDD%'].iloc[0]):.1f}%) | "
      f"MDD-50이내 최적 N={float(safe.명목N.iloc[0]) if len(safe) else float('nan'):.2f}")

P("\n[4] 권장안 월별 성과 — x=400bp, 명목 2배")
A = agg(400)
N = 2.0; dist = (1 / N - 0.004) * 1e4
pnl = pd.Series(np.where(A.m >= dist, -1.0, N * A.r / 1e4), index=A.index)
mon = pnl.resample("ME").apply(lambda x: np.prod(1 + x) - 1) * 100
P("  최근 18개월: " + " | ".join(f"{i.strftime('%y-%m')} {v:+.1f}%" for i, v in mon.tail(18).items()))
l12 = mon.tail(12)
P(f"  최근 12개월 합계 {(np.prod(1+l12/100)-1)*100:+.1f}% | 월평균 {l12.mean():+.2f}% | 양수 {int((l12>0).sum())}/12 | 최악 {l12.min():+.1f}%")
te = pnl[pnl.index >= TE]
eq = np.cumprod(1 + te.values); yrs = (te.index[-1] - te.index[0]).days / 365
P(f"  테스트 전체: 연 {(eq[-1]**(1/yrs)-1)*100:+.1f}% | MDD {float((eq/np.maximum.accumulate(eq)-1).min())*100:.1f}% | 거래 {len(te)} 주 {len(te)/((te.index[-1]-te.index[0]).days/7):.2f}회")
