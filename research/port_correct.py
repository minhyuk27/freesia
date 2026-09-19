"""정확한 포트폴리오 시뮬레이션 — 자본 배분·동시 진입·분할 자본을 모두 반영.
규칙: 총자본 1. 4다리에 각 1/4 고정 배정. 다리 내 3단 분할이면 배정액의 1/3씩.
각 다리는 독립 복리, 최종 합산. 동시 진입해도 자본 중복 없음.
사용: python research/port_correct.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from lib_dip import bars
TE = pd.Timestamp("2024-01-01", tz="UTC")
rng = np.random.default_rng(283)
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
LEGS = [("BTCUSDT", 240), ("BTCUSDT", 480), ("ETHUSDT", 240), ("ETHUSDT", 480)]
D = {}
for (sym, B) in LEGS:
    d = bars(sym, B); s = pd.Series
    d["rv"] = (s(np.log(d["bc"])).diff().rolling(20).std()).shift(1).values
    D[(sym, B)] = d


@numba.njit(cache=True)
def leg_pnl(o, h, l, c, B, x1, x2, x3, fee, N):
    """반환: 봉별 (자본 대비 수익, 최대 역행 비율). 배정 자본을 3등분해 단계별 투입.
    각 단계 명목 = 배정자본/3 × N. 청산은 다리 배정자본 기준."""
    n = len(c) // B
    pnl = np.zeros(n); mae = np.zeros(n); used = np.zeros(n)
    for b in range(n):
        if np.isnan(x1[b]) or x1[b] <= 0:
            continue
        s = b * B; e = s + B
        O = o[s]
        t1 = O * (1 - x1[b]); t2 = O * (1 - x2[b]); t3 = O * (1 - x3[b])
        f1 = -1; f2 = -1; f3 = -1
        for t in range(s, e):
            if f1 < 0 and l[t] <= t1:
                f1 = t
            if f2 < 0 and l[t] <= t2:
                f2 = t
            if f3 < 0 and l[t] <= t3:
                f3 = t
        if f1 < 0:
            continue
        C = c[e - 1]
        # 단계별 수익(각 배정자본의 1/3에 명목 N배)
        tot = 0.0; cnt = 0.0; wavg = 0.0
        r1 = (C / t1 - 1.0) - fee / 1e4
        tot += r1 / 3.0; cnt += 1.0; wavg += t1
        if f2 >= 0:
            r2 = (C / t2 - 1.0) - fee / 1e4
            tot += r2 / 3.0; cnt += 1.0; wavg += t2
        if f3 >= 0:
            r3 = (C / t3 - 1.0) - fee / 1e4
            tot += r3 / 3.0; cnt += 1.0; wavg += t3
        avg = wavg / cnt
        lo_ = t1
        for t in range(f1, e):
            if l[t] < lo_:
                lo_ = l[t]
        # 청산 판정: 배정자본 대비 손실이 100%면 청산. 명목 = (cnt/3)*N
        notional = (cnt / 3.0) * N
        adverse = 1.0 - lo_ / avg
        pnl[b] = tot * N
        mae[b] = adverse * notional      # 배정자본 대비 최대 평가손실
        used[b] = cnt / 3.0
    return pnl, mae, used


def portfolio(split, N, fee=7.0, w=None):
    """4다리 각 1/4 배정, 독립 복리 후 합산"""
    if w is None:
        w = [0.25] * 4
    eqs = []
    allm = []
    for i, key in enumerate(LEGS):
        d = D[key]
        k1, k2, k3 = split
        p, m, u = leg_pnl(d["o"], d["h"], d["l"], d["c"], d["B"],
                          np.asarray(d["rv"] * k1, float), np.asarray(d["rv"] * k2, float),
                          np.asarray(d["rv"] * k3, float), fee, float(N))
        nz = u > 0
        ser = pd.Series(p[nz], index=d["bi"][nz])
        mse = pd.Series(m[nz], index=d["bi"][nz])
        # 청산: 배정자본 대비 손실 100% 초과
        liq = mse >= 1.0
        ser = ser.where(~liq, -1.0)
        eqs.append((ser, w[i], int(liq.sum())))
        allm.append(mse)
    # 다리별 독립 복리 → 총자본 = sum(w_i * eq_i)
    idx = sorted(set().union(*[s.index for s, _, _ in eqs]))
    cur = np.array([w_ for _, w_, _ in eqs], float)
    hist = []
    for t in idx:
        for i, (s_, _, _) in enumerate(eqs):
            if t in s_.index:
                v = s_.loc[t]
                v = v.iloc[0] if hasattr(v, "iloc") and not np.isscalar(v) else v
                cur[i] *= (1 + v)
                if cur[i] < 1e-9:
                    cur[i] = 0.0
        hist.append(cur.sum())
    eq = pd.Series(hist, index=pd.DatetimeIndex(idx))
    liqs = sum(l for _, _, l in eqs)
    return eq, liqs, pd.concat(allm).max()


P("[1] 정확한 자본 배분 적용 — 분할·명목별 (4다리 각 25%)")
rows = []
for split in ((3.0, 4.5, 6.0), (4.0, 5.5, 7.0), (2.5, 4.0, 5.5), (5.0, 6.5, 8.0)):
    for N in (1, 2, 3, 5, 8):
        eq, liqs, mx = portfolio(split, N)
        if len(eq) < 100:
            continue
        yrs = (eq.index[-1] - eq.index[0]).days / 365
        te = eq[eq.index >= TE]
        base = te.iloc[0]
        yrs2 = (te.index[-1] - te.index[0]).days / 365
        rows.append((str(split), N, len(eq), liqs,
                     (eq.iloc[-1] ** (1 / yrs) - 1) * 100 if eq.iloc[-1] > 0 else -100,
                     float((eq / eq.cummax() - 1).min()) * 100,
                     ((te.iloc[-1] / base) ** (1 / yrs2) - 1) * 100 if te.iloc[-1] > 0 else -100,
                     float((te / te.cummax() - 1).min()) * 100,
                     len(eq) / ((eq.index[-1] - eq.index[0]).days / 7)))
P(pd.DataFrame(rows, columns=["분할", "명목", "진입봉수", "청산", "전체연%", "전체MDD%", "테스트연%", "테스트MDD%", "주당"]).round(1).to_string(index=False))

P("\n[2] 최적안 상세 — 분할 (4.0,5.5,7.0), 명목 3배")
eq, liqs, mx = portfolio((4.0, 5.5, 7.0), 3)
yrs = (eq.index[-1] - eq.index[0]).days / 365
P(f"  전체 {eq.iloc[-1]:.1f}배 (연 {(eq.iloc[-1]**(1/yrs)-1)*100:+.1f}%) MDD {float((eq/eq.cummax()-1).min())*100:.1f}% | 청산 {liqs}건")
te = eq[eq.index >= TE]; yrs2 = (te.index[-1] - te.index[0]).days / 365
P(f"  테스트 {te.iloc[-1]/te.iloc[0]:.2f}배 (연 {((te.iloc[-1]/te.iloc[0])**(1/yrs2)-1)*100:+.1f}%) MDD {float((te/te.cummax()-1).min())*100:.1f}%")
yr = eq.resample("YE").last() / eq.resample("YE").first() - 1
P("  연도별: " + " | ".join(f"{i.year} {v*100:+.0f}%" for i, v in yr.items()))
mon = eq.resample("ME").last() / eq.resample("ME").first() - 1
l12 = mon.tail(12)
P("  최근 12개월: " + " | ".join(f"{i.strftime('%y-%m')} {v*100:+.1f}%" for i, v in l12.items()))
P(f"  최근 12개월 합계 {(np.prod(1+l12)-1)*100:+.1f}% | 월평균 {l12.mean()*100:+.2f}% | 양수 {int((l12>0).sum())}/12")

P("\n[3] 비용 민감도 (분할 4.0/5.5/7.0, 명목 3배)")
for fee in (7, 12, 20, 30):
    eq, liqs, mx = portfolio((4.0, 5.5, 7.0), 3, fee=fee)
    yrs = (eq.index[-1] - eq.index[0]).days / 365
    te = eq[eq.index >= TE]; yrs2 = (te.index[-1] - te.index[0]).days / 365
    P(f"  {fee}bp: 전체 연 {(eq.iloc[-1]**(1/yrs)-1)*100:+7.1f}% MDD {float((eq/eq.cummax()-1).min())*100:6.1f}% | 테스트 연 {((te.iloc[-1]/te.iloc[0])**(1/yrs2)-1)*100:+7.1f}%")
