"""공유 자본 시뮬레이터 — 다리별 고정배분 대신 전체 자본을 공유. 교차마진 청산 판정.
이벤트: (진입분, 청산분, 수익률, 최대역행). 시간순 처리하며 자본·열린포지션 추적.
진입 시 명목 = 현재자본 × w. 동시 명목 합 상한 Nmax. 청산: 미실현 손실 합 ≥ 자본.
사용: python research/shared_cap.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from lib_dip import bars
TE = pd.Timestamp("2024-01-01", tz="UTC")
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
LEGS = [(s, b) for s in ("BTCUSDT", "ETHUSDT") for b in (60, 240, 480)]
D = {}
for key in LEGS:
    d = bars(key[0], key[1]); s = pd.Series
    d["rv"] = (s(np.log(d["bc"])).diff().rolling(20).std()).shift(1).values
    D[key] = d


@numba.njit(cache=True)
def events(o, h, l, c, B, x1, x2, x3, fee, off):
    """반환: 진입분(글로벌), 청산분, 수익률, 최대역행비율 (모두 명목 1단위 기준)"""
    n = len(c) // B
    ei = np.full(n, -1, np.int64); ex = np.full(n, -1, np.int64)
    rr = np.zeros(n); mm = np.zeros(n)
    k = 0
    for b in range(n):
        if np.isnan(x1[b]) or x1[b] <= 0:
            continue
        s = b * B; e = s + B
        O = o[s]; C = c[e - 1]
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
        tot = 0.0; cnt = 0.0; wavg = 0.0
        for tt, ff in ((t1, f1), (t2, f2), (t3, f3)):
            if ff >= 0:
                tot += ((C / tt - 1.0) - fee / 1e4) / 3.0; cnt += 1.0; wavg += tt
        avg = wavg / cnt
        lo_ = t1
        for t in range(f1, e):
            if l[t] < lo_:
                lo_ = l[t]
        ei[k] = f1 + off; ex[k] = e - 1 + off
        rr[k] = tot * 3.0 / 3.0 * (cnt / cnt)      # 평균 단계 수익(명목 1단위=배정액 전체)
        rr[k] = tot
        mm[k] = (1.0 - lo_ / avg) * (cnt / 3.0)
        k += 1
    return ei[:k], ex[:k], rr[:k], mm[:k]


def build(split, fee=7.0):
    E = []
    for key in LEGS:
        d = D[key]
        k1, k2, k3 = split
        ei, ex, rr, mm = events(d["o"], d["h"], d["l"], d["c"], d["B"],
                                np.asarray(d["rv"] * k1, float), np.asarray(d["rv"] * k2, float),
                                np.asarray(d["rv"] * k3, float), fee, 0)
        ts = d["idx"]
        for a, b, r, m in zip(ei, ex, rr, mm):
            E.append((int(a), int(b), float(r), float(m), ts[int(a)]))
    E.sort(key=lambda z: z[0])
    return E


def sim(E, w, Nmax, mmr=0.004):
    """w: 진입당 명목/자본 비율. Nmax: 동시 명목 합 상한(자본배수)."""
    cap = 1.0
    open_ = []          # (청산분, 명목, 수익률, 최대역행)
    eqs = []; times = []
    liq = 0
    for a, b, r, m, ts in E:
        # 만기 도달한 포지션 정산
        done = [p for p in open_ if p[0] <= a]
        for p in done:
            cap += p[1] * p[2]
            open_.remove(p)
        if cap <= 1e-9:
            break
        cur_notional = sum(p[1] for p in open_)
        avail = max(cap * Nmax - cur_notional, 0.0)
        notional = min(cap * w, avail)
        if notional <= 1e-9:
            continue
        # 청산 판정: 이 포지션의 최대 평가손실 + 기존 포지션 최대 손실 합이 자본 초과
        worst = notional * m + sum(p[1] * p[3] for p in open_)
        if worst >= cap * (1 - mmr):
            cap = 0.0
            liq += 1
            break
        open_.append((b, notional, r, m))
        eqs.append(cap + sum(p[1] * p[2] for p in open_) * 0)   # 실현 기준
        times.append(ts)
    for p in open_:
        cap += p[1] * p[2]
    eqs.append(cap); times.append(E[-1][4] if E else pd.Timestamp("2026-01-01", tz="UTC"))
    return pd.Series(eqs, index=pd.DatetimeIndex(times)), liq


P("[1] 공유 자본 — 진입당 명목 비율 w × 동시 명목 상한 Nmax (분할 5.0/6.5/8.0)")
E = build((5.0, 6.5, 8.0))
P(f"  총 진입 이벤트 {len(E)}건")
rows = []
for w in (0.5, 1.0, 2.0, 3.0, 5.0, 8.0):
    for Nmax in (1, 2, 3, 5, 8, 12):
        if w > Nmax:
            continue
        eq, liq = sim(E, w, Nmax)
        if eq.iloc[-1] <= 1e-6:
            rows.append((w, Nmax, "파산", np.nan, np.nan, liq)); continue
        yrs = (eq.index[-1] - eq.index[0]).days / 365
        te = eq[eq.index >= TE]
        rows.append((w, Nmax, f"{(eq.iloc[-1]**(1/yrs)-1)*100:+.0f}%",
                     float((eq / eq.cummax() - 1).min()) * 100,
                     f"{((te.iloc[-1]/te.iloc[0])**(1/((te.index[-1]-te.index[0]).days/365))-1)*100:+.0f}%" if len(te) > 30 else "-",
                     liq))
P(pd.DataFrame(rows, columns=["진입명목w", "동시상한Nmax", "전체연", "전체MDD%", "테스트연", "청산"]).to_string(index=False))

P("\n[2] 분할 설정별 (w=3, Nmax=8)")
for sp in ((3.0, 4.5, 6.0), (4.0, 5.5, 7.0), (5.0, 6.5, 8.0), (6.0, 7.5, 9.0)):
    E2 = build(sp)
    eq, liq = sim(E2, 3.0, 8)
    if eq.iloc[-1] <= 1e-6:
        P(f"  {str(sp):18s} 파산(청산 {liq})"); continue
    yrs = (eq.index[-1] - eq.index[0]).days / 365
    te = eq[eq.index >= TE]
    P(f"  {str(sp):18s} 이벤트 {len(E2):5d} | 전체 연 {(eq.iloc[-1]**(1/yrs)-1)*100:+8.0f}% MDD {float((eq/eq.cummax()-1).min())*100:6.1f}% | "
      f"테스트 연 {((te.iloc[-1]/te.iloc[0])**(1/((te.index[-1]-te.index[0]).days/365))-1)*100:+8.0f}% | 청산 {liq}")
