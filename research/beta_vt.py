"""완전히 새로운 계열 — 변동성 타겟 레버리지 롱 (+추세 필터). 펀딩 비용 포함, 청산 판정 포함.
지금까지 '그냥 베타'라며 기각했던 것을 수익원으로 정면 검증한다.
신호는 전부 shift(1) — 룩어헤드 없음.
사용: python research/beta_vt.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
SYM = ("BTCUSDT", "ETHUSDT")
TE = pd.Timestamp("2024-01-01", tz="UTC"); REC = pd.Timestamp("2025-01-01", tz="UTC")
MMR = 0.004

D = {}
for s in SYM:
    k = store.load("klines_1m", s).set_index("ts")
    d = pd.DataFrame({"o": k.open, "h": k.high, "l": k.low, "c": k.close})
    dd = d.resample("D").agg({"o": "first", "h": "max", "l": "min", "c": "last"}).dropna()
    f = store.load("funding", s).set_index("ts").rate
    fd = f.resample("D").sum().reindex(dd.index).fillna(0.0)   # 일별 펀딩 합(롱이 지불하면 양수)
    dd["fund"] = fd
    D[s] = dd
    P(f"{s}: 일봉 {len(dd)}일 {dd.index[0].date()}~{dd.index[-1].date()} | 펀딩 일평균 {fd.mean()*1e4:+.2f}bp")


@numba.njit(cache=True)
def sim(o, h, l, c, fund, expo, fee_bp, mmr):
    """expo[t]: t일 시작 시점의 목표 명목/자본 배수(전일까지 정보로 계산).
    일중 저가로 청산 판정. 펀딩은 명목에 비례해 차감."""
    n = len(c)
    cap = 1.0
    eq = np.zeros(n); eq[0] = 1.0
    prev = 0.0
    liq = 0
    for t in range(1, n):
        if cap <= 0.0:
            eq[t] = 0.0; continue
        e = expo[t]
        if e < 0.0: e = 0.0
        # 리밸런싱 비용
        turn = abs(e - prev)
        cap -= cap*turn*fee_bp/1e4
        notional = cap*e
        # 일중 청산 판정: 저가까지 갔을 때 평가손실
        loss = notional*(1.0 - l[t]/o[t])
        if e > 0.0 and loss >= cap*(1.0-mmr):
            cap = 0.0; liq += 1; eq[t] = 0.0; continue
        # 일간 손익 + 펀딩
        cap += notional*(c[t]/o[t] - 1.0) - notional*fund[t]
        if cap < 0.0: cap = 0.0
        prev = e
        eq[t] = cap
    return eq, liq


def expo_series(dd, N, vt, tgt, lo, hi, trend, warm=60):
    c = dd.c.values
    lr = pd.Series(np.log(c)).diff()
    rv = lr.rolling(30).std()*np.sqrt(365)       # 연율화 실현변동성
    rv = rv.shift(1)                              # 룩어헤드 방지
    e = np.full(len(c), float(N))
    if vt:
        f = np.clip(tgt/rv.values, lo, hi)
        e = N*f
    if trend > 0:
        ma = pd.Series(c).rolling(trend).mean().shift(1).values
        cp = pd.Series(c).shift(1).values
        e = np.where(cp > ma, e, 0.0)
    e[:warm] = 0.0
    return np.nan_to_num(e, nan=0.0)


def rep(eq, idx, lab, liq):
    s = pd.Series(eq, index=idx)
    if s.iloc[-1] <= 1e-6:
        P(f"  {lab:38s} 파산 (청산 {liq})"); return None
    y = (s.index[-1]-s.index[0]).days/365
    cg = (s.iloc[-1]**(1/y)-1)*100
    mdd = float((s/s.cummax()-1).min())*100
    sg = lambda t0: (lambda x: ((x.iloc[-1]/x.iloc[0])**(365/max((x.index[-1]-x.index[0]).days,1))-1)*100 if len(x) > 60 else np.nan)(s[s.index >= t0])
    mo = (s.resample("ME").last()/s.resample("ME").first()-1).dropna()
    P(f"  {lab:38s} 연 {cg:+8.1f}% MDD {mdd:6.1f}% | 24~ {sg(TE):+7.1f}% | 25~ {sg(REC):+7.1f}% | "
      f"월승률 {float((mo>0).mean()):.2f} 월중앙 {mo.median()*100:+5.1f}%")
    return cg, mdd, s, mo


P("\n[1] 기준 — 단순 매수보유와 고정 레버리지 (펀딩 포함)")
for s in SYM:
    dd = D[s]
    for N in (1.0, 1.5, 2.0, 3.0, 5.0):
        e = expo_series(dd, N, False, 0, 0, 0, 0)
        eq, liq = sim(dd.o.values, dd.h.values, dd.l.values, dd.c.values, dd.fund.values, e, 5.0, MMR)
        rep(eq, dd.index, f"{s[:3]} 고정 {N:.1f}x", liq)

P("\n[2] 변동성 타겟 (목표 연변동성 × 배수 제한)")
for s in SYM:
    P(f"  ── {s} ──")
    dd = D[s]
    for tgt in (0.4, 0.6, 0.8, 1.0):
        for N, hi in ((1.0, 3.0), (1.0, 5.0), (2.0, 3.0), (2.0, 5.0)):
            e = expo_series(dd, N, True, tgt, 0.2, hi, 0)
            eq, liq = sim(dd.o.values, dd.h.values, dd.l.values, dd.c.values, dd.fund.values, e, 5.0, MMR)
            rep(eq, dd.index, f"목표변동성{tgt:.1f} N={N:.0f} 상한{hi:.0f} (평균노출 {e[60:].mean():.2f}x)", liq)

P("\n[3] 추세 필터 추가 (이동평균 위에서만 보유)")
for s in SYM:
    P(f"  ── {s} ──")
    dd = D[s]
    for tr in (20, 50, 100, 200):
        for tgt, N, hi in ((0.6, 2.0, 5.0), (0.8, 2.0, 5.0), (0.6, 3.0, 5.0)):
            e = expo_series(dd, N, True, tgt, 0.2, hi, tr)
            eq, liq = sim(dd.o.values, dd.h.values, dd.l.values, dd.c.values, dd.fund.values, e, 5.0, MMR)
            rep(eq, dd.index, f"MA{tr} 목표{tgt:.1f} N={N:.0f} (노출 {e[60:].mean():.2f}x)", liq)

P("\n[4] BTC+ETH 결합 (각 절반 배정)")
bt, et = D["BTCUSDT"], D["ETHUSDT"]
ix = bt.index.intersection(et.index)
for tr in (0, 50, 100, 200):
    for tgt, N, hi in ((0.6, 2.0, 5.0), (0.8, 2.0, 5.0), (0.6, 3.0, 5.0), (0.8, 3.0, 5.0)):
        eqs = []
        liqs = 0
        for s in SYM:
            dd = D[s].reindex(ix).dropna()
            e = expo_series(dd, N, True, tgt, 0.2, hi, tr)
            eq, liq = sim(dd.o.values, dd.h.values, dd.l.values, dd.c.values, dd.fund.values, e, 5.0, MMR)
            eqs.append(pd.Series(eq, index=dd.index)); liqs += liq
        comb = (eqs[0]+eqs[1])/2
        rep(comb.values, comb.index, f"결합 MA{tr} 목표{tgt:.1f} N={N:.0f}", liqs)
