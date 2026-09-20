"""단일 포지션 집중 — 겹치는 신호를 받지 않고 한 번에 하나만, 전체 자본으로 큰 명목.
켈리 N*=8.5에 접근 가능한가. 청산은 MAE >= 1/N - mmr 로 판정(교차마진).
사용: python research/single_kelly.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from anchor import build_a
from freesia import store
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
SYM = ("BTCUSDT", "ETHUSDT")
TE = pd.Timestamp("2024-01-01", tz="UTC"); REC = pd.Timestamp("2025-01-01", tz="UTC")
MMR = 0.004
Z = build_a("open", ext=0.10)
PM = {s: store.load("premium_1m", s).set_index("ts").close for s in SYM}
PQ = []
for a, b, r, m, ts, key, rv, wg in Z:
    ser = PM[key[0]]; i = ser.index.searchsorted(ts)
    if i <= 1: PQ.append(np.nan); continue
    pv = float(ser.iloc[i-1]); past = ser.iloc[max(0, i-1-525600):i-1]
    PQ.append(float((past < pv).mean()) if len(past) > 10000 else np.nan)
PQ = np.array(PQ)
P(f"이벤트 {len(Z)}건")


def sim_single(N, maxpos=1, sel="first", amp_th=0.10, amp=1.0, wait=0):
    """maxpos: 동시 허용 포지션 수(각각 N/maxpos 명목). sel: first=먼저 온 신호, deep=가장 깊은 단계 우선."""
    cap = 1.0; open_ = []; ts_l = []; eq_l = []; fl_l = []; liq = 0; taken = 0; skipped = 0
    dist = 1.0/N - MMR
    for j, (a, b, r, m, ts, key, rv, wg) in enumerate(Z):
        for p in [p for p in open_ if p[0] <= a]:
            cap += p[1]*p[2]; open_.remove(p)
        if cap <= 1e-9: break
        if len(open_) >= maxpos:
            skipped += 1
            continue
        f = amp if (np.isfinite(PQ[j]) and PQ[j] <= amp_th) else 1.0
        notional = cap*(N/maxpos)*f
        # 교차마진: 기존 포지션 평가손실 + 이번 최대역행 합이 자본 초과 시 청산
        worst = notional*m + sum(p[1]*p[3] for p in open_)
        if worst >= cap*(1-MMR):
            cap = 0.0; liq += 1; break
        open_.append((b, notional, r, m))
        taken += 1
        ts_l.append(ts); eq_l.append(cap); fl_l.append(cap - sum(p[1]*p[3] for p in open_))
    for p in open_: cap += p[1]*p[2]
    if not ts_l: return None
    ts_l.append(Z[-1][4]); eq_l.append(cap); fl_l.append(cap)
    eq = pd.Series(eq_l, index=pd.DatetimeIndex(ts_l)).groupby(level=0).last().resample("D").last().ffill()
    fl = pd.Series(fl_l, index=pd.DatetimeIndex(ts_l)).groupby(level=0).min().resample("D").min().ffill()
    if eq.iloc[-1] <= 1e-6: return dict(bust=True, liq=liq, taken=taken)
    y = (eq.index[-1]-eq.index[0]).days/365
    sg = lambda t0: (lambda s: ((s.iloc[-1]/s.iloc[0])**(365/max((s.index[-1]-s.index[0]).days,1))-1)*100 if len(s) > 60 else np.nan)(eq[eq.index >= t0])
    mo = (eq.resample("ME").last()/eq.resample("ME").first()-1).dropna()
    mo = mo[mo != 0]
    return dict(bust=False, cagr=(eq.iloc[-1]**(1/y)-1)*100,
                mdd=float((eq/eq.cummax()-1).min())*100,
                floor=float((fl.reindex(eq.index).ffill()/eq.cummax()-1).min())*100,
                t24=sg(TE), t25=sg(REC), liq=liq, taken=taken, skipped=skipped,
                eq=eq, mon=mo, mult=eq.iloc[-1])


P("\n[1] 단일 포지션 집중 — 명목 N별")
P(f"  {'N':>5s} {'채택':>5s} {'스킵':>5s} {'전체연':>10s} {'월환산':>8s} {'MDD':>7s} {'최악여유':>8s} {'24~':>9s} {'25~':>9s} {'청산':>4s}")
for N in (2, 3, 4, 5, 6, 7, 8, 8.5, 9, 10):
    d = sim_single(N)
    if d is None: continue
    if d["bust"]:
        P(f"  {N:5.1f} {d['taken']:5d} {'':5s} {'파산':>10s} (청산 {d['liq']})"); continue
    mo = (1+d["cagr"]/100)**(1/12)-1
    P(f"  {N:5.1f} {d['taken']:5d} {d['skipped']:5d} {d['cagr']:+9.0f}% {mo*100:+7.1f}% {d['mdd']:6.1f}% {d['floor']:7.1f}% {d['t24']:+8.0f}% {d['t25']:+8.0f}% {d['liq']:4d}")

P("\n[2] 동시 포지션 허용 수 (총명목 N 고정, 각 N/maxpos)")
for N in (5, 8):
    for mp in (1, 2, 3, 5):
        d = sim_single(N, maxpos=mp)
        if d is None or d["bust"]:
            P(f"  N={N} 동시{mp}: 파산"); continue
        mo = (1+d["cagr"]/100)**(1/12)-1
        P(f"  N={N} 동시{mp}: 채택 {d['taken']:3d} 연 {d['cagr']:+8.0f}% (월 {mo*100:+5.1f}%) 최악 {d['floor']:6.1f}% 25~ {d['t25']:+8.0f}%")

P("\n[3] 프리미엄 증폭 결합 (단일 포지션)")
for N in (4, 5, 6, 7):
    for amp in (1.0, 1.3, 1.5):
        d = sim_single(N, amp=amp)
        if d is None or d["bust"]:
            P(f"  N={N} 증폭{amp}: 파산"); continue
        mo = (1+d["cagr"]/100)**(1/12)-1
        P(f"  N={N} 증폭{amp}: 연 {d['cagr']:+9.0f}% (월 {mo*100:+5.1f}%) 최악 {d['floor']:6.1f}% 24~ {d['t24']:+8.0f}% 25~ {d['t25']:+8.0f}%")

P("\n[4] 최적 후보 상세 — 월별 분포")
for N in (5, 6, 7):
    d = sim_single(N)
    if d is None or d["bust"]: continue
    mo = d["mon"]
    P(f"\n  ── N={N} (연 {d['cagr']:+.0f}%, 최종 {d['mult']:.3g}배, 최악여유 {d['floor']:.1f}%) ──")
    yr = d["eq"].resample("YE").last()/d["eq"].resample("YE").first()-1
    P("     연도별: " + " | ".join(f"{i.year} {v*100:+.0f}%" for i, v in yr.items()))
    P(f"     거래월 {len(mo)}/81 | 중앙 {mo.median()*100:+.0f}% 평균 {mo.mean()*100:+.0f}% 최대 {mo.max()*100:+.0f}%")
    for th in (0.3, 0.5, 0.8, 1.0):
        P(f"       월 +{th*100:.0f}% 이상: {int((mo>=th).sum())}개월 ({int((mo>=th).sum())/81*100:.0f}% of 81개월, {int((mo>=th).sum())/len(mo)*100:.0f}% of 거래월)")

P("\n[5] 안전 한계 — 단일 거래 최대역행 분포")
mae = np.array([x[3] for x in Z])
P(f"  MAE 평균 {mae.mean()*100:.2f}% 중앙 {np.median(mae)*100:.2f}% 95분위 {np.quantile(mae,0.95)*100:.2f}% "
  f"99분위 {np.quantile(mae,0.99)*100:.2f}% 최대 {mae.max()*100:.2f}%")
for N in (3, 4, 5, 6, 7, 8, 10):
    dist = 1.0/N - MMR
    P(f"  N={N:2d}: 청산거리 {dist*100:5.2f}% → 과거 {int((mae>=dist).sum())}건 청산 ({(mae>=dist).mean()*100:.1f}%)")
