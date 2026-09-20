"""z0.3(패닉) + 추세(상승장) 결합 — 시간대가 겹치지 않는 두 수익원.
추세 전략 MDD를 낮추는 변형도 함께. 워크포워드 포함.
사용: python research/combo_z3t.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
import exit_tier as X
from absfloor import met
from anchor import build_a
from beta_vt import D as DD, sim as tsim, expo_series, MMR
from freesia import store
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
SYM = ("BTCUSDT", "ETHUSDT")
TE = pd.Timestamp("2024-01-01", tz="UTC"); REC = pd.Timestamp("2025-01-01", tz="UTC")

Z = build_a("open", ext=0.10)
PMd = {s: store.load("premium_1m", s).set_index("ts").close for s in SYM}
ZA = []
for x in Z:
    ser = PMd[x[5][0]]; i = ser.index.searchsorted(x[4]); f = 1.0
    if i > 1:
        pv = float(ser.iloc[i-1]); past = ser.iloc[max(0, i-1-525600):i-1]
        if len(past) > 10000 and float((past < pv).mean()) <= 0.10: f = 1.5
    ZA.append((x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]*f))
P(f"z0.3+증폭 {len(ZA)}건")


def trend_eq(tr, tgt, N, hi=3.0, lo=0.2):
    eqs = []
    for s in SYM:
        dd = DD[s]
        e = expo_series(dd, N, True, tgt, lo, hi, tr)
        eq, liq = tsim(dd.o.values, dd.h.values, dd.l.values, dd.c.values, dd.fund.values, e, 5.0, MMR)
        eqs.append(pd.Series(eq, index=dd.index))
    ix = eqs[0].index.intersection(eqs[1].index)
    return (eqs[0].reindex(ix)+eqs[1].reindex(ix))/2


def zeq(w, Nm):
    eq, fl, liq = X.sim(ZA, w, Nm); m = met(eq, fl, liq)
    return (m[5], m[2], liq) if m else (None, None, liq)


def rep(s, lab):
    s = s.dropna()
    if len(s) < 100 or s.iloc[-1] <= 1e-9:
        P(f"  {lab:40s} 파산/무효"); return None
    y = (s.index[-1]-s.index[0]).days/365
    cg = (s.iloc[-1]**(1/y)-1)*100
    mdd = float((s/s.cummax()-1).min())*100
    sg = lambda t0: (lambda x: ((x.iloc[-1]/x.iloc[0])**(365/max((x.index[-1]-x.index[0]).days,1))-1)*100 if len(x) > 60 else np.nan)(s[s.index >= t0])
    mo = (s.resample("ME").last()/s.resample("ME").first()-1).dropna()
    P(f"  {lab:40s} 연 {cg:+7.1f}% MDD {mdd:6.1f}% | 24~ {sg(TE):+7.1f}% | 25~ {sg(REC):+7.1f}% | "
      f"월승률 {float((mo>0).mean()):.2f} 중앙 {mo.median()*100:+5.1f}% | 월+80%↑ {int((mo>=0.8).sum())}")
    return cg, mdd, s, mo


P("\n[1] 추세 전략 MDD 낮추기 — 목표변동성·상한 조정")
for tr in (30, 50, 70):
    for tgt in (0.3, 0.4, 0.5, 0.6):
        for N, hi in ((1.0, 2.0), (1.5, 2.0), (2.0, 2.0), (1.5, 3.0)):
            rep(trend_eq(tr, tgt, N, hi), f"MA{tr} 목표{tgt:.1f} N={N:.1f} 상한{hi:.0f}")

P("\n[2] z0.3 단독 (비교 기준)")
for w, Nm, tag in ((1.0, 1.0, "무레버"), (3.0, 5.0, "중간"), (5.0, 8.0, "공격")):
    s, fl, liq = zeq(w, Nm)
    if s is not None: rep(s, f"z0.3 {tag}")

P("\n[3] 결합 — 자본을 나눠 독립 운용")
TCFG = [(50, 0.4, 1.5, 2.0), (50, 0.5, 1.5, 2.0), (50, 0.3, 2.0, 2.0), (30, 0.4, 1.5, 2.0)]
best = None
for tr, tgt, N, hi in TCFG:
    te = trend_eq(tr, tgt, N, hi)
    for zw, zn, ztag in ((3.0, 5.0, "중간"), (5.0, 8.0, "공격")):
        ze, fl, liq = zeq(zw, zn)
        if ze is None: continue
        ix = te.index.intersection(ze.index)
        for a in (0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0):
            comb = a*te.reindex(ix).ffill() + (1-a)*ze.reindex(ix).ffill()
            r = rep(comb, f"MA{tr}목표{tgt:.1f}N{N:.1f} {a*100:.0f}% + z0.3{ztag} {(1-a)*100:.0f}%")
            if r and (best is None or r[0] > best[0]): best = (r[0], r[1], tr, tgt, N, hi, a, ztag, r[2], r[3])

if best:
    P(f"\n★ 최고 결합: MA{best[2]} 목표{best[3]:.1f} N={best[4]:.1f} 비중 {best[6]*100:.0f}% + z0.3 {best[7]} {(1-best[6])*100:.0f}%")
    P(f"   연 {best[0]:+.1f}% MDD {best[1]:.1f}%")
    s = best[8]; mo = best[9]
    yr = s.resample("YE").last()/s.resample("YE").first()-1
    P("   연도별: " + " | ".join(f"{i.year} {v*100:+.0f}%" for i, v in yr.items()))
    P(f"   월 분포: 중앙 {mo.median()*100:+.1f}% 평균 {mo.mean()*100:+.1f}% 최대 {mo.max()*100:+.0f}% 최소 {mo.min()*100:+.0f}%")
    for th in (0.1, 0.3, 0.5, 0.8):
        P(f"     월 +{th*100:.0f}% 이상 {int((mo>=th).sum())}개월 / {len(mo)}")

P("\n[4] 두 전략의 월별 상관 — 정말 겹치지 않나")
te = trend_eq(50, 0.4, 1.5, 2.0)
ze, _, _ = zeq(3.0, 5.0)
ix = te.index.intersection(ze.index)
mt = (te.reindex(ix).resample("ME").last()/te.reindex(ix).resample("ME").first()-1).dropna()
mz = (ze.reindex(ix).resample("ME").last()/ze.reindex(ix).resample("ME").first()-1).dropna()
j = pd.concat([mt.rename("추세"), mz.rename("z03")], axis=1).fillna(0)
P(f"  월별 상관 {j.추세.corr(j.z03):+.3f}")
P(f"  z0.3 무거래 달 {int((j.z03==0).sum())}개 중 추세 양수 {int(((j.z03==0)&(j.추세>0)).sum())}개, 그 달 추세 평균 {j[j.z03==0].추세.mean()*100:+.1f}%")
P(f"  추세 손실 달 {int((j.추세<0).sum())}개 중 z0.3 양수 {int(((j.추세<0)&(j.z03>0)).sum())}개")

P("\n[5] 추세 전략 워크포워드 (MA·목표변동성을 학습 구간에서 선택)")
for y in pd.date_range("2022-01-01", "2026-01-01", freq="YS", tz="UTC"):
    tr0, tr1, te1 = y-pd.DateOffset(years=2), y, y+pd.DateOffset(years=1)
    bw = None
    for tr in (20, 30, 50, 70, 100, 200):
        for tgt in (0.3, 0.4, 0.5, 0.6):
            s = trend_eq(tr, tgt, 1.5, 2.0)
            sl = s[(s.index >= tr0) & (s.index < tr1)]
            if len(sl) < 200 or sl.iloc[-1] <= 1e-9: continue
            c = (sl.iloc[-1]/sl.iloc[0])**(365/(sl.index[-1]-sl.index[0]).days)-1
            if bw is None or c > bw[0]: bw = (c, tr, tgt)
    if bw is None: continue
    s = trend_eq(bw[1], bw[2], 1.5, 2.0)
    sl = s[(s.index >= tr1) & (s.index < te1)]
    if len(sl) < 100: continue
    c = (sl.iloc[-1]/sl.iloc[0])**(365/(sl.index[-1]-sl.index[0]).days)-1
    P(f"  {y.year}: 학습최적 MA{bw[1]} 목표{bw[2]:.1f} → 검증 연 {c*100:+8.1f}% (MDD {float((sl/sl.cummax()-1).min())*100:.1f}%)")
