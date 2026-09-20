"""z0.6b 최종 — 수수료와 체결엄격도를 분리한 민감도, MAE 상한 처리, z0.3 결합.
사용: python research/z06_final.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from z06b import build
from anchor import build_a
import exit_tier as X
from absfloor import met
from freesia import store
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
SYM = ("BTCUSDT", "ETHUSDT")
TE = pd.Timestamp("2024-01-01", tz="UTC"); REC = pd.Timestamp("2025-01-01", tz="UTC")
Q, KX, W, H = 0.005, 1.0, 30, 240

P("[1] 수수료 vs 체결엄격도 분리 (어느 쪽이 죽이나)")
P(f"  {'수수료':>16s} {'관통':>6s} {'n':>6s} {'평균bp':>8s} {'무레버 연':>10s} {'MDD':>7s}")
for fm, ft in ((2, 5), (3, 6), (4, 7)):
    for eps in (0.0, 0.0002, 0.0005, 0.001):
        E = build(Q, KX, W, H, 30, fm=fm, ft=ft, eps=eps)
        r = np.array([x[2] for x in E])
        eq, fl, liq = X.sim(E, 1.0, 1.0); m = met(eq, fl, liq)
        P(f"  메{fm}/테{ft:<11d} {eps*1e4:5.0f}bp {len(r):6d} {r.mean()*1e4:+8.2f} "
          f"{m[0] if m else float('nan'):+9.1f}% {m[1] if m else float('nan'):6.1f}%")

P("\n[2] MAE 꼬리 처리 — 보유 중 손실 상한(강제 청산)을 두면 레버리지가 열리나")
import numba
@numba.njit(cache=True)
def clip_mae(rr, mm, cap):
    n = len(rr)
    r2 = np.empty(n); m2 = np.empty(n)
    for i in range(n):
        if mm[i] >= cap:
            r2[i] = -cap - 7.0/1e4
            m2[i] = cap
        else:
            r2[i] = rr[i]; m2[i] = mm[i]
    return r2, m2

E0 = build(Q, KX, W, H, 30)
rr = np.array([x[2] for x in E0]); mm = np.array([x[3] for x in E0])
P(f"  원본 MAE: 평균 {mm.mean()*100:.2f}% 95분위 {np.quantile(mm,0.95)*100:.2f}% 99분위 {np.quantile(mm,0.99)*100:.2f}% 최대 {mm.max()*100:.2f}%")
for cap in (1.0, 0.15, 0.10, 0.07, 0.05, 0.03):
    r2, m2 = clip_mae(rr, mm, cap)
    E2 = [(x[0], x[1], float(r2[i]), float(m2[i]), x[4], x[5], x[6], x[7]) for i, x in enumerate(E0)]
    row = f"  손실상한 {'없음' if cap>=1 else f'{cap*100:.0f}%':>5s} 평균 {r2.mean()*1e4:+6.2f}bp |"
    for w, Nm, tag in ((1.0, 1.0, "무레버"), (2.0, 3.0, "저레버"), (3.0, 5.0, "중간"), (5.0, 8.0, "공격")):
        eq, fl, liq = X.sim(E2, w, Nm); m = met(eq, fl, liq)
        row += f" {tag} {'파산' if m is None else f'{m[0]:+6.1f}%'}"
    P(row)

P("\n[3] 저레버리지 구간 세밀 (손실상한 7%)")
r2, m2 = clip_mae(rr, mm, 0.07)
E2 = [(x[0], x[1], float(r2[i]), float(m2[i]), x[4], x[5], x[6], x[7]) for i, x in enumerate(E0)]
for w in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
    for Nm in (2.0, 3.0, 5.0):
        if Nm < w: continue
        eq, fl, liq = X.sim(E2, w, Nm); m = met(eq, fl, liq)
        if m is None: continue
        mo = (1+m[0]/100)**(1/12)-1
        mon = (m[5].resample("ME").last()/m[5].resample("ME").first()-1).dropna()
        P(f"  w={w:.1f} Nmax={Nm:.0f}: 연 {m[0]:+7.1f}% (월 {mo*100:+5.1f}%) MDD {m[1]:6.1f}% 최악 {m[2]:6.1f}% | "
          f"24~ {m[3]:+7.1f}% | 25~ {m[4]:+7.1f}% | 월승률 {float((mon>0).mean()):.2f}")

P("\n[4] z0.3과의 상관 · 결합")
Z = build_a("open", ext=0.10)
PMd = {s: store.load("premium_1m", s).set_index("ts").close for s in SYM}
ZA = []
for x in Z:
    ser = PMd[x[5][0]]; i = ser.index.searchsorted(x[4]); f = 1.0
    if i > 1:
        pv = float(ser.iloc[i-1]); past = ser.iloc[max(0, i-1-525600):i-1]
        if len(past) > 10000 and float((past < pv).mean()) <= 0.10: f = 1.5
    ZA.append((x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]*f))

m6 = pd.DataFrame([(x[4], x[2]) for x in E2], columns=["ts", "r"]).set_index("ts").resample("ME").sum()
m3 = pd.DataFrame([(x[4], x[2]) for x in ZA], columns=["ts", "r"]).set_index("ts").resample("ME").sum()
j = pd.concat([m3.r.rename("z03"), m6.r.rename("z06")], axis=1).fillna(0)
P(f"  월별 상관 {j.z03.corr(j.z06):+.3f}")
P(f"  z0.3 무거래 달 {int((j.z03==0).sum())}개 중 z0.6 양수 {int(((j.z03==0)&(j.z06>0)).sum())}개, 평균 {j[j.z03==0].z06.mean()*1e4:+.0f}bp")

def eqof(E, w, Nm):
    eq, fl, liq = X.sim(E, w, Nm); m = met(eq, fl, liq)
    return m[5] if m else None

z6 = eqof(E2, 2.0, 3.0)
for zw, zn, ztag in ((3.0, 5.0, "중간"), (5.0, 8.0, "공격")):
    z3 = eqof(ZA, zw, zn)
    if z3 is None or z6 is None: continue
    ix = z3.index.intersection(z6.index)
    P(f"  ── z0.3 {ztag} + z0.6(저레버 2x/3x) ──")
    for a in (0.0, 0.25, 0.5, 0.75, 1.0):
        comb = (a*z6.reindex(ix).ffill() + (1-a)*z3.reindex(ix).ffill()).dropna()
        y = (comb.index[-1]-comb.index[0]).days/365
        cg = (comb.iloc[-1]**(1/y)-1)*100
        mdd = float((comb/comb.cummax()-1).min())*100
        sg = lambda t0: (lambda x: ((x.iloc[-1]/x.iloc[0])**(365/max((x.index[-1]-x.index[0]).days,1))-1)*100 if len(x) > 60 else np.nan)(comb[comb.index >= t0])
        mon = (comb.resample("ME").last()/comb.resample("ME").first()-1).dropna()
        P(f"    z0.6 {a*100:3.0f}% : 연 {cg:+7.1f}% MDD {mdd:6.1f}% | 24~ {sg(TE):+7.1f}% | 25~ {sg(REC):+7.1f}% | "
          f"월승률 {float((mon>0).mean()):.2f} 중앙 {mon.median()*100:+5.1f}%")
