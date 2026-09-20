"""절대 손절로 MAE 상한을 고정 → 명목 배수 확대. 앞선 rv비례 손절과 달리 최악 몇 건만 건드린다.
사용: python research/hardstop.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
import exit_tier as X
from absfloor import met
from exit_tier import RAW, SYM, prep, CACHE
from freesia import store
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
BL6 = (240, 360, 480, 720, 960, 1440)
DEP = {10.0: 1.0, 14.0: 3.0, 20.0: 6.0}
FLO = {10.0: 0.05, 14.0: 0.07, 20.0: 0.10}
MMR = 0.004
TE = pd.Timestamp("2024-01-01", tz="UTC"); REC = pd.Timestamp("2025-01-01", tz="UTC")


@numba.njit(cache=True)
def ev_hs(o, h, l, c, st, B, x, ext, stop, eps, fm, ft):
    """stop>0: 진입가 대비 stop 비율 하락 시 시장가 손절. MAE가 stop으로 상한."""
    n = len(st)
    ei = np.full(n, -1, np.int64); ex = np.full(n, -1, np.int64)
    rr = np.zeros(n); mm = np.zeros(n); sf = np.zeros(n); k = 0
    for j in range(n):
        if np.isnan(x[j]) or x[j] <= 0: continue
        s0 = st[j]; e = s0+B
        if e > len(c): break
        T = o[s0]*(1-x[j])
        f = -1; lo_ = 1e18; hit = -1
        for t in range(s0, e):
            if f < 0:
                if l[t] <= T*(1-eps): f = t
                else: continue
            if l[t] < lo_: lo_ = l[t]
            if stop > 0.0 and l[t] <= T*(1.0-stop):
                hit = t; break
        if f < 0: continue
        end = e-1
        if hit < 0 and ext > 0.0 and (f-s0) > B*(1.0-ext):
            e2 = e+B
            if e2 <= len(c):
                end = e2-1
                for t in range(e, e2):
                    if l[t] < lo_: lo_ = l[t]
                    if stop > 0.0 and l[t] <= T*(1.0-stop):
                        hit = t; break
        if hit >= 0:
            end = hit
            px = T*(1.0-stop)
            if lo_ < T*(1.0-stop): lo_ = T*(1.0-stop)
            sf[k] = 1.0
        else:
            px = c[end]
        ei[k] = f; ex[k] = end
        rr[k] = (px/T-1.0) - (fm+ft)/1e4
        mm[k] = 1.0 - lo_/T
        k += 1
    return ei[:k], ex[:k], rr[:k], mm[:k], sf[:k]


def build_hs(stop, ext=0.10, eps=0.0002, fm=2.0, ft=5.0):
    E = []; ns = 0
    for s in SYM:
        d = RAW[s]
        for B in BL6:
            if (s, B) not in CACHE: CACHE[(s, B)] = prep(s, B)
            st, rv = CACHE[(s, B)]
            for k, wgt in DEP.items():
                xx = np.maximum(rv*k, FLO[k]); xx = np.where(np.isnan(rv), np.nan, xx)
                a_, b_, r_, m_, f_ = ev_hs(d["o"], d["h"], d["l"], d["c"], st.astype(np.int64), B,
                                           np.asarray(xx, float), ext, stop, eps, fm, ft)
                ns += int(f_.sum())
                for a, bb, r, m in zip(a_, b_, r_, m_):
                    E.append((int(a), int(bb), float(r), float(m), d["idx"][int(a)], (s, B, k), float(rv[int(a)//B]), wgt))
    E.sort(key=lambda z: z[0])
    return E, ns


PMd = {s: store.load("premium_1m", s).set_index("ts").close for s in SYM}
def amp(E, th=0.10, a=1.5):
    out = []
    for x in E:
        ser = PMd[x[5][0]]; i = ser.index.searchsorted(x[4])
        f = 1.0
        if i > 1:
            pv = float(ser.iloc[i-1]); past = ser.iloc[max(0, i-1-525600):i-1]
            if len(past) > 10000 and float((past < pv).mean()) <= th: f = a
        out.append((x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]*f))
    return out


def scan(E, lab):
    r = np.array([x[2] for x in E]); mae = np.array([x[3] for x in E])
    best = None
    for w in (3, 5, 8, 12, 20, 30):
        for Nm in (5, 8, 12, 20, 30):
            if Nm < w: continue
            eq, fl, liq = X.sim(E, float(w), float(Nm)); m = met(eq, fl, liq)
            if m is None or liq: continue
            if best is None or m[0] > best[0]: best = (m[0], m[2], m[3], m[4], w, Nm, m[5])
    s = f"  {lab:22s} 거래당 {r.mean()*1e4:+5.0f}bp 승률 {(r>0).mean():.3f} MAE최대 {mae.max()*100:5.2f}% |"
    if best:
        mo = (1+best[0]/100)**(1/12)-1
        s += f" 최대 w={best[4]:2d}/{best[5]:2d} 연 {best[0]:+7.0f}%(월 {mo*100:+5.1f}%) 최악 {best[1]:6.1f}% 25~ {best[3]:+7.0f}%"
    else:
        s += " 무청산 구성 없음"
    P(s)
    return best


P("[1] 절대 손절 폭별 (증폭 없음)")
for stop in (0.0, 0.25, 0.20, 0.15, 0.12, 0.10, 0.08):
    E, ns = build_hs(stop)
    scan(E, f"손절 {'없음' if stop==0 else f'{stop*100:.0f}%'} (발동 {ns}건)")

P("\n[2] 절대 손절 + 프리미엄 증폭 1.5배")
for stop in (0.0, 0.20, 0.15, 0.12, 0.10):
    E, ns = build_hs(stop)
    scan(amp(E), f"손절 {'없음' if stop==0 else f'{stop*100:.0f}%'} +증폭")

P("\n[3] 최적 후보의 3단 성과 + 월 분포")
for stop in (0.0, 0.15, 0.12):
    E, ns = build_hs(stop); E = amp(E)
    P(f"\n  ── 손절 {'없음' if stop==0 else f'{stop*100:.0f}%'} + 증폭 (손절 발동 {ns}건) ──")
    for w, Nm, tag in ((1.0, 1.0, "무레버"), (3.0, 5.0, "중간"), (5.0, 8.0, "공격"), (8.0, 12.0, "초공격"), (12.0, 20.0, "극단")):
        eq, fl, liq = X.sim(E, w, Nm); m = met(eq, fl, liq)
        if m is None or liq: P(f"     {tag:6s} 파산"); continue
        mo = (1+m[0]/100)**(1/12)-1
        mon = (m[5].resample("ME").last()/m[5].resample("ME").first()-1).dropna(); mon = mon[mon != 0]
        P(f"     {tag:6s} 연 {m[0]:+8.0f}% (월 {mo*100:+5.1f}%) 최악 {m[2]:6.1f}% | 24~ {m[3]:+8.0f}% | 25~ {m[4]:+8.0f}% | "
          f"거래월 {len(mon)}/81 중앙 {mon.median()*100:+4.0f}% 월+80%↑ {int((mon>=0.8).sum())}개")
