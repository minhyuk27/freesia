"""봉 경계 위상(phase) 연구 — UTC 정각 고정을 풀면 (a) 엣지가 경계의 우연인지 검증되고
(b) 진입 기회가 배수로 늘어 표본·자본회전이 커진다.
사용: python research/phase.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
EPS = 0.0002; MMR = 0.004
TE = pd.Timestamp("2024-01-01", tz="UTC"); REC = pd.Timestamp("2025-01-01", tz="UTC")
SYM = ("BTCUSDT", "ETHUSDT")
RAW = {}
for s in SYM:
    k = store.load("klines_1m", s).set_index("ts")
    RAW[s] = dict(o=k.open.values, h=k.high.values, l=k.low.values, c=k.close.values, idx=k.index)


def bars_off(s, B, off):
    """위상 off(분)만큼 어긋난 봉. off < B."""
    d = RAW[s]; c = d["c"]
    n = (len(c) - off) // B
    st = off + np.arange(n)*B
    bc = c[st + B - 1]
    rv = pd.Series(np.log(bc)).diff().rolling(20).std().shift(1).values
    return st, rv


@numba.njit(cache=True)
def ev_off(o, h, l, c, st, B, x, eps, fm, ft):
    n = len(st)
    ei = np.full(n, -1, np.int64); ex = np.full(n, -1, np.int64)
    rr = np.zeros(n); mm = np.zeros(n); k = 0
    for b in range(n):
        if np.isnan(x[b]) or x[b] <= 0: continue
        s0 = st[b]; e = s0 + B
        if e > len(c): break
        O = o[s0]; C = c[e-1]; T = O*(1-x[b])
        f = -1; lo_ = 1e18
        for t in range(s0, e):
            if f < 0:
                if l[t] <= T*(1-eps): f = t
                else: continue
            if l[t] < lo_: lo_ = l[t]
        if f < 0: continue
        ei[k] = f; ex[k] = e-1
        rr[k] = (C/T - 1.0) - (fm+ft)/1e4
        mm[k] = 1.0 - lo_/T
        k += 1
    return ei[:k], ex[:k], rr[:k], mm[:k]


def build(barlist, depths, offs_per_bar, eps=EPS, fm=2.0, ft=5.0):
    E = []
    for s in SYM:
        d = RAW[s]
        for B in barlist:
            noff = offs_per_bar
            for j in range(noff):
                off = int(round(B*j/noff))
                st, rv = bars_off(s, B, off)
                for k, wgt in depths.items():
                    a_, b_, r_, m_ = ev_off(d["o"], d["h"], d["l"], d["c"], st.astype(np.int64), B,
                                            np.asarray(rv*k, float), eps, fm, ft)
                    key = (s, B, off, k)
                    for a, bb, r, m in zip(a_, b_, r_, m_):
                        E.append((int(a), int(bb), float(r), float(m), d["idx"][int(a)], key,
                                  float(rv[np.searchsorted(st, a, "right")-1]), wgt))
    E.sort(key=lambda z: z[0])
    return E


def sim(E, w, Nmax, lo=0.2, hi=3.0, warm=30):
    cap = 1.0; open_ = []; rvh = {}; liq = 0; ts_l = []; eq_l = []; fl_l = []
    for a, b, r, m, ts, key, rvb, wgt in E:
        for p in [p for p in open_ if p[0] <= a]:
            cap += p[1]*p[2]; open_.remove(p)
        if cap <= 1e-9: break
        if not np.isfinite(rvb) or rvb <= 0: continue
        rh = rvh.setdefault(key, [])
        f = float(np.clip(np.median(rh)/rvb, lo, hi)) if len(rh) >= warm else lo
        rh.append(rvb)
        n_ = min(cap*w*wgt*f, max(cap*Nmax - sum(p[1] for p in open_), 0.0))
        if n_ <= 1e-9: continue
        if n_*m + sum(p[1]*p[3] for p in open_) >= cap*(1-MMR):
            cap = 0.0; liq += 1; break
        open_.append((b, n_, r, m))
        ts_l.append(ts); eq_l.append(cap); fl_l.append(cap - sum(p[1]*p[3] for p in open_))
    for p in open_: cap += p[1]*p[2]
    if not ts_l: return None, None, liq
    ts_l.append(E[-1][4]); eq_l.append(cap); fl_l.append(cap)
    eq = pd.Series(eq_l, index=pd.DatetimeIndex(ts_l)).groupby(level=0).last().resample("D").last().ffill()
    fl = pd.Series(fl_l, index=pd.DatetimeIndex(ts_l)).groupby(level=0).min().resample("D").min().ffill()
    return eq, fl, liq


def show(eq, fl, liq, lab, quiet=False):
    if eq is None or eq.iloc[-1] <= 1e-6:
        if not quiet: P(f"  {lab:34s} 파산 (청산 {liq})")
        return None
    y = (eq.index[-1]-eq.index[0]).days/365
    cg = (eq.iloc[-1]**(1/y)-1)*100
    mdd = float((eq/eq.cummax()-1).min())*100
    fld = float((fl.reindex(eq.index).ffill()/eq.cummax()-1).min())*100
    sg = lambda t0: (lambda s: ((s.iloc[-1]/s.iloc[0])**(365/max((s.index[-1]-s.index[0]).days,1))-1)*100 if len(s) > 60 else np.nan)(eq[eq.index >= t0])
    if not quiet:
        P(f"  {lab:34s} 연 {cg:+7.0f}% MDD {mdd:6.1f}%(최악 {fld:6.1f}%) | 24~ {sg(TE):+7.0f}% | 25~ {sg(REC):+7.0f}% | 청산 {liq}")
    return cg, mdd, fld, sg(TE), sg(REC), eq

BL = (240, 360, 480, 720, 960, 1440)
DEP = {8.0: 1.0, 12.0: 3.0}

P("[1] 위상 단독 검정 — 4시간봉을 0/1/2/3시간 어긋나게 했을 때 거래당 수익이 유지되나")
for off_h in (0, 1, 2, 3):
    accs = []
    for s in SYM:
        d = RAW[s]
        st, rv = bars_off(s, 240, off_h*60)
        for k in (8.0, 12.0):
            a_, b_, r_, m_ = ev_off(d["o"], d["h"], d["l"], d["c"], st.astype(np.int64), 240,
                                    np.asarray(rv*k, float), EPS, 2.0, 5.0)
            accs.append(r_)
    r = np.concatenate(accs)
    P(f"  4h 위상 {off_h}시간: n={len(r):4d} 평균 {r.mean()*1e4:+7.0f}bp 중앙 {np.median(r)*1e4:+6.0f}bp 승률 {(r>0).mean():.3f}")

P("\n[2] 위상 다중화 — 봉당 위상 개수를 늘리면")
for npo in (1, 2, 3, 4, 6):
    E = build(BL, DEP, npo)
    r = np.array([e[2] for e in E])
    P(f"  위상 {npo}개/봉: 이벤트 {len(E):5d}건 ({len(E)/6.7:.0f}/년) 평균 {r.mean()*1e4:+6.0f}bp 승률 {(r>0).mean():.3f}")
    for w, Nm in ((2.0, 5.0), (3.0, 5.0), (5.0, 5.0), (3.0, 8.0), (5.0, 8.0)):
        eq, fl, liq = sim(E, w, Nm)
        show(eq, fl, liq, f"    w={w:.0f} Nmax={Nm:.0f}")

P("\n[3] 최적 위상 구성에서 안전여유 제약(−70% 초과) 최적점")
E = build(BL, DEP, 4)
rows = []
for w in (1.0, 1.5, 2.0, 3.0, 5.0, 8.0):
    for Nm in (3.0, 5.0, 8.0, 12.0):
        if Nm < w: continue
        eq, fl, liq = sim(E, w, Nm)
        r = show(eq, fl, liq, "", quiet=True)
        if r: rows.append((w, Nm, *[round(x, 1) for x in r[:5]]))
t = pd.DataFrame(rows, columns=["w", "Nmax", "연%", "MDD%", "최악%", "24~%", "25~%"])
P(t.to_string(index=False))
sel = t[t["최악%"] > -70].sort_values("연%", ascending=False)
if len(sel):
    w, Nm = float(sel.iloc[0].w), float(sel.iloc[0].Nmax)
    P(f"\n★ 안전제약 통과 최적: w={w:.1f} Nmax={Nm:.0f}")
    eq, fl, liq = sim(E, w, Nm)
    r = show(eq, fl, liq, "  최종")
    dly = r[5]
    yr = dly.resample("YE").last()/dly.resample("YE").first()-1
    mon = dly.resample("ME").last()/dly.resample("ME").first()-1
    P("   연도별: " + " | ".join(f"{i.year} {v*100:+.0f}%" for i, v in yr.items()))
    P("   최근 12개월: " + " | ".join(f"{i.strftime('%y-%m')} {v*100:+.1f}%" for i, v in mon.tail(12).items()))
    l12 = mon.tail(12)
    P(f"   최근 12개월 누적 {(np.prod(1+l12)-1)*100:+.1f}% 양수 {int((l12>0).sum())}/12 | 월승률 {float((mon>0).mean()):.3f}")
