"""z0.3 제대로 된 플라시보 + 표본 확대(봉 길이·깊이 세분화).
플라시보: 같은 다리·같은 보유길이로 '무작위 분'에 진입. 조건의 가치만 분리.
사용: python research/z03_valid.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from lib_dip import bars
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
EPS = 0.0002; MMR = 0.004
TE = pd.Timestamp("2024-01-01", tz="UTC"); REC = pd.Timestamp("2025-01-01", tz="UTC")
SYM = ("BTCUSDT", "ETHUSDT")
BARS2 = (120, 240, 360, 480, 720, 960, 1440)
D2 = {}
for s in SYM:
    for b in BARS2:
        d = bars(s, b)
        d["rv"] = (pd.Series(np.log(d["bc"])).diff().rolling(20).std()).shift(1).values
        D2[(s, b)] = d
from z03b import ev1


@numba.njit(cache=True)
def rand_ev(o, h, l, c, starts, holds, fee_bp):
    """무작위 분 진입 → 같은 보유길이 후 청산. 플라시보 기준."""
    out = np.zeros(len(starts))
    for i in range(len(starts)):
        s = starts[i]; e = s + holds[i]
        if e >= len(c): e = len(c)-1
        out[i] = (c[e]/o[s] - 1.0) - fee_bp/1e4
    return out


def build2(bars_, depths, kg, fm=2.0, ft=5.0, eps=EPS):
    E = []
    for s in SYM:
        for b in bars_:
            d = D2[(s, b)]; rv = d["rv"]; B = d["B"]; ts = d["idx"]
            for k, wgt in depths.items():
                a_, b_, r_, m_ = ev1(d["o"], d["h"], d["l"], d["c"], B, np.asarray(rv*k, float),
                                     float(kg), np.asarray(rv, float), fm, ft, eps)
                for a, bb, r, m in zip(a_, b_, r_, m_):
                    E.append((int(a), int(bb), float(r), float(m), ts[int(a)], (s, b, k), float(rv[int(a)//B]), wgt))
    E.sort(key=lambda z: z[0])
    return E


def simple_sim(E, w, Nmax, lo=0.2, hi=3.0, warm=30):
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
        open_.append((b, n_, r, m, key))
        ts_l.append(ts); eq_l.append(cap); fl_l.append(cap - sum(p[1]*p[3] for p in open_))
    for p in open_: cap += p[1]*p[2]
    if not ts_l: return None, None, liq
    ts_l.append(E[-1][4]); eq_l.append(cap); fl_l.append(cap)
    eq = pd.Series(eq_l, index=pd.DatetimeIndex(ts_l)).groupby(level=0).last().resample("D").last().ffill()
    fl = pd.Series(fl_l, index=pd.DatetimeIndex(ts_l)).groupby(level=0).min().resample("D").min().ffill()
    return eq, fl, liq


def show(eq, fl, liq, lab, quiet=False):
    if eq is None or eq.iloc[-1] <= 1e-6:
        if not quiet: P(f"  {lab:28s} 파산 (청산 {liq})")
        return None
    y = (eq.index[-1]-eq.index[0]).days/365
    cg = (eq.iloc[-1]**(1/y)-1)*100
    mdd = float((eq/eq.cummax()-1).min())*100
    fld = float((fl.reindex(eq.index).ffill()/eq.cummax()-1).min())*100
    def sg(t0):
        s = eq[eq.index >= t0]
        return ((s.iloc[-1]/s.iloc[0])**(365/max((s.index[-1]-s.index[0]).days, 1))-1)*100 if len(s) > 60 else np.nan
    if not quiet:
        P(f"  {lab:28s} 연 {cg:+8.0f}% MDD {mdd:6.1f}%(최악 {fld:6.1f}%) | 24~ {sg(TE):+8.0f}% | 25~ {sg(REC):+8.0f}% | 청산 {liq}")
    return cg, mdd, fld, sg(TE), sg(REC), eq


P("[1] 제대로 된 플라시보 — 실제 깊은딥 진입 vs 같은 다리·같은 보유길이의 무작위 분 진입")
DEP = {8.0: 1.0, 12.0: 3.0}
E = build2((240, 480, 720, 1440), DEP, 0.0)
rr = np.array([e[2] for e in E]); hold = np.array([e[1]-e[0] for e in E])
P(f"  실제: n={len(E)} 평균 {rr.mean()*1e4:+.0f}bp 중앙 {np.median(rr)*1e4:+.0f}bp 승률 {(rr>0).mean():.3f} 보유중앙 {np.median(hold):.0f}분")
rng = np.random.default_rng(11)
nul = []
for it in range(500):
    acc = []
    for s in SYM:
        d = D2[(s, 480)]; N = len(d["c"])
        cnt = len(E)//len(SYM)
        st = rng.integers(0, N-1500, cnt)
        hd = rng.choice(hold, cnt).astype(np.int64)
        acc.append(rand_ev(d["o"], d["h"], d["l"], d["c"], st.astype(np.int64), hd, 7.0))
    nul.append(np.concatenate(acc).mean()*1e4)
nul = np.array(nul)
P(f"  플라시보 500회: 평균 {nul.mean():+.0f}bp 95분위 {np.quantile(nul,0.95):+.0f}bp 최대 {nul.max():+.0f}bp")
P(f"  → 실제 {rr.mean()*1e4:+.0f}bp 의 백분위 {(nul < rr.mean()*1e4).mean()*100:.1f}% "
  f"({'통과' if (nul < rr.mean()*1e4).mean() > 0.99 else '미달'})")

P("\n[2] 표본 확대 — 봉 길이 세분화 (2h~1d 7종)")
for bs, lab in (((240, 480, 720, 1440), "4종 4h·8h·12h·1d"),
                ((120, 240, 360, 480, 720, 960, 1440), "7종 2h~1d"),
                ((240, 360, 480, 720, 960, 1440), "6종 4h~1d")):
    for dep, dl in (({8.0: 1.0, 12.0: 3.0}, "8·12(1:3)"),
                    ({7.0: 1.0, 9.0: 2.0, 12.0: 3.0}, "7·9·12(1:2:3)"),
                    ({6.0: 1.0, 8.0: 2.0, 10.0: 3.0, 13.0: 4.0}, "6·8·10·13")):
        E2 = build2(bs, dep, 0.0)
        r_ = np.array([e[2] for e in E2])
        eq, fl, liq = simple_sim(E2, 5.0, 5.0)
        show(eq, fl, liq, f"{lab} {dl} n={len(E2)}")

P("\n[3] 확대 표본에서 명목 스윕 (7종 2h~1d, 6·8·10·13)")
E3 = build2(BARS2, {6.0: 1.0, 8.0: 2.0, 10.0: 3.0, 13.0: 4.0}, 0.0)
best = None
for w in (1.0, 2.0, 3.0, 5.0, 8.0):
    for Nm in (3.0, 5.0, 8.0, 12.0):
        if Nm < w: continue
        eq, fl, liq = simple_sim(E3, w, Nm)
        r = show(eq, fl, liq, f"w={w:.0f} Nmax={Nm:.0f}", quiet=(liq > 0 or eq is None))
        if r and r[2] > -75 and (best is None or r[0] > best[0]): best = (*r, w, Nm)
if best:
    cg, mdd, fld, t24, t25, eq, w, Nm = best
    P(f"\n★ 확대판 최적: w={w:.0f} Nmax={Nm:.0f} → 연 {cg:+.0f}% MDD {mdd:.1f}%(최악 {fld:.1f}%) 24~ {t24:+.0f}% 25~ {t25:+.0f}%")
    yr = eq.resample("YE").last()/eq.resample("YE").first()-1
    mon = eq.resample("ME").last()/eq.resample("ME").first()-1
    P("   연도별: " + " | ".join(f"{i.year} {v*100:+.0f}%" for i, v in yr.items()))
    P("   최근 12개월: " + " | ".join(f"{i.strftime('%y-%m')} {v*100:+.1f}%" for i, v in mon.tail(12).items()))
    l12 = mon.tail(12)
    P(f"   최근 12개월 누적 {(np.prod(1+l12)-1)*100:+.1f}% 양수 {int((l12>0).sum())}/12 | 거래 {len(E3)}건 ({len(E3)/6.7:.0f}/년)")
