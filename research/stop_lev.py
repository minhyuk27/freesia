"""손절 도입 → MAE 상한 확정 → 명목 대폭 확대.
손절가 = 시가 × (1 − rv×ks), ks > k3. 저가가 닿으면 체결된 단계 전부 손절가에 청산(테이커).
같은 분에 진입·손절이 겹치면 '진입 후 손절'로 보수 처리.
사용: python research/stop_lev.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from lib_dip import bars
TE = pd.Timestamp("2024-01-01", tz="UTC")
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
MMR = 0.004
ALL = [(s, b) for s in ("BTCUSDT", "ETHUSDT") for b in (60, 120, 240, 480)]
D = {}
for key in ALL:
    d = bars(key[0], key[1])
    d["rv"] = (pd.Series(np.log(d["bc"])).diff().rolling(20).std()).shift(1).values
    D[key] = d


@numba.njit(cache=True)
def ev_stop(o, h, l, c, B, x1, x2, x3, xs, fee_m, fee_t):
    """xs: 손절 깊이(0이면 손절 없음). 반환 진입분·청산분·수익률·MAE·손절여부"""
    n = len(c) // B
    ei = np.full(n, -1, np.int64); ex = np.full(n, -1, np.int64)
    rr = np.zeros(n); mm = np.zeros(n); sf = np.zeros(n)
    k = 0
    for b in range(n):
        if np.isnan(x1[b]) or x1[b] <= 0:
            continue
        s = b * B; e = s + B
        O = o[s]; C = c[e - 1]
        t1 = O * (1 - x1[b]); t2 = O * (1 - x2[b]); t3 = O * (1 - x3[b])
        SL = O * (1 - xs[b]) if xs[b] > 0 else 0.0
        f1 = -1; f2 = -1; f3 = -1; hitsl = -1
        for t in range(s, e):
            if f1 < 0 and l[t] <= t1: f1 = t
            if f2 < 0 and l[t] <= t2: f2 = t
            if f3 < 0 and l[t] <= t3: f3 = t
            if SL > 0 and hitsl < 0 and l[t] <= SL:
                hitsl = t
                break
        if f1 < 0:
            continue
        end = hitsl if hitsl >= 0 else e - 1
        px = SL if hitsl >= 0 else C
        fee = (fee_m + fee_t) if hitsl >= 0 else (fee_m + fee_t)
        tot = 0.0; cnt = 0.0; wsum = 0.0
        for tt, ff in ((t1, f1), (t2, f2), (t3, f3)):
            if ff >= 0 and ff <= end:
                tot += ((px / tt - 1.0) - fee / 1e4) / 3.0; cnt += 1.0; wsum += tt
        if cnt == 0.0:
            continue
        avg = wsum / cnt
        lo_ = t1
        for t in range(f1, end + 1):
            if l[t] < lo_: lo_ = l[t]
        if hitsl >= 0 and lo_ < SL: lo_ = SL   # 손절가에서 끊김
        ei[k] = f1; ex[k] = end
        rr[k] = tot
        mm[k] = (1.0 - lo_ / avg) * (cnt / 3.0)
        sf[k] = 1.0 if hitsl >= 0 else 0.0
        k += 1
    return ei[:k], ex[:k], rr[:k], mm[:k], sf[:k]


def build(legs, split, ks, fee_m=2.0, fee_t=5.0):
    E = []
    ns = 0
    for key in legs:
        d = D[key]
        k1, k2, k3 = split
        rv = d["rv"]
        xs = np.asarray(rv * ks, float) if ks > 0 else np.zeros(len(rv))
        ei, ex, rr, mm, sf = ev_stop(d["o"], d["h"], d["l"], d["c"], d["B"],
                                     np.asarray(rv*k1, float), np.asarray(rv*k2, float),
                                     np.asarray(rv*k3, float), xs, fee_m, fee_t)
        ts = d["idx"]; ns += int(sf.sum())
        for a, b, r, m, s_ in zip(ei, ex, rr, mm, sf):
            E.append((int(a), int(b), float(r), float(m), ts[int(a)], key))
    E.sort(key=lambda z: z[0])
    return E, ns


def sim(E, w, Nmax, mmr=MMR):
    cap = 1.0; open_ = []; eqs = []; times = []; liq = 0
    for a, b, r, m, ts, key in E:
        for p in [p for p in open_ if p[0] <= a]:
            cap += p[1] * p[2]; open_.remove(p)
        if cap <= 1e-9: break
        cur = sum(p[1] for p in open_)
        notional = min(cap * w, max(cap * Nmax - cur, 0.0))
        if notional <= 1e-9: continue
        worst = notional * m + sum(p[1] * p[3] for p in open_)
        if worst >= cap * (1 - mmr):
            cap = 0.0; liq += 1; break
        open_.append((b, notional, r, m, key))
        eqs.append(cap); times.append(ts)
    for p in open_: cap += p[1] * p[2]
    eqs.append(cap); times.append(E[-1][4])
    return pd.Series(eqs, index=pd.DatetimeIndex(times)), liq


def stat(eq, liq, label, quiet=False):
    if eq.iloc[-1] <= 1e-6:
        if not quiet: P(f"  {label:32s} 파산 (청산 {liq})")
        return None
    yrs = (eq.index[-1] - eq.index[0]).days / 365
    te = eq[eq.index >= TE]; t = np.nan
    if len(te) > 30:
        t = ((te.iloc[-1]/te.iloc[0])**(1/((te.index[-1]-te.index[0]).days/365))-1)*100
    cagr = (eq.iloc[-1] ** (1/yrs) - 1) * 100
    mdd = float((eq/eq.cummax()-1).min())*100
    if not quiet:
        P(f"  {label:32s} 전체 연 {cagr:+8.0f}% MDD {mdd:6.1f}% | 테스트 {t:+7.0f}% | 청산 {liq}")
    return cagr, mdd, t, eq


BASE = [(s, b) for s in ("BTCUSDT", "ETHUSDT") for b in (60, 240, 480)]
SP = (5.0, 6.5, 8.0)

P("[1] 손절 깊이별 — 거래 단위 통계 (명목 1배)")
for ks in (0, 10, 12, 15, 20, 25, 30):
    E, ns = build(BASE, SP, ks)
    r = np.array([e[2] for e in E]); m = np.array([e[3] for e in E])
    P(f"  ks={ks:2d} ({'손절없음' if ks==0 else f'rv×{ks}'})  n={len(E):5d} 손절 {ns:4d}({ns/max(len(E),1)*100:4.1f}%) "
      f"평균 {r.mean()*1e4:+6.1f}bp 승률 {(r>0).mean():.3f} | MAE 평균 {m.mean()*100:5.2f}% 99분위 {np.quantile(m,0.99)*100:5.2f}% 최대 {m.max()*100:6.2f}%")

P("\n[2] 손절 깊이 × 명목 (Nmax=명목×3)")
best = None
for ks in (0, 12, 15, 20, 25, 30):
    E, ns = build(BASE, SP, ks)
    P(f"  -- ks={ks} --")
    for w in (1.0, 2.0, 3.0, 5.0, 8.0, 12.0):
        for Nm in (w*2, w*3, w*5):
            eq, liq = sim(E, w, Nm)
            r = stat(eq, liq, f"w={w:.0f} Nmax={Nm:.0f}", quiet=(eq.iloc[-1] <= 1e-6))
            if r and (best is None or r[0] > best[0]):
                best = (r[0], r[1], r[2], ks, w, Nm, r[3], liq)
if best:
    P(f"\n  ★ 최고: ks={best[3]} w={best[4]:.0f} Nmax={best[5]:.0f} → 전체 연 {best[0]:+.0f}% MDD {best[1]:.1f}% 테스트 {best[2]:+.0f}%")
    eq = best[6]
    yr = eq.resample("YE").last()/eq.resample("YE").first()-1
    mon = eq.resample("ME").last()/eq.resample("ME").first()-1
    l12 = mon.tail(12)
    P("    연도별: " + " | ".join(f"{i.year} {v*100:+.0f}%" for i, v in yr.items()))
    P(f"    최근 12개월 {(np.prod(1+l12)-1)*100:+.1f}% 양수 {int((l12>0).sum())}/12 | 월승률 {float((mon>0).mean()):.3f}")
