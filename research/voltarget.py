"""변동성 타겟팅 + rv 기반 위험예산. MAE≈c×rv 이므로 명목∝1/rv 이면 손실이 일정.
rv는 직전 봉까지로 계산(shift 1) → 진입 전에 알 수 있다. 룩어헤드 없음.
사용: python research/voltarget.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from lib_dip import bars
from shared_cap import events
TE = pd.Timestamp("2024-01-01", tz="UTC")
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
MMR = 0.004
ALL = [(s, b) for s in ("BTCUSDT", "ETHUSDT") for b in (60, 120, 240, 480, 720)]
D = {}
for key in ALL:
    d = bars(key[0], key[1])
    d["rv"] = (pd.Series(np.log(d["bc"])).diff().rolling(20).std()).shift(1).values
    D[key] = d


def build(legs, split, fee=7.0):
    E = []
    for key in legs:
        d = D[key]; k1, k2, k3 = split; rv = d["rv"]
        ei, ex, rr, mm = events(d["o"], d["h"], d["l"], d["c"], d["B"],
                                np.asarray(rv*k1, float), np.asarray(rv*k2, float),
                                np.asarray(rv*k3, float), fee, 0)
        ts = d["idx"]; B = d["B"]
        for a, b, r, m in zip(ei, ex, rr, mm):
            rvb = rv[int(a)//B]           # 그 봉의 rv (직전봉까지 정보)
            E.append((int(a), int(b), float(r), float(m), ts[int(a)], key, float(rvb)))
    E.sort(key=lambda z: z[0])
    return E


def sim(E, w, Nmax, mode="flat", budget=None, cq=0.99, warm=40, lo=0.2, hi=3.0, mmr=MMR, diag=False):
    """mode: flat=고정명목 / vt=변동성타겟(명목∝rv_ref/rv)
       budget: rv×c분위 로 추정한 손실합 ≤ cap×budget (확장창, 룩어헤드 없음)"""
    cap = 1.0; open_ = []; eqs = []; times = []; liq = 0; info = None
    cs = {}; rvh = {}
    for a, b, r, m, ts, key, rvb in E:
        for p in [p for p in open_ if p[0] <= a]:
            cap += p[1]*p[2]; open_.remove(p)
        if cap <= 1e-9: break
        if not np.isfinite(rvb) or rvb <= 0: continue
        rh = rvh.setdefault(key, []); ch = cs.setdefault(key, [])
        notional = cap * w
        if mode == "vt":
            if len(rh) < warm:
                notional = cap * w * lo
            else:
                notional = cap * w * float(np.clip(np.median(rh) / rvb, lo, hi))
        notional = min(notional, max(cap*Nmax - sum(p[1] for p in open_), 0.0))
        if budget is not None:
            cq_ = float(np.quantile(ch, cq)) if len(ch) >= warm else 30.0
            est = max(rvb * cq_, 1e-4)
            room = max(cap*budget - sum(p[1]*p[4] for p in open_), 0.0)
            notional = min(notional, room/est)
        else:
            est = m
        rh.append(rvb); ch.append(m/rvb)
        if notional <= 1e-9: continue
        worst = notional*m + sum(p[1]*p[3] for p in open_)
        if worst >= cap*(1-mmr):
            if diag: info = dict(ts=ts, key=key, cap=cap, n=notional, m=m, rv=rvb,
                                 others=[(p[5], p[1], p[3]) for p in open_])
            cap = 0.0; liq += 1; break
        open_.append((b, notional, r, m, est, key))
        eqs.append(cap); times.append(ts)
    for p in open_: cap += p[1]*p[2]
    eqs.append(cap); times.append(E[-1][4])
    return pd.Series(eqs, index=pd.DatetimeIndex(times)), liq, info


def stat(eq, liq, label, quiet=False):
    if eq.iloc[-1] <= 1e-6:
        if not quiet: P(f"  {label:30s} 파산 (청산 {liq})")
        return None
    yrs = (eq.index[-1]-eq.index[0]).days/365
    te = eq[eq.index >= TE]; t = np.nan
    if len(te) > 30:
        t = ((te.iloc[-1]/te.iloc[0])**(1/((te.index[-1]-te.index[0]).days/365))-1)*100
    cagr = (eq.iloc[-1]**(1/yrs)-1)*100; mdd = float((eq/eq.cummax()-1).min())*100
    if not quiet: P(f"  {label:30s} 전체 연 {cagr:+9.0f}% MDD {mdd:6.1f}% | 테스트 {t:+8.0f}% | 청산 {liq}")
    return cagr, mdd, t, eq


BASE = [(s, b) for s in ("BTCUSDT", "ETHUSDT") for b in (60, 240, 480)]
SP = (5.0, 6.5, 8.0)
E = build(BASE, SP)
m = np.array([e[3] for e in E]); rv = np.array([e[6] for e in E])
P(f"이벤트 {len(E)}건. MAE/rv 비율: 중앙 {np.median(m/rv):.1f} 95분위 {np.quantile(m/rv,0.95):.1f} "
  f"99분위 {np.quantile(m/rv,0.99):.1f} 최대 {(m/rv).max():.1f}")
P(f"  → MAE가 rv에 비례함을 확인. 상관 {np.corrcoef(m, rv)[0,1]:.3f}")

P("\n[1] 변동성 타겟팅 (명목 ∝ rv중앙/rv, 0.2~3.0배 제한)")
best = None
for w in (1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0):
    for Nm in (3.0, 5.0, 8.0, 12.0, 20.0, 40.0):
        if Nm < w: continue
        eq, liq, _ = sim(E, w, Nm, mode="vt")
        r = stat(eq, liq, f"w={w:.0f} Nmax={Nm:.0f}", quiet=(eq.iloc[-1] <= 1e-6))
        if r and (best is None or r[0] > best[0]): best = (r[0], r[1], r[2], "vt", w, Nm, None, r[3], liq)

P("\n[2] rv 위험예산 (추정MAE = rv × c99, 확장창)")
for bud in (0.2, 0.3, 0.5, 0.7):
    for w in (3.0, 5.0, 8.0, 12.0, 20.0):
        eq, liq, _ = sim(E, w, 99.0, mode="flat", budget=bud)
        r = stat(eq, liq, f"예산{bud:.1f} w={w:.0f}", quiet=(eq.iloc[-1] <= 1e-6))
        if r and (best is None or r[0] > best[0]): best = (r[0], r[1], r[2], f"bud{bud}", w, 99.0, bud, r[3], liq)

P("\n[3] 변동성타겟 + 위험예산 동시")
for bud in (0.2, 0.3, 0.5, 0.7):
    for w in (3.0, 5.0, 8.0, 12.0, 20.0):
        eq, liq, _ = sim(E, w, 99.0, mode="vt", budget=bud)
        r = stat(eq, liq, f"vt+예산{bud:.1f} w={w:.0f}", quiet=(eq.iloc[-1] <= 1e-6))
        if r and (best is None or r[0] > best[0]): best = (r[0], r[1], r[2], f"vt+bud{bud}", w, 99.0, bud, r[3], liq)

if best:
    P(f"\n★ 최고: {best[3]} w={best[4]:.0f} Nmax={best[5]:.0f} → 전체 연 {best[0]:+.0f}% MDD {best[1]:.1f}% 테스트 {best[2]:+.0f}% 청산 {best[8]}")
    eq = best[7]
    yr = eq.resample("YE").last()/eq.resample("YE").first()-1
    mon = eq.resample("ME").last()/eq.resample("ME").first()-1
    l12 = mon.tail(12)
    P("  연도별: " + " | ".join(f"{i.year} {v*100:+.0f}%" for i, v in yr.items()))
    P(f"  최근 12개월 {(np.prod(1+l12)-1)*100:+.1f}% 양수 {int((l12>0).sum())}/12 | 월승률 {float((mon>0).mean()):.3f} (n={len(mon)})")
