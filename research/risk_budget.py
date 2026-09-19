"""청산 사건 해부 + 위험예산 사이징 + 다리 확장.
위험예산: 진입 시점에 '그 다리의 과거 MAE 99분위'(확장창, 룩어헤드 없음)로 최악을 추정해
   Σ(명목 × 추정MAE) ≤ 자본 × budget 이 되도록 명목을 깎는다. 청산 판정은 실제 MAE로.
사용: python research/risk_budget.py"""
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

ALL = [(s, b) for s in ("BTCUSDT", "ETHUSDT") for b in (30, 60, 120, 240, 480, 720, 1440)]
D = {}
for key in ALL:
    d = bars(key[0], key[1])
    d["rv"] = (pd.Series(np.log(d["bc"])).diff().rolling(20).std()).shift(1).values
    D[key] = d


def build(legs, split, fee=7.0):
    E = []
    for key in legs:
        d = D[key]
        k1, k2, k3 = split
        ei, ex, rr, mm = events(d["o"], d["h"], d["l"], d["c"], d["B"],
                                np.asarray(d["rv"] * k1, float), np.asarray(d["rv"] * k2, float),
                                np.asarray(d["rv"] * k3, float), fee, 0)
        ts = d["idx"]
        for a, b, r, m in zip(ei, ex, rr, mm):
            E.append((int(a), int(b), float(r), float(m), ts[int(a)], key))
    E.sort(key=lambda z: z[0])
    return E


def sim(E, w, Nmax, budget=None, q=0.99, warm=50, mmr=MMR, diag=False):
    """budget=None이면 위험예산 없이 w·Nmax만. budget이 주어지면 과거 MAE q분위로 명목 제한."""
    cap = 1.0; open_ = []; eqs = []; times = []; liq = 0; info = None
    hist = {}
    for a, b, r, m, ts, key in E:
        for p in [p for p in open_ if p[0] <= a]:
            cap += p[1] * p[2]; open_.remove(p)
        if cap <= 1e-9:
            break
        cur = sum(p[1] for p in open_)
        notional = min(cap * w, max(cap * Nmax - cur, 0.0))
        if budget is not None:
            hk = hist.setdefault(key, [])
            if len(hk) < warm:
                est = 0.05
            else:
                est = float(np.quantile(hk, q))
            est = max(est, 1e-4)
            used = sum(p[1] * p[4] for p in open_)
            room = max(cap * budget - used, 0.0)
            notional = min(notional, room / est)
        else:
            est = m
        hist.setdefault(key, []).append(m)
        if notional <= 1e-9:
            continue
        worst = notional * m + sum(p[1] * p[3] for p in open_)
        if worst >= cap * (1 - mmr):
            if diag:
                info = dict(ts=ts, key=key, cap=cap, notional=notional, m=m,
                            others=[(p[5], p[1], p[3]) for p in open_])
            cap = 0.0; liq += 1; break
        open_.append((b, notional, r, m, est, key))
        eqs.append(cap); times.append(ts)
    for p in open_:
        cap += p[1] * p[2]
    eqs.append(cap); times.append(E[-1][4])
    return pd.Series(eqs, index=pd.DatetimeIndex(times)), liq, info


def stat(eq, liq, label):
    if eq.iloc[-1] <= 1e-6:
        P(f"  {label:34s} 파산 (청산 {liq})"); return None
    yrs = (eq.index[-1] - eq.index[0]).days / 365
    te = eq[eq.index >= TE]
    t = "-"
    if len(te) > 30:
        y2 = (te.index[-1] - te.index[0]).days / 365
        t = f"{((te.iloc[-1]/te.iloc[0])**(1/y2)-1)*100:+7.0f}%"
    cagr = (eq.iloc[-1] ** (1 / yrs) - 1) * 100
    mdd = float((eq / eq.cummax() - 1).min()) * 100
    P(f"  {label:34s} 전체 연 {cagr:+8.0f}% MDD {mdd:6.1f}% | 테스트 {t} | 청산 {liq}")
    return cagr, mdd, eq


BASE = [(s, b) for s in ("BTCUSDT", "ETHUSDT") for b in (60, 240, 480)]
SP = (5.0, 6.5, 8.0)
E = build(BASE, SP)
P(f"기본 6다리(1h·4h·8h × BTC·ETH) 이벤트 {len(E)}건")

P("\n[1] 청산 사건 해부 (w=1.0, Nmax=8)")
eq, liq, info = sim(E, 1.0, 8, diag=True)
if info:
    P(f"  시점 {info['ts']}  다리 {info['key']}  자본 {info['cap']:.3f}")
    P(f"  이번 진입 명목 {info['notional']:.3f} × 실제MAE {info['m']*100:.2f}% = {info['notional']*info['m']*100:.1f}% 자본")
    P("  동시 보유:")
    for k, nn, mm_ in info["others"]:
        P(f"    {str(k):22s} 명목 {nn:6.3f} × MAE {mm_*100:6.2f}% = {nn*mm_*100:5.1f}%")
    P(f"  합계 손실 {info['notional']*info['m']*100 + sum(nn*mm_ for _,nn,mm_ in info['others'])*100:.1f}% ≥ 자본 100%")

P("\n[2] 위험예산 사이징 (과거 MAE 99분위, 룩어헤드 없음)")
for bud in (0.3, 0.5, 0.7, 0.9):
    for w in (1.0, 2.0, 3.0, 5.0, 8.0):
        eq, liq, _ = sim(E, w, 99, budget=bud)
        stat(eq, liq, f"예산{bud:.1f} w={w:.0f}")

P("\n[3] MAE 분위수 선택 (예산 0.7, w=5)")
for q in (0.95, 0.98, 0.99, 0.995, 1.0):
    eq, liq, _ = sim(E, 5.0, 99, budget=0.7, q=q)
    stat(eq, liq, f"q={q}")

P("\n[4] 다리 확장 (위험예산 0.7, q=0.99, w=5)")
SETS = {
    "6다리 1h·4h·8h": BASE,
    "8다리 +12h": BASE + [(s, 720) for s in ("BTCUSDT", "ETHUSDT")],
    "10다리 +12h·1d": BASE + [(s, b) for s in ("BTCUSDT", "ETHUSDT") for b in (720, 1440)],
    "8다리 +2h": BASE + [(s, 120) for s in ("BTCUSDT", "ETHUSDT")],
    "10다리 +2h·12h": BASE + [(s, b) for s in ("BTCUSDT", "ETHUSDT") for b in (120, 720)],
    "12다리 +30m·2h·12h": BASE + [(s, b) for s in ("BTCUSDT", "ETHUSDT") for b in (30, 120, 720)],
    "14다리 전부": ALL,
}
best = None
for nm, lg in SETS.items():
    E2 = build(lg, SP)
    eq, liq, _ = sim(E2, 5.0, 99, budget=0.7)
    r = stat(eq, liq, f"{nm} ({len(E2)}건)")
    if r and (best is None or r[0] > best[0]):
        best = (r[0], nm, lg, r[2], liq)
if best:
    P(f"\n  → 최고 {best[1]}  연 {best[0]:+.0f}%")
