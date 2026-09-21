"""z23 — 딥을 최대로 민다. 폭발력의 원천은 딥이다.

z22 결과: 다리 제거 분석에서 딥만 기여(−2.93%p), 배리어 다리는 전부 희석(+0.06~1.58%p).
         샤프: 딥 0.507 vs 배리어 0.12~0.17.
         켈리(공분산)도 A1에 0.48, 나머지 대부분 0을 준다.

그런데 A1의 내부 레버리지를 `w=3/Nmax=5`로 임의 고정해뒀다. 이걸 최적화한다.
그리고 BTC 딥과 ETH 딥을 분리하면 다리가 늘어난다(상관이 낮으면 이득).

탐색:
  ①exit_tier의 (w, Nmax) 격자
  ②호가 문턱
  ③봉 길이(15/30/60분)
  ④BTC/ETH 분리 후 켈리 결합
"""
import sys, warnings, glob
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R/"research"))
import exit_tier as X
from absfloor import met


def P(*a, end="\n"):
    print(" ".join(str(x) for x in a), end=end, flush=True)


SYM = ("BTCUSDT", "ETHUSDT")


def bookslope(sym, idx):
    fs = sorted(glob.glob(str(R/"data"/"parquet"/"bookdepth"/sym/"*.parquet")))
    if not fs: return None
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").drop_duplicates("ts")
    s = df.set_index("ts")[["n-1", "n-5"]]
    ti = pd.DatetimeIndex(s.index).floor("min").as_unit("ms")
    lag = int(max(1, np.median(np.diff(ti.asi8)/60000.0)))
    pos = idx.searchsorted(ti); N = len(idx)
    o = {}
    for c in ("n-1", "n-5"):
        a = np.full(N, np.nan); v = s[c].values.astype(float)
        ok = (pos >= 0) & (pos < N)
        a[pos[ok]] = v[ok]
        o[c] = pd.Series(a).ffill().shift(lag).values
    return o["n-5"]/np.maximum(o["n-1"], 1)


BOOK = {s: bookslope(s, X.RAW[s]["idx"]) for s in SYM}
P("호가 준비 완료\n")


def expq(v, w0=300, st=100):
    v = np.asarray(v, float); q = np.full(len(v), np.nan)
    for i in range(w0, len(v), st):
        x = v[:i]; x = x[np.isfinite(x)]
        if len(x) < 200: continue
        q[i:i+st] = np.searchsorted(np.sort(x), v[i:i+st])/len(x)
    return q


def filt(E, thr):
    E = sorted(E, key=lambda z: z[0])
    vals = np.array([np.nan if BOOK.get(e[5][0]) is None or e[0] >= len(BOOK[e[5][0]])
                     else BOOK[e[5][0]][e[0]] for e in E], float)
    have = np.isfinite(vals)
    q = np.full(len(vals), np.nan)
    if have.sum() > 400: q[have] = expq(vals[have])
    keep = (~have) | (~np.isfinite(q)) | (q >= thr)
    return [e for e, k in zip(E, keep) if k]


def only(E, sym):
    return [e for e in E if e[5][0] == sym]


def build15(bl=(15,), dep=None):
    X.BL = bl; X.CACHE.clear()
    E = X.build(dep or {6.0: 1.0, 8.0: 3.0, 12.0: 6.0}, 0)
    X.BL = (240, 360, 480, 720, 960, 1440); X.CACHE.clear()
    return E


E15 = build15()
X.BL = (240, 360, 480, 720, 960, 1440); X.CACHE.clear()
E03 = X.build({10.0: 1.0, 14.0: 3.0, 20.0: 6.0}, 0)


def run(E, w, Nm):
    eq, fl, liq = X.sim(E, float(w), float(Nm))
    m = met(eq, fl, liq)
    if m is None: return None, None
    mon = (eq.resample("ME").last()/eq.resample("ME").first()-1).dropna()
    return m, mon


P("="*128); P("[1] (w, Nmax) 격자 — 딥+호가50 + z0.3 공유자본"); P("="*128)
AB = sorted(filt(E15, 0.5)+E03, key=lambda z: z[0])
P(f"  {len(AB)}건")
P(f"  {'w＼Nmax':>8s} " + " ".join(f"{n:>17s}" for n in ("1", "2", "3", "5", "8", "12", "16")))
BEST = None
for w in (1, 1.5, 2, 3, 4, 5, 6, 8):
    row = []
    for Nm in (1, 2, 3, 5, 8, 12, 16):
        if Nm < w: row.append("        -        "); continue
        m, mon = run(AB, w, Nm)
        if m is None: row.append("       파산      "); continue
        mo = ((1+m[0]/100)**(1/12)-1)*100
        row.append(f"{mo:+7.2f}%/{m[1]:6.0f}%")
        if BEST is None or mo > BEST[0]: BEST = (mo, w, Nm, m, mon)
    P(f"  {w:7.1f} " + " ".join(f"{x:>17s}" for x in row))
P("  (셀 = 월% / MDD%)")
if BEST:
    mo, w, Nm, m, mon = BEST
    P(f"\n  ★ 최고: w={w} Nmax={Nm} → 월 {mo:+.2f}%, 연 {m[0]:+.0f}%, MDD {m[1]:.1f}%, "
      f"최악 {m[2]:.1f}%, 24~ {m[3]:+.0f}%, 25~ {m[4]:+.0f}%")

P("\n"+"="*128); P("[2] 호가 문턱 × (w,Nmax) — 최적 조합"); P("="*128)
P(f"  {'문턱':>8s} " + " ".join(f"{'w'+str(a)+'/N'+str(b):>16s}"
                               for a, b in ((2, 3), (3, 5), (4, 6), (5, 8), (6, 10))))
B2 = None
for thr in (0.0, 0.3, 0.5, 0.6, 0.7):
    AB2 = sorted(filt(E15, thr)+E03, key=lambda z: z[0])
    row = []
    for a, b in ((2, 3), (3, 5), (4, 6), (5, 8), (6, 10)):
        m, mon = run(AB2, a, b)
        if m is None: row.append("      파산      "); continue
        mo = ((1+m[0]/100)**(1/12)-1)*100
        row.append(f"{mo:+7.2f}%/{m[1]:6.0f}%")
        if B2 is None or mo > B2[0]: B2 = (mo, thr, a, b, m)
    P(f"  상위{(1-thr)*100:4.0f}% " + " ".join(f"{x:>16s}" for x in row))

P("\n"+"="*128); P("[3] 봉 길이 — 15/30/60분 딥"); P("="*128)
P(f"  {'봉':>6s} {'n':>6s} {'거래당bp':>9s} {'w3/N5 월%':>11s} {'MDD%':>8s} {'w5/N8 월%':>11s} {'MDD%':>8s}")
SERS = {}
for bl in (15, 30, 60):
    Ex = build15(bl=(bl,))
    Ef = filt(Ex, 0.5)
    r = np.array([e[2] for e in Ef])
    row = [f"{bl:5d}분 {len(Ef):6d} {r.mean()*1e4:+9.1f}"]
    for a, b in ((3, 5), (5, 8)):
        m, mon = run(sorted(Ef+E03, key=lambda z: z[0]), a, b)
        if m is None: row.append("       파산        "); continue
        row.append(f"{((1+m[0]/100)**(1/12)-1)*100:11.2f} {m[1]:8.1f}")
        if a == 3: SERS[f"딥{bl}분"] = mon
    P("  " + " ".join(row))

P("\n"+"="*128); P("[4] BTC 딥 · ETH 딥 분리 후 결합"); P("="*128)
Ef = filt(E15, 0.5)
LEG = {}
for s in SYM:
    Es = sorted(only(Ef, s)+only(E03, s), key=lambda z: z[0])
    m, mon = run(Es, 3, 5)
    if m is None: continue
    LEG[f"{s[:3]}딥"] = mon
    P(f"  {s[:3]} 딥: {len(Es)}건, w3/N5 연 {m[0]:+.0f}% 월 {((1+m[0]/100)**(1/12)-1)*100:+.2f}% MDD {m[1]:.0f}%")
if len(LEG) == 2:
    df = pd.concat(LEG, axis=1).fillna(0)
    P(f"  월 상관 {df.corr().iloc[0,1]:.3f}")
    P(f"\n  {'구성':>16s} " + " ".join(f"{'×'+str(n):>16s}" for n in (1, 1.5, 2, 3, 5)))
    mu = df.mean().values; sd = df.std().values
    kel = np.maximum(mu, 0)/np.maximum(sd**2, 1e-12); kel = kel/kel.sum()
    for lab, w_ in (("균등", np.array([0.5, 0.5])), ("켈리", kel),
                    ("BTC만", np.array([1.0, 0.0])), ("ETH만", np.array([0.0, 1.0]))):
        row = []
        for nom in (1, 1.5, 2, 3, 5):
            tot = (df*w_).sum(axis=1)*nom
            eq = (1+tot).cumprod()
            if (eq <= 0.01).any(): row.append("      파산      "); continue
            a = (eq.iloc[-1]**(12/len(tot))-1)*100
            row.append(f"{((1+a/100)**(1/12)-1)*100:+7.2f}%/{(eq/eq.cummax()-1).min()*100:6.0f}%")
        P(f"  {lab:>16s} " + " ".join(f"{x:>16s}" for x in row))
    P(f"  켈리 가중: BTC {kel[0]:.2f} / ETH {kel[1]:.2f}")

P("\n"+"="*128); P("[5] 최종 — 최고 딥 구성의 3단 레버리지 + 연도별"); P("="*128)
if BEST:
    mo, w, Nm, m, mon = BEST
    P(f"  구성: 딥+호가상위50% + z0.3, 공유자본 w={w} Nmax={Nm}")
    P(f"\n  {'구간':>14s} {'w/Nmax':>10s} {'연수익%':>12s} {'월%':>9s} {'MDD%':>8s} {'최악%':>8s}")
    for lb, a, b in (("무레버리지", 1, 1), ("중간", 2, 3), ("중간+", 3, 5),
                     ("공격", 4, 6), ("최대공격", w, Nm), ("극단", 6, 12)):
        mm, _ = run(AB, a, b)
        P(f"  {lb:>14s} {f'{a}/{b}':>10s} " + ("        파산" if mm is None
                                                else f"{mm[0]:12.0f} {((1+mm[0]/100)**(1/12)-1)*100:9.2f} "
                                                     f"{mm[1]:8.1f} {mm[2]:8.1f}"))
    out = []
    for yr in range(2020, 2027):
        sub = mon[mon.index.year == yr]
        out.append(f"{yr}  n/a" if len(sub) == 0 else f"{yr} {((1+sub).prod()-1)*100:+8.0f}%")
    P(f"  연도별(w{w}/N{Nm}): " + " | ".join(out))
    P(f"  월승률 {np.mean(mon>0)*100:.1f}%  |  월 중앙 {np.median(mon)*100:+.2f}%  최악 월 {mon.min()*100:+.1f}%")
