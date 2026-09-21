"""z22 — 다리를 늘리고 켈리로 사이징한다. 폭발적 수익의 정당한 경로.

z21은 다리 4개에 고정 가중(2:1:1)을 줬다. 그건 각 다리의 엣지 크기를 무시한다.
켈리는 각 다리를 (기대수익/분산)에 비례해 배분하라고 한다.
그리고 상관이 낮은 다리를 더 넣으면 같은 MDD에서 총명목을 더 키울 수 있다.

다리 후보(전부 검증된 메커니즘):
  A1 z1.4 딥 BTC+ETH × 호가 상위50%
  A2 z1.4 딥 × 호가 상위20%  (거래당 +35.4bp, 빈도 낮음)
  B  z0.3 깊은 딥
  C  BTC롱 스윙 ±4%
  D  ETH숏 ±4%
  E  BTC롱 스윙 ±2%   (다른 배리어 = 다른 보유시간 = 다른 베팅)
  F  BTC롱 스윙 ±8%
  G  ETH롱 스윙 ±2%   (ETH롱은 ±4%에서 죽었지만 ±2%는 별개)

가중: ①균등 ②켈리(평균/분산) ③위험패리티(1/변동성) ④최소분산
"""
import sys, warnings, glob
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R/"research"))
import exit_tier as X
from absfloor import met
import z2_lib as L2
from z2_final import build_sigs, exp_z


def P(*a, end="\n"):
    print(" ".join(str(x) for x in a), end=end, flush=True)


SYM = ("BTCUSDT", "ETHUSDT")
warm = 200000


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


P("준비...")
BOOK = {s: bookslope(s, X.RAW[s]["idx"]) for s in SYM}


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


@numba.njit(cache=True)
def barrier(o, h, l, c, sg, th, bar, maxh, step, start, n_end, side):
    K = 40000
    ea = np.zeros(K, np.int64); er = np.zeros(K); k = 0; t = start
    while t < n_end-maxh-2:
        i = t//step
        if i >= len(sg) or not np.isfinite(sg[i]): t += step; continue
        if side == 1 and sg[i] < th: t += step; continue
        if side == -1 and sg[i] > th: t += step; continue
        f = t+1; px = o[f]; e = -1; res = 0.0; end = min(f+maxh, n_end-1)
        if side == 1:
            U = px*(1+bar); D = px*(1-bar)
            for j in range(f, end+1):
                if l[j] <= D: e = j; res = -bar; break
                if h[j] >= U: e = j; res = bar; break
            if e < 0: e = end; res = c[e]/px-1
        else:
            U = px*(1-bar); D = px*(1+bar)
            for j in range(f, end+1):
                if h[j] >= D: e = j; res = -bar; break
                if l[j] <= U: e = j; res = bar; break
            if e < 0: e = end; res = px/c[e]-1
        ea[k] = f; er[k] = res-10.0/1e4; k += 1
        t = e+1
        if k >= K-2: break
    return ea[:k], er[:k]


SIGC = {}
for S in SYM:
    d = L2.D[S]
    SG = build_sigs(S)
    Z = {n: exp_z(np.asarray(SG[n], float)) for n in SG}
    ml = min(min(len(Z[n]) for n in Z), len(d["c"])//L2.STEP)
    SIGC[S] = np.nanmean(np.vstack([Z[n][:ml] for n in Z if n.endswith("4320")]), axis=0)


def bar_ser(S, side, bar, q=0.10, maxh=43200):
    d = L2.D[S]; sig = SIGC[S]
    fin = sig[np.isfinite(sig)]
    th = float(np.quantile(fin, 1-q if side == 1 else q))
    ea, er = barrier(d["o"], d["h"], d["l"], d["c"], sig, th, bar, maxh, L2.STEP,
                     warm, len(d["c"]), side)
    if len(er) < 60: return None, None
    return pd.Series(er, index=pd.DatetimeIndex(d["idx"][ea])).resample("ME").sum(), (len(er), er.mean()*1e4, np.mean(er > 0)*100)


LEGS = {}
# 딥 다리 — exit_tier 공유자본
X.BL = (15,); X.CACHE.clear()
E14 = X.build({6.0: 1.0, 8.0: 3.0, 12.0: 6.0}, 0)
X.BL = (240, 360, 480, 720, 960, 1440); X.CACHE.clear()
E03 = X.build({10.0: 1.0, 14.0: 3.0, 20.0: 6.0}, 0)
for thr, tag in ((0.5, "A1 딥+호가50"), (0.8, "A2 딥+호가20")):
    EE = sorted(filt(E14, thr)+E03, key=lambda z: z[0])
    eq, fl, liq = X.sim(EE, 3.0, 5.0)
    if eq is None: continue
    m = (eq.resample("ME").last()/eq.resample("ME").first()-1).dropna()
    LEGS[tag] = m
    mm = met(eq, fl, liq)
    P(f"  {tag}: {len(EE)}건, w3/N5 연 {mm[0]:+.0f}% MDD {mm[1]:.0f}%")
X.CACHE.clear()

# 배리어 다리
for S, side, bar, q, tag in (("BTCUSDT", 1, 0.04, 0.10, "C BTC롱4%"),
                             ("ETHUSDT", -1, 0.04, 0.10, "D ETH숏4%"),
                             ("BTCUSDT", 1, 0.02, 0.10, "E BTC롱2%"),
                             ("BTCUSDT", 1, 0.08, 0.10, "F BTC롱8%"),
                             ("ETHUSDT", 1, 0.02, 0.10, "G ETH롱2%"),
                             ("BTCUSDT", -1, 0.04, 0.10, "H BTC숏4%")):
    s_, meta = bar_ser(S, side, bar, q)
    if s_ is None: continue
    LEGS[tag] = s_
    P(f"  {tag}: {meta[0]}건, 거래당 {meta[1]:+.1f}bp, 승률 {meta[2]:.1f}%")

df = pd.concat(LEGS, axis=1).fillna(0)
P(f"\n  다리 {len(df.columns)}개, 기간 {len(df)}개월")

P("\n"+"="*128); P("[1] 다리별 월 통계"); P("="*128)
P(f"  {'다리':>16s} {'월평균%':>9s} {'월표준편차%':>11s} {'샤프(월)':>9s} {'월승률%':>8s} {'거래월':>7s}")
for c in df.columns:
    s_ = df[c]
    nz = s_[s_ != 0]
    P(f"  {c:>16s} {s_.mean()*100:9.2f} {s_.std()*100:11.2f} "
      f"{s_.mean()/max(s_.std(),1e-9):9.3f} {np.mean(s_>0)*100:8.1f} {len(nz):7d}")

P("\n"+"="*128); P("[2] 상관행렬"); P("="*128)
cr = df.corr()
P("                " + " ".join(f"{c[:6]:>8s}" for c in cr.columns))
for i, r in cr.iterrows():
    P(f"  {i[:14]:>14s} " + " ".join(f"{r[c]:8.3f}" for c in cr.columns))
tri = cr.values[np.triu_indices(len(cr), 1)]
P(f"\n  평균 쌍상관 {tri.mean():.3f}, 최대 {tri.max():.3f}, 최소 {tri.min():.3f}")


def wsets(df):
    n = len(df.columns)
    mu = df.mean().values; sd = df.std().values
    cov = df.cov().values
    out = {}
    out["균등"] = np.ones(n)/n
    rp = 1.0/np.maximum(sd, 1e-9); out["위험패리티"] = rp/rp.sum()
    kel = np.maximum(mu, 0)/np.maximum(sd**2, 1e-12)
    out["켈리(대각)"] = kel/max(kel.sum(), 1e-9)
    try:
        inv = np.linalg.pinv(cov)
        kf = inv @ mu
        kf = np.maximum(kf, 0)
        if kf.sum() > 0: out["켈리(공분산)"] = kf/kf.sum()
        ones = np.ones(n)
        mv = inv @ ones
        mv = np.maximum(mv, 0)
        if mv.sum() > 0: out["최소분산"] = mv/mv.sum()
    except Exception:
        pass
    return out


def pf(w, nom):
    tot = (df*w).sum(axis=1)*nom
    eq = (1+tot).cumprod()
    if (eq <= 0.01).any(): return None
    a = (eq.iloc[-1]**(12/len(tot))-1)*100
    return dict(ann=a, mon=((1+a/100)**(1/12)-1)*100,
                mdd=(eq/eq.cummax()-1).min()*100, mwr=np.mean(tot > 0)*100)


P("\n"+"="*128); P("[3] 가중 방식 × 명목"); P("="*128)
WS = wsets(df)
P(f"  {'방식':>14s} " + " ".join(f"{'×'+str(n):>17s}" for n in (1, 2, 3, 5, 8, 12)))
BEST = None
for lab, w in WS.items():
    row = []
    for nom in (1, 2, 3, 5, 8, 12):
        r = pf(w, nom)
        row.append("       파산      " if r is None else f"{r['mon']:+7.2f}%/{r['mdd']:6.0f}%")
        if r and (BEST is None or r["mon"] > BEST[0]): BEST = (r["mon"], lab, w, nom, r)
    P(f"  {lab:>14s} " + " ".join(f"{x:>17s}" for x in row))
P("  (셀 = 월% / MDD%)")

P("\n  가중값:")
for lab, w in WS.items():
    P(f"    {lab:>14s}: " + " ".join(f"{c[:6]}={v:.2f}" for c, v in zip(df.columns, w)))

P("\n"+"="*128); P("[4] 최고 구성 3단 레버리지 + 연도별"); P("="*128)
if BEST:
    mon, lab, w, nom, r = BEST
    P(f"  최고: {lab} ×{nom} → 월 {mon:+.2f}%, 연 {r['ann']:+.0f}%, MDD {r['mdd']:.1f}%")
    P(f"\n  {'구간':>12s} {'명목':>6s} {'연수익%':>13s} {'월%':>9s} {'MDD%':>8s} {'월승률%':>8s}")
    for lb, n2 in (("무레버리지", 1), ("중간", 2), ("중간+", 3), ("공격", 5), ("최대공격", 8), ("극단", 12)):
        rr = pf(w, n2)
        P(f"  {lb:>12s} {n2:5d}x " + ("         파산" if rr is None
                                       else f"{rr['ann']:13.0f} {rr['mon']:9.2f} {rr['mdd']:8.1f} {rr['mwr']:8.1f}"))
    tot = (df*w).sum(axis=1)
    for n2 in (3, nom):
        out = []
        for yr in range(2020, 2027):
            sub = (tot*n2)[tot.index.year == yr]
            out.append(f"{yr}  n/a" if len(sub) == 0 else f"{yr} {((1+sub).prod()-1)*100:+8.0f}%")
        P(f"  연도별({n2}배): " + " | ".join(out))

P("\n"+"="*128); P("[5] 다리를 하나씩 빼면 — 무엇이 기여하나 (균등가중 ×3)"); P("="*128)
base = pf(np.ones(len(df.columns))/len(df.columns), 3)
P(f"  전체: 월 {base['mon']:+.2f}%, MDD {base['mdd']:.1f}%")
for c in df.columns:
    sub = df.drop(columns=[c])
    tot = sub.mean(axis=1)*3
    eq = (1+tot).cumprod()
    if (eq <= 0.01).any(): P(f"    −{c:>14s}: 파산"); continue
    a = (eq.iloc[-1]**(12/len(tot))-1)*100
    m = ((1+a/100)**(1/12)-1)*100
    P(f"    −{c:>14s}: 월 {m:+6.2f}% ({m-base['mon']:+5.2f}%p)  MDD {(eq/eq.cummax()-1).min()*100:6.1f}%")
