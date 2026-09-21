"""z21 — 검증된 조각들을 하나로 묶고 명목 상한을 찾는다.

묶을 것(전부 독립 검증 통과, 회계 정상):
  A. z1.4 + 호가기울기 상위50%  — 15분봉 깊은 딥, 연 ~400건
  B. z0.3 깊은 딥               — 4h~1d봉, 연 17건, 거래당 +707bp
  C. z4 스윙 (±4% 배리어)        — 신호 상위10%, 연 47건
  D. ETH숏 (±4% 배리어)          — 신호 하위10%, 연 81건

A·B는 같은 exit_tier 공유자본 엔진에 함께 넣으면 자본 충돌까지 정확히 계산된다.
C·D는 배리어 전략이라 월별 수익으로 합산한다.

목표: 검증된 것만으로 도달 가능한 최대 명목과 그때의 월수익."""
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
    out = {}
    for c in ("n-1", "n-5"):
        a = np.full(N, np.nan); v = s[c].values.astype(float)
        ok = (pos >= 0) & (pos < N)
        a[pos[ok]] = v[ok]
        out[c] = pd.Series(a).ffill().shift(lag).values
    return out["n-5"]/np.maximum(out["n-1"], 1)


P("준비...")
BOOK = {s: bookslope(s, X.RAW[s]["idx"]) for s in SYM}


def expq(v, w0=300, st=100):
    v = np.asarray(v, float); q = np.full(len(v), np.nan)
    for i in range(w0, len(v), st):
        x = v[:i]; x = x[np.isfinite(x)]
        if len(x) < 200: continue
        q[i:i+st] = np.searchsorted(np.sort(x), v[i:i+st])/len(x)
    return q


def filt_book(E, thr=0.5):
    """호가기울기 상위(1-thr) 만 남긴다. 호가 없는 구간은 그대로 통과."""
    E = sorted(E, key=lambda z: z[0])
    vals = []
    for e in E:
        bk = BOOK.get(e[5][0])
        vals.append(np.nan if bk is None or e[0] >= len(bk) else bk[e[0]])
    vals = np.array(vals, float)
    have = np.isfinite(vals)
    q = np.full(len(vals), np.nan)
    hv = np.where(have)[0]
    if len(hv) > 400:
        qq = expq(vals[hv])
        q[hv] = qq
    keep = (~have) | (~np.isfinite(q)) | (q >= thr)
    return [e for e, k in zip(E, keep) if k], int(have.sum()), int(keep.sum())


# A: z1.4 (15분봉)
X.BL = (15,); X.CACHE.clear()
A_all = X.build({6.0: 1.0, 8.0: 3.0, 12.0: 6.0}, 0)
A, nb, nk = filt_book(A_all, 0.5)
P(f"  A z1.4+호가상위50%: 전체 {len(A_all)} → {len(A)}건 (호가가용 {nb}, 통과 {nk})")

# B: z0.3 (4h~1d봉)
X.BL = (240, 360, 480, 720, 960, 1440); X.CACHE.clear()
B = X.build({10.0: 1.0, 14.0: 3.0, 20.0: 6.0}, 0)
P(f"  B z0.3 깊은딥: {len(B)}건")


def perf(E, w, Nm):
    eq, fl, liq = X.sim(E, w, Nm)
    m = met(eq, fl, liq)
    return (None, None) if m is None else (m, m[5])


P("\n"+"="*126); P("[1] A·B 단독 vs 공유자본 결합 (exit_tier 한 엔진)"); P("="*126)
P(f"  {'구성':>22s} {'n':>6s} " + " ".join(f"{'w'+str(w)+'/N'+str(n):>16s}"
                                          for w, n in ((1, 1), (2, 3), (3, 5), (5, 8), (8, 12))))
SETS = [("A z1.4+호가", A), ("B z0.3", B), ("A+B 공유자본", sorted(A+B, key=lambda z: z[0]))]
CURVES = {}
for lab, E in SETS:
    row = []
    for w, Nm in ((1, 1), (2, 3), (3, 5), (5, 8), (8, 12)):
        m, eq = perf(E, float(w), float(Nm))
        row.append("      파산      " if m is None else f"{m[0]:+8.0f}%/{m[1]:6.0f}%")
        if lab == "A+B 공유자본" and w == 3: CURVES["AB"] = eq
    P(f"  {lab:>22s} {len(E):6d} " + " ".join(f"{x:>16s}" for x in row))
P("  (셀 = 연수익% / MDD%)")

P("\n"+"="*126); P("[2] A+B 명목 상한 탐색"); P("="*126)
AB = sorted(A+B, key=lambda z: z[0])
P(f"  {'w':>5s} {'Nmax':>6s} {'연수익%':>10s} {'MDD%':>8s} {'최악%':>8s} {'24~':>8s} {'25~':>8s} {'청산':>5s}")
best = None
for w in (1, 2, 3, 4, 5, 6, 8, 10, 12):
    for Nm in (1, 2, 3, 5, 8, 12, 16):
        if Nm < w: continue
        eq, fl, liq = X.sim(AB, float(w), float(Nm))
        m = met(eq, fl, liq)
        if m is None: continue
        if best is None or m[0] > best[0]: best = (m[0], w, Nm, m)
        if (w, Nm) in ((1, 1), (2, 3), (3, 5), (4, 6), (5, 8), (6, 10), (8, 12), (10, 16), (12, 16)):
            P(f"  {w:5d} {Nm:6d} {m[0]:10.0f} {m[1]:8.1f} {m[2]:8.1f} {m[3]:8.0f} {m[4]:8.0f} {liq:5d}")
if best:
    a, w, Nm, m = best
    P(f"\n  ★ 최고: w={w} Nmax={Nm} → 연 {a:+.0f}% (월 {((1+a/100)**(1/12)-1)*100:+.2f}%), "
      f"MDD {m[1]:.1f}%, 최악 {m[2]:.1f}%")

P("\n"+"="*126); P("[3] C 스윙 · D ETH숏 추가 (월별 합산)"); P("="*126)


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


warm = 200000
SER = {}
for S, side, lab in (("BTCUSDT", 1, "C 스윙(BTC롱)"), ("ETHUSDT", -1, "D ETH숏")):
    d = L2.D[S]
    SG = build_sigs(S)
    Z = {n: exp_z(np.asarray(SG[n], float)) for n in SG}
    ml = min(min(len(Z[n]) for n in Z), len(d["c"])//L2.STEP)
    sig = np.nanmean(np.vstack([Z[n][:ml] for n in Z if n.endswith("4320")]), axis=0)
    fin = sig[np.isfinite(sig)]
    th = float(np.quantile(fin, 0.90 if side == 1 else 0.10))
    ea, er = barrier(d["o"], d["h"], d["l"], d["c"], sig, th, 0.04, 43200, L2.STEP,
                     warm, len(d["c"]), side)
    SER[lab] = pd.Series(er, index=pd.DatetimeIndex(d["idx"][ea])).resample("ME").sum()
    P(f"  {lab}: {len(er)}건, 거래당 {er.mean()*1e4:+.1f}bp, 승률 {np.mean(er>0)*100:.2f}%")

if "AB" in CURVES and CURVES["AB"] is not None:
    ab = CURVES["AB"]
    abm = (ab.resample("ME").last()/ab.resample("ME").first()-1).dropna()
    SER["AB(w3/N5)"] = abm
df = pd.concat(SER, axis=1).fillna(0)
P(f"\n  월별 상관:")
cr = df.corr()
P("             " + " ".join(f"{c[:10]:>12s}" for c in cr.columns))
for i, r in cr.iterrows():
    P(f"  {i[:11]:>11s} " + " ".join(f"{r[c]:12.3f}" for c in cr.columns))


def pf(ws, nom):
    tot = (df*np.array(ws)).sum(axis=1)*nom
    eq = (1+tot).cumprod()
    if (eq <= 0.01).any(): return None
    yy = len(tot)/12
    a = (eq.iloc[-1]**(1/yy)-1)*100
    return dict(ann=a, mon=((1+a/100)**(1/12)-1)*100,
                mdd=(eq/eq.cummax()-1).min()*100, mwr=np.mean(tot > 0)*100)


P(f"\n  {'구성':>24s} " + " ".join(f"{'×'+str(n):>16s}" for n in (1, 2, 3, 5, 8)))
cols = list(df.columns)
CFG = [("AB 단독", [1.0 if "AB" in c else 0.0 for c in cols]),
       ("AB+C 균등", [0.5 if ("AB" in c or "스윙" in c) else 0.0 for c in cols]),
       ("AB+D 균등", [0.5 if ("AB" in c or "ETH숏" in c) else 0.0 for c in cols]),
       ("셋 균등", [1/3]*3),
       ("AB 2 : C 1 : D 1", [0.25 if "AB" not in c else 0.5 for c in cols])]
for lab, ws in CFG:
    row = []
    for nom in (1, 2, 3, 5, 8):
        r = pf(ws, nom)
        row.append("      파산      " if r is None else f"{r['ann']:+8.0f}%/{r['mdd']:6.0f}%")
    P(f"  {lab:>24s} " + " ".join(f"{x:>16s}" for x in row))
P("  (셀 = 연수익% / MDD%)")

P("\n"+"="*126); P("[4] 최종 3단 레버리지"); P("="*126)
bestpf = None
for lab, ws in CFG:
    for nom in (1, 2, 3, 4, 5, 6, 8, 10):
        r = pf(ws, nom)
        if r and (bestpf is None or r["mon"] > bestpf[0]): bestpf = (r["mon"], lab, ws, nom, r)
if bestpf:
    mon, lab, ws, nom, r = bestpf
    P(f"  최고 조합: {lab}")
    P(f"\n  {'구간':>12s} {'명목':>6s} {'연수익%':>12s} {'월%':>9s} {'MDD%':>8s} {'월승률%':>8s}")
    for lb, n2 in (("무레버리지", 1), ("중간", 2), ("중간+", 3), ("공격", 5), ("최대공격", 8), ("극단", 10)):
        rr = pf(ws, n2)
        P(f"  {lb:>12s} {n2:5d}x " + ("        파산" if rr is None
                                       else f"{rr['ann']:12.0f} {rr['mon']:9.2f} {rr['mdd']:8.1f} {rr['mwr']:8.1f}"))
    tot = (df*np.array(ws)).sum(axis=1)
    out = []
    for yr in range(2020, 2027):
        sub = (tot*3)[tot.index.year == yr]
        out.append(f"{yr}  n/a" if len(sub) == 0 else f"{yr} {((1+sub).prod()-1)*100:+7.0f}%")
    P("  연도별(3배): " + " | ".join(out))
