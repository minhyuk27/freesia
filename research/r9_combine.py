"""R9 독립 정보원 결합 — 호가기울기 + z0.20 가격신호 + 역발상 포지셔닝.

R6 발견(±4% 배리어, 기저 53.78%):
  호가기울기매수  59.01% (위 68.72%), z0.20과 상관 0.002 ← 완전 독립
  호가기울기차    56.26% (위 66.85%), z0.20과 상관 0.001
  전체롱숏비      44.15% → 뒤집으면 55.85% (역발상)
  베이시스/공포탐욕/코인베이스프리미엄 55~56%

독립적이므로 결합하면 승률이 실제로 오를 수 있다.
천장 = exp(N_독립 × g)에서 g를 올리는 유일한 길.

주의: 호가 데이터는 2023-01부터라 표본이 3.7년뿐. 전 구간 비교를 함께 낸다."""
import sys, warnings, glob
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R/"research"))
import z0_20_lib as L2
from z0_20_final import build_sigs, exp_z as ez


def P(*a, end="\n"):
    print(" ".join(str(x) for x in a), end=end, flush=True)


S = "BTCUSDT"; Y = L2.YRS; MMR = L2.MMR; STEP = L2.STEP
d = L2.D[S]; idx = d["idx"]; rv = d["rv"]
O, H, Lo, C = d["o"], d["h"], d["l"], d["c"]
N = len(C); warm = 200000
G = N//STEP


def load(kind):
    fs = sorted(glob.glob(str(R/"data"/"parquet"/kind/S/"*.parquet")))
    if not fs: return None
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.sort_values("ts").drop_duplicates("ts")


def to_grid(df, cols):
    """15분 격자로. 데이터 주기만큼 지연 강제(z6에서 확인한 룩어헤드 함정)."""
    s = df.set_index("ts")[cols]
    ti = pd.DatetimeIndex(s.index).floor("min").as_unit("ms")
    gaps = np.diff(ti.asi8)/60000.0
    lag = int(max(1, np.median(gaps))) if len(gaps) else 1
    pos = idx.searchsorted(ti)
    out = {}
    for c in cols:
        a = np.full(N, np.nan)
        v = s[c].values.astype(float)
        ok = (pos >= 0) & (pos < N)
        a[pos[ok]] = v[ok]
        out[c] = pd.Series(a).ffill().shift(lag).values[::STEP][:G].astype(np.float32)
    return out, lag


P("정보원 구축 중...")
F = {}
bd = load("bookdepth")
if bd is not None:
    cols = [c for c in bd.columns if c != "ts"]
    bm, lag = to_grid(bd, cols)
    P(f"  bookdepth {len(bd):,}행, 지연 {lag}분, 시작 {bd.ts.min().date()}")
    if "n-1" in bm and "n-5" in bm:
        F["호가기울기매수"] = bm["n-5"]/np.maximum(bm["n-1"], 1)
    if "n+1" in bm and "n+5" in bm and "호가기울기매수" in F:
        F["호가기울기차"] = F["호가기울기매수"] - bm["n+5"]/np.maximum(bm["n+1"], 1)
    for lv in (3, 4, 5):
        a, b = bm.get(f"n+{lv}"), bm.get(f"n-{lv}")
        if a is not None and b is not None:
            F[f"호가불균형{lv}"] = (b-a)/(a+b)

m5 = load("metrics_5m")
if m5 is not None:
    cols = [c for c in m5.columns if c != "ts"]
    mm, lag = to_grid(m5, cols)
    P(f"  metrics_5m 지연 {lag}분")
    if "count_long_short_ratio" in mm:
        F["역_전체롱숏비"] = -mm["count_long_short_ratio"]
    if "count_toptrader_long_short_ratio" in mm:
        F["역_상위트레이더수"] = -mm["count_toptrader_long_short_ratio"]
    if "sum_open_interest_value" in mm:
        F["OI변화4320"] = pd.Series(mm["sum_open_interest_value"]).pct_change(4320//STEP).values

for fn, nm, pick in (("basis_btc", "베이시스", "next_ann"), ("fng", "공포탐욕", None),
                     ("cb_btc_1d", "코인베이스프리미엄", None)):
    f = R/"data"/"ext"/f"{fn}.parquet"
    if not f.exists(): continue
    df = pd.read_parquet(f).reset_index()
    tcol = next((c for c in ("ts", "time", "date", "index") if c in df.columns), df.columns[0])
    df = df.loc[:, ~df.columns.duplicated()]
    df["__ts"] = pd.to_datetime(df[tcol], utc=True, errors="coerce")
    df = df.dropna(subset=["__ts"]).drop(columns=[tcol], errors="ignore").rename(columns={"__ts": "ts"})
    df = df.sort_values("ts").drop_duplicates("ts")
    num = [c for c in df.columns if c != "ts" and pd.api.types.is_numeric_dtype(df[c])]
    if pick and pick in num: num = [pick]
    else: num = num[:1]
    gg, lag = to_grid(df, num)
    P(f"  {fn} 지연 {lag}분")
    v = gg[num[0]]
    s_ = pd.Series(v)
    F[f"{nm}z"] = ((s_-s_.rolling(10080//STEP).mean())/s_.rolling(10080//STEP).std()).values

SG = build_sigs(S)
names = list(SG.keys())
Zn = {n: ez(np.asarray(SG[n], float)) for n in names}
ml = min(min(len(Zn[n]) for n in names), G)
F["z2가격"] = np.nanmean(np.vstack([Zn[n][:ml] for n in names if n.endswith("4320")]), axis=0)
for k in list(F):
    F[k] = np.asarray(F[k], float)[:ml]
P(f"  정보원 {len(F)}개, 격자 {ml:,}\n")


def egz(v, wm=200000//STEP, st=5000//STEP):
    v = np.asarray(v, float); z = np.full(len(v), np.nan)
    for i in range(wm, len(v), st):
        x = v[:i]; x = x[np.isfinite(x)]
        if len(x) < 500: continue
        mu, sd = x.mean(), x.std()
        if sd > 0: z[i:i+st] = (v[i:i+st]-mu)/sd
    return z


ZF = {k: egz(v) for k, v in F.items()}


@numba.njit(cache=True)
def lab(o, h, l, c, bar, maxh, step, start, n_end, m):
    y = np.full(m, -1, np.int8)
    for i in range(start//step, m):
        t = i*step
        if t+maxh >= n_end: break
        f = t+1; px = o[f]; U = px*(1+bar); D = px*(1-bar)
        for j in range(f, min(f+maxh, n_end-1)+1):
            if l[j] <= D: y[i] = 0; break
            if h[j] >= U: y[i] = 1; break
    return y


BAR = 0.04
y = lab(O, H, Lo, C, BAR, 43200, STEP, warm, N, ml)
okmask = y >= 0
P("="*126); P(f"[1] 개별 정보원 (±{BAR*100:.0f}%, 상위10% 진입) — 표본 구간 통일 전/후"); P("="*126)
both = okmask & np.isfinite(ZF["호가기울기매수"]) & np.isfinite(ZF["z2가격"])
P(f"  호가 가용구간 {both.sum():,}점 (전체 {okmask.sum():,}점의 {both.sum()/okmask.sum()*100:.0f}%)")
P(f"  그 구간 기저 상승우선 {np.mean(y[both]==1)*100:.2f}%  |  전체 구간 기저 {np.mean(y[okmask]==1)*100:.2f}%")
P(f"\n  {'정보원':>20s} {'전체구간%':>10s} {'호가구간%':>10s} {'n':>7s}")
for k, z in ZF.items():
    o1 = okmask & np.isfinite(z)
    if o1.sum() < 2000: continue
    th = np.quantile(z[o1], 0.90)
    s1 = o1 & (z >= th)
    o2 = both & np.isfinite(z)
    th2 = np.quantile(z[o2], 0.90) if o2.sum() > 500 else np.nan
    s2 = o2 & (z >= th2) if np.isfinite(th2) else o2 & False
    a1 = np.mean(y[s1] == 1)*100 if s1.sum() > 100 else np.nan
    a2 = np.mean(y[s2] == 1)*100 if s2.sum() > 100 else np.nan
    P(f"  {k:>20s} {a1:10.2f} {a2:10.2f} {s2.sum():7d}")

P("\n"+"="*126); P("[2] 결합 — z2가격 + 독립 정보원 (호가 가용구간에서 공정 비교)"); P("="*126)
COMBOS = [
    ("z2가격 단독", ["z2가격"]),
    ("호가기울기매수 단독", ["호가기울기매수"]),
    ("z2 + 호가기울기매수", ["z2가격", "호가기울기매수"]),
    ("z2 + 호가기울기차", ["z2가격", "호가기울기차"]),
    ("z2 + 호가2개", ["z2가격", "호가기울기매수", "호가기울기차"]),
    ("z2 + 역롱숏비", ["z2가격", "역_전체롱숏비"]),
    ("z2 + 베이시스", ["z2가격", "베이시스z"]),
    ("z2 + 공포탐욕", ["z2가격", "공포탐욕z"]),
    ("z2 + 호가 + 역롱숏", ["z2가격", "호가기울기매수", "역_전체롱숏비"]),
    ("z2 + 호가 + 베이시스", ["z2가격", "호가기울기매수", "베이시스z"]),
    ("전부", ["z2가격", "호가기울기매수", "호가기울기차", "역_전체롱숏비", "베이시스z", "공포탐욕z"]),
]
P(f"  {'구성':>22s} " + " ".join(f"{'상위'+str(int(q*100))+'%':>10s}" for q in (0.02, 0.05, 0.10, 0.20)))
BEST = None
for lab_, ks in COMBOS:
    mats = [ZF[k] for k in ks if k in ZF]
    if not mats: continue
    cb = np.nanmean(np.vstack(mats), axis=0)
    row = []
    for q in (0.02, 0.05, 0.10, 0.20):
        o1 = both & np.isfinite(cb)
        if o1.sum() < 1000: row.append("      -   "); continue
        th = np.quantile(cb[o1], 1-q)
        s1 = o1 & (cb >= th)
        if s1.sum() < 80: row.append("      -   "); continue
        a = np.mean(y[s1] == 1)*100
        row.append(f"{a:8.2f}/{s1.sum():d}"[:10].rjust(10))
        if BEST is None or a > BEST[0]: BEST = (a, lab_, ks, q, s1.sum())
    P(f"  {lab_:>22s} " + " ".join(f"{x:>10s}" for x in row))
P("  (셀 = 정확도% / 표본수)")
if BEST:
    P(f"\n  ★ 최고: {BEST[1]} 상위{BEST[3]*100:.0f}% → {BEST[0]:.2f}% (n={BEST[4]})")
    P(f"     같은 구간 z2가격 단독 대비, 기저 {np.mean(y[both]==1)*100:.2f}%")

P("\n"+"="*126); P("[3] 정보원 간 상관 — 진짜 독립인가"); P("="*126)
ks = [k for k in ("z2가격", "호가기울기매수", "호가기울기차", "역_전체롱숏비", "베이시스z", "공포탐욕z") if k in ZF]
M = np.vstack([ZF[k] for k in ks])
ok = both & np.all(np.isfinite(M), axis=0)
cr = np.corrcoef(M[:, ok])
P("        " + " ".join(f"{k[:8]:>9s}" for k in ks))
for i, k in enumerate(ks):
    P(f"  {k[:6]:>6s} " + " ".join(f"{cr[i,j]:9.3f}" for j in range(len(ks))))

P("\n"+"="*126); P("[4] 실제 백테스트 — 결합 신호로 z4 구조 (±4%, 단일 포지션)"); P("="*126)


@numba.njit(cache=True)
def bt(o, h, l, c, sg, th, bar, maxh, step, nom, mmr, fee, start, n_end):
    cap = 1.0; peak = 1.0; mdd = 0.0; ntr = 0; nwin = 0; sr = 0.0
    t = start
    while t < n_end-maxh-2:
        i = t//step
        if i >= len(sg) or not np.isfinite(sg[i]) or sg[i] < th: t += step; continue
        f = t+1; px = o[f]; U = px*(1+bar); D = px*(1-bar)
        e = -1; res = 0.0; end = min(f+maxh, n_end-1)
        for j in range(f, end+1):
            if l[j] <= D: e = j; res = -bar; break
            if h[j] >= U: e = j; res = bar; nwin += 1; break
        if e < 0: e = end; res = c[e]/px-1.0
        pnl = cap*nom*(res-fee/1e4)
        if pnl < -cap: pnl = -cap
        cap += pnl
        if cap > peak: peak = cap
        dd = cap/peak-1
        if dd < mdd: mdd = dd
        ntr += 1; nwin += 0; sr += res
        t = e+1
        if cap < 0.01: break
    return cap, mdd, ntr, nwin, sr


st0 = int(np.nanargmax(np.isfinite(ZF["호가기울기매수"])))*STEP
st0 = max(st0, warm)
yrs2 = (N-st0)/525600
P(f"  호가 가용 시작 {idx[st0].date()}, 검증 {yrs2:.2f}년")
P(f"\n  {'구성':>22s} {'문턱':>7s} {'거래':>5s} {'연거래':>6s} {'승률%':>7s} "
  + " ".join(f"{'명목'+str(n)+'배':>12s}" for n in (1, 2, 3, 5)))
for lab_, ks2 in COMBOS:
    mats = [ZF[k] for k in ks2 if k in ZF]
    if not mats: continue
    cb = np.nanmean(np.vstack(mats), axis=0)
    fin = cb[np.isfinite(cb)]
    for q in (0.05, 0.10):
        th = float(np.quantile(fin, 1-q))
        row = []
        base = None
        for nom in (1, 2, 3, 5):
            cap, mdd, ntr, nwin, sr = bt(O, H, Lo, C, cb, th, BAR, 43200, STEP,
                                         float(nom), MMR, 10.0, st0, N)
            if ntr < 10: row.append("        -   "); continue
            if base is None: base = (ntr, nwin/ntr*100)
            row.append("        파산" if cap < 0.01 else f"{(cap**(1/yrs2)-1)*100:+12.1f}")
        if base is None: continue
        P(f"  {lab_:>22s} {q*100:6.0f}% {base[0]:5d} {base[0]/yrs2:6.1f} {base[1]:7.2f} "
          + " ".join(f"{x:>12s}" for x in row))
