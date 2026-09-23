"""R6 새 정보원 전수 조사 — ±4% 배리어 라벨에 대해 미사용 데이터를 전부 재측정.

왜 다시 하나: z0.20은 ±1% 배리어 라벨로 지표를 골랐다. 배리어가 4배 넓어지면
잡는 현상 자체가 달라진다(±1%는 마이크로구조, ±4%는 수일 스윙).
호가창은 ±1%에서 무효였지만 ±4%에서는 다를 수 있다.

데이터: metrics_5m(미결제약정·상위트레이더 롱숏비·테이커비율), funding, premium,
       bookdepth(±1~5% 잔량), ext(DVOL·베이시스·코인베이스 프리미엄·공포탐욕·스테이블코인)
전부 shift로 룩어헤드 차단, 확장창 z로 정규화."""
import sys, warnings, glob
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R/"research"))
import z0_20_lib as L2


def P(*a, end="\n"):
    print(" ".join(str(x) for x in a), end=end, flush=True)


S = "BTCUSDT"; Y = L2.YRS; STEP = L2.STEP
d = L2.D[S]; idx = d["idx"]
O, H, Lo, C = d["o"], d["h"], d["l"], d["c"]
N = len(C)
warm = 200000


def load(kind, sym=S):
    fs = sorted(glob.glob(str(R/"data"/"parquet"/kind/sym/"*.parquet")))
    if not fs: return None
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.sort_values("ts").drop_duplicates("ts")


def to_min(df, cols):
    """분 단위 인덱스에 맞춰 채우되, **데이터 자체의 주기만큼 지연**시킨다.

    ★ 룩어헤드 방지의 핵심: 일봉 데이터에서 D일 00:00 타임스탬프의 값은
      보통 D일이 끝나야 확정된다. 그걸 00:00부터 쓰면 최대 24시간을 미리 보는 것이다.
      (이 함정에 한 번 빠졌다 — 온체인 PriceUSD가 ±2% 방향을 80% 맞혔다)
      따라서 관측 간격의 중앙값(분)만큼 lag를 주고, 최소 1분은 항상 민다."""
    s = df.set_index("ts")[cols]
    ti = pd.DatetimeIndex(s.index).floor("min").as_unit("ms")
    gaps = np.diff(ti.asi8)/60000.0
    lag = int(max(1, np.median(gaps))) if len(gaps) else 1
    out = {"__lag": lag}
    pos = idx.searchsorted(ti)
    for c in cols:
        a = np.full(N, np.nan)
        v = s[c].values.astype(float)
        ok = (pos >= 0) & (pos < N)
        a[pos[ok]] = v[ok]
        out[c] = pd.Series(a).ffill().shift(lag).values
    return out


P("데이터 로드 중...")
FE = {}


def add(name, arr):
    """메모리 보호: 분 단위 배열을 만들자마자 15분 간격 float32로 줄여 보관한다.
    (라벨도 15분 격자이므로 정보 손실 없음. 150특징×350만행=4GB → 0.1GB)"""
    a = np.asarray(arr, dtype=np.float64)
    if len(a) != N: return
    if np.isfinite(a).sum() < 100000: return
    FE[name] = a[::STEP].astype(np.float32)
    del a


# --- metrics_5m ---
m5 = load("metrics_5m")
if m5 is not None:
    cols = [c for c in m5.columns if c != "ts"]
    mm = to_min(m5, cols)
    P(f"  metrics_5m: {len(m5):,}행, {m5.ts.min().date()}~, 지연 {mm['__lag']}분")
    oi = mm.get("sum_open_interest_value")
    if oi is not None:
        for w in (60, 240, 1440, 4320, 10080):
            add(f"OI변화{w}", pd.Series(oi).pct_change(w).values)
            add(f"OI가격괴리{w}", (pd.Series(oi).pct_change(w).values
                                - pd.Series(C).pct_change(w).shift(1).values))
    for key, nm in (("sum_toptrader_long_short_ratio", "상위트레이더비율"),
                    ("count_toptrader_long_short_ratio", "상위트레이더수비율"),
                    ("count_long_short_ratio", "전체롱숏비"),
                    ("sum_taker_long_short_vol_ratio", "테이커롱숏비")):
        v = mm.get(key)
        if v is None: continue
        add(f"{nm}", v)
        for w in (240, 1440, 4320):
            add(f"{nm}변화{w}", pd.Series(v).diff(w).values)
            add(f"{nm}z{w}", ((pd.Series(v) - pd.Series(v).rolling(w).mean())
                              / pd.Series(v).rolling(w).std()).values)

# --- funding ---
fd = load("funding")
if fd is not None:
    _f = to_min(fd, ["rate"]); P(f"  funding: {len(fd):,}행, 지연 {_f['__lag']}분")
    ff = _f["rate"]
    add("펀딩", ff)
    for w in (1440, 4320, 10080, 43200):
        add(f"펀딩누적{w}", pd.Series(ff).rolling(w).sum().values)
        add(f"펀딩z{w}", ((pd.Series(ff)-pd.Series(ff).rolling(w).mean())
                         / pd.Series(ff).rolling(w).std()).values)

# --- premium ---
pm = load("premium_1m")
if pm is not None:
    _p = to_min(pm, ["close"]); P(f"  premium_1m: {len(pm):,}행, 지연 {_p['__lag']}분")
    pp = _p["close"]
    add("프리미엄", pp)
    for w in (60, 240, 1440, 4320):
        add(f"프리미엄ma{w}", pd.Series(pp).rolling(w).mean().values)
        add(f"프리미엄z{w}", ((pd.Series(pp)-pd.Series(pp).rolling(w).mean())
                            / pd.Series(pp).rolling(w).std()).values)

# --- bookdepth ---
bd = load("bookdepth")
if bd is not None:
    have = [c for c in bd.columns if c != "ts"]
    bm = to_min(bd, have)
    P(f"  bookdepth: {len(bd):,}행, {bd.ts.min().date()}~, 지연 {bm['__lag']}분")
    for lvl in (1, 2, 3, 4, 5):
        pos_, neg_ = f"n+{lvl}", f"n-{lvl}"
        if pos_ in bm and neg_ in bm:
            a, b = bm[pos_], bm[neg_]
            add(f"호가불균형{lvl}", (b-a)/(a+b))
            for w in (240, 1440):
                imb = pd.Series((b-a)/(a+b))
                add(f"호가불균형{lvl}z{w}", ((imb-imb.rolling(w).mean())/imb.rolling(w).std()).values)
    if "n+1" in bm and "n+5" in bm:
        add("호가기울기매도", bm["n+5"]/np.maximum(bm["n+1"], 1))
    if "n-1" in bm and "n-5" in bm:
        add("호가기울기매수", bm["n-5"]/np.maximum(bm["n-1"], 1))
        if "n+1" in bm and "n+5" in bm:
            add("호가기울기차", bm["n-5"]/np.maximum(bm["n-1"], 1)
                - bm["n+5"]/np.maximum(bm["n+1"], 1))
    tot = None
    for lvl in (1, 2, 3, 4, 5):
        for sgn in ("n+", "n-"):
            k = f"{sgn}{lvl}"
            if k in bm: tot = bm[k] if tot is None else tot+bm[k]
    if tot is not None:
        add("호가총잔량", tot)
        for w in (1440, 4320):
            t_ = pd.Series(tot)
            add(f"호가총잔량z{w}", ((t_-t_.rolling(w).mean())/t_.rolling(w).std()).values)

# --- ext ---
for fn, nm, col in (("dvol_btc_1h", "내재변동성", None), ("basis_btc", "베이시스", None),
                    ("cb_btc_1d", "코인베이스프리미엄", None), ("fng", "공포탐욕", None),
                    ("stablecoins", "스테이블코인", None), ("cm_btc", "온체인", None)):
    f = R/"data"/"ext"/f"{fn}.parquet"
    if not f.exists(): continue
    df = pd.read_parquet(f).reset_index()
    tcol = None
    for cand in ("ts", "time", "date", "index", "timestamp"):
        if cand in df.columns: tcol = cand; break
    if tcol is None: tcol = df.columns[0]
    df = df.loc[:, ~df.columns.duplicated()]
    df["__ts"] = pd.to_datetime(df[tcol], utc=True, errors="coerce")
    df = df.dropna(subset=["__ts"]).drop(columns=[tcol], errors="ignore")
    df = df.rename(columns={"__ts": "ts"}).sort_values("ts").drop_duplicates("ts")
    num = [c for c in df.columns if c != "ts" and pd.api.types.is_numeric_dtype(df[c])]
    if not num: continue
    mm2 = to_min(df, num[:6])
    P(f"  {fn}: {len(df):,}행, 열 {len(num)}개, 지연 {mm2['__lag']}분")
    for c in num[:6]:
        v = mm2[c]
        add(f"{nm}_{c}", v)
        for w in (1440, 10080):
            s_ = pd.Series(v)
            add(f"{nm}_{c}z{w}", ((s_-s_.rolling(w).mean())/s_.rolling(w).std()).values)
            add(f"{nm}_{c}변화{w}", s_.pct_change(w).values)

P(f"\n총 특징 {len(FE)}개 생성\n")


@numba.njit(cache=True)
def label(o, h, l, c, up, dn, maxh, step, start, n_end):
    m = n_end//step
    y = np.full(m, -1, np.int8)
    for i in range(start//step, m):
        t = i*step
        if t+maxh >= n_end: break
        f = t+1; px = o[f]; U = px*(1.0+up); D = px*(1.0-dn)
        for j in range(f, min(f+maxh, n_end-1)+1):
            if l[j] <= D: y[i] = 0; break
            if h[j] >= U: y[i] = 1; break
    return y


def exp_z(v, warmn=200000//STEP, stp=5000//STEP):
    """15분 격자 배열용 확장창 z (시점 t까지의 평균·표준편차만 사용)."""
    v = np.asarray(v, dtype=np.float64)
    z = np.full(len(v), np.nan)
    for i in range(warmn, len(v), stp):
        x = v[:i]; x = x[np.isfinite(x)]
        if len(x) < 500: continue
        mu = x.mean(); sd = x.std()
        if sd <= 0: continue
        z[i:i+stp] = (v[i:i+stp]-mu)/sd
    return z


P("="*130); P("[1] 특징별 ±배리어 방향 정확도 (확장창 z, 상하위 10% 진입 기준)"); P("="*130)
RES = {}
for bar, mh in ((0.02, 20160), (0.04, 43200)):
    y = label(O, H, Lo, C, bar, bar, mh, STEP, warm, N)
    base = np.mean(y[y >= 0] == 1)*100
    P(f"\n  ══ 배리어 ±{bar*100:.0f}%  (판정가능 {np.sum(y>=0):,}, 기저 상승우선 {base:.2f}%) ══")
    rows = []
    for nm, v in FE.items():
        zz = exp_z(v)[:len(y)]
        ok = (y >= 0) & np.isfinite(zz)
        if ok.sum() < 3000: continue
        hi = np.quantile(zz[ok], 0.90); lo = np.quantile(zz[ok], 0.10)
        su = ok & (zz >= hi); sd_ = ok & (zz <= lo)
        if su.sum() < 300 or sd_.sum() < 300: continue
        au = np.mean(y[su] == 1)*100
        ad = np.mean(y[sd_] == 0)*100
        rows.append(((au+ad)/2, au, ad, nm, su.sum()))
    rows.sort(reverse=True)
    RES[bar] = rows
    P(f"  {'특징':>28s} {'합%':>7s} {'위쪽%':>7s} {'아래쪽%':>8s} {'n':>7s}")
    for a, au, ad, nm, k in rows[:15]:
        P(f"  {nm:>28s} {a:7.2f} {au:7.2f} {ad:8.2f} {k:7d}")
    P(f"  … 하위 5개:")
    for a, au, ad, nm, k in rows[-5:]:
        P(f"  {nm:>28s} {a:7.2f} {au:7.2f} {ad:8.2f} {k:7d}")

P("\n"+"="*130); P("[2] z2 신호와 겹치지 않는가 — 상위 특징들의 상관"); P("="*130)
from z0_20_final import build_sigs, exp_z as ez2
SG = build_sigs(S)
nm4 = [n for n in SG if n.endswith("4320")]
Zc = [ez2(np.asarray(SG[n], float)) for n in nm4]
ml = min(min(len(z) for z in Zc), len(C)//STEP)
SIG = np.nanmean(np.vstack([z[:ml] for z in Zc]), axis=0)
top = [r[3] for r in RES[0.04][:8]]
P(f"  {'특징':>28s} {'z2신호와 상관':>14s}")
for nm in top:
    z = exp_z(FE[nm])[:ml]
    ok = np.isfinite(z) & np.isfinite(SIG[:len(z)])
    if ok.sum() < 1000: continue
    c_ = np.corrcoef(z[ok], SIG[:len(z)][ok])[0, 1]
    P(f"  {nm:>28s} {c_:14.3f}")
