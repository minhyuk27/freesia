"""z14 — z12 딥에 필터를 건다. 어떤 딥이 더 잘 되돌아오는가.

논리: 켈리는 승률에 비선형으로 반응한다(승률 55%→65%면 켈리 1.9배→9.15배).
      z12는 이미 승률 60.6%(BTC)·64.5%(ETH). 여기서 몇 %p만 올려도 폭발적이다.

가설: 딥이 체결된 **그 시점의 시장 상태**가 되돌림 강도를 예측한다.
  - 호가기울기(깊은 매수벽/얕은 매수벽) — z2와 상관 0.002인 독립 정보원
  - 프리미엄(무기한 vs 지수) — '급함의 가격' 직접 관측
  - 펀딩, 미결제약정 변화, 테이커 롱숏비
  - 그 봉의 딥 깊이 자체, rv 분위

전부 체결 시점 기준, 데이터 주기만큼 지연 강제(z6에서 배운 함정)."""
import sys, warnings, glob
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R/"research"))
import z2_lib as L2


def P(*a, end="\n"):
    print(" ".join(str(x) for x in a), end=end, flush=True)


Y = L2.YRS
EPS, SLIP, FEE = 5.0, 10.0, 10.0
warm = 200000
BAR = 720


@numba.njit(cache=True)
def dip(o, h, l, c, rvv, barmin, k0, k1, k2, w0, w1, w2, eps, slip, fee, start, n_end):
    K = 200000
    ea = np.zeros(K, np.int64); er = np.zeros(K); dp = np.zeros(K); mae = np.zeros(K)
    k = 0
    nb = (n_end-start)//barmin
    for b in range(nb):
        t0 = start+b*barmin; t1 = t0+barmin-1
        if t1 >= n_end-1: break
        r = rvv[t0]
        if np.isnan(r) or r <= 0: continue
        px0 = o[t0]
        tot = 0.0; wsum = 0.0; ft = -1; dsum = 0.0; worst = 0.0
        for li in range(3):
            kk = k0 if li == 0 else (k1 if li == 1 else k2)
            ww = w0 if li == 0 else (w1 if li == 1 else w2)
            T = px0*(1.0 - r*kk); trig = T*(1.0-eps/1e4)
            got = -1
            for j in range(t0, t1+1):
                if l[j] <= trig: got = j; break
            if got >= 0:
                fill = T*(1.0+slip/1e4)
                tot += ww*(c[t1]/fill-1.0-fee/1e4); wsum += ww
                dsum += ww*(r*kk)
                lo = fill
                for j in range(got, t1+1):
                    if l[j] < lo: lo = l[j]
                dd = lo/fill-1.0
                if dd < worst: worst = dd
                if ft < 0: ft = got
        if wsum > 0:
            ea[k] = ft; er[k] = tot/wsum; dp[k] = dsum/wsum; mae[k] = worst; k += 1
            if k >= K-2: break
    return ea[:k], er[:k], dp[:k], mae[:k]


def load(kind, sym):
    fs = sorted(glob.glob(str(R/"data"/"parquet"/kind/sym/"*.parquet")))
    if not fs: return None
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.sort_values("ts").drop_duplicates("ts")


def to_min(df, cols, idx, N):
    """데이터 주기만큼 지연 강제."""
    s = df.set_index("ts")[cols]
    ti = pd.DatetimeIndex(s.index).floor("min").as_unit("ms")
    gaps = np.diff(ti.asi8)/60000.0
    lag = int(max(1, np.median(gaps))) if len(gaps) else 1
    pos = idx.searchsorted(ti)
    out = {}
    for cname in cols:
        a = np.full(N, np.nan)
        v = s[cname].values.astype(float)
        ok = (pos >= 0) & (pos < N)
        a[pos[ok]] = v[ok]
        out[cname] = pd.Series(a).ffill().shift(lag).values
    return out, lag


def build(S):
    d = L2.D[S]; idx = d["idx"]; N = len(d["c"])
    F = {}
    bd = load("bookdepth", S)
    if bd is not None:
        bm, lg = to_min(bd, [c for c in bd.columns if c != "ts"], idx, N)
        if "n-1" in bm and "n-5" in bm:
            F["호가기울기"] = bm["n-5"]/np.maximum(bm["n-1"], 1)
        for lv in (1, 5):
            a, b = bm.get(f"n+{lv}"), bm.get(f"n-{lv}")
            if a is not None and b is not None:
                F[f"호가불균형{lv}"] = (b-a)/(a+b)
        tot = None
        for lv in (1, 2, 3, 4, 5):
            for sg in ("n+", "n-"):
                kk = f"{sg}{lv}"
                if kk in bm: tot = bm[kk] if tot is None else tot+bm[kk]
        if tot is not None: F["호가총잔량"] = tot
    pm = load("premium_1m", S)
    if pm is not None:
        pp, _ = to_min(pm, ["close"], idx, N)
        v = pp["close"]
        F["프리미엄"] = v
        F["프리미엄ma60"] = pd.Series(v).rolling(60).mean().values
    fd = load("funding", S)
    if fd is not None:
        ff, _ = to_min(fd, ["rate"], idx, N)
        F["펀딩"] = ff["rate"]
        F["펀딩누적3일"] = pd.Series(ff["rate"]).rolling(4320).sum().values
    m5 = load("metrics_5m", S)
    if m5 is not None:
        mm, _ = to_min(m5, [c for c in m5.columns if c != "ts"], idx, N)
        if "sum_open_interest_value" in mm:
            F["OI변화1일"] = pd.Series(mm["sum_open_interest_value"]).pct_change(1440).values
        for kk, nm in (("sum_taker_long_short_vol_ratio", "테이커롱숏비"),
                       ("count_long_short_ratio", "전체롱숏비")):
            if kk in mm: F[nm] = mm[kk]
    lr = pd.Series(np.log(d["c"])).diff()
    F["rv분위"] = pd.Series(d["rv"]).rolling(43200).rank(pct=True).shift(1).values
    F["직전1일수익"] = pd.Series(d["c"]).pct_change(1440).shift(1).values
    F["직전3일수익"] = pd.Series(d["c"]).pct_change(4320).shift(1).values
    return F


for S in ("BTCUSDT", "ETHUSDT"):
    d = L2.D[S]; idx = d["idx"]; N = len(d["c"])
    KS = (15.0, 21.0, 30.0)
    ea, er, dp, mae = dip(d["o"], d["h"], d["l"], d["c"], d["rv"], BAR, *KS,
                          1.0, 3.0, 6.0, EPS, SLIP, FEE, warm, N)
    F = build(S)
    F["딥깊이"] = None  # 체결 기준 값은 따로
    P("="*126); P(f"[{S}] 딥 {len(er):,}건, 거래당 {er.mean()*1e4:+.1f}bp, 승률 {np.mean(er>0)*100:.2f}%")
    P("="*126)
    rows = []
    for nm, arr in F.items():
        if arr is None: continue
        v = np.asarray(arr, float)[ea]
        ok = np.isfinite(v)
        if ok.sum() < 300: continue
        # 확장창 분위로 상/하위 30% 비교 (룩어헤드 없음: 과거 체결분 기준)
        q = np.full(len(v), np.nan)
        for i in range(200, len(v), 50):
            x = v[:i]; x = x[np.isfinite(x)]
            if len(x) < 100: continue
            q[i:i+50] = np.searchsorted(np.sort(x), v[i:i+50])/len(x)
        m = np.isfinite(q)
        hi = m & (q >= 0.7); lo = m & (q <= 0.3)
        if hi.sum() < 100 or lo.sum() < 100: continue
        rows.append((er[hi].mean()*1e4 - er[lo].mean()*1e4, nm,
                     er[hi].mean()*1e4, np.mean(er[hi] > 0)*100, int(hi.sum()),
                     er[lo].mean()*1e4, np.mean(er[lo] > 0)*100, int(lo.sum())))
    # 딥 깊이 자체
    q = np.full(len(dp), np.nan)
    for i in range(200, len(dp), 50):
        x = dp[:i]
        if len(x) < 100: continue
        q[i:i+50] = np.searchsorted(np.sort(x), dp[i:i+50])/len(x)
    m = np.isfinite(q); hi = m & (q >= 0.7); lo = m & (q <= 0.3)
    rows.append((er[hi].mean()*1e4-er[lo].mean()*1e4, "딥깊이(체결)",
                 er[hi].mean()*1e4, np.mean(er[hi] > 0)*100, int(hi.sum()),
                 er[lo].mean()*1e4, np.mean(er[lo] > 0)*100, int(lo.sum())))
    rows.sort(reverse=True)
    P(f"  {'필터':>16s} {'차이bp':>8s} | {'상위30% bp':>11s} {'승률%':>7s} {'n':>6s} | "
      f"{'하위30% bp':>11s} {'승률%':>7s} {'n':>6s}")
    for diff, nm, hb, hw, hn, lb, lw, ln in rows:
        P(f"  {nm:>16s} {diff:+8.1f} | {hb:+11.1f} {hw:7.2f} {hn:6d} | {lb:+11.1f} {lw:7.2f} {ln:6d}")
    P("")
