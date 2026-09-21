"""z25 — 캐스케이드 지문을 '진입 트리거'가 아니라 '사전 준비'에 쓴다.

z24 기각: 캐스케이드 탐지 후 시장가 진입은 25개 지문 전부 음수(−5.7~−15.2bp).
         이유는 볼트에 두 번 기록된 원리 — 지문은 사건이 끝난 뒤에 관측되고,
         급락을 시장가로 쫓으면 최악의 가격에 체결된다.

올바른 구현: 지문을 **사전 준비**에 쓴다.
  캐스케이드가 임박한 국면이면 → 지정가를 더 깊게 깔고 / 더 크게 건다.
  사건이 터지면 내 지정가가 이미 거기 있다.

측정: 봉 시작 시점의 지문 상태 → 그 봉의 딥 거래 성과.
      (지문은 봉 시작 시점 값만 쓴다 = 미리 정하는 것)
"""
import sys, warnings, glob
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R/"research"))
import z2_lib as L2
from freesia import store


def P(*a, end="\n"):
    print(" ".join(str(x) for x in a), end=end, flush=True)


S = "BTCUSDT"
d = L2.D[S]; idx = d["idx"]; N = len(d["c"])
O, H, Lo, C, RV = d["o"], d["h"], d["l"], d["c"], d["rv"]
warm = 200000; Y = L2.YRS

k = store.load("klines_1m", S).set_index("ts").reindex(idx)
VOL = k["volume"].values.astype(float)
TRD = k["trades"].values.astype(float)
TBV = k["taker_buy_volume"].values.astype(float)


def load_p(kind):
    fs = sorted(glob.glob(str(R/"data"/"parquet"/kind/S/"*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.sort_values("ts").drop_duplicates("ts")


def to_min(df, cols):
    s = df.set_index("ts")[cols]
    ti = pd.DatetimeIndex(s.index).floor("min").as_unit("ms")
    lag = int(max(1, np.median(np.diff(ti.asi8)/60000.0)))
    pos = idx.searchsorted(ti)
    out = {}
    for c in cols:
        a = np.full(N, np.nan); v = s[c].values.astype(float)
        ok = (pos >= 0) & (pos < N)
        a[pos[ok]] = v[ok]
        out[c] = pd.Series(a).ffill().shift(lag).values
    return out


m5 = load_p("metrics_5m")
mm = to_min(m5, [c for c in m5.columns if c != "ts"])
bd = load_p("bookdepth")
bm = to_min(bd, [c for c in bd.columns if c != "ts"])
P("데이터 준비 완료\n")

# ── 사전 준비용 지문: '캐스케이드가 터질 조건이 갖춰졌는가' ──
F = {}
oi = pd.Series(mm["sum_open_interest_value"])
F["OI수준z"] = ((oi-oi.rolling(43200).mean())/oi.rolling(43200).std()).values
F["OI증가1주"] = oi.pct_change(10080).values
F["레버리지축적"] = (oi.pct_change(10080).values
                 / np.maximum(pd.Series(C).pct_change(10080).abs().values, 1e-4))
ls = mm.get("count_long_short_ratio")
if ls is not None:
    s_ = pd.Series(ls)
    F["롱쏠림z"] = ((s_-s_.rolling(43200).mean())/s_.rolling(43200).std()).values
tls = mm.get("sum_taker_long_short_vol_ratio")
if tls is not None:
    s_ = pd.Series(tls)
    F["테이커롱쏠림z"] = ((s_-s_.rolling(43200).mean())/s_.rolling(43200).std()).values
if "n-1" in bm and "n-5" in bm:
    F["호가기울기"] = bm["n-5"]/np.maximum(bm["n-1"], 1)
    deep = pd.Series(bm["n-5"])
    F["깊은벽수준z"] = ((deep-deep.rolling(1440).mean())/deep.rolling(1440).std()).values
    tot = None
    for lv in (1, 2, 3, 4, 5):
        for sg in ("n+", "n-"):
            kk = f"{sg}{lv}"
            if kk in bm: tot = bm[kk] if tot is None else tot+bm[kk]
    t_ = pd.Series(tot)
    F["총잔량z"] = ((t_-t_.rolling(1440).mean())/t_.rolling(1440).std()).values
    F["잔량얇음"] = -F["총잔량z"]
F["rv분위"] = pd.Series(RV).rolling(43200).rank(pct=True).shift(1).values
F["직전1일수익"] = pd.Series(C).pct_change(1440).shift(1).values
sz = pd.Series(VOL/np.maximum(TRD, 1))
F["평균체결크기z"] = ((sz-sz.rolling(43200).mean())/sz.rolling(43200).std()).shift(1).values
P(f"사전 지문 {len(F)}개\n")


@numba.njit(cache=True)
def dip(o, h, l, c, rvv, barmin, k0, k1, k2, w0, w1, w2, eps, slip, fee, start, n_end):
    """총자본 기준(정직한 회계). 봉 시작 시점 인덱스도 함께 반환."""
    K = 200000
    ea = np.zeros(K, np.int64); t0a = np.zeros(K, np.int64); er = np.zeros(K)
    k = 0; wtot = w0+w1+w2
    nb = (n_end-start)//barmin
    for b in range(nb):
        t0 = start+b*barmin; t1 = t0+barmin-1
        if t1 >= n_end-1: break
        r = rvv[t0]
        if np.isnan(r) or r <= 0: continue
        px0 = o[t0]
        tot = 0.0; wsum = 0.0; ft = -1
        for li in range(3):
            kk = k0 if li == 0 else (k1 if li == 1 else k2)
            ww = w0 if li == 0 else (w1 if li == 1 else w2)
            T = px0*(1.0-r*kk); trig = T*(1.0-eps/1e4)
            got = -1
            for j in range(t0, t1+1):
                if l[j] <= trig: got = j; break
            if got >= 0:
                fill = T*(1.0+slip/1e4)
                tot += ww*(c[t1]/fill-1.0-fee/1e4); wsum += ww
                if ft < 0: ft = got
        if wsum > 0:
            ea[k] = ft; t0a[k] = t0; er[k] = tot/wtot; k += 1
            if k >= K-2: break
    return ea[:k], t0a[:k], er[:k]


BAR = 15
ea, t0a, er = dip(O, H, Lo, C, RV, BAR, 6.0, 8.0, 12.0, 1.0, 3.0, 6.0,
                  5.0, 10.0, 10.0, warm, N)
P(f"기준 딥(15분봉 rv×6·8·12, 총자본 기준): {len(er):,}건 (연 {len(er)/Y:.0f}), "
  f"거래당 {er.mean()*1e4:+.2f}bp, 승률 {np.mean(er>0)*100:.2f}%\n")


def expq_at(vals):
    v = np.asarray(vals, float); q = np.full(len(v), np.nan)
    for i in range(500, len(v), 200):
        x = v[:i]; x = x[np.isfinite(x)]
        if len(x) < 300: continue
        q[i:i+200] = np.searchsorted(np.sort(x), v[i:i+200])/len(x)
    return q


P("="*126); P("[1] 봉 시작 시점 지문 → 그 봉 딥의 성과 (미리 정하는 것)"); P("="*126)
P(f"  {'지문':>16s} {'차이bp':>8s} | {'상위30% bp':>11s} {'승률%':>7s} {'n':>6s} | "
  f"{'하위30% bp':>11s} {'승률%':>7s} {'n':>6s}")
ROWS = []
for nm, arr in F.items():
    v = np.asarray(arr, float)[t0a]
    if np.isfinite(v).sum() < 800: continue
    q = expq_at(v)
    m = np.isfinite(q)
    hi = m & (q >= 0.7); lo = m & (q <= 0.3)
    if hi.sum() < 200 or lo.sum() < 200: continue
    ROWS.append((er[hi].mean()*1e4-er[lo].mean()*1e4, nm,
                 er[hi].mean()*1e4, np.mean(er[hi] > 0)*100, int(hi.sum()),
                 er[lo].mean()*1e4, np.mean(er[lo] > 0)*100, int(lo.sum())))
ROWS.sort(reverse=True)
for diff, nm, hb, hw, hn, lb, lw, ln in ROWS:
    P(f"  {nm:>16s} {diff:+8.2f} | {hb:+11.2f} {hw:7.2f} {hn:6d} | {lb:+11.2f} {lw:7.2f} {ln:6d}")

P("\n"+"="*126); P("[2] 상위 지문으로 '깊이를 키우면' 어떻게 되나"); P("="*126)
P("  가설: 캐스케이드 임박 국면에서는 더 깊게 깔아야 한다(더 큰 프리미엄을 받는다)")
TOP = [r[1] for r in ROWS[:3]] + [r[1] for r in ROWS[-2:]]
P(f"\n  {'지문':>16s} {'국면':>8s} " + " ".join(f"{'rv×'+str(m):>14s}" for m in (4, 6, 9, 12, 18)))
for nm in TOP:
    v = np.asarray(F[nm], float)[t0a]
    q = expq_at(v)
    for lab, msk in (("상위30%", np.isfinite(q) & (q >= 0.7)),
                     ("하위30%", np.isfinite(q) & (q <= 0.3))):
        row = []
        for mult in (4, 6, 9, 12, 18):
            ea2, t02, er2 = dip(O, H, Lo, C, RV, BAR, float(mult), mult*1.35, mult*2.0,
                                1.0, 3.0, 6.0, 5.0, 10.0, 10.0, warm, N)
            v2 = np.asarray(F[nm], float)[t02]
            q2 = expq_at(v2)
            m2 = (np.isfinite(q2) & (q2 >= 0.7)) if lab == "상위30%" else (np.isfinite(q2) & (q2 <= 0.3))
            if m2.sum() < 150: row.append("      -       "); continue
            row.append(f"{er2[m2].mean()*1e4:+7.2f}/{np.mean(er2[m2]>0)*100:5.1f}")
        P(f"  {nm:>16s} {lab:>8s} " + " ".join(f"{x:>14s}" for x in row))
P("  (셀 = 거래당bp / 승률%)")

P("\n"+"="*126); P("[3] 최선 조합 — 지문으로 국면을 고르고 깊이를 맞춤"); P("="*126)
best = None
for nm in [r[1] for r in ROWS[:5]]:
    for thr in (0.5, 0.7, 0.8):
        for mult in (4, 6, 9, 12, 18):
            ea2, t02, er2 = dip(O, H, Lo, C, RV, BAR, float(mult), mult*1.35, mult*2.0,
                                1.0, 3.0, 6.0, 5.0, 10.0, 10.0, warm, N)
            v2 = np.asarray(F[nm], float)[t02]
            q2 = expq_at(v2)
            m2 = np.isfinite(q2) & (q2 >= thr)
            if m2.sum() < 300: continue
            bp = er2[m2].mean()*1e4
            if best is None or bp > best[0]:
                best = (bp, nm, thr, mult, m2.sum(), np.mean(er2[m2] > 0)*100)
if best:
    bp, nm, thr, mult, n, wr = best
    P(f"  ★ {nm} 상위{(1-thr)*100:.0f}% × 깊이 rv×{mult} → 거래당 {bp:+.2f}bp, "
      f"승률 {wr:.2f}%, {n}건 (연 {n/Y:.0f})")
    P(f"     기준(전체·rv×6) 거래당 {er.mean()*1e4:+.2f}bp 대비 {bp-er.mean()*1e4:+.2f}bp")
P("\n  ※ 총자본 기준 회계(z18에서 정정한 것). 체결층 정규화가 아니다.")
