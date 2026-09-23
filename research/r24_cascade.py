"""R24 — 완전히 새로운 방향: 청산 캐스케이드 직접 관측.

지금까지의 딥 전략은 "가격이 떨어졌다"는 **결과**만 본다.
이 전략은 **왜 떨어졌는지**를 구분한다:
  - 강제 청산(레버리지 포지션이 터진 것) → 되돌아온다
  - 정보성 매도(누가 알고 파는 것)      → 안 되돌아온다

청산 캐스케이드의 관측 가능한 지문:
  ①미결제약정(OI) 급감 — 포지션이 강제로 닫혔다는 직접 증거
  ②대량 체결 + 큰 평균 체결크기 — 청산 엔진이 시장가로 던진다
  ③테이커 매도 비율 극단 — 청산은 전부 테이커
  ④호가 잔량 급감(벽이 먹힘) → 그리고 급속 회복(다시 채워짐)
  ⑤가격 급락

완전히 새로운 것: **호가 잔량의 시간 변화(delta)**. 지금까지 수준(level)만 봤다.

공격적 구조: 캐스케이드 탐지 즉시 시장가 진입 → 짧게 보유(15~240분) → 배리어 청산.
"""
import sys, warnings, glob
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R/"research"))
import z0_20_lib as L2
from freesia import store


def P(*a, end="\n"):
    print(" ".join(str(x) for x in a), end=end, flush=True)


S = "BTCUSDT"
d = L2.D[S]; idx = d["idx"]; N = len(d["c"])
O, H, Lo, C = d["o"], d["h"], d["l"], d["c"]
warm = 200000

P("원천 데이터 로드...")
k = store.load("klines_1m", S).set_index("ts")
k = k.reindex(idx)
VOL = k["volume"].values.astype(float)
TRD = k["trades"].values.astype(float) if "trades" in k.columns else None
TBV = k["taker_buy_volume"].values.astype(float) if "taker_buy_volume" in k.columns else None
P(f"  klines: volume, trades={TRD is not None}, taker_buy={TBV is not None}")


def load_p(kind, sym=S):
    fs = sorted(glob.glob(str(R/"data"/"parquet"/kind/sym/"*.parquet")))
    if not fs: return None
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
    return out, lag


m5 = load_p("metrics_5m")
mm, lag5 = to_min(m5, [c for c in m5.columns if c != "ts"])
OI = mm.get("sum_open_interest_value")
P(f"  metrics_5m 지연 {lag5}분, OI {'있음' if OI is not None else '없음'}")

bd = load_p("bookdepth")
bm, lagb = to_min(bd, [c for c in bd.columns if c != "ts"])
P(f"  bookdepth 지연 {lagb}분\n")

# ── 캐스케이드 지문 구성 (전부 shift로 확정값만) ──
F = {}
lr = pd.Series(np.log(C)).diff()
rv = d["rv"]
F["급락5분"] = pd.Series(C).pct_change(5).shift(1).values
F["급락15분"] = pd.Series(C).pct_change(15).shift(1).values
F["급락_rv정규"] = F["급락5분"]/np.maximum(rv*np.sqrt(5), 1e-9)
vz = pd.Series(VOL)
F["거래량z"] = ((vz-vz.rolling(1440).mean())/vz.rolling(1440).std()).shift(1).values
if TRD is not None:
    sz = VOL/np.maximum(TRD, 1)
    s_ = pd.Series(sz)
    F["평균체결크기z"] = ((s_-s_.rolling(1440).mean())/s_.rolling(1440).std()).shift(1).values
if TBV is not None:
    tb = TBV/np.maximum(VOL, 1e-9)
    F["테이커매수비"] = pd.Series(tb).shift(1).values
    F["테이커매도쏠림"] = pd.Series(1.0-tb).rolling(5).mean().shift(1).values
if OI is not None:
    oi = pd.Series(OI)
    F["OI변화30분"] = oi.pct_change(30).values
    F["OI변화60분"] = oi.pct_change(60).values
    F["OI급감"] = -oi.pct_change(30).values
    # OI가 줄면서 가격도 내리면 = 롱 청산
    F["롱청산지표"] = (-oi.pct_change(30).values) * (-F["급락5분"])

# ★ 새로운 것: 호가 잔량의 시간 변화
if "n-1" in bm and "n-5" in bm:
    deep = pd.Series(bm["n-5"]); shal = pd.Series(bm["n-1"])
    F["호가기울기"] = (bm["n-5"]/np.maximum(bm["n-1"], 1))
    for w in (5, 15, 60):
        F[f"깊은벽변화{w}"] = deep.pct_change(w).values
        F[f"얕은벽변화{w}"] = shal.pct_change(w).values
        F[f"벽소멸{w}"] = -deep.pct_change(w).values
    tot = None
    for lv in (1, 2, 3, 4, 5):
        for sg in ("n+", "n-"):
            kk = f"{sg}{lv}"
            if kk in bm: tot = bm[kk] if tot is None else tot+bm[kk]
    if tot is not None:
        t_ = pd.Series(tot)
        for w in (5, 15, 60):
            F[f"총잔량변화{w}"] = t_.pct_change(w).values
        F["총잔량급감"] = -t_.pct_change(15).values
P(f"지문 {len(F)}개 구성\n")


@numba.njit(cache=True)
def fwd(o, h, l, c, starts, up, dn, maxh, cool, fee, n_end):
    K = 200000
    ea = np.zeros(K, np.int64); er = np.zeros(K); k = 0; free = 0
    for i in range(len(starts)):
        t = starts[i]
        if t < free or t >= n_end-maxh-2: continue
        f = t+1; px = o[f]; U = px*(1+up); D = px*(1-dn)
        e = -1; res = 0.0; end = min(f+maxh, n_end-1)
        for j in range(f, end+1):
            if l[j] <= D: e = j; res = -dn; break
            if h[j] >= U: e = j; res = up; break
        if e < 0: e = end; res = c[e]/px-1.0
        ea[k] = f; er[k] = res-fee/1e4; k += 1
        free = e+1+cool
        if k >= K-2: break
    return ea[:k], er[:k]


def expq(v, w0=200000, st=5000):
    v = np.asarray(v, float); q = np.full(len(v), np.nan)
    for i in range(w0, len(v), st):
        x = v[:i]; x = x[np.isfinite(x)]
        if len(x) < 5000: continue
        q[i:i+st] = np.searchsorted(np.sort(x), v[i:i+st])/len(x)
    return q


P("="*126); P("[1] 지문별 단독 성과 — 상위 0.5% 시점에서 시장가 진입, ±1% 배리어 240분"); P("="*126)
P(f"  {'지문':>18s} {'연거래':>6s} {'승률%':>7s} {'거래당bp':>9s} {'무레버 연%':>11s}")
Y = L2.YRS
ROWS = []
for nm, arr in F.items():
    q = expq(arr)
    st = np.where(np.isfinite(q) & (q >= 0.995))[0]
    st = st[st >= warm]
    if len(st) < 200: continue
    ea, er = fwd(O, H, Lo, C, st.astype(np.int64), 0.01, 0.01, 240, 60, 10.0, N)
    if len(er) < 100: continue
    m = pd.Series(er, index=pd.DatetimeIndex(idx[ea])).resample("ME").sum()
    eq = (1+m).cumprod()
    ann = np.nan if (eq <= 0.01).any() else (eq.iloc[-1]**(12/len(m))-1)*100
    ROWS.append((er.mean()*1e4, nm, len(er)/Y, np.mean(er > 0)*100, ann))
ROWS.sort(reverse=True)
for bp, nm, n, wr, a in ROWS:
    P(f"  {nm:>18s} {n:6.0f} {wr:7.2f} {bp:+9.1f} " + ("       파산" if not np.isfinite(a) else f"{a:11.1f}"))

P("\n"+"="*126); P("[2] 상위 지문 — 문턱 × 배리어 (최고 3개)"); P("="*126)
for bp0, nm, _, _, _ in ROWS[:3]:
    q = expq(F[nm])
    P(f"\n  ── {nm} (전체 상위0.5% 거래당 {bp0:+.1f}bp) ──")
    P(f"  {'문턱':>9s} " + " ".join(f"{'±'+str(b*100)+'%':>16s}" for b in (0.005, 0.01, 0.02, 0.04)))
    for thq in (0.999, 0.995, 0.99, 0.98, 0.95):
        st = np.where(np.isfinite(q) & (q >= thq))[0]
        st = st[st >= warm]
        if len(st) < 150: continue
        row = []
        for bar in (0.005, 0.01, 0.02, 0.04):
            mh = int(bar*24000)
            ea, er = fwd(O, H, Lo, C, st.astype(np.int64), bar, bar, mh, 60, 10.0, N)
            if len(er) < 80: row.append("        -       "); continue
            row.append(f"{er.mean()*1e4:+7.1f}bp/{np.mean(er>0)*100:4.1f}")
        P(f"  상위{(1-thq)*100:5.2f}% " + " ".join(f"{x:>16s}" for x in row))
    P("  (셀 = 거래당bp / 승률%)")

P("\n"+"="*126); P("[3] 캐스케이드 결합 — 여러 지문 동시 극단"); P("="*126)
CAND = [nm for _, nm, _, _, _ in ROWS[:8]]
QS = {nm: expq(F[nm]) for nm in CAND}
P(f"  {'조합':>44s} {'연거래':>6s} {'승률%':>7s} {'거래당bp':>9s}")
import itertools
best = None
for r in (1, 2, 3):
    for combo in itertools.combinations(CAND, r):
        sc = np.nanmean(np.vstack([QS[c] for c in combo]), axis=0)
        st = np.where(np.isfinite(sc) & (sc >= 0.98))[0]
        st = st[st >= warm]
        if len(st) < 150: continue
        ea, er = fwd(O, H, Lo, C, st.astype(np.int64), 0.01, 0.01, 240, 60, 10.0, N)
        if len(er) < 100: continue
        bp = er.mean()*1e4
        if best is None or bp > best[0]:
            best = (bp, combo, len(er)/Y, np.mean(er > 0)*100)
        if r <= 2 and bp > 15:
            P(f"  {'+'.join(c[:12] for c in combo):>44s} {len(er)/Y:6.0f} "
              f"{np.mean(er>0)*100:7.2f} {bp:+9.1f}")
if best:
    P(f"\n  ★ 최고 조합: {' + '.join(best[1])}")
    P(f"     연 {best[2]:.0f}건, 승률 {best[3]:.2f}%, 거래당 {best[0]:+.1f}bp")
