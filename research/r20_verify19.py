"""R20 — R19(z0.14 딥 × 호가기울기) 확정 검증.

R19 결과: 호가기울기 분위별 거래당 −1 → −2 → +17 → +33bp (완벽 단조),
         상위50%만 거래하면 공격 +149% → +316%, 대조군(하위30%) 공격 −22%.

확인할 것:
  ①플라시보 — 같은 건수를 무작위 선택 (필터 값 셔플)
  ②문턱 고원 — 상위 30~80% 전 구간
  ③비용 민감도
  ④연도별 (호가 구간 2023~2026)
  ⑤다른 호가 특징 (불균형, 총잔량)과의 비교
  ⑥z0.14의 다른 깊이·봉 조합에서도 먹히나
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
FEATS = {}


def bookload(sym, idx):
    fs = sorted(glob.glob(str(R/"data"/"parquet"/"bookdepth"/sym/"*.parquet")))
    if not fs: return {}
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").drop_duplicates("ts")
    cols = [c for c in df.columns if c != "ts"]
    s = df.set_index("ts")[cols]
    ti = pd.DatetimeIndex(s.index).floor("min").as_unit("ms")
    gaps = np.diff(ti.asi8)/60000.0
    lag = int(max(1, np.median(gaps))) if len(gaps) else 1
    pos = idx.searchsorted(ti); N = len(idx)
    raw = {}
    for c in cols:
        a = np.full(N, np.nan); v = s[c].values.astype(float)
        ok = (pos >= 0) & (pos < N)
        a[pos[ok]] = v[ok]
        raw[c] = pd.Series(a).ffill().shift(lag).values
    out = {}
    if "n-1" in raw and "n-5" in raw:
        out["호가기울기"] = raw["n-5"]/np.maximum(raw["n-1"], 1)
    if "n+1" in raw and "n+5" in raw:
        out["호가기울기매도"] = raw["n+5"]/np.maximum(raw["n+1"], 1)
        if "호가기울기" in out:
            out["호가기울기차"] = out["호가기울기"]-out["호가기울기매도"]
    for lv in (1, 3, 5):
        a, b = raw.get(f"n+{lv}"), raw.get(f"n-{lv}")
        if a is not None and b is not None:
            out[f"호가불균형{lv}"] = (b-a)/(a+b)
    tot = None
    for lv in (1, 2, 3, 4, 5):
        for sg in ("n+", "n-"):
            k = f"{sg}{lv}"
            if k in raw: tot = raw[k] if tot is None else tot+raw[k]
    if tot is not None: out["호가총잔량"] = tot
    return out


P("호가 특징 로드...")
for s in SYM:
    FEATS[s] = bookload(s, X.RAW[s]["idx"])
P(f"  특징 {list(FEATS[SYM[0]].keys())}\n")

X.BL = (15,)
X.CACHE.clear()
D14 = {6.0: 1.0, 8.0: 3.0, 12.0: 6.0}
E = X.build(D14, 0)
E.sort(key=lambda z: z[0])


def attach(E, fname):
    out = []
    for e in E:
        a = e[0]; s = e[5][0]
        f = FEATS.get(s, {}).get(fname)
        v = np.nan if f is None or a >= len(f) else f[a]
        out.append((e, v))
    return out


def expq(vals, w0=300, st=100):
    v = np.asarray(vals, float); q = np.full(len(v), np.nan)
    for i in range(w0, len(v), st):
        x = v[:i]; x = x[np.isfinite(x)]
        if len(x) < 200: continue
        q[i:i+st] = np.searchsorted(np.sort(x), v[i:i+st])/len(x)
    return q


def perf(sel, quiet=True):
    if len(sel) < 60: return None
    r = np.array([x[2] for x in sel])
    res = {"n": len(sel), "bp": r.mean()*1e4, "wr": np.mean(r > 0)*100}
    for w, Nm, t in ((1.0, 1.0, "무레버"), (2.0, 3.0, "저레버"), (3.0, 5.0, "중간"), (5.0, 8.0, "공격")):
        eq, fl, liq = X.sim(sel, w, Nm)
        m = met(eq, fl, liq)
        res[t] = None if m is None else m[0]
        if t == "공격" and m: res["공격MDD"] = m[1]
    return res


AT = attach(E, "호가기울기")
have = [(e, v) for e, v in AT if np.isfinite(v)]
have.sort(key=lambda z: z[0][0])
q = expq([v for _, v in have])
base = [e for e, _ in have]
rb = perf(base)
P("="*128); P(f"[0] 기준 — 호가 가용 구간 {len(base)}건"); P("="*128)
P(f"  필터 없음: 거래당 {rb['bp']:+.1f}bp 승률 {rb['wr']:.2f}% | "
  f"무레버 {rb['무레버']:+.0f}% 저레버 {rb['저레버']:+.0f}% 중간 {rb['중간']:+.0f}% 공격 {rb['공격']:+.0f}%")

P("\n"+"="*128); P("[1] 문턱 고원 — 상위 x%만 거래"); P("="*128)
P(f"  {'문턱':>9s} {'n':>6s} {'거래당bp':>9s} {'승률%':>7s} {'무레버%':>8s} {'저레버%':>8s} {'중간%':>8s} {'공격%':>8s}")
for lo in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
    sel = [e for (e, v), qq in zip(have, q) if np.isfinite(qq) and qq >= lo]
    r = perf(sel)
    if r is None: continue
    f = lambda k: "    파산" if r.get(k) is None else f"{r[k]:8.0f}"
    P(f"  상위{(1-lo)*100:4.0f}% {r['n']:6d} {r['bp']:+9.1f} {r['wr']:7.2f} "
      f"{f('무레버')} {f('저레버')} {f('중간')} {f('공격')}")

P("\n"+"="*128); P("[2] 플라시보 — 같은 건수를 무작위 선택 (300회)"); P("="*128)
rng = np.random.default_rng(2024)
sel50 = [e for (e, v), qq in zip(have, q) if np.isfinite(qq) and qq >= 0.5]
r50 = perf(sel50)
nsel = len(sel50)
pb, pa = [], []
for it in range(300):
    ix = rng.choice(len(base), nsel, replace=False)
    sub = [base[i] for i in sorted(ix)]
    rr = np.array([x[2] for x in sub])
    pb.append(rr.mean()*1e4)
    if it < 60:
        eq, fl, liq = X.sim(sub, 5.0, 8.0)
        m = met(eq, fl, liq)
        pa.append(-100.0 if m is None else m[0])
pb = np.array(pb); pa = np.array(pa)
P(f"  거래당bp 실제 {r50['bp']:+7.2f} | 무작위 평균 {pb.mean():+6.2f} 최대 {pb.max():+6.2f} "
  f"→ 백분위 {np.mean(pb < r50['bp'])*100:5.1f}%")
P(f"  공격 연%  실제 {r50['공격']:+7.0f} | 무작위 평균 {pa.mean():+6.0f} 최대 {pa.max():+6.0f} "
  f"→ 백분위 {np.mean(pa < r50['공격'])*100:5.1f}%  (60회)")

P("\n"+"="*128); P("[3] 다른 호가 특징 비교 (상위50%)"); P("="*128)
P(f"  {'특징':>14s} {'n':>6s} {'거래당bp':>9s} {'승률%':>7s} {'무레버%':>8s} {'중간%':>8s} {'공격%':>8s}")
for fn in FEATS[SYM[0]].keys():
    at = attach(E, fn)
    hv = [(e, v) for e, v in at if np.isfinite(v)]
    if len(hv) < 400: continue
    hv.sort(key=lambda z: z[0][0])
    qq = expq([v for _, v in hv])
    sel = [e for (e, v), x2 in zip(hv, qq) if np.isfinite(x2) and x2 >= 0.5]
    r = perf(sel)
    if r is None: continue
    f = lambda k: "    파산" if r.get(k) is None else f"{r[k]:8.0f}"
    P(f"  {fn:>14s} {r['n']:6d} {r['bp']:+9.1f} {r['wr']:7.2f} {f('무레버')} {f('중간')} {f('공격')}")

P("\n"+"="*128); P("[4] 비용 민감도 (상위50%)"); P("="*128)
for fm, ft, eps, lab in ((2, 5, 0.0002, "기준 메2/테5"), (3, 6, 0.0005, "보수"),
                         (4, 7, 0.001, "비관"), (5, 8, 0.002, "최악")):
    X.CACHE.clear()
    E2 = X.build(D14, 0, eps=eps, fm=fm, ft=ft)
    E2.sort(key=lambda z: z[0])
    at = attach(E2, "호가기울기")
    hv = [(e, v) for e, v in at if np.isfinite(v)]
    hv.sort(key=lambda z: z[0][0])
    qq = expq([v for _, v in hv])
    sel = [e for (e, v), x2 in zip(hv, qq) if np.isfinite(x2) and x2 >= 0.5]
    r = perf(sel); r0 = perf([e for e, _ in hv])
    f = lambda rr, k: "  파산" if rr is None or rr.get(k) is None else f"{rr[k]:+6.0f}"
    P(f"  {lab:>14s}: 필터없음 거래당{r0['bp']:+6.1f}bp 공격{f(r0,'공격')}% | "
      f"상위50% {r['bp']:+6.1f}bp 공격{f(r,'공격')}%")
X.CACHE.clear()

P("\n"+"="*128); P("[5] 연도별 (호가 구간 2023~)"); P("="*128)
E3 = X.build(D14, 0)
E3.sort(key=lambda z: z[0])
at = attach(E3, "호가기울기")
hv = [(e, v) for e, v in at if np.isfinite(v)]
hv.sort(key=lambda z: z[0][0])
qq = expq([v for _, v in hv])
P(f"  {'연도':>6s} {'필터없음 n':>10s} {'bp':>7s} {'승률%':>7s} | {'상위50% n':>10s} {'bp':>7s} {'승률%':>7s}")
for yr in (2023, 2024, 2025, 2026):
    a0 = [e for e, _ in hv if e[4].year == yr]
    a1 = [e for (e, v), x2 in zip(hv, qq) if np.isfinite(x2) and x2 >= 0.5 and e[4].year == yr]
    if len(a0) < 30: continue
    r0 = np.array([x[2] for x in a0]); r1 = np.array([x[2] for x in a1]) if a1 else np.array([0.0])
    P(f"  {yr:6d} {len(a0):10d} {r0.mean()*1e4:+7.1f} {np.mean(r0>0)*100:7.2f} | "
      f"{len(a1):10d} {r1.mean()*1e4:+7.1f} {np.mean(r1>0)*100:7.2f}")

P("\n"+"="*128); P("[6] 다른 깊이 조합에서도 먹히나"); P("="*128)
P(f"  {'깊이':>22s} {'필터없음 공격%':>14s} {'상위50% 공격%':>14s} {'배수':>7s}")
for lab, dep in (("6·8·12 (1:3:6)", {6.0: 1.0, 8.0: 3.0, 12.0: 6.0}),
                 ("8·12·16 (1:3:6)", {8.0: 1.0, 12.0: 3.0, 16.0: 6.0}),
                 ("5·7·10 (1:3:6)", {5.0: 1.0, 7.0: 3.0, 10.0: 6.0}),
                 ("6·10 (1:3)", {6.0: 1.0, 10.0: 3.0}),
                 ("8·12·16·20", {8.0: 1.0, 12.0: 2.0, 16.0: 4.0, 20.0: 6.0})):
    X.CACHE.clear()
    Ex = X.build(dep, 0)
    Ex.sort(key=lambda z: z[0])
    at = attach(Ex, "호가기울기")
    hv2 = [(e, v) for e, v in at if np.isfinite(v)]
    if len(hv2) < 400: continue
    hv2.sort(key=lambda z: z[0][0])
    q2 = expq([v for _, v in hv2])
    r0 = perf([e for e, _ in hv2])
    r1 = perf([e for (e, v), x2 in zip(hv2, q2) if np.isfinite(x2) and x2 >= 0.5])
    if r0 is None or r1 is None: continue
    a0 = r0.get("공격"); a1 = r1.get("공격")
    f = lambda v: "         파산" if v is None else f"{v:14.0f}"
    mult = "   -" if (a0 is None or a1 is None or a0 <= 0) else f"{a1/a0:6.2f}배"
    P(f"  {lab:>22s} {f(a0)} {f(a1)} {mult:>7s}")
X.BL = (240, 360, 480, 720, 960, 1440)
X.CACHE.clear()
