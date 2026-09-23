"""R19 — 검증된 z0.3/z1.4 딥에 검증된 호가창 필터를 붙인다.

둘 다 독립 검증을 통과했고 회계도 정상이다(exit_tier는 명목을 먼저 정한다).
  z0.3/z1.4: 깊은 딥 유동성 공급. exit_tier 공유자본 엔진.
  호가기울기(n−5/n−1): ±4% 배리어에서 59.01%, z0.20 가격신호와 상관 0.002.

가설: 깊은 매수벽이 두터운 자리의 딥이 더 잘 되돌아온다.
      호가기울기는 딥 메커니즘과 같은 것(유동성)을 다른 각도로 본다.

주의: 호가 데이터는 2023-01부터. 같은 구간 비교를 반드시 함께 낸다."""
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


def bookfeat(sym, idx):
    fs = sorted(glob.glob(str(R/"data"/"parquet"/"bookdepth"/sym/"*.parquet")))
    if not fs: return None
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").drop_duplicates("ts")
    s = df.set_index("ts")[[c for c in df.columns if c != "ts"]]
    ti = pd.DatetimeIndex(s.index).floor("min").as_unit("ms")
    gaps = np.diff(ti.asi8)/60000.0
    lag = int(max(1, np.median(gaps))) if len(gaps) else 1
    pos = idx.searchsorted(ti)
    N = len(idx)
    out = {}
    for c in ("n-1", "n-5", "n+1", "n+5"):
        if c not in s.columns: continue
        a = np.full(N, np.nan)
        v = s[c].values.astype(float)
        ok = (pos >= 0) & (pos < N)
        a[pos[ok]] = v[ok]
        out[c] = pd.Series(a).ffill().shift(lag).values
    if "n-1" in out and "n-5" in out:
        return out["n-5"]/np.maximum(out["n-1"], 1), lag
    return None, lag


P("호가 특징 구축...")
BOOK = {}
for s in SYM:
    idx = X.RAW[s]["idx"]
    b, lag = bookfeat(s, idx)
    if b is not None:
        BOOK[s] = b
        first = int(np.argmax(np.isfinite(b)))
        P(f"  {s}: 지연 {lag}분, 시작 {idx[first].date()}")
P("")


def expq_by(vals):
    """확장창 분위 — 과거 관측만 사용."""
    v = np.asarray(vals, float)
    q = np.full(len(v), np.nan)
    for i in range(300, len(v), 100):
        x = v[:i]; x = x[np.isfinite(x)]
        if len(x) < 200: continue
        q[i:i+100] = np.searchsorted(np.sort(x), v[i:i+100])/len(x)
    return q


def tag(E):
    """각 거래의 체결 시점 호가기울기 분위를 붙인다."""
    out = []
    for e in E:
        a, bb, r, m, ts, key, rvb, wgt = e
        s = key[0]
        bk = BOOK.get(s)
        v = np.nan if bk is None or a >= len(bk) else bk[a]
        out.append(e + (v,))
    return out


def run(E, lab, full=False):
    if len(E) < 60:
        P(f"  {lab:40s} 표본부족({len(E)})"); return None
    r = np.array([x[2] for x in E])
    line = (f"  {lab:40s} n={len(E):5d} {r.mean()*1e4:+7.0f}bp 승률{(r>0).mean():.3f} |")
    best = None
    for w, Nm, t2 in ((1.0, 1.0, "무레버"), (2.0, 3.0, "저레버"), (3.0, 5.0, "중간"), (5.0, 8.0, "공격")):
        eq, fl, liq = X.sim(E, w, Nm)
        mm = met(eq, fl, liq)
        line += f" {t2}{'파산' if mm is None else f'{mm[0]:+8.0f}%'}"
        if mm and (best is None or mm[0] > best[0]): best = (mm[0], mm[1], mm[2], t2)
    P(line)
    if full and best:
        P(f"      최고 {best[3]}: MDD {best[1]:.0f}% 최악 {best[2]:.0f}%")
    return best


D03 = {10.0: 1.0, 14.0: 3.0, 20.0: 6.0}
P("="*128); P("[1] z0.3 깊은 딥 × 호가기울기 필터 (출구 A 봉종가)"); P("="*128)
E = X.build(D03, 0)
ET = tag(E)
bk = np.array([x[-1] for x in ET], float)
P(f"  전체 {len(ET)}건 중 호가 가용 {int(np.isfinite(bk).sum())}건 "
  f"({np.isfinite(bk).sum()/len(ET)*100:.0f}%)")
have = [e for e in ET if np.isfinite(e[-1])]
have.sort(key=lambda z: z[0])
q = expq_by([e[-1] for e in have])
P(f"\n  호가 가용 구간({len(have)}건)에서:")
run([e[:8] for e in have], "필터 없음(호가구간)")
for lo in (0.3, 0.5, 0.7):
    sel = [e[:8] for e, qq in zip(have, q) if np.isfinite(qq) and qq >= lo]
    run(sel, f"호가기울기 상위{(1-lo)*100:.0f}%")
for hi in (0.3, 0.5):
    sel = [e[:8] for e, qq in zip(have, q) if np.isfinite(qq) and qq <= hi]
    run(sel, f"호가기울기 하위{hi*100:.0f}% (대조)")

P("\n"+"="*128); P("[2] z1.4 초단기 딥(15분봉) × 호가기울기"); P("="*128)
BL_OLD = X.BL
X.BL = (15,)
X.CACHE.clear()
D14 = {6.0: 1.0, 8.0: 3.0, 12.0: 6.0}
E2 = X.build(D14, 0)
ET2 = tag(E2)
have2 = [e for e in ET2 if np.isfinite(e[-1])]
have2.sort(key=lambda z: z[0])
P(f"  전체 {len(ET2)}건 중 호가 가용 {len(have2)}건")
if len(have2) > 200:
    q2 = expq_by([e[-1] for e in have2])
    run([e[:8] for e in have2], "필터 없음(호가구간)")
    for lo in (0.3, 0.5, 0.7):
        sel = [e[:8] for e, qq in zip(have2, q2) if np.isfinite(qq) and qq >= lo]
        run(sel, f"호가기울기 상위{(1-lo)*100:.0f}%")
    for hi in (0.3,):
        sel = [e[:8] for e, qq in zip(have2, q2) if np.isfinite(qq) and qq <= hi]
        run(sel, f"호가기울기 하위{hi*100:.0f}% (대조)")
X.BL = BL_OLD
X.CACHE.clear()

P("\n"+"="*128); P("[3] 거래당 수익으로 직접 비교 — 호가기울기 분위별"); P("="*128)
for lab, hv in (("z0.3 깊은딥", have), ("z1.4 15분딥", have2)):
    if len(hv) < 200: continue
    qq = expq_by([e[-1] for e in hv])
    r = np.array([e[2] for e in hv])
    P(f"\n  {lab} (n={len(hv)}):")
    P(f"    {'분위':>10s} {'n':>6s} {'거래당bp':>10s} {'승률%':>7s}")
    for a, b, nm in ((0.0, 0.25, "하위25%"), (0.25, 0.5, "25~50%"),
                     (0.5, 0.75, "50~75%"), (0.75, 1.01, "상위25%")):
        m = np.isfinite(qq) & (qq >= a) & (qq < b)
        if m.sum() < 30: continue
        P(f"    {nm:>10s} {int(m.sum()):6d} {r[m].mean()*1e4:+10.0f} {np.mean(r[m]>0)*100:7.2f}")
