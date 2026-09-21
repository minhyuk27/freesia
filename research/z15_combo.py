"""z15 — z12 딥 + 필터 결합. 승률을 올려 켈리를 폭증시킨다.

z14 발견 (양 종목 공통):
  rv분위↑   BTC +49.2bp / ETH +74.1bp  ← 압도적 1위 (고변동 딥이 잘 되돌아온다)
  펀딩·프리미엄↑ BTC +38.5/+22.3 / ETH +41.4/+52.2
  직전1일수익↓  BTC −27.3 / ETH −29.5  (이미 오른 뒤의 딥은 나쁘다)
  호가기울기↑   BTC +22.5 / ETH +1.1

전부 확장창 분위로 결합(룩어헤드 없음). 필터는 진입 문턱과 사이징 둘 다로 써본다."""
import sys, warnings, glob
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R/"research"))
import z2_lib as L2
from z14_dipfilter import dip, build, EPS, SLIP, FEE, warm, BAR


def P(*a, end="\n"):
    print(" ".join(str(x) for x in a), end=end, flush=True)


Y = L2.YRS
KS = (15.0, 21.0, 30.0)


def expq(v, w0=200, st=50):
    """확장창 분위 — 시점 i까지의 과거 체결분만 사용."""
    v = np.asarray(v, float)
    q = np.full(len(v), np.nan)
    for i in range(w0, len(v), st):
        x = v[:i]; x = x[np.isfinite(x)]
        if len(x) < 100: continue
        q[i:i+st] = np.searchsorted(np.sort(x), v[i:i+st])/len(x)
    return q


def curve(er, ea, idx, nom=1.0, w=None):
    if len(er) < 30: return None
    ww = np.ones(len(er)) if w is None else w
    m = pd.Series(nom*ww*er, index=pd.DatetimeIndex(idx[ea])).resample("ME").sum()
    eq = (1+m).cumprod()
    if (eq <= 0.01).any(): return None
    return dict(ann=(eq.iloc[-1]**(12/len(m))-1)*100,
                mon=((eq.iloc[-1]**(12/len(m)))**(1/12)-1)*100,
                mdd=(eq/eq.cummax()-1).min()*100, mwr=np.mean(m > 0)*100, ser=m)


RES = {}
for S in ("BTCUSDT", "ETHUSDT"):
    d = L2.D[S]; idx = d["idx"]; N = len(d["c"])
    ea, er, dp, mae = dip(d["o"], d["h"], d["l"], d["c"], d["rv"], BAR, *KS,
                          1.0, 3.0, 6.0, EPS, SLIP, FEE, warm, N)
    F = build(S)
    Q = {}
    for nm in ("rv분위", "펀딩", "펀딩누적3일", "프리미엄", "프리미엄ma60", "호가기울기",
               "직전1일수익", "직전3일수익", "OI변화1일", "테이커롱숏비", "호가총잔량"):
        a = F.get(nm)
        if a is None: continue
        v = np.asarray(a, float)[ea]
        if np.isfinite(v).sum() < 400: continue
        Q[nm] = expq(v)
    RES[S] = (ea, er, idx, Q)
    P(f"{S}: 딥 {len(er):,}건, 필터 {len(Q)}개 준비")
P("")

COMBOS = {
    "A 필터없음": [],
    "B rv분위만": [("rv분위", 1)],
    "C rv+펀딩": [("rv분위", 1), ("펀딩", 1)],
    "D rv+프리미엄": [("rv분위", 1), ("프리미엄", 1)],
    "E rv+직전수익역": [("rv분위", 1), ("직전1일수익", -1)],
    "F rv+펀딩+직전역": [("rv분위", 1), ("펀딩", 1), ("직전1일수익", -1)],
    "G rv+프리+직전역": [("rv분위", 1), ("프리미엄", 1), ("직전1일수익", -1)],
    "H 넷 다": [("rv분위", 1), ("펀딩", 1), ("프리미엄", 1), ("직전1일수익", -1)],
    "I +호가기울기": [("rv분위", 1), ("펀딩", 1), ("프리미엄", 1), ("직전1일수익", -1), ("호가기울기", 1)],
}


def score(Q, parts):
    if not parts: return None
    acc = None; n = 0
    for nm, sgn in parts:
        if nm not in Q: continue
        v = Q[nm] if sgn > 0 else (1.0-Q[nm])
        acc = v if acc is None else np.nansum(np.vstack([acc, v]), axis=0)
        n += 1
    if acc is None: return None
    return acc/n


for S in ("BTCUSDT", "ETHUSDT"):
    ea, er, idx, Q = RES[S]
    P("="*126); P(f"[{S}] 필터 결합 — 상위 q%만 진입 (무레버 기준)"); P("="*126)
    P(f"  {'구성':>18s} " + " ".join(f"{'상위'+str(int(q*100))+'%':>19s}" for q in (0.2, 0.3, 0.5, 0.7, 1.0)))
    BEST = None
    for lab, parts in COMBOS.items():
        sc = score(Q, parts)
        row = []
        for q in (0.2, 0.3, 0.5, 0.7, 1.0):
            if sc is None:
                if q < 1.0: row.append("         -         "); continue
                m = np.ones(len(er), bool)
            else:
                m = np.isfinite(sc) & (sc >= 1.0-q)
            if m.sum() < 80: row.append("         -         "); continue
            r = curve(er[m], ea[m], idx)
            if r is None: row.append("       파산        "); continue
            row.append(f"{r['ann']:+8.1f}%/{np.mean(er[m]>0)*100:4.1f}/{m.sum()/Y:4.0f}")
            if BEST is None or r["ann"] > BEST[0]: BEST = (r["ann"], lab, q, m, r)
        P(f"  {lab:>18s} " + " ".join(f"{x:>19s}" for x in row))
    P("  (셀 = 연수익% / 승률% / 연거래)")
    if BEST:
        a, lab, q, m, r = BEST
        P(f"\n  ★ 최고: {lab} 상위{q*100:.0f}% → 연 {a:+.1f}%, 승률 {np.mean(er[m]>0)*100:.2f}%, "
          f"연 {m.sum()/Y:.0f}건, MDD {r['mdd']:.1f}%")
        P(f"\n  {'구간':>12s} {'명목':>6s} {'연수익%':>12s} {'월%':>9s} {'MDD%':>8s} {'월승률%':>8s}")
        for lb, nom in (("무레버리지", 1), ("중간", 2), ("중간+", 3), ("공격", 5), ("최대공격", 8)):
            rr = curve(er[m], ea[m], idx, nom)
            P(f"  {lb:>12s} {nom:5d}x " + ("        파산" if rr is None
                                            else f"{rr['ann']:12.1f} {rr['mon']:9.2f} {rr['mdd']:8.1f} {rr['mwr']:8.1f}"))
        out = []
        for yr in range(2020, 2027):
            sub = r["ser"][r["ser"].index.year == yr]
            out.append(f"{yr}  n/a" if len(sub) == 0 else f"{yr} {((1+sub).prod()-1)*100:+7.0f}%")
        P("  연도별(무레버): " + " | ".join(out))
        for y0 in (2023, 2025):
            mm = m & np.array([idx[a2].year >= y0 for a2 in ea])
            if mm.sum() < 40: continue
            rr = curve(er[mm], ea[mm], idx)
            P(f"  {y0}년 이후: {mm.sum()}건, 승률 {np.mean(er[mm]>0)*100:.2f}%, 거래당 {er[mm].mean()*1e4:+.1f}bp, "
              + ("파산" if rr is None else f"연 {rr['ann']:+.1f}%"))
    P("")

P("="*126); P("[결합] BTC+ETH 병행 — 각 최선 필터, 자본 절반씩"); P("="*126)
sers = {}
for S in ("BTCUSDT", "ETHUSDT"):
    ea, er, idx, Q = RES[S]
    sc = score(Q, COMBOS["H 넷 다"])
    m = np.isfinite(sc) & (sc >= 0.5)
    sers[S] = pd.Series(er[m], index=pd.DatetimeIndex(idx[ea[m]])).resample("ME").sum()
    P(f"  {S}: 상위50% 필터 → 연 {m.sum()/Y:.0f}건, 승률 {np.mean(er[m]>0)*100:.2f}%")
df = pd.concat(sers, axis=1).fillna(0)
P(f"  월 상관 {df.corr().iloc[0,1]:.3f}")
P(f"\n  {'명목':>6s} {'연수익%':>12s} {'월%':>9s} {'MDD%':>8s} {'월승률%':>8s}")
for nom in (1, 2, 3, 5, 8):
    tot = df.mean(axis=1)*nom
    eq = (1+tot).cumprod()
    if (eq <= 0.01).any(): P(f"  {nom:5d}x         파산"); continue
    a = (eq.iloc[-1]**(12/len(tot))-1)*100
    P(f"  {nom:5d}x {a:12.1f} {((1+a/100)**(1/12)-1)*100:9.2f} "
      f"{(eq/eq.cummax()-1).min()*100:8.1f} {np.mean(tot>0)*100:8.1f}")
