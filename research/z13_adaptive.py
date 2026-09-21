"""z13 — z12 감쇠 대응: 깊이를 변동성에 덜 민감하게 만들면 막을 수 있나.

문제: rv 중앙값이 2021 10.27bp → 2025 5.18bp로 반토막.
      딥 깊이 = rv×k 이므로 딥도 반, 수익도 반인데 **비용은 고정**.
      거래당 2021 +62.7bp → 2025 +11.3bp, 승률 66.16% → 48.98%.

후보:
  A. 고정 bp 깊이 (rv 무관)
  B. rv^0.5 비례 (감쇠 완화)
  C. 확장창 rv 중앙값 기준 정규화 — 깊이 = k × rv중앙(항상 같은 절대 깊이)
  D. 혼합: k × (rv × rv중앙)^0.5  = 기하평균
  E. 저변동일수록 k를 키움: k × (rv중앙/rv)^a
  F. 봉 길이를 늘려 절대 깊이 확보 (rv는 그대로)
"""
import sys, warnings
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
DAT = {}
for S in ("BTCUSDT", "ETHUSDT"):
    d = L2.D[S]; rv = d["rv"]
    med = np.full(len(rv), np.nan)
    for i in range(warm, len(rv), 5000):
        x = rv[:i]; x = x[np.isfinite(x)]
        if len(x) >= 1000: med[i:i+5000] = np.median(x)
    DAT[S] = dict(o=d["o"], h=d["h"], l=d["l"], c=d["c"], idx=d["idx"], rv=rv, med=med)
P("데이터 준비 완료\n")


@numba.njit(cache=True)
def dip(o, h, l, c, rvv, medv, barmin, k0, k1, k2, w0, w1, w2,
        eps, slip, fee, mode, alpha, start, n_end):
    """mode 0=rv비례(z12) 1=고정bp 2=rv^0.5 3=rv중앙고정 4=기하평균 5=저변동가중"""
    K = 200000
    ea = np.zeros(K, np.int64); er = np.zeros(K); dp = np.zeros(K)
    k = 0
    nb = (n_end-start)//barmin
    for b in range(nb):
        t0 = start+b*barmin; t1 = t0+barmin-1
        if t1 >= n_end-1: break
        r = rvv[t0]; mm = medv[t0]
        if np.isnan(r) or r <= 0 or np.isnan(mm) or mm <= 0: continue
        if mode == 0: base = r
        elif mode == 1: base = 1.0
        elif mode == 2: base = np.sqrt(r*1e4)/1e4*np.sqrt(1e4*mm)/np.sqrt(mm*1e4)*np.sqrt(mm)
        elif mode == 3: base = mm
        elif mode == 4: base = np.sqrt(r*mm)
        else:
            f = mm/r
            if f < 0.3: f = 0.3
            if f > 3.0: f = 3.0
            base = r*(f**alpha)
        px0 = o[t0]
        tot = 0.0; wsum = 0.0; ft = -1; dsum = 0.0
        for li in range(3):
            kk = k0 if li == 0 else (k1 if li == 1 else k2)
            ww = w0 if li == 0 else (w1 if li == 1 else w2)
            depth = base*kk
            if mode == 1: depth = kk/1e4
            T = px0*(1.0-depth); trig = T*(1.0-eps/1e4)
            got = -1
            for j in range(t0, t1+1):
                if l[j] <= trig: got = j; break
            if got >= 0:
                fill = T*(1.0+slip/1e4)
                tot += ww*(c[t1]/fill-1.0-fee/1e4); wsum += ww
                dsum += ww*depth
                if ft < 0: ft = got
        if wsum > 0:
            ea[k] = ft; er[k] = tot/wsum; dp[k] = dsum/wsum; k += 1
            if k >= K-2: break
    return ea[:k], er[:k], dp[:k]


def ann(er, ea, idx, nom=1.0):
    if len(er) < 30: return None
    m = pd.Series(nom*er, index=pd.DatetimeIndex(idx[ea])).resample("ME").sum()
    eq = (1+m).cumprod()
    if (eq <= 0.01).any(): return None
    return dict(ann=(eq.iloc[-1]**(12/len(m))-1)*100,
                mdd=(eq/eq.cummax()-1).min()*100, ser=m)


def recent(er, ea, idx, yr0):
    m = np.array([idx[a].year >= yr0 for a in ea])
    if m.sum() < 30: return None, None, None
    r = ann(er[m], ea[m], idx)
    return (None if r is None else r["ann"]), er[m].mean()*1e4, np.mean(er[m] > 0)*100


S = "BTCUSDT"; D = DAT[S]
BAR = 720
P("="*128); P("[1] 깊이 방식 × 시기 — 감쇠를 막는 것이 있나 (BTC, 720분봉)"); P("="*128)
P(f"  {'방식':>20s} {'연거래':>6s} {'전체 연%':>9s} {'전체bp':>8s} "
  f"{'2023~ 연%':>10s} {'2023~bp':>9s} {'2025~ 연%':>10s} {'2025~bp':>9s} {'2025~승률':>9s}")
MODES = [("A rv비례(z12)", 0, (15.0, 21.0, 30.0), 0.0),
         ("B 고정 100bp", 1, (100.0, 140.0, 200.0), 0.0),
         ("C 고정 70bp", 1, (70.0, 98.0, 140.0), 0.0),
         ("D 고정 150bp", 1, (150.0, 210.0, 300.0), 0.0),
         ("E rv중앙 고정", 3, (15.0, 21.0, 30.0), 0.0),
         ("F 기하평균", 4, (15.0, 21.0, 30.0), 0.0),
         ("G 저변동가중 a=0.5", 5, (15.0, 21.0, 30.0), 0.5),
         ("H 저변동가중 a=1.0", 5, (15.0, 21.0, 30.0), 1.0)]
for lab, mode, ks, al in MODES:
    ea, er, dpv = dip(D["o"], D["h"], D["l"], D["c"], D["rv"], D["med"], BAR, *ks,
                      1.0, 3.0, 6.0, EPS, SLIP, FEE, mode, al, warm, len(D["c"]))
    if len(er) < 50: continue
    r0 = ann(er, ea, D["idx"])
    a3, b3, _ = recent(er, ea, D["idx"], 2023)
    a5, b5, w5 = recent(er, ea, D["idx"], 2025)
    f = lambda v: "    파산" if v is None else f"{v:9.1f}"
    P(f"  {lab:>20s} {len(er)/Y:6.0f} {f(None if r0 is None else r0['ann'])} {er.mean()*1e4:8.1f} "
      f"{f(a3):>10s} {b3:9.1f} {f(a5):>10s} {b5:9.1f} {w5:8.2f}%")

P("\n"+"="*128); P("[2] 고정 bp 깊이 — 값 스윕 (BTC 720분봉)"); P("="*128)
P(f"  {'1층 깊이':>9s} {'연거래':>6s} {'전체 연%':>10s} {'2023~':>10s} {'2025~':>10s} {'2025~bp':>9s} {'2025~승률':>9s}")
for base_bp in (40, 60, 80, 100, 120, 150, 200, 300):
    ks = (base_bp, base_bp*1.4, base_bp*2.0)
    ea, er, _ = dip(D["o"], D["h"], D["l"], D["c"], D["rv"], D["med"], BAR, *ks,
                    1.0, 3.0, 6.0, EPS, SLIP, FEE, 1, 0.0, warm, len(D["c"]))
    if len(er) < 50: continue
    r0 = ann(er, ea, D["idx"])
    a3, _, _ = recent(er, ea, D["idx"], 2023)
    a5, b5, w5 = recent(er, ea, D["idx"], 2025)
    f = lambda v: "     파산" if v is None else f"{v:10.1f}"
    P(f"  {base_bp:8d}bp {len(er)/Y:6.0f} {f(None if r0 is None else r0['ann'])} {f(a3)} {f(a5)} "
      f"{b5:9.1f} {w5:8.2f}%")

P("\n"+"="*128); P("[3] 봉 길이로 절대 깊이 확보 (rv비례 유지, 2025년 기준)"); P("="*128)
P(f"  {'봉':>7s} " + " ".join(f"{'rv×'+str(k):>13s}" for k in (12, 15, 20, 28, 40)))
for bm in (720, 1440, 2880, 5760, 10080):
    row = []
    for mult in (12, 15, 20, 28, 40):
        ks = (mult, mult*1.4, mult*2.0)
        ea, er, _ = dip(D["o"], D["h"], D["l"], D["c"], D["rv"], D["med"], bm, *ks,
                        1.0, 3.0, 6.0, EPS, SLIP, FEE, 0, 0.0, warm, len(D["c"]))
        if len(er) < 40: row.append("      -      "); continue
        a5, b5, w5 = recent(er, ea, D["idx"], 2025)
        row.append("    파산     " if a5 is None else f"{a5:+7.1f}%/{w5:4.1f}")
    P(f"  {bm:6d}분 " + " ".join(f"{x:>13s}" for x in row))
P("  (셀 = 2025년 이후 연수익% / 승률%)")

P("\n"+"="*128); P("[4] 최선안 — 전 기간·최근 동시 비교 (BTC + ETH)"); P("="*128)
BESTS = [("z12 원본 rv×15 720분", 0, (15.0, 21.0, 30.0), 0.0, 720),
         ("고정 100bp 720분", 1, (100.0, 140.0, 200.0), 0.0, 720),
         ("rv×20 1440분", 0, (20.0, 28.0, 40.0), 0.0, 1440),
         ("rv×28 2880분", 0, (28.0, 39.2, 56.0), 0.0, 2880),
         ("저변동가중 a=1.0", 5, (15.0, 21.0, 30.0), 1.0, 720)]
for S2 in ("BTCUSDT", "ETHUSDT"):
    D2 = DAT[S2]
    P(f"\n  ══ {S2} ══")
    P(f"  {'구성':>22s} {'연거래':>6s} {'전체 연%':>10s} {'MDD%':>8s} {'2025~ 연%':>10s} {'2025~승률':>9s}")
    for lab, mode, ks, al, bm in BESTS:
        ea, er, _ = dip(D2["o"], D2["h"], D2["l"], D2["c"], D2["rv"], D2["med"], bm, *ks,
                        1.0, 3.0, 6.0, EPS, SLIP, FEE, mode, al, warm, len(D2["c"]))
        if len(er) < 50: continue
        r0 = ann(er, ea, D2["idx"])
        a5, b5, w5 = recent(er, ea, D2["idx"], 2025)
        f = lambda v: "     파산" if v is None else f"{v:10.1f}"
        P(f"  {lab:>22s} {len(er)/Y:6.0f} {f(None if r0 is None else r0['ann'])} "
          f"{(0.0 if r0 is None else r0['mdd']):8.1f} {f(a5)} {w5:8.2f}%")

P("\n"+"="*128); P("[5] 왜 고정 깊이가 감쇠를 막는가 — 실제 깊이 추이"); P("="*128)
for lab, mode, ks in (("rv비례 rv×15", 0, (15.0, 21.0, 30.0)),
                      ("고정 100bp", 1, (100.0, 140.0, 200.0))):
    ea, er, dpv = dip(D["o"], D["h"], D["l"], D["c"], D["rv"], D["med"], BAR, *ks,
                      1.0, 3.0, 6.0, EPS, SLIP, FEE, mode, 0.0, warm, len(D["c"]))
    df = pd.DataFrame({"y": pd.DatetimeIndex(D["idx"][ea]).year, "d": dpv, "r": er})
    P(f"\n  {lab}:")
    P("    " + " ".join(f"{y:>9d}" for y in range(2020, 2027)))
    for nm, fn in (("평균깊이%", lambda g: f"{g.d.mean()*100:.3f}"),
                   ("거래당bp", lambda g: f"{g.r.mean()*1e4:+.1f}"),
                   ("거래수", lambda g: f"{len(g)}")):
        cells = []
        for y in range(2020, 2027):
            g = df[df.y == y]
            cells.append(fn(g) if len(g) else "-")
        P(f"    {nm:>8s}" + " ".join(f"{c:>9s}" for c in cells))
