"""R17 — 여러 봉 길이를 동시 운용. 독립 베팅 수를 실제로 늘린다.

천장 분석(R4): 월수익 = exp(N_독립 × g). N_독립이 제약.
같은 종목·같은 메커니즘이라도 **봉 길이가 다르면 진입 시각과 보유 구간이 달라져**
겹치지 않는 베팅이 된다. 720/1440/2880/5760분봉을 동시에 굴린다.

추가로 봉 위상도 분리한다(같은 720분봉이라도 0시/6시 시작 두 갈래).
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R/"research"))
import z0_20_lib as L2


def P(*a, end="\n"):
    print(" ".join(str(x) for x in a), end=end, flush=True)


Y = L2.YRS; MMR = L2.MMR
EPS, SLIP, FEE = 5.0, 10.0, 10.0
warm = 200000


@numba.njit(cache=True)
def dip(o, h, l, c, rvv, barmin, k0, k1, k2, w0, w1, w2, eps, slip, fee,
        start, n_end, phase):
    """phase: 봉 시작 위상(분). 같은 봉 길이라도 위상이 다르면 다른 베팅."""
    K = 200000
    ea = np.zeros(K, np.int64); er = np.zeros(K); mae = np.zeros(K); nomv = np.zeros(K)
    k = 0
    wtot = w0+w1+w2
    nb = (n_end-start)//barmin
    for b in range(nb):
        t0 = start+b*barmin+phase; t1 = t0+barmin-1
        if t1 >= n_end-1 or t0 < start: continue
        r = rvv[t0]
        if np.isnan(r) or r <= 0: continue
        px0 = o[t0]
        tot = 0.0; wsum = 0.0; ft = -1; worst = 0.0
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
                lo = fill
                for j in range(got, t1+1):
                    if l[j] < lo: lo = l[j]
                dd = lo/fill-1.0
                if dd < worst: worst = dd
                if ft < 0: ft = got
        if wsum > 0:
            ea[k] = ft; er[k] = tot/wtot; nomv[k] = wsum/wtot; mae[k] = worst; k += 1
            if k >= K-2: break
    return ea[:k], er[:k], mae[:k], nomv[:k]


def mser(ea, er, idx):
    return pd.Series(er, index=pd.DatetimeIndex(idx[ea])).resample("ME").sum()


def curve(ser, nom=1.0):
    eq = (1+ser*nom).cumprod()
    if (eq <= 0.01).any(): return None
    a = (eq.iloc[-1]**(12/len(ser))-1)*100
    return dict(ann=a, mon=((1+a/100)**(1/12)-1)*100,
                mdd=(eq/eq.cummax()-1).min()*100, mwr=np.mean(ser > 0)*100)


# 봉 길이별 최적 깊이 (z13 [3] 표 기준)
LEGS = [(720, 15.0), (1440, 20.0), (2880, 28.0), (5760, 40.0)]

for S in ("BTCUSDT", "ETHUSDT"):
    d = L2.D[S]; idx = d["idx"]; N = len(d["c"])
    P("="*126); P(f"[{S}] 봉 길이별 단독 성과 (보수적: 10단위 준비 방식)"); P("="*126)
    SER = {}; META = {}
    for bm, mult in LEGS:
        ks = (mult, mult*1.4, mult*2.0)
        ea, er, mae, nomv = dip(d["o"], d["h"], d["l"], d["c"], d["rv"], bm, *ks,
                                1.0, 3.0, 6.0, EPS, SLIP, FEE, warm, N, 0)
        if len(er) < 50: continue
        s = mser(ea, er, idx)
        SER[f"{bm}분"] = s
        META[f"{bm}분"] = (len(er)/Y, np.mean(er > 0)*100, er.mean()*1e4, nomv.mean())
        r = curve(s)
        P(f"  {bm:5d}분 rv×{mult:4.1f}: 연 {len(er)/Y:4.0f}건  승률 {np.mean(er>0)*100:5.2f}%  "
          f"거래당 {er.mean()*1e4:+6.1f}bp  실효명목 {nomv.mean():.2f}  "
          + ("파산" if r is None else f"연 {r['ann']:+8.1f}%  MDD {r['mdd']:6.1f}%"))
    # 위상 분리 (720분봉 4위상)
    for ph in (180, 360, 540):
        ks = (15.0, 21.0, 30.0)
        ea, er, mae, nomv = dip(d["o"], d["h"], d["l"], d["c"], d["rv"], 720, *ks,
                                1.0, 3.0, 6.0, EPS, SLIP, FEE, warm, N, ph)
        if len(er) > 50: SER[f"720분@{ph}"] = mser(ea, er, idx)

    df = pd.concat(SER, axis=1).fillna(0)
    P(f"\n  월별 수익 상관:")
    cr = df.corr()
    P("           " + " ".join(f"{c:>10s}" for c in cr.columns))
    for i, row in cr.iterrows():
        P(f"  {i:>9s} " + " ".join(f"{row[c]:10.3f}" for c in cr.columns))

    P(f"\n  포트폴리오 (각 다리 균등, 보수적 기준):")
    P(f"  {'구성':>28s} " + " ".join(f"{'명목'+str(n)+'배':>16s}" for n in (1, 2, 3, 5, 8)))
    CFG = [("720분 단독", ["720분"]),
           ("봉길이 4종", [f"{b}분" for b, _ in LEGS if f"{b}분" in SER]),
           ("720분 4위상", [c for c in SER if c.startswith("720")]),
           ("전부", list(SER.keys()))]
    for lab, ks_ in CFG:
        ks_ = [k for k in ks_ if k in df.columns]
        if not ks_: continue
        tot = df[ks_].mean(axis=1)
        row = []
        for nom in (1, 2, 3, 5, 8):
            r = curve(tot, nom)
            row.append("      파산      " if r is None
                       else f"{r['ann']:+8.1f}%/{r['mdd']:6.1f}%")
        P(f"  {lab:>28s} " + " ".join(f"{x:>16s}" for x in row))
    P("  (셀 = 연수익% / MDD%)")

    best = None
    for lab, ks_ in CFG:
        ks_ = [k for k in ks_ if k in df.columns]
        if not ks_: continue
        tot = df[ks_].mean(axis=1)
        for nom in (1, 2, 3, 5, 8, 12):
            r = curve(tot, nom)
            if r and (best is None or r["mon"] > best[0]): best = (r["mon"], lab, nom, r, tot)
    if best:
        mon, lab, nom, r, tot = best
        P(f"\n  ★ 최고: {lab} 명목 {nom}배 → 월 {mon:+.2f}%, 연 {r['ann']:+.1f}%, "
          f"MDD {r['mdd']:.1f}%, 월승률 {r['mwr']:.1f}%")
        out = []
        for yr in range(2020, 2027):
            sub = (tot*nom)[tot.index.year == yr]
            out.append(f"{yr}  n/a" if len(sub) == 0 else f"{yr} {((1+sub).prod()-1)*100:+7.0f}%")
        P("     연도별: " + " | ".join(out))
    P("")

P("="*126); P("[BTC+ETH 전부 결합]"); P("="*126)
ALL = {}
for S in ("BTCUSDT", "ETHUSDT"):
    d = L2.D[S]; idx = d["idx"]; N = len(d["c"])
    for bm, mult in LEGS:
        ks = (mult, mult*1.4, mult*2.0)
        ea, er, mae, nomv = dip(d["o"], d["h"], d["l"], d["c"], d["rv"], bm, *ks,
                                1.0, 3.0, 6.0, EPS, SLIP, FEE, warm, N, 0)
        if len(er) > 50: ALL[f"{S[:3]}{bm}"] = mser(ea, er, idx)
dfa = pd.concat(ALL, axis=1).fillna(0)
P(f"  다리 {len(dfa.columns)}개, 평균 쌍상관 "
  f"{(dfa.corr().values[np.triu_indices(len(dfa.columns),1)]).mean():.3f}")
tot = dfa.mean(axis=1)
P(f"\n  {'명목':>6s} {'연수익%':>12s} {'월%':>9s} {'MDD%':>8s} {'월승률%':>8s}")
for nom in (1, 2, 3, 5, 8, 12, 16):
    r = curve(tot, nom)
    P(f"  {nom:5d}x " + ("        파산" if r is None
                          else f"{r['ann']:12.1f} {r['mon']:9.2f} {r['mdd']:8.1f} {r['mwr']:8.1f}"))
out = []
for yr in range(2020, 2027):
    sub = (tot*5)[tot.index.year == yr]
    out.append(f"{yr}  n/a" if len(sub) == 0 else f"{yr} {((1+sub).prod()-1)*100:+7.0f}%")
P("  연도별(5배): " + " | ".join(out))
for y0 in (2023, 2025):
    sub = tot[tot.index.year >= y0]
    for nom in (3, 5):
        r = curve(sub, nom)
        P(f"  {y0}년 이후 {nom}배: " + ("파산" if r is None
                                       else f"연 {r['ann']:+9.1f}%  월 {r['mon']:+6.2f}%  MDD {r['mdd']:6.1f}%"))
