"""z0.2 최적 구성 정밀 검증 — 1h+4h+8h 기준. 명목·비용·가중치·워크포워드·위약·월별.
사용: python research/z0_2_final.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from z0_2_extend import D, leg3, port, rep, ALL_LEGS
TE = pd.Timestamp("2024-01-01", tz="UTC")
rng = np.random.default_rng(293)
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
LG = [(s, b) for s in ("BTCUSDT", "ETHUSDT") for b in (60, 240, 480)]

P("[1] 명목 × 분할 격자 (1h+4h+8h)")
rows = []
for sp in ((3.0, 4.5, 6.0), (4.0, 5.5, 7.0), (5.0, 6.5, 8.0), (3.5, 5.0, 6.5)):
    for N in (1, 2, 3, 4, 5):
        eq = port(LG, sp, N)
        if len(eq) < 100 or eq.iloc[-1] <= 1e-6:
            rows.append((str(sp), N, "파산", np.nan, np.nan, np.nan, np.nan)); continue
        yrs = (eq.index[-1] - eq.index[0]).days / 365
        te = eq[eq.index >= TE]; y2 = (te.index[-1] - te.index[0]).days / 365
        rows.append((str(sp), N, f"{(eq.iloc[-1]**(1/yrs)-1)*100:+.1f}%", float((eq / eq.cummax() - 1).min()) * 100,
                     f"{((te.iloc[-1]/te.iloc[0])**(1/y2)-1)*100:+.1f}%", float((te / te.cummax() - 1).min()) * 100,
                     len(eq) / ((eq.index[-1] - eq.index[0]).days / 7)))
d = pd.DataFrame(rows, columns=["분할", "명목", "전체연", "전체MDD%", "테스트연", "테스트MDD%", "주당"])
P(d.round(1).to_string(index=False))

P("\n[2] 비용 민감도 (분할 4.0/5.5/7.0, 명목 3배) — 1h 포함이라 거래 많음")
for fee in (7, 10, 12, 16, 20, 25):
    eq = port(LG, (4.0, 5.5, 7.0), 3, fee=fee)
    if eq.iloc[-1] <= 1e-6:
        P(f"  {fee}bp: 파산"); continue
    yrs = (eq.index[-1] - eq.index[0]).days / 365
    te = eq[eq.index >= TE]; y2 = (te.index[-1] - te.index[0]).days / 365
    P(f"  {fee:2d}bp: 전체 연 {(eq.iloc[-1]**(1/yrs)-1)*100:+7.1f}% MDD {float((eq/eq.cummax()-1).min())*100:6.1f}% | "
      f"테스트 연 {((te.iloc[-1]/te.iloc[0])**(1/y2)-1)*100:+7.1f}% MDD {float((te/te.cummax()-1).min())*100:6.1f}%")

P("\n[3] 워크포워드 (1h+4h+8h, 정확 자본배분)")
CAND = [((3.0, 4.5, 6.0), 2), ((4.0, 5.5, 7.0), 2), ((4.0, 5.5, 7.0), 3), ((4.0, 5.5, 7.0), 4),
        ((5.0, 6.5, 8.0), 3), ((5.0, 6.5, 8.0), 4), ((3.5, 5.0, 6.5), 3)]
rows = []
for y in range(2022, 2027):
    sD = pd.Timestamp(f"{y}-01-01", tz="UTC"); eD = pd.Timestamp(f"{y+1}-01-01", tz="UTC")
    best = None
    for sp, N in CAND:
        e_ = port(LG, sp, N, end=sD)
        if len(e_) < 150 or e_.iloc[-1] <= 1e-6:
            continue
        yrs = (e_.index[-1] - e_.index[0]).days / 365
        mdd = float((e_ / e_.cummax() - 1).min())
        g = (e_.iloc[-1] ** (1 / yrs) - 1)
        if mdd < -0.5:
            continue
        if best is None or g > best[0]:
            best = (g, sp, N)
    if best is None:
        continue
    _, sp, N = best
    e_ = port(LG, sp, N)
    seg = e_[(e_.index >= sD) & (e_.index < eD)]
    if len(seg) < 20:
        continue
    rows.append((y, str(sp), N, (seg.iloc[-1] / seg.iloc[0] - 1) * 100,
                 float((seg / seg.cummax() - 1).min()) * 100, len(seg)))
wf = pd.DataFrame(rows, columns=["연", "선택 분할", "명목", "그해%", "그해MDD%", "진입"])
P("  " + wf.round(1).to_string(index=False).replace("\n", "\n  "))
if len(wf):
    P(f"  워크포워드 평균 {wf['그해%'].mean():+.1f}%/년, 전부 양수 {int((wf['그해%']>0).sum())}/{len(wf)}, 최악 MDD {wf['그해MDD%'].min():.1f}%")

P("\n[4] 최적안 월별 — 분할 4.0/5.5/7.0, 명목 3배")
eq = port(LG, (4.0, 5.5, 7.0), 3)
mon = eq.resample("ME").last() / eq.resample("ME").first() - 1
yr = eq.resample("YE").last() / eq.resample("YE").first() - 1
P("  연도별: " + " | ".join(f"{i.year} {v*100:+.0f}%" for i, v in yr.items()))
l12 = mon.tail(12)
P("  최근 12개월: " + " | ".join(f"{i.strftime('%y-%m')} {v*100:+.1f}%" for i, v in l12.items()))
P(f"  최근 12개월 합계 {(np.prod(1+l12)-1)*100:+.1f}% | 월평균 {l12.mean()*100:+.2f}% | 양수 {int((l12>0).sum())}/12 | 최악 {l12.min()*100:+.1f}%")
P(f"  전체 월 승률 {float((mon>0).mean()):.3f} (n={len(mon)})")

P("\n[5] 위약 — 각 다리에서 같은 봉의 무작위 시점 진입 (정확 자본배분) 100회")
real = port(LG, (4.0, 5.5, 7.0), 3)
yrs = (real.index[-1] - real.index[0]).days / 365
realg = (real.iloc[-1] ** (1 / yrs) - 1) * 100
pl = []
for _ in range(100):
    eqs = []
    for key in LG:
        d = D[key]; B = d["B"]; n = d["n"]
        j = rng.integers(0, B, n)
        ent = d["c"][np.arange(n) * B + j]
        ex = d["c"][np.arange(n) * B + B - 1]
        r_ = (ex / ent - 1) - 7 / 1e4
        eqs.append(pd.Series(r_ * 3 / 3, index=d["bi"]))   # 명목3배, 배정 1/6씩은 아래서 처리
    idx = sorted(set().union(*[s.index for s in eqs]))
    cur = np.full(len(eqs), 1.0 / len(eqs))
    hist = []
    for t in idx:
        for i, s_ in enumerate(eqs):
            if t in s_.index:
                cur[i] = max(cur[i] * (1 + float(s_.loc[t]) * 3), 0.0)
        hist.append(cur.sum())
    e2 = pd.Series(hist, index=pd.DatetimeIndex(idx))
    if e2.iloc[-1] > 1e-9:
        y2 = (e2.index[-1] - e2.index[0]).days / 365
        pl.append((e2.iloc[-1] ** (1 / y2) - 1) * 100)
pl = np.array(pl)
P(f"  실제 연 {realg:+.1f}% | 위약 평균 {pl.mean():+.1f}% 95% {np.percentile(pl,95):+.1f}% 최대 {pl.max():+.1f}% → 백분위 {(pl<realg).mean()*100:.1f}%")
