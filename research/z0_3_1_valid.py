"""z0.3 깊이 10·14·20 구성 독립 검증: 플라시보·워크포워드·체결/비용·가중 민감도·가뭄 구조.
사용: python research/z0_3_1_valid.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from exit_tier import build, sim, show, RAW, SYM, BL, TE, REC
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
DEP = {10.0: 1.0, 14.0: 3.0, 20.0: 6.0}


@numba.njit(cache=True)
def rnd(o, c, st, hd, fee):
    out = np.zeros(len(st))
    for i in range(len(st)):
        e = st[i]+hd[i]
        if e >= len(c): e = len(c)-1
        out[i] = (c[e]/o[st[i]]-1.0) - fee/1e4
    return out


E = build(DEP, 0)
r = np.array([e[2] for e in E]); hd = np.array([e[1]-e[0] for e in E])
P(f"[0] 기본: n={len(E)} 평균 {r.mean()*1e4:+.0f}bp 중앙 {np.median(r)*1e4:+.0f}bp 승률 {(r>0).mean():.3f} "
  f"최악 {r.min()*1e4:+.0f}bp 보유중앙 {np.median(hd):.0f}분")

P("\n[1] 플라시보 — 같은 보유길이, 무작위 분 진입 1000회")
rng = np.random.default_rng(23)
nul = []
for _ in range(1000):
    acc = []
    for s in SYM:
        d = RAW[s]; N = len(d["c"]); cnt = len(E)//2
        acc.append(rnd(d["o"], d["c"], rng.integers(0, N-2000, cnt).astype(np.int64),
                       rng.choice(hd, cnt).astype(np.int64), 7.0))
    nul.append(np.concatenate(acc).mean()*1e4)
nul = np.array(nul)
P(f"  위약 평균 {nul.mean():+.0f}bp 95분위 {np.quantile(nul,0.95):+.0f} 99분위 {np.quantile(nul,0.99):+.0f} 최대 {nul.max():+.0f}")
P(f"  실제 {r.mean()*1e4:+.0f}bp → 백분위 {(nul < r.mean()*1e4).mean()*100:.1f}% ({'통과' if (nul<r.mean()*1e4).mean()>=0.999 else '미달'})")

P("\n[2] 깊이 격자 전체 — 10·14·20 이 고원 위에 있나 (w=5 Nmax=5)")
rows = []
for k1 in (7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 14.0):
    for d2 in (3.0, 4.0, 5.0):
        dep = {k1: 1.0, k1+d2: 3.0, k1+2*d2: 6.0}
        E2 = build(dep, 0)
        if len(E2) < 40: continue
        eq, fl, liq = sim(E2, 5.0, 5.0)
        rr = show(eq, fl, liq, "", quiet=True)
        rows.append((k1, k1+d2, k1+2*d2, len(E2), *( [round(x,1) for x in rr[:5]] if rr else [np.nan]*5 )))
t = pd.DataFrame(rows, columns=["k1", "k2", "k3", "n", "연%", "MDD%", "최악%", "24~%", "25~%"])
P(t.to_string(index=False))

P("\n[3] 가중 민감도 (깊이 10·14·20)")
for wg in ((1, 1, 1), (1, 2, 3), (1, 3, 6), (1, 4, 10), (1, 2, 4), (3, 2, 1)):
    dep = {10.0: float(wg[0]), 14.0: float(wg[1]), 20.0: float(wg[2])}
    E2 = build(dep, 0)
    for w, Nm in ((5.0, 5.0), (5.0, 8.0)):
        eq, fl, liq = sim(E2, w, Nm)
        show(eq, fl, liq, f"  가중{wg} w={w:.0f} Nmax={Nm:.0f}")

P("\n[4] 워크포워드 (2년 학습 → 1년 검증)")
def seg(E, s, e): return [x for x in E if (s is None or x[4] >= s) and (e is None or x[4] < e)]
tot = []
for y in pd.date_range("2022-01-01", "2026-01-01", freq="YS", tz="UTC"):
    tr = seg(E, y-pd.DateOffset(years=2), y); te = seg(E, y, y+pd.DateOffset(years=1))
    if len(tr) < 15 or len(te) < 3: continue
    bw = None
    for w in (2.0, 3.0, 5.0, 8.0):
        for Nm in (3.0, 5.0, 8.0):
            if Nm < w: continue
            eq, fl, liq = sim(tr, w, Nm)
            rr = show(eq, fl, liq, "", quiet=True)
            if rr and rr[2] > -70 and (bw is None or rr[0] > bw[0]): bw = (rr[0], w, Nm)
    if bw is None: continue
    eq, fl, liq = sim(te, bw[1], bw[2])
    rr = show(eq, fl, liq, "", quiet=True)
    c = rr[0] if rr else -100.0
    tot.append(c); P(f"  {y.year}: 학습최적 w={bw[1]:.0f} Nmax={bw[2]:.0f} → 검증 연 {c:+8.0f}% (거래 {len(te)}건)")
if tot: P(f"  → 평균 {np.mean(tot):+.0f}% 양수 {sum(1 for x in tot if x>0)}/{len(tot)}")

P("\n[5] 체결·비용 민감도 (w=5 Nmax=5)")
for eps, fm, ft, lab in ((0.0, 2.0, 5.0, "낙관"), (0.0002, 2.0, 5.0, "기준"), (0.0005, 3.0, 6.0, "보수"),
                         (0.001, 4.0, 7.0, "비관"), (0.002, 5.0, 8.0, "최악"), (0.005, 8.0, 12.0, "극단")):
    E2 = build(DEP, 0, eps=eps, fm=fm, ft=ft)
    eq, fl, liq = sim(E2, 5.0, 5.0)
    show(eq, fl, liq, f"  {lab} eps={eps*1e4:.0f}bp 메{fm:.0f}/테{ft:.0f} (n={len(E2)})")

P("\n[6] 거래 분포 — 가뭄과 몰림 구조")
d = pd.DataFrame([(e[4], e[2]) for e in E], columns=["ts", "r"]).set_index("ts").sort_index()
yc = d.groupby(d.index.year).r
P("  연도별 건수/평균bp: " + " | ".join(f"{y} {c}건 {m*1e4:+.0f}bp" for (y, c), (_, m) in zip(yc.size().items(), yc.mean().items())))
gap = d.index.to_series().diff().dt.days.dropna()
P(f"  거래 간격(일): 중앙 {gap.median():.0f} 평균 {gap.mean():.0f} 90분위 {gap.quantile(0.9):.0f} 최대 {gap.max():.0f}")
mon = d.resample("ME").size()
P(f"  거래 있는 달 {int((mon>0).sum())}/{len(mon)} ({(mon>0).mean()*100:.0f}%) | 마지막 거래 {d.index[-1].date()}")
top = d.r.nlargest(5)
P(f"  상위 5거래가 총수익에서 차지: {top.sum()/d.r.sum()*100:.0f}% | 상위 10건 {d.r.nlargest(10).sum()/d.r.sum()*100:.0f}%")
P("  상위 5거래: " + " | ".join(f"{i.date()} {v*1e4:+.0f}bp" for i, v in top.items()))

P("\n[7] 공격 구성 상세 (w=5 Nmax=8)")
eq, fl, liq = sim(E, 5.0, 8.0)
rr = show(eq, fl, liq, "  ")
if rr:
    dly = rr[5]
    yr = dly.resample("YE").last()/dly.resample("YE").first()-1
    P("   연도별: " + " | ".join(f"{i.year} {v*100:+.0f}%" for i, v in yr.items()))
