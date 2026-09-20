"""z0.3 최종본(보유 연장 포함) 전체 재검증 + 3구간 성과표."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
import exit_tier as X
from absfloor import met
from anchor import build_a
from exit_tier import RAW, SYM
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
EXT = 0.10


@numba.njit(cache=True)
def rnd(o, c, st, hd, fee):
    out = np.zeros(len(st))
    for i in range(len(st)):
        e = st[i]+hd[i]
        if e >= len(c): e = len(c)-1
        out[i] = (c[e]/o[st[i]]-1.0) - fee/1e4
    return out


P("[1] 연장 비율 고원 확인")
for ext in (0.0, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30):
    E = build_a("open", ext=ext)
    r = np.array([x[2] for x in E])
    row = f"  ext={ext*100:5.1f}% 평균 {r.mean()*1e4:+6.0f}bp 승률 {(r>0).mean():.3f} |"
    for w, Nm, tag in ((1.0, 1.0, "무레버"), (3.0, 5.0, "중간"), (5.0, 8.0, "공격")):
        eq, fl, liq = X.sim(E, w, Nm); m = met(eq, fl, liq)
        row += f" {tag} {'파산' if m is None else f'{m[0]:+5.0f}%'}" + (f"(최악{m[2]:5.1f}%)" if m else "")
    P(row)

E = build_a("open", ext=EXT)
r = np.array([x[2] for x in E]); hd = np.array([x[1]-x[0] for x in E])
P(f"\n[2] 최종 구성: n={len(r)} 평균 {r.mean()*1e4:+.0f}bp 중앙 {np.median(r)*1e4:+.0f}bp "
  f"승률 {(r>0).mean():.3f} 최악 {r.min()*1e4:+.0f}bp 보유중앙 {np.median(hd):.0f}분")

P("\n[3] 플라시보 1000회")
rng = np.random.default_rng(77); nul = []
for _ in range(1000):
    acc = []
    for s in SYM:
        d = RAW[s]; N = len(d["c"]); cnt = len(E)//2
        acc.append(rnd(d["o"], d["c"], rng.integers(0, N-3000, cnt).astype(np.int64),
                       rng.choice(hd, cnt).astype(np.int64), 7.0))
    nul.append(np.concatenate(acc).mean()*1e4)
nul = np.array(nul)
P(f"  위약 평균 {nul.mean():+.0f}bp 99분위 {np.quantile(nul,0.99):+.0f} 최대 {nul.max():+.0f} | "
  f"실제 {r.mean()*1e4:+.0f}bp → 백분위 {(nul < r.mean()*1e4).mean()*100:.1f}%")

P("\n[4] 워크포워드 (2년 학습 → 1년 검증)")
seg = lambda Ex, s, e: [x for x in Ex if (s is None or x[4] >= s) and (e is None or x[4] < e)]
tot = []
for y in pd.date_range("2022-01-01", "2026-01-01", freq="YS", tz="UTC"):
    tr = seg(E, y-pd.DateOffset(years=2), y); te = seg(E, y, y+pd.DateOffset(years=1))
    if len(tr) < 15 or len(te) < 3: continue
    bw = None
    for w in (1.0, 2.0, 3.0, 5.0, 8.0):
        for Nm in (3.0, 5.0, 8.0):
            if Nm < w: continue
            eq, fl, liq = X.sim(tr, w, Nm); m = met(eq, fl, liq)
            if m and m[2] > -70 and (bw is None or m[0] > bw[0]): bw = (m[0], w, Nm)
    if bw is None: continue
    eq, fl, liq = X.sim(te, bw[1], bw[2]); m = met(eq, fl, liq)
    c = m[0] if m else -100.0; tot.append(c)
    P(f"  {y.year}: 학습최적 w={bw[1]:.0f} Nmax={bw[2]:.0f} → 검증 연 {c:+8.0f}% ({len(te)}건)")
if tot: P(f"  → 평균 {np.mean(tot):+.0f}% 양수 {sum(1 for x in tot if x>0)}/{len(tot)}")

P("\n[5] 실행 스트레스 재확인 (핵심만)")
for eps, fm, ft, lab in ((0.0002, 2.0, 5.0, "기준"), (0.0005, 3.0, 6.0, "보수"),
                         (0.002, 5.0, 8.0, "최악 관통20bp 메5/테8"), (0.005, 8.0, 12.0, "극단 관통50bp 메8/테12")):
    E2 = build_a("open", ext=EXT, eps=eps, fm=fm, ft=ft)
    row = f"  {lab:24s} n={len(E2):4d} |"
    for w, Nm, tag in ((1.0, 1.0, "무레버"), (3.0, 5.0, "중간"), (5.0, 8.0, "공격")):
        eq, fl, liq = X.sim(E2, w, Nm); m = met(eq, fl, liq)
        row += f" {tag} {'파산' if m is None else f'{m[0]:+5.0f}%'}"
    P(row)

P("\n[6] 상위 거래 제외 강건성")
order = np.argsort(-r)
for drop in (0, 1, 2, 5, 10, 20):
    E2 = [x for i, x in enumerate(E) if i not in set(order[:drop].tolist())]
    row = f"  상위 {drop:2d}건 제외 (n={len(E2):4d}) |"
    for w, Nm, tag in ((1.0, 1.0, "무레버"), (3.0, 5.0, "중간"), (5.0, 8.0, "공격")):
        eq, fl, liq = X.sim(E2, w, Nm); m = met(eq, fl, liq)
        row += f" {tag} {'파산' if m is None else f'{m[0]:+5.0f}%'}"
    P(row)

P("\n[7] ★ 최종 3구간 성과표")
P("=" * 100)
P(f"  {'구간':16s} {'명목':16s} {'전체연':>8s} {'2024~':>8s} {'2025~':>8s} {'최악여유':>9s} {'청산':>5s}")
P("=" * 100)
rows = {}
for w, Nm, tag, nm2 in ((1.0, 1.0, "① 무레버리지", "총 ≤ 1x"), (2.0, 2.0, "  참고", "총 ≤ 2x"),
                        (3.0, 5.0, "② 중간", "진입 3x / 총 5x"), (5.0, 5.0, "  중간+", "진입 5x / 총 5x"),
                        (5.0, 8.0, "③ 공격", "진입 5x / 총 8x"), (8.0, 12.0, "  한계초과", "총 12x")):
    eq, fl, liq = X.sim(E, w, Nm); m = met(eq, fl, liq)
    if m is None: P(f"  {tag:16s} {nm2:16s} {'파산':>8s} {'':>8s} {'':>8s} {'':>9s} {liq:5d}"); continue
    rows[tag] = m
    P(f"  {tag:16s} {nm2:16s} {m[0]:+7.0f}% {m[3]:+7.0f}% {m[4]:+7.0f}% {m[2]:8.1f}% {liq:5d}")
P("=" * 100)
P("\n  연도별 (%)")
P(f"  {'구간':16s} " + " ".join(f"{y:>7d}" for y in range(2020, 2027)))
for tag, m in rows.items():
    yr = m[5].resample("YE").last()/m[5].resample("YE").first()-1
    dd = {i.year: v*100 for i, v in yr.items()}
    P(f"  {tag:16s} " + " ".join(f"{dd.get(y, float('nan')):+7.0f}" for y in range(2020, 2027)))
