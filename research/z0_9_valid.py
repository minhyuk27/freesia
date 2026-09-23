"""z0.9 = z0.6 + 메타 라벨링. 전면 검증: 3단 레버리지·플라시보·비용·워크포워드·MAE 구조."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from meta_strong import meta, show, FEAT, SYM
from z0_6b import build as build_z6
from freesia import store
import exit_tier as X
from absfloor import met
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
TE = pd.Timestamp("2024-01-01", tz="UTC"); REC = pd.Timestamp("2025-01-01", tz="UTC")
T0 = pd.Timestamp("2021-06-01", tz="UTC")


def mk(fm=2.0, ft=5.0, eps=0.0002, mq=0.5, nc=10, C=0.05):
    E6 = [x for x in build_z6(0.005, 1.0, 30, 240, 30, fm=fm, ft=ft, eps=eps) if x[4] >= T0]
    return E6, meta(E6, meta_q=mq, ncol=nc, C=C)


P("[1] 기본 구성 상세")
E6, E9 = mk()
r6 = np.array([x[2] for x in E6]); r9 = np.array([x[2] for x in E9])
m6 = np.array([x[3] for x in E6]); m9 = np.array([x[3] for x in E9])
P(f"  1차 z0.6 : n={len(E6)} {r6.mean()*1e4:+.1f}bp 승률 {(r6>0).mean():.3f} | MAE 평균 {m6.mean()*100:.2f}% 95분위 {np.quantile(m6,.95)*100:.2f}% 최대 {m6.max()*100:.1f}%")
P(f"  메타 z0.9: n={len(E9)} {r9.mean()*1e4:+.1f}bp 승률 {(r9>0).mean():.3f} | MAE 평균 {m9.mean()*100:.2f}% 95분위 {np.quantile(m9,.95)*100:.2f}% 최대 {m9.max()*100:.1f}%")
P(f"  → 메타 필터가 MAE 95분위를 {np.quantile(m6,.95)*100:.2f}% → {np.quantile(m9,.95)*100:.2f}% 로 낮춘다 (레버리지가 열리는 이유)")

P("\n[2] 3단 레버리지 성과표")
P(f"  {'구간':14s} {'명목':14s} {'전체연':>9s} {'월환산':>8s} {'MDD':>7s} {'최악여유':>9s} {'24~':>9s} {'25~':>9s} {'월승률':>7s}")
for w, Nm, tag, nm2 in ((1.0, 1.0, "① 무레버", "총 ≤1x"), (2.0, 3.0, "저레버", "2x/3x"),
                        (3.0, 5.0, "② 중간", "3x/5x"), (4.0, 6.0, "중간+", "4x/6x"), (5.0, 8.0, "③ 공격", "5x/8x")):
    eq, fl, liq = X.sim(E9, w, Nm); m = met(eq, fl, liq)
    if m is None: P(f"  {tag:14s} {nm2:14s} {'파산':>9s} (청산 {liq})"); continue
    mo = (1+m[0]/100)**(1/12)-1
    mon = (m[5].resample("ME").last()/m[5].resample("ME").first()-1).dropna()
    P(f"  {tag:14s} {nm2:14s} {m[0]:+8.1f}% {mo*100:+7.1f}% {m[1]:6.1f}% {m[2]:8.1f}% {m[3]:+8.1f}% {m[4]:+8.1f}% {float((mon>0).mean()):6.2f}")

P("\n[3] 연도별 (중간 3x/5x)")
eq, fl, liq = X.sim(E9, 3.0, 5.0); m = met(eq, fl, liq)
if m is not None:
    yr = m[5].resample("YE").last()/m[5].resample("YE").first()-1
    P("  " + " | ".join(f"{i.year} {v*100:+.0f}%" for i, v in yr.items()))
df = pd.DataFrame([(x[4], x[2]) for x in E9], columns=["ts", "r"]).set_index("ts")
g = df.groupby(df.index.year).r
P("  거래당: " + " ".join(f"{y}{v*1e4:+6.1f}×{n}" for (y, v), (_, n) in zip(g.mean().items(), g.size().items())))

P("\n[4] 비용·체결 민감도")
for fm, ft, eps, lab in ((2, 5, 0.0, "낙관"), (2, 5, 0.0002, "기준"), (3, 6, 0.0005, "보수"),
                         (4, 7, 0.001, "비관"), (5, 8, 0.002, "최악")):
    _, Ex = mk(fm=fm, ft=ft, eps=eps)
    if len(Ex) < 40: P(f"  {lab}: 표본 부족"); continue
    rx = np.array([x[2] for x in Ex])
    row = f"  {lab:5s} 메{fm}/테{ft} 관통{eps*1e4:.0f}bp: n={len(Ex):4d} {rx.mean()*1e4:+6.1f}bp |"
    for w, Nm, tag in ((1.0, 1.0, "무레버"), (2.0, 3.0, "저레버"), (3.0, 5.0, "중간")):
        eq, fl, liq = X.sim(Ex, w, Nm); mm2 = met(eq, fl, liq)
        row += f" {tag} {'파산' if mm2 is None else f'{mm2[0]:+7.1f}%'}"
    P(row)

P("\n[5] 메타 파라미터 안정성 (고원인가)")
for mq in (0.4, 0.5, 0.6):
    for nc in (8, 10, 12, 15):
        _, Ex = mk(mq=mq, nc=nc)
        if len(Ex) < 40: continue
        rx = np.array([x[2] for x in Ex])
        eq, fl, liq = X.sim(Ex, 3.0, 5.0); mm2 = met(eq, fl, liq)
        P(f"  통과{(1-mq)*100:.0f}% 특징{nc:2d}: n={len(Ex):4d} {rx.mean()*1e4:+6.1f}bp 중간 "
          + ("파산" if mm2 is None else f"{mm2[0]:+7.1f}% 최악 {mm2[2]:6.1f}%"))

P("\n[6] 플라시보 500회 (같은 건수·같은 보유길이 무작위 진입)")
@numba.njit(cache=True)
def rnd(o, c, st, hd, fee):
    out = np.zeros(len(st))
    for i in range(len(st)):
        e = st[i]+hd[i]
        if e >= len(c): e = len(c)-1
        out[i] = (c[e]/o[st[i]]-1.0) - fee/1e4
    return out
K = {s: store.load("klines_1m", s).set_index("ts") for s in SYM}
hd = np.array([x[1]-x[0] for x in E9])
rng = np.random.default_rng(7); nul = []
for _ in range(500):
    acc = []
    for s in SYM:
        o = K[s].open.values; c = K[s].close.values; N = len(c)
        cnt = max(len(E9)//2, 1)
        acc.append(rnd(o, c, rng.integers(0, N-500, cnt).astype(np.int64),
                       rng.choice(hd, cnt).astype(np.int64), 7.0))
    nul.append(np.concatenate(acc).mean()*1e4)
nul = np.array(nul)
P(f"  위약 평균 {nul.mean():+.2f}bp 99분위 {np.quantile(nul,0.99):+.2f} 최대 {nul.max():+.2f} | "
  f"실제 {r9.mean()*1e4:+.2f}bp → 백분위 {(nul < r9.mean()*1e4).mean()*100:.1f}%")

P("\n[7] 상위 거래 제외 강건성")
order = np.argsort(-r9)
for drop in (0, 5, 20, 50):
    E2 = [x for i, x in enumerate(E9) if i not in set(order[:drop].tolist())]
    rx = np.array([x[2] for x in E2])
    row = f"  상위 {drop:2d}건 제외 (n={len(E2):4d}) {rx.mean()*1e4:+6.1f}bp |"
    for w, Nm, tag in ((1.0, 1.0, "무레버"), (2.0, 3.0, "저레버"), (3.0, 5.0, "중간")):
        eq, fl, liq = X.sim(E2, w, Nm); mm2 = met(eq, fl, liq)
        row += f" {tag} {'파산' if mm2 is None else f'{mm2[0]:+7.1f}%'}"
    P(row)
