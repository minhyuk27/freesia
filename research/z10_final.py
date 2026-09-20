"""z1.0 최종 검증 — 상위 3후보 플라시보·상위거래제외·파라미터 고원·3단 성과표."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from z10_synth import synth, prep, D0
from meta_strong import SYM
from freesia import store
import exit_tier as X
from absfloor import met
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
TE = pd.Timestamp("2024-01-01", tz="UTC"); REC = pd.Timestamp("2025-01-01", tz="UTC")

CAND = {
    "A 저변동+켈리": dict(use_lowvol=True, use_meta=False, use_kelly=True, size="flat"),
    "B 저변동+MAE사이징": dict(use_lowvol=True, use_meta=False, use_kelly=False, size="mae"),
    "C 저변동30%+메타": dict(use_lowvol=True, use_meta=True, use_kelly=False, size="flat", lv_q=0.3),
    "D 메타단독(z0.9)": dict(use_lowvol=False, use_meta=True, use_kelly=False, size="flat"),
}

@numba.njit(cache=True)
def rnd(o, c, st, hd, fee):
    out = np.zeros(len(st))
    for i in range(len(st)):
        e = st[i]+hd[i]
        if e >= len(c): e = len(c)-1
        out[i] = (c[e]/o[st[i]]-1.0) - fee/1e4
    return out
K = {s: store.load("klines_1m", s).set_index("ts") for s in SYM}

for nm, cfg in CAND.items():
    E = synth(**cfg)
    if len(E) < 60: continue
    r = np.array([x[2] for x in E]); mae = np.array([x[3] for x in E])
    P("\n" + "="*130)
    P(f"★ {nm}  n={len(E)} ({len(E)/5.3:.0f}/년) 거래당 {r.mean()*1e4:+.1f}bp 중앙 {np.median(r)*1e4:+.1f}bp "
      f"승률 {(r>0).mean():.3f} | MAE 평균 {mae.mean()*100:.2f}% 95분위 {np.quantile(mae,.95)*100:.2f}%")
    P("="*130)
    P(f"  {'구간':12s} {'명목':10s} {'전체연':>9s} {'월환산':>8s} {'MDD':>7s} {'최악여유':>9s} {'24~':>9s} {'25~':>9s} {'월승률':>7s}")
    for w, Nm, tag, n2 in ((1.0, 1.0, "① 무레버", "≤1x"), (2.0, 3.0, "저레버", "2x/3x"),
                           (3.0, 5.0, "② 중간", "3x/5x"), (4.0, 6.0, "중간+", "4x/6x"), (5.0, 8.0, "③ 공격", "5x/8x")):
        eq, fl, liq = X.sim(E, w, Nm); m = met(eq, fl, liq)
        if m is None: P(f"  {tag:12s} {n2:10s} {'파산':>9s}"); continue
        mo = (1+m[0]/100)**(1/12)-1
        mon = (m[5].resample("ME").last()/m[5].resample("ME").first()-1).dropna()
        P(f"  {tag:12s} {n2:10s} {m[0]:+8.1f}% {mo*100:+7.1f}% {m[1]:6.1f}% {m[2]:8.1f}% {m[3]:+8.1f}% {m[4]:+8.1f}% {float((mon>0).mean()):6.2f}")
    df = pd.DataFrame([(x[4], x[2]) for x in E], columns=["ts", "r"]).set_index("ts")
    g = df.groupby(df.index.year).r
    P("  연도별 거래당: " + " ".join(f"{y}{v*1e4:+6.0f}×{n}" for (y, v), (_, n) in zip(g.mean().items(), g.size().items())))
    # 플라시보
    hd = np.array([x[1]-x[0] for x in E]); rng = np.random.default_rng(17); nul = []
    for _ in range(400):
        acc = []
        for s in SYM:
            o = K[s].open.values; c = K[s].close.values; N = len(c)
            cnt = max(len(E)//2, 1)
            acc.append(rnd(o, c, rng.integers(0, N-500, cnt).astype(np.int64),
                           rng.choice(hd, cnt).astype(np.int64), 7.0))
        nul.append(np.concatenate(acc).mean()*1e4)
    nul = np.array(nul)
    P(f"  플라시보 400회: 위약 평균 {nul.mean():+.2f}bp 최대 {nul.max():+.2f} | 실제 {r.mean()*1e4:+.2f}bp → 백분위 {(nul < r.mean()*1e4).mean()*100:.1f}%")
    # 상위 거래 제외
    order = np.argsort(-r)
    row = "  상위거래 제외: "
    for drop in (5, 20, 50):
        E2 = [x for i, x in enumerate(E) if i not in set(order[:drop].tolist())]
        r2 = np.array([x[2] for x in E2])
        eq, fl, liq = X.sim(E2, 3.0, 5.0); m2 = met(eq, fl, liq)
        row += f"{drop}건→{r2.mean()*1e4:+5.1f}bp/중간 {'파산' if m2 is None else f'{m2[0]:+6.1f}%'}  "
    P(row)

P("\n" + "="*130)
P("[파라미터 고원] 저변동 임계 × 재학습 주기 × 특징 수 (A 구성)")
P("="*130)
for lq in (0.3, 0.4, 0.5, 0.6):
    row = f"  저변동 하위{lq*100:.0f}%: "
    for nc in (8, 10, 15):
        E = synth(use_lowvol=True, use_meta=False, use_kelly=True, size="flat", lv_q=lq, ncol=nc)
        if len(E) < 60: row += f"특징{nc} 표본부족  "; continue
        eq, fl, liq = X.sim(E, 3.0, 5.0); m = met(eq, fl, liq)
        row += f"특징{nc} {'파산' if m is None else f'{m[0]:+6.1f}%'}  "
    P(row)
for rd in (14, 30, 60):
    E = synth(use_lowvol=True, use_meta=False, use_kelly=True, size="flat", refit_days=rd)
    if len(E) < 60: continue
    eq, fl, liq = X.sim(E, 3.0, 5.0); m = met(eq, fl, liq)
    P(f"  재학습 {rd}일: n={len(E)} 중간 " + ("파산" if m is None else f"{m[0]:+.1f}% 최악 {m[2]:.1f}%"))
