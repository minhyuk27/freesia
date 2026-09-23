"""z0.7 최종 검증 — 두 갈래(A: 극단·짧은보유·레버리지 가능 / B: 고빈도·무레버) 전면 검증."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from z0_7_dev import build, D, SYM
import exit_tier as X
from absfloor import met
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
TE = pd.Timestamp("2024-01-01", tz="UTC"); REC = pd.Timestamp("2025-01-01", tz="UTC")
A = dict(q=0.001, dn=30, hold=30, cool=30, mkt=1)            # 극단·짧은보유·시장가
B = dict(q=0.01, dn=15, hold=240, cool=30, mkt=0, kx=1.0)    # 고빈도·지정가


def full(E, w, Nm):
    eq, fl, liq = X.sim(E, w, Nm); return met(eq, fl, liq), liq


def show(E, w, Nm, tag):
    m, liq = full(E, w, Nm)
    if m is None: P(f"     {tag:16s} 파산 (청산 {liq})"); return None
    mo = (1+m[0]/100)**(1/12)-1
    mon = (m[5].resample("ME").last()/m[5].resample("ME").first()-1).dropna()
    P(f"     {tag:16s} 연 {m[0]:+8.1f}% (월 {mo*100:+5.1f}%) MDD {m[1]:6.1f}% 최악 {m[2]:6.1f}% | "
      f"24~ {m[3]:+8.1f}% | 25~ {m[4]:+8.1f}% | 월승률 {float((mon>0).mean()):.2f}")
    return m


for nm, cfg in (("A 극단·30분보유·시장가", A), ("B 고빈도·240분보유·지정가", B)):
    E = build(**cfg)
    r = np.array([x[2] for x in E]); mae = np.array([x[3] for x in E])
    df = pd.DataFrame([(x[4], x[2]) for x in E], columns=["ts", "r"]).set_index("ts")
    g = df.groupby(df.index.year).r
    P(f"\n{'='*100}\n★ {nm}  n={len(r)} ({len(r)/6.7:.0f}/년) 거래당 {r.mean()*1e4:+.1f}bp 중앙 {np.median(r)*1e4:+.1f}bp "
      f"승률 {(r>0).mean():.3f}\n   MAE 평균 {mae.mean()*100:.2f}% 95분위 {np.quantile(mae,0.95)*100:.2f}% 최대 {mae.max()*100:.2f}%\n{'='*100}")
    P("   연도별: " + " ".join(f"{y}{v*1e4:+6.1f}×{n}" for (y, v), (_, n) in zip(g.mean().items(), g.size().items())))
    P("   3단 레버리지:")
    for w, Nm, tag in ((1.0, 1.0, "① 무레버"), (2.0, 3.0, "저레버2x/3x"), (3.0, 5.0, "② 중간3x/5x"), (5.0, 8.0, "③ 공격5x/8x")):
        show(E, w, Nm, tag)
    P("   세밀 스윕:")
    best = None
    for w in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
        for Nm in (1.0, 2.0, 3.0, 5.0):
            if Nm < w: continue
            m, liq = full(E, w, Nm)
            if m is None or liq: continue
            if best is None or m[0] > best[0]: best = (m[0], m[1], m[2], m[3], m[4], w, Nm)
    if best:
        P(f"     최대 무청산 w={best[5]:.1f} Nmax={best[6]:.0f} → 연 {best[0]:+.1f}% MDD {best[1]:.1f}% 최악 {best[2]:.1f}% | 24~ {best[3]:+.1f}% | 25~ {best[4]:+.1f}%")

    P("   비용 민감도 (무레버):")
    for fm, ft, eps, lab in ((2, 5, 0.0002, "기준"), (3, 6, 0.0005, "보수"), (4, 7, 0.001, "비관"), (5, 8, 0.002, "최악")):
        E2 = build(**cfg, fm=fm, ft=ft, eps=eps)
        r2 = np.array([x[2] for x in E2])
        m, _ = full(E2, 1.0, 1.0)
        P(f"     {lab:6s} 메{fm}/테{ft} 관통{eps*1e4:.0f}bp: 거래당 {r2.mean()*1e4:+6.1f}bp 무레버 연 {m[0] if m else float('nan'):+7.1f}%")

    P("   플라시보 500회:")
    @numba.njit(cache=True)
    def rnd(o, c, st, hd, fee):
        out = np.zeros(len(st))
        for i in range(len(st)):
            e = st[i]+hd[i]
            if e >= len(c): e = len(c)-1
            out[i] = (c[e]/o[st[i]]-1.0) - fee/1e4
        return out
    hd = np.array([x[1]-x[0] for x in E]); rng = np.random.default_rng(41); nul = []
    fee = 10.0 if cfg.get("mkt", 1) else 7.0
    for _ in range(500):
        acc = []
        for s in SYM:
            d = D[s]; N = len(d["c"]); cnt = max(len(E)//2, 1)
            acc.append(rnd(d["o"], d["c"], rng.integers(0, N-500, cnt).astype(np.int64),
                           rng.choice(hd, cnt).astype(np.int64), fee))
        nul.append(np.concatenate(acc).mean()*1e4)
    nul = np.array(nul)
    P(f"     위약 평균 {nul.mean():+.2f}bp 99분위 {np.quantile(nul,0.99):+.2f} 최대 {nul.max():+.2f} | "
      f"실제 {r.mean()*1e4:+.2f}bp → 백분위 {(nul < r.mean()*1e4).mean()*100:.1f}%")

    P("   워크포워드 (2년 학습 → 1년 검증, w·Nmax 선택):")
    seg = lambda Ex, s0, e0: [x for x in Ex if (s0 is None or x[4] >= s0) and (e0 is None or x[4] < e0)]
    tot = []
    for y in pd.date_range("2022-01-01", "2026-01-01", freq="YS", tz="UTC"):
        tr = seg(E, y-pd.DateOffset(years=2), y); te = seg(E, y, y+pd.DateOffset(years=1))
        if len(tr) < 30 or len(te) < 10: continue
        bw = None
        for w in (1.0, 1.5, 2.0, 3.0):
            for Nm in (1.0, 2.0, 3.0, 5.0):
                if Nm < w: continue
                m, liq = full(tr, w, Nm)
                if m is None or liq: continue
                if bw is None or m[0] > bw[0]: bw = (m[0], w, Nm)
        if bw is None: continue
        m, liq = full(te, bw[1], bw[2])
        c = m[0] if m else -100.0
        tot.append(c)
        P(f"     {y.year}: 학습최적 w={bw[1]:.1f}/{bw[2]:.0f} → 검증 연 {c:+8.1f}% ({len(te)}건)")
    if tot: P(f"     → 평균 {np.mean(tot):+.1f}% 양수 {sum(1 for x in tot if x>0)}/{len(tot)}")
