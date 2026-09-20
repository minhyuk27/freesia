"""방향1 심화 — 불균형 바 위의 되돌림. 정렬 수정판.
r[i] = 바 i→i+1 수익. 진입은 바 i+1 종가(관측 완료 시점), 청산은 i+1+H 바 종가.
사용: python research/imbbar_deep.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
from infobars import imbalance_bars, dollar_bars, D, SYM
import exit_tier as X
from absfloor import met
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
TE = pd.Timestamp("2024-01-01", tz="UTC"); REC = pd.Timestamp("2025-01-01", tz="UTC")


def make_imb(s, mult):
    d = D[s]
    imb_tot = np.abs(2*d["tb"] - d["qv"]).sum()
    n_target = len(d["c"])//60
    thr = imb_tot/n_target*mult
    ii, io, ih, il, ic, iv = imbalance_bars(d["c"], d["o"], d["h"], d["l"], d["qv"], d["tb"], thr)
    return ii, ic, io, ih, il, iv


P("[1] 불균형 바 임계 × 되돌림 강도 (정렬 수정)")
P(f"  {'종목':6s} {'배수':>5s} {'바수':>7s} {'평균간격':>8s} {'조건':>9s} {'H':>3s} {'n':>6s} {'수익bp':>9s} {'승률':>6s}")
STORE = {}
for s in SYM:
    d = D[s]
    for mult in (0.25, 0.5, 1.0, 2.0):
        ii, ic, io, ih, il, iv = make_imb(s, mult)
        if len(ic) < 2000: continue
        STORE[(s, mult)] = (ii, ic, io, ih, il)
        r = np.diff(np.log(ic))              # r[i] = 바 i→i+1
        gap = np.mean(np.diff(ii))
        q = pd.Series(r).rank(pct=True).values
        for lo_q, lab in ((0.005, "하위0.5%"), (0.01, "하위1%"), (0.05, "하위5%")):
            for H in (1, 3, 5):
                # 진입: 바 i+1 종가(=ic[i+1]), 청산: ic[i+1+H]
                base = ic[1:]                 # 길이 len(ic)-1, base[i] = ic[i+1]
                fwd = np.full(len(base), np.nan)
                if H < len(base):
                    fwd[:-H] = base[H:]/base[:-H] - 1.0
                sel = np.isfinite(fwd) & np.isfinite(r) & (q <= lo_q)
                if sel.sum() < 60: continue
                P(f"  {s[:3]:6s} {mult:5.2f} {len(ic):7d} {gap:7.1f}분 {lab:>9s} {H:3d} {int(sel.sum()):6d} "
                  f"{fwd[sel].mean()*1e4:+8.2f} {(fwd[sel] > 0).mean():6.3f}")

P("\n[2] 시간 바 동일 조건 대조 (같은 개수·같은 분위)")
for s in SYM:
    d = D[s]
    for gap in (60, 160):
        cl = d["c"][::gap]
        r = np.diff(np.log(cl))
        q = pd.Series(r).rank(pct=True).values
        base = cl[1:]
        for lo_q, lab in ((0.01, "하위1%"),):
            for H in (1, 3, 5):
                fwd = np.full(len(base), np.nan)
                if H < len(base): fwd[:-H] = base[H:]/base[:-H] - 1.0
                sel = np.isfinite(fwd) & (q <= lo_q)
                if sel.sum() < 60: continue
                P(f"  {s[:3]} 시간{gap}분 {lab} H={H}: n={int(sel.sum()):5d} {fwd[sel].mean()*1e4:+8.2f}bp 승률 {(fwd[sel]>0).mean():.3f}")

P("\n[3] 불균형 바 전략화 — 확장창 분위 임계, 3단 레버리지")
def build_imb(s, mult, q_th, H, fee=10.0):
    if (s, mult) not in STORE: return []
    ii, ic, io, ih, il = STORE[(s, mult)]
    d = D[s]
    r = np.diff(np.log(ic))
    n = len(r)
    th = np.full(n, np.nan)
    warm = max(500, n//10)
    for i in range(warm, n, 200):
        th[i:i+200] = np.nanquantile(r[:i], q_th)
    E = []
    for i in range(warm, n-H-1):
        if not np.isfinite(th[i]) or r[i] > th[i]: continue
        a = int(ii[i+1]); b = int(ii[i+1+H])
        if b >= len(d["c"]): break
        px = ic[i+1]; ex = ic[i+1+H]
        lo_ = d["l"][a:b+1].min() if b > a else px
        E.append((a, b, float(ex/px - 1.0 - fee/1e4), float(max(1.0-lo_/px, 0)),
                  d["idx"][a], (s, mult), 0.01, 1.0))
    return E

for mult in (0.25, 0.5, 1.0):
    for q_th in (0.01, 0.03):
        for H in (1, 3):
            E = sorted(sum([build_imb(s, mult, q_th, H) for s in SYM], []), key=lambda z: z[0])
            if len(E) < 100: continue
            r = np.array([x[2] for x in E]); mae = np.array([x[3] for x in E])
            out = f"  배수{mult} q={q_th} H={H}: n={len(E):5d}({len(E)/6.7:4.0f}/년) {r.mean()*1e4:+7.2f}bp " \
                  f"승률{(r>0).mean():.3f} MAE95 {np.quantile(mae,.95)*100:5.2f}% |"
            for w, Nm, tag in ((1.0, 1.0, "무레버"), (2.0, 3.0, "저레버"), (3.0, 5.0, "중간")):
                eq, fl, liq = X.sim(E, w, Nm); m = met(eq, fl, liq)
                out += f" {tag} {'파산' if m is None else f'{m[0]:+7.1f}%'}"
            P(out)
            eq, fl, liq = X.sim(E, 2.0, 3.0); m = met(eq, fl, liq)
            if m is not None:
                df = pd.DataFrame([(x[4], x[2]) for x in E], columns=["ts", "r"]).set_index("ts")
                g = df.groupby(df.index.year).r
                P(f"      저레버 24~ {m[3]:+7.1f}% 25~ {m[4]:+7.1f}% 최악 {m[2]:6.1f}% | " +
                  " ".join(f"{y}{v*1e4:+6.0f}×{n}" for (y, v), (_, n) in zip(g.mean().items(), g.size().items())))
