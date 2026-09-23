"""z0.20 — 롱 전용 최적화. 위쪽 예측이 아래쪽보다 정확한 비대칭을 활용.
사용: python research/z0_20_long.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from z0_20_final import build_sigs, exp_z
from z0_20_adaptive import D, SYM, STEP, TH_MKT, TH_LIM
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)


def ev_side(sig, y, lab, side=1, qs=(0.01, 0.02, 0.05, 0.10, 0.20), show=True):
    """side=1: 상위 분위에서 롱만. side=-1: 하위 분위에서 숏만."""
    n = min(len(sig), len(y)); sig = sig[:n]; y = y[:n]
    dec = (y != 0) & np.isfinite(sig)
    if dec.sum() < 2000: return None
    q = pd.Series(np.where(dec, sig, np.nan)).rank(pct=True).values
    best = 0; row = f"    {lab:34s}"
    for qq in qs:
        sel = dec & ((q >= 1-qq) if side == 1 else (q <= qq))
        if sel.sum() < 100: continue
        acc = (y[sel] == side).mean()*100
        row += f" | {qq*100:.0f}% {acc:5.2f}%(n={int(sel.sum())//1000}k)"
        best = max(best, acc)
    if show: P(row + (f"  ★{best:.2f}%" if best > TH_MKT else ""))
    return best


P("[1] 롱 전용 vs 숏 전용 — 방향별 정확도 분해")
ST = {}
for s in SYM:
    S = build_sigs(s); d = D[s]; y = d["y"]
    Z = {nm: exp_z(np.asarray(v, float)) for nm, v in S.items()}
    ST[s] = (S, Z)
    P(f"\n  ══ {s} ══")
    cands = ["고저비대칭4320", "고저비대칭8640", "절대값8640", "반변동성4320", "OFI1440", "효율비1440"]
    for nm in cands:
        if nm not in S: continue
        P(f"   {nm}")
        ev_side(np.asarray(S[nm], float), y, "     롱(상위분위)", side=1)
        ev_side(np.asarray(S[nm], float), y, "     숏(하위분위)", side=-1)

P("\n[2] 결합의 롱 전용 성과")
FIXED = {
    "고저4320+절대8640": ["고저비대칭4320", "절대값8640"],
    "고저3개": ["고저비대칭2880", "고저비대칭4320", "고저비대칭8640"],
    "고저4320+절대8640+OFI1440": ["고저비대칭4320", "절대값8640", "OFI1440"],
    "고저4320+반변동4320": ["고저비대칭4320", "반변동성4320"],
    "5개결합": ["고저비대칭4320", "고저비대칭8640", "절대값8640", "반변동성4320", "OFI1440"],
}
CB = {}
for s in SYM:
    S, Z = ST[s]; d = D[s]; y = d["y"]
    P(f"\n  ══ {s} ══")
    for lab, names in FIXED.items():
        names = [x for x in names if x in Z]
        n = min(min(len(Z[x]) for x in names), len(y))
        c = np.nanmean(np.vstack([Z[x][:n] for x in names]), axis=0)
        CB[(s, lab)] = c
        ev_side(c, y, f"롱 {lab}", side=1)

P("\n[3] 최고 롱 구성의 연도별 안정성")
for s in SYM:
    d = D[s]; y = d["y"]
    P(f"\n  ══ {s} ══")
    for lab in FIXED:
        c = CB[(s, lab)]
        n = min(len(c), len(y))
        ts = d["idx"][::STEP][:n]
        df = pd.DataFrame({"sig": c[:n], "y": y[:n]}, index=ts)
        for qq in (0.02, 0.05):
            row = f"    {lab:26s} 상위{qq*100:.0f}% "
            ok = True
            for yr, g in df.groupby(df.index.year):
                gg = g[(g.y != 0) & np.isfinite(g.sig)]
                if len(gg) < 3000: continue
                sel = gg[gg.sig.rank(pct=True) >= 1-qq]
                if len(sel) < 30: continue
                a = (sel.y == 1).mean()*100
                row += f"{yr}:{a:5.1f} "
                if a < TH_MKT: ok = False
            P(row + ("  ★전년통과" if ok else ""))

P("\n[4] 확신 극단(상위1%)에서의 최대 정확도 + 표본")
for s in SYM:
    d = D[s]; y = d["y"]
    P(f"\n  ══ {s} ══")
    S, Z = ST[s]
    allc = dict(CB)
    for nm in ("고저비대칭4320", "절대값8640", "반변동성4320"):
        if nm in Z: allc[(s, nm)] = Z[nm]
    for (ss, lab), c in allc.items():
        if ss != s: continue
        n = min(len(c), len(y))
        cc = c[:n]; yy = y[:n]
        dec = (yy != 0) & np.isfinite(cc)
        q = pd.Series(np.where(dec, cc, np.nan)).rank(pct=True).values
        row = f"    {lab:28s}"
        for qq in (0.005, 0.01, 0.02):
            sel = dec & (q >= 1-qq)
            if sel.sum() < 50: continue
            row += f" | 상위{qq*100:.1f}% {(yy[sel]==1).mean()*100:5.2f}% (n={int(sel.sum())})"
        P(row)
