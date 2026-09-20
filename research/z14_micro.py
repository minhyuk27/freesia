"""z1.4 — 초단기 깊은 딥(15·30분봉). 되돌림에 필요한 최소 시간의 경계 위에서.
사용: python research/z14_micro.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from micro import build_depth, D, SYM
import exit_tier as X
from absfloor import met
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
TE = pd.Timestamp("2024-01-01", tz="UTC"); REC = pd.Timestamp("2025-01-01", tz="UTC")


def mk(B, kx, H=0, fm=2.0, ft=5.0, eps=0.0002):
    return sorted(sum([build_depth(s, B, kx, H, fee_m=fm, fee_t=ft, eps=eps) for s in SYM], []),
                  key=lambda z: z[0])


def rep(E, lab, full=False):
    if len(E) < 60: P(f"  {lab:44s} 표본부족({len(E)})"); return None
    r = np.array([x[2] for x in E]); mae = np.array([x[3] for x in E])
    out = f"  {lab:44s} n={len(E):5d}({len(E)/6.7:4.0f}/년) {r.mean()*1e4:+6.1f}bp 승률{(r>0).mean():.3f} |"
    best = None
    for w, Nm, tag in ((1.0, 1.0, "무레버"), (2.0, 3.0, "저레버"), (3.0, 5.0, "중간"),
                       (4.0, 6.0, "중간+"), (5.0, 8.0, "공격")):
        eq, fl, liq = X.sim(E, w, Nm); m = met(eq, fl, liq)
        out += f" {tag}{'파산' if m is None else f'{m[0]:+7.1f}%'}"
        if m and (best is None or m[0] > best[0]): best = (m[0], m[1], m[2], m[3], m[4], tag, m[5])
    P(out)
    if full and best:
        mon = (best[6].resample("ME").last()/best[6].resample("ME").first()-1).dropna()
        P(f"      최고 {best[5]}: MDD {best[1]:.0f}% 최악 {best[2]:.0f}% 24~ {best[3]:+.0f}% 25~ {best[4]:+.0f}% "
          f"월승률 {float((mon>0).mean()):.2f} | MAE평균 {mae.mean()*100:.2f}% 95분위 {np.quantile(mae,.95)*100:.2f}%")
        df = pd.DataFrame([(x[4], x[2]) for x in E], columns=["ts", "r"]).set_index("ts")
        g = df.groupby(df.index.year).r
        P("      연도별 " + " ".join(f"{y}{v*1e4:+5.0f}×{n}" for (y, v), (_, n) in zip(g.mean().items(), g.size().items())))
    return best

P("[1] 깊이 미세 격자 (15·20·30·45·60분봉)")
best_all = []
for B in (15, 20, 30, 45, 60):
    for kx in (4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0):
        E = mk(B, kx)
        b = rep(E, f"{B}분봉 rv×{kx}")
        if b: best_all.append((b[0], B, kx))

P("\n[2] 상위 후보 상세")
best_all.sort(reverse=True)
for sc_, B, kx in best_all[:4]:
    rep(mk(B, kx), f"{B}분봉 rv×{kx}", full=True)

P("\n[3] 비용·체결 민감도 (최상위)")
if best_all:
    _, B, kx = best_all[0]
    for fm, ft, eps, lab in ((2, 2, 0.0002, "메이커왕복4bp"), (2, 5, 0.0, "낙관 관통0"),
                             (2, 5, 0.0002, "기준"), (3, 6, 0.0005, "보수"),
                             (4, 7, 0.001, "비관"), (5, 8, 0.002, "최악")):
        rep(mk(B, kx, fm=fm, ft=ft, eps=eps), f"  {lab} 메{fm}/테{ft} 관통{eps*1e4:.0f}bp")

P("\n[4] 플라시보 400회")
@numba.njit(cache=True)
def rnd(o, c, st, hd, fee):
    out = np.zeros(len(st))
    for i in range(len(st)):
        e = st[i]+hd[i]
        if e >= len(c): e = len(c)-1
        out[i] = (c[e]/o[st[i]]-1.0) - fee/1e4
    return out
if best_all:
    _, B, kx = best_all[0]
    E = mk(B, kx); r = np.array([x[2] for x in E]); hd = np.array([x[1]-x[0] for x in E])
    rng = np.random.default_rng(31); nul = []
    for _ in range(400):
        acc = []
        for s in SYM:
            o = D[s]["o"]; c = D[s]["c"]; N = len(c)
            cnt = max(len(E)//2, 1)
            acc.append(rnd(o, c, rng.integers(0, N-200, cnt).astype(np.int64),
                           rng.choice(hd, cnt).astype(np.int64), 7.0))
        nul.append(np.concatenate(acc).mean()*1e4)
    nul = np.array(nul)
    P(f"  {B}분봉 rv×{kx}: 위약 평균 {nul.mean():+.2f}bp 최대 {nul.max():+.2f} | 실제 {r.mean()*1e4:+.2f}bp → 백분위 {(nul < r.mean()*1e4).mean()*100:.1f}%")
    order = np.argsort(-r)
    row = "  상위거래 제외: "
    for drop in (5, 20, 50):
        E2 = [x for i, x in enumerate(E) if i not in set(order[:drop].tolist())]
        r2 = np.array([x[2] for x in E2])
        eq, fl, liq = X.sim(E2, 3.0, 5.0); m2 = met(eq, fl, liq)
        row += f"{drop}건→{r2.mean()*1e4:+5.1f}bp/중간{'파산' if m2 is None else f'{m2[0]:+6.1f}%'}  "
    P(row)

P("\n[5] 다단 분할 — 여러 깊이 동시 (z0.3 방식)")
for B in (15, 30):
    for ks, wgts in (((5.0, 7.0, 10.0), (1, 2, 4)), ((6.0, 8.0, 12.0), (1, 3, 6)),
                     ((4.0, 6.0, 9.0), (1, 2, 4))):
        E = []
        for k, w in zip(ks, wgts):
            for x in mk(B, k):
                E.append((x[0], x[1], x[2], x[3], x[4], (x[5][0], B, k), x[6], float(w)))
        E.sort(key=lambda z: z[0])
        rep(E, f"{B}분봉 다단 {ks} 가중{wgts}", full=True)
