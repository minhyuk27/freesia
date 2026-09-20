"""z1.3b — 패턴 트리거 + 지정가 대기. 시장가 추격을 버린다.
트리거(전저점/라운드 하향 이탈) 발생 → 그 아래 지정가를 걸고 기다림 → 체결되면 H봉 보유.
사용: python research/z13b_limit.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from z13_break import bars, D, SYM
import exit_tier as X
from absfloor import met
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
TE = pd.Timestamp("2024-01-01", tz="UTC"); REC = pd.Timestamp("2025-01-01", tz="UTC")


def build_lim(s, B, mode, LB=50, H=4, rounddiv=1.0, kx=1.0, W=1,
              fm=2.0, ft=5.0, eps=0.0002, cool=1, lowvol=None):
    """kx: 트리거 봉 종가 아래 rv60×kx 지점에 지정가. W: 유효 봉 수."""
    b = bars(s, B); d = D[s]
    cl = b["c"]; n = len(cl)
    lr = pd.Series(np.log(cl)).diff()
    rv = lr.rolling(50).std().shift(1).values          # 봉 스케일 변동성
    rvr = (lr.rolling(10).std()/lr.rolling(100).std()).shift(1).values
    prev_low = pd.Series(b["l"]).rolling(LB).min().shift(1).values
    broke = b["l"] < prev_low
    unrecov = b["c"] < prev_low
    mag = 10**np.floor(np.log10(np.maximum(cl, 1)))
    div = mag*rounddiv
    below = cl < np.round(cl/div)*div
    pb = np.roll(below, 1); pb[0] = below[0]
    rbreak = below & ~pb
    sig = (broke & unrecov) if mode == "prevlow" else (rbreak if mode == "round" else ((broke & unrecov) | rbreak))
    # 저변동 국면 임계(확장창)
    lv_th = np.full(n, np.inf)
    if lowvol is not None:
        for i in range(200, n, 50):
            lv_th[i:i+50] = np.nanquantile(rvr[:i][np.isfinite(rvr[:i])], lowvol) if np.isfinite(rvr[:i]).sum() > 50 else np.inf
    E = []; last = -10**9
    for i in range(max(LB, 120), n-H-W-1):
        if not sig[i] or not np.isfinite(rv[i]) or rv[i] <= 0: continue
        if lowvol is not None and not (np.isfinite(rvr[i]) and rvr[i] <= lv_th[i]): continue
        if i - last < cool: continue
        last = i
        T = cl[i]*(1.0 - rv[i]*kx)
        a0 = int(b["st"][i]) + B          # 다음 분부터
        a1 = int(b["st"][i+W]) + B - 1
        if a1 >= len(d["c"]): break
        f = -1
        for u in range(a0, a1+1):
            if d["l"][u] <= T*(1-eps): f = u; break
        if f < 0: continue
        e_i = i + W + H
        if e_i >= n: break
        bb = int(b["st"][e_i]) + B - 1
        if bb >= len(d["c"]) or bb <= f: continue
        lo_ = d["l"][f:bb+1].min()
        E.append((f, bb, float(cl[e_i]/T - 1.0 - (fm+ft)/1e4), float(max(1.0-lo_/T, 0)),
                  d["idx"][f], (s, mode, B), 0.01, 1.0))
    return E


def rep(E, lab, full=False):
    if len(E) < 60: P(f"  {lab:48s} 표본부족({len(E)})"); return None
    r = np.array([x[2] for x in E])
    out = f"  {lab:48s} n={len(E):5d}({len(E)/6.7:4.0f}/년) {r.mean()*1e4:+6.1f}bp 승률{(r>0).mean():.3f} |"
    best = None
    for w, Nm, tag in ((1.0, 1.0, "무레버"), (2.0, 3.0, "저레버"), (3.0, 5.0, "중간"), (5.0, 8.0, "공격")):
        eq, fl, liq = X.sim(E, w, Nm); m = met(eq, fl, liq)
        out += f" {tag}{'파산' if m is None else f'{m[0]:+7.1f}%'}"
        if m and (best is None or m[0] > best[0]): best = (m[0], m[1], m[2], m[3], m[4], tag, m[5])
    P(out)
    if full and best:
        mon = (best[6].resample("ME").last()/best[6].resample("ME").first()-1).dropna()
        P(f"      최고 {best[5]}: MDD {best[1]:.0f}% 최악 {best[2]:.0f}% 24~ {best[3]:+.0f}% 25~ {best[4]:+.0f}% 월승률 {float((mon>0).mean()):.2f}")
        df = pd.DataFrame([(x[4], x[2]) for x in E], columns=["ts", "r"]).set_index("ts")
        g = df.groupby(df.index.year).r
        P("      연도별 " + " ".join(f"{y}{v*1e4:+5.0f}×{n}" for (y, v), (_, n) in zip(g.mean().items(), g.size().items())))
    return best


P("[1] 지정가 깊이 × 유효시간 (60분봉 라운드 이탈)")
for kx in (0.5, 1.0, 2.0, 3.0):
    for W in (1, 2):
        for H in (4, 8):
            E = sorted(sum([build_lim(s, 60, "round", H=H, kx=kx, W=W) for s in SYM], []), key=lambda z: z[0])
            rep(E, f"60분 라운드 k={kx} 유효{W}봉 보유{H}봉")

P("\n[2] 전저점 이탈 + 지정가")
for kx in (1.0, 2.0):
    for H in (4, 8):
        E = sorted(sum([build_lim(s, 60, "prevlow", LB=50, H=H, kx=kx, W=2) for s in SYM], []), key=lambda z: z[0])
        rep(E, f"60분 전저점50 k={kx} 보유{H}봉")
        E = sorted(sum([build_lim(s, 240, "prevlow", LB=50, H=H, kx=kx, W=2) for s in SYM], []), key=lambda z: z[0])
        rep(E, f"240분 전저점50 k={kx} 보유{H}봉")

P("\n[3] 저변동 국면 필터 추가 (5대 방향 탐색의 발견 적용)")
for md in ("round", "prevlow", "both"):
    for lv in (0.3, 0.5):
        E = sorted(sum([build_lim(s, 60, md, LB=50, H=8, kx=2.0, W=2, lowvol=lv) for s in SYM], []), key=lambda z: z[0])
        rep(E, f"60분 {md} k=2 보유8봉 +저변동{lv*100:.0f}%")

P("\n[4] 상위 후보 상세 + 비용 민감도")
CAND = [(60, "round", 2.0, 2, 8, None), (60, "round", 2.0, 2, 8, 0.5),
        (60, "both", 2.0, 2, 8, 0.5), (60, "prevlow", 2.0, 2, 8, 0.5)]
res = []
for B, md, kx, W, H, lv in CAND:
    E = sorted(sum([build_lim(s, B, md, LB=50, H=H, kx=kx, W=W, lowvol=lv) for s in SYM], []), key=lambda z: z[0])
    b = rep(E, f"{B}분 {md} k={kx} W={W} H={H} lv={lv}", full=True)
    if b: res.append((b[0], B, md, kx, W, H, lv))
if res:
    res.sort(reverse=True)
    _, B, md, kx, W, H, lv = res[0]
    P(f"\n  ── 비용 민감도: {B}분 {md} k={kx} lv={lv} ──")
    for fm, ft, eps, lab in ((2, 5, 0.0002, "기준"), (3, 6, 0.0005, "보수"), (4, 7, 0.001, "비관"), (5, 8, 0.002, "최악")):
        E = sorted(sum([build_lim(s, B, md, LB=50, H=H, kx=kx, W=W, lowvol=lv, fm=fm, ft=ft, eps=eps) for s in SYM], []), key=lambda z: z[0])
        rep(E, f"    {lab} 메{fm}/테{ft} 관통{eps*1e4:.0f}bp")
