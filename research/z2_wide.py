"""z2 넓은 배리어 탐색 + 최근 약화 진단.
고원 표에서 배리어 2.5%가 1.5%보다 나았다. 빈도 요건(연 180건)과 함께 확인한다."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R/"research"))
import z2_lib as L
import numba

S = "BTCUSDT"; Y = L.YRS
d = L.D[S]; idx = d["idx"]; rv = d["rv"]; sig = d["sig"]
warm = 200000; _st = 5000
med = np.full(len(rv), np.nan)
for i in range(warm, len(rv), _st):
    x = rv[:i]; x = x[np.isfinite(x)]
    if len(x) >= 1000: med[i:i+_st] = np.median(x)
sq = np.full(len(sig), np.nan); _sw = warm//L.STEP; _ss = _st//L.STEP
for i in range(_sw, len(sig), _ss):
    x = sig[:i]; x = x[np.isfinite(x)]
    if len(x) < 1000: continue
    sq[i:i+_ss] = np.searchsorted(np.sort(x), sig[i:i+_ss])/len(x)


@numba.njit(cache=True)
def tr_at(o, h, l, c, starts, up, dn, maxh, cool):
    n = len(c); K = len(starts)
    ei = np.zeros(K, np.int64); ex = np.zeros(K, np.int64); rr = np.zeros(K)
    k = 0; free = 0
    for si in range(K):
        t = starts[si]
        if t < free or t >= n-maxh-2: continue
        f = t+1; px = o[f]; U = px*(1.0+up); Dn = px*(1.0-dn)
        e = -1; res = 0.0; end = min(f+maxh, n-1)
        for j in range(f, end+1):
            if l[j] <= Dn: e = j; res = -dn; break
            if h[j] >= U: e = j; res = up; break
        if e < 0: e = end; res = c[e]/px-1.0
        ei[k] = f; ex[k] = e; rr[k] = res; k += 1
        free = e+1+cool
    return ei[:k], ex[:k], rr[:k]


def sig_starts(q):
    th = np.quantile(d["fin"], 1-q)
    ok = np.where(np.isfinite(sig) & (sig >= th))[0]*L.STEP
    return ok[ok >= warm].astype(np.int64)


def evaluate(ei, ex, rr, lev, use_size=True, dn=0.015, yr=None):
    cap = 1e7; cur = []
    for a, b, res in zip(ei, ex, rr):
        mu = 1.0
        if use_size:
            r_ = rv[a-1] if a > 0 else np.nan; mm = med[a]
            if np.isfinite(r_) and np.isfinite(mm) and r_ > 0:
                mu *= float(np.clip(mm/r_, 0.1, 4.0))
            s_ = sq[(a-1)//L.STEP] if (a-1)//L.STEP < len(sq) else np.nan
            if np.isfinite(s_):
                mu *= (2.0 if s_ >= 0.97 else (1.0 if s_ >= 0.90 else 0.5))
        lv = lev*mu
        if lv <= 0: cur.append(cap); continue
        if yr is not None and idx[a].year != yr: continue
        liq = 1.0/lv - L.MMR; r = res
        if r <= -dn and liq < dn: r = -1.0/lv
        pnl = cap*lv*(r-10.0/1e4)
        if pnl < -cap: pnl = -cap
        cap += pnl; cur.append(cap)
        if cap <= 1e5: break
    if not cur: return None
    eq = np.array(cur)
    if eq[-1] <= 2e5: return None
    yy = Y if yr is None else 1.0
    peak = np.maximum.accumulate(np.concatenate([[1e7], eq]))[1:]
    return dict(ann=((eq[-1]/1e7)**(1/yy)-1)*100, mdd=(eq/peak-1).min()*100, fin=eq[-1]/1e7)


def P(*a, end="\n"):
    print(" ".join(str(x) for x in a), end=end, flush=True)


P("="*126); P("[1] 넓은 배리어 × 분위 — 연거래 함께 (3배). 셀 = 연수익% / 연거래 / 승률%"); P("="*126)
QS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40)
P(f"  {'배리어':>7s} " + " ".join(f"{'상위'+str(int(q*100))+'%':>19s}" for q in QS))
GRID = []
for bar in (0.015, 0.02, 0.025, 0.03, 0.04, 0.05):
    row = []
    for q in QS:
        a2, b2, r2 = tr_at(d["o"], d["h"], d["l"], d["c"], sig_starts(q), bar, bar, L.MAXH, 60)
        v = evaluate(a2, b2, r2, 3, dn=bar)
        n = len(a2)/Y
        row.append(f"{v['ann']:+7.1f}/{n:4.0f}/{np.mean(r2>0)*100:4.1f}" if v else f"  파산/{n:4.0f}/{np.mean(r2>0)*100:4.1f}")
        if v and n >= 180: GRID.append((v['ann'], bar, q, n, v['mdd'], np.mean(r2>0)*100))
    P(f"  {bar*100:6.2f}% " + " ".join(f"{x:>19s}" for x in row))

P("\n  연 180건 이상 중 상위:")
for a, bar, q, n, mdd, wr in sorted(GRID, reverse=True)[:8]:
    P(f"    배리어 {bar*100:.2f}% 상위{q*100:.0f}%: 연 {a:+7.1f}%  MDD {mdd:6.1f}%  연거래 {n:.0f}  승률 {wr:.1f}%")

P("\n"+"="*126); P("[2] 넓은 배리어는 왜 나은가 — 보유시간·손익분기 정확도"); P("="*126)
P(f"  {'배리어':>7s} {'승률%':>7s} {'손익분기%':>9s} {'여유%p':>7s} {'보유중앙':>9s} {'보유평균':>9s} {'연거래':>6s} {'노출%':>7s}")
for bar in (0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05):
    a2, b2, r2 = tr_at(d["o"], d["h"], d["l"], d["c"], sig_starts(0.15), bar, bar, L.MAXH, 60)
    hold = b2-a2
    be = 50 + 10.0/(2*bar*1e4)*100
    wr = np.mean(r2 > 0)*100
    P(f"  {bar*100:6.2f}% {wr:7.2f} {be:9.2f} {wr-be:+7.2f} {np.median(hold):8.0f}분 {hold.mean():8.0f}분 "
      f"{len(a2)/Y:6.0f} {hold.sum()/(len(d['c'])-warm)*100:7.1f}")

P("\n"+"="*126); P("[3] 최근 약화 진단 — 연도별·반기별 거래당 수익"); P("="*126)
for bar, q in ((0.015, 0.15), (0.025, 0.15), (0.025, 0.20)):
    a2, b2, r2 = tr_at(d["o"], d["h"], d["l"], d["c"], sig_starts(q), bar, bar, L.MAXH, 60)
    ts = idx[a2]
    df = pd.DataFrame({"t": ts, "r": r2})
    P(f"\n  ── 배리어 {bar*100:.1f}% 상위{q*100:.0f}% (연 {len(a2)/Y:.0f}건) ──")
    P("    " + " ".join(f"{y:>12d}" for y in range(2020, 2027)))
    for lab, fn in (("거래수", lambda g: f"{len(g):d}건"),
                    ("승률", lambda g: f"{np.mean(g.r>0)*100:.1f}%"),
                    ("거래당", lambda g: f"{g.r.mean()*1e4:+.0f}bp")):
        cells = []
        for y in range(2020, 2027):
            g = df[df.t.dt.year == y]
            cells.append(fn(g) if len(g) else "-")
        P(f"    {lab:>4s}" + " ".join(f"{c:>12s}" for c in cells))

P("\n"+"="*126); P("[4] 최종 후보 3단 레버리지 (연 180건 이상 최고 구성)"); P("="*126)
if GRID:
    _, bar, q, n, _, _ = sorted(GRID, reverse=True)[0]
    a2, b2, r2 = tr_at(d["o"], d["h"], d["l"], d["c"], sig_starts(q), bar, bar, L.MAXH, 60)
    P(f"BTC 상위{q*100:.0f}% 시장가 ±{bar*100:.2f}% 쿨60 + 변동성×신호계단")
    P(f"  {len(a2)}건 = 연 {len(a2)/Y:.0f}건 = 이틀당 {len(a2)/Y/182.5:.2f}건, 승률 {np.mean(r2>0)*100:.2f}%")
    P(f"\n  {'구간':>12s} {'배수':>6s} {'연수익%':>10s} {'MDD%':>8s} {'최종배수':>11s}")
    for lb, lev in (("무레버리지", 1), ("중간", 3), ("중간+", 5), ("공격", 8), ("최대공격", 12), ("극단", 20)):
        v = evaluate(a2, b2, r2, lev, dn=bar)
        P(f"  {lb:>12s} {lev:5d}x " + (f"{v['ann']:10.1f} {v['mdd']:8.1f} {v['fin']:11.2f}" if v else "        파산"))
    for lv in (1, 3, 5):
        out = []
        for yr in range(2020, 2027):
            v = evaluate(a2, b2, r2, lv, dn=bar, yr=yr)
            out.append(f"{yr} {(v['fin']-1)*100:+7.0f}%" if v else f"{yr}   파산")
        P(f"  연도별({lv}x): " + " | ".join(out))
    hold = b2-a2
    gap = a2[1:]-b2[:-1]
    P(f"\n  보유 중앙 {np.median(hold):.0f}분 평균 {hold.mean():.0f}분 | 공백 중앙 {np.median(gap):.0f}분 "
      f"| 노출 {hold.sum()/(len(d['c'])-warm)*100:.1f}%")
    mm = pd.Series(1, index=pd.DatetimeIndex(idx[a2])).resample("ME").sum()
    P(f"  월 거래 중앙 {mm.median():.0f} 최소 {mm.min():.0f} | 거래 0건 월 {int((mm==0).sum())}/{len(mm)}")
    o = np.argsort(-r2)
    for k in (10, 30):
        keep = np.ones(len(r2), bool); keep[o[:k]] = False
        v = evaluate(a2[keep], b2[keep], r2[keep], 3, dn=bar)
        P(f"  상위 {k}건 제외 3배: " + (f"{v['ann']:+.1f}%" if v else "파산"))
