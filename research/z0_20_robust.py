"""z0.20 신호 강건성 — 선택을 전혀 안 한 신호가 고른 신호만큼 좋은가.
'학습 상위 5개(54.45%) < 전체 평균(55.83%)' 발견에 대한 후속.
지표 선택이 과적합이라면, 전 지표 균등 결합이 현행 3개 조합만큼 나와야 한다."""
import sys, warnings, itertools
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R/"research"))
import z0_20_lib as L
from z0_20_final import build_sigs, exp_z


def P(*a, end="\n"):
    print(" ".join(str(x) for x in a), end=end, flush=True)


S = "BTCUSDT"; Y = L.YRS
d = L.D[S]; idx = d["idx"]; rv = d["rv"]
STEP = L.STEP; MAXH = L.MAXH
warm = 200000; _st = 5000
med = np.full(len(rv), np.nan)
for i in range(warm, len(rv), _st):
    x = rv[:i]; x = x[np.isfinite(x)]
    if len(x) >= 1000: med[i:i+_st] = np.median(x)


@numba.njit(cache=True)
def tr_at(o, h, l, c, starts, up, dn, maxh, cool):
    n = len(c); K = len(starts)
    ei = np.zeros(K, np.int64); ex = np.zeros(K, np.int64); rr = np.zeros(K)
    k = 0; free = 0
    for si in range(K):
        t = starts[si]
        if t < free or t >= n-maxh-2: continue
        f = t+1; px = o[f]; U = px*(1.0+up); D = px*(1.0-dn)
        e = -1; res = 0.0; end = min(f+maxh, n-1)
        for j in range(f, end+1):
            if l[j] <= D: e = j; res = -dn; break
            if h[j] >= U: e = j; res = up; break
        if e < 0: e = end; res = c[e]/px-1.0
        ei[k] = f; ex[k] = e; rr[k] = res; k += 1
        free = e+1+cool
    return ei[:k], ex[:k], rr[:k]


P("지표 계산 중...")
SG = build_sigs(S)
names = list(SG.keys())
Z = {n: exp_z(np.asarray(SG[n], float)) for n in names}
mlen = min(min(len(Z[n]) for n in names), len(d["c"])//STEP)
P(f"  {len(names)}개 지표, 길이 {mlen:,}\n")


def bt(sigv, q, bar=0.015, cool=60, levs=(1, 3, 5), fee=10.0, yr=None):
    sq = np.full(len(sigv), np.nan)
    _sw = warm//STEP; _ss = _st//STEP
    for i in range(_sw, len(sigv), _ss):
        x = sigv[:i]; x = x[np.isfinite(x)]
        if len(x) < 1000: continue
        sq[i:i+_ss] = np.searchsorted(np.sort(x), sigv[i:i+_ss])/len(x)
    fin = sigv[np.isfinite(sigv)]
    if len(fin) < 1000: return None
    th = np.quantile(fin, 1-q)
    st = np.where(np.isfinite(sigv) & (sigv >= th))[0]*STEP
    st = st[st >= warm].astype(np.int64)
    ei, ex, rr = tr_at(d["o"], d["h"], d["l"], d["c"], st, bar, bar, MAXH, cool)
    if len(ei) < 30: return None
    out = {"n": len(ei), "wr": np.mean(rr > 0)*100, "bp": rr.mean()*1e4}
    for lev in levs:
        cap = 1e7; cur = []
        for a, b, res in zip(ei, ex, rr):
            if yr is not None and idx[a].year != yr: continue
            mu = 1.0
            r_ = rv[a-1]; mm = med[a]
            if np.isfinite(r_) and np.isfinite(mm) and r_ > 0: mu *= float(np.clip(mm/r_, 0.1, 4.0))
            s_ = sq[(a-1)//STEP]
            if np.isfinite(s_): mu *= (2.0 if s_ >= 0.97 else (1.0 if s_ >= 0.90 else 0.5))
            lv = lev*mu
            if lv <= 0: continue
            liq = 1.0/lv-L.MMR; r = res
            if r <= -bar and liq < bar: r = -1.0/lv
            pnl = cap*lv*(r-fee/1e4)
            if pnl < -cap: pnl = -cap
            cap += pnl; cur.append(cap)
            if cap <= 1e5: break
        if not cur or cur[-1] <= 2e5:
            out[f"a{lev}"] = None; out[f"m{lev}"] = None; continue
        eq = np.array(cur); yy = Y if yr is None else 1.0
        pk = np.maximum.accumulate(np.concatenate([[1e7], eq]))[1:]
        out[f"a{lev}"] = ((eq[-1]/1e7)**(1/yy)-1)*100
        out[f"m{lev}"] = (eq/pk-1).min()*100
        out[f"f{lev}"] = eq[-1]/1e7
    return out


def comb(ns):
    return np.nanmean(np.vstack([Z[n][:mlen] for n in ns]), axis=0)


CUR = ["고저비대칭4320", "절대값8640", "OFI1440"]
ALL = names
FAM = {}
for n in names:
    base = "".join(ch for ch in n if not ch.isdigit())
    FAM.setdefault(base, []).append(n)

P("="*124); P("[1] 선택 없는 신호 vs 고른 신호 (상위20%, ±1.5%, 쿨60)"); P("="*124)
P(f"  {'신호':>26s} {'개수':>4s} {'연거래':>6s} {'승률%':>7s} {'거래당bp':>9s} {'1x%':>8s} {'3x%':>8s} {'5x%':>8s}")
CASES = [("현행 3개(고른 것)", CUR), ("전 지표 균등(선택 없음)", ALL)]
for fam, ns in sorted(FAM.items()):
    if len(ns) >= 3: CASES.append((f"{fam} 계열 전부", ns))
for lab, ns in CASES:
    v = bt(comb(ns), 0.20)
    if not v: continue
    f_ = lambda k: f"{v[k]:+8.1f}" if v.get(k) is not None else "    파산"
    P(f"  {lab:>26s} {len(ns):4d} {v['n']/Y:6.0f} {v['wr']:7.2f} {v['bp']:+9.1f} "
      f"{f_('a1')} {f_('a3')} {f_('a5')}")

P("\n"+"="*124); P("[2] 무작위 3개 조합 100개 — 현행 3개가 특별한가"); P("="*124)
rng = np.random.default_rng(11)
vals = []
for it in range(100):
    ns = list(rng.choice(names, 3, replace=False))
    v = bt(comb(ns), 0.20)
    if v and v.get("a3") is not None: vals.append((v["a3"], v["wr"], ns))
vals.sort(reverse=True)
arr = np.array([x[0] for x in vals])
cur = bt(comb(CUR), 0.20)
P(f"  현행 3개: 3배 연 {cur['a3']:+.1f}%  승률 {cur['wr']:.2f}%")
P(f"  무작위 3개 100회: 평균 {arr.mean():+.1f}%  중앙 {np.median(arr):+.1f}%  "
  f"최대 {arr.max():+.1f}%  최소 {arr.min():+.1f}%")
P(f"  → 현행의 백분위 {np.mean(arr < cur['a3'])*100:.0f}%  (100%면 특별, 50%면 평범)")
P(f"  무작위 조합 중 양수 비율 {np.mean(arr > 0)*100:.0f}%")
P("\n  무작위 상위 5:")
for a, wr, ns in vals[:5]:
    P(f"    {a:+8.1f}%  승률 {wr:5.2f}%  {', '.join(ns)}")
P("  무작위 하위 3:")
for a, wr, ns in vals[-3:]:
    P(f"    {a:+8.1f}%  승률 {wr:5.2f}%  {', '.join(ns)}")

P("\n"+"="*124); P("[3] 전 지표 균등 신호의 3단 레버리지 + 연도별 (선택편향 0)"); P("="*124)
SA = comb(ALL)
for q in (0.15, 0.20, 0.25, 0.30):
    v = bt(SA, q, levs=(1, 3, 5, 8))
    if not v: continue
    f_ = lambda k: f"{v[k]:+8.1f}" if v.get(k) is not None else "    파산"
    P(f"  상위{q*100:2.0f}%: 연거래 {v['n']/Y:4.0f} 승률 {v['wr']:5.2f}% | "
      f"1x {f_('a1')} 3x {f_('a3')} 5x {f_('a5')} 8x {f_('a8')}")
P("")
best_q = 0.20
v = bt(SA, best_q, levs=(1, 3, 5, 8, 12))
P(f"  ── 전 지표 균등 상위{best_q*100:.0f}% ──")
P(f"  {'구간':>12s} {'배수':>6s} {'연수익%':>10s} {'MDD%':>8s} {'최종배수':>11s}")
for lb, lev in (("무레버리지", 1), ("중간", 3), ("중간+", 5), ("공격", 8), ("최대공격", 12)):
    if v.get(f"a{lev}") is None: P(f"  {lb:>12s} {lev:5d}x        파산"); continue
    P(f"  {lb:>12s} {lev:5d}x {v[f'a{lev}']:10.1f} {v[f'm{lev}']:8.1f} {v[f'f{lev}']:11.2f}")
for lv in (1, 3, 5):
    out = []
    for yr in range(2020, 2027):
        vy = bt(SA, best_q, levs=(lv,), yr=yr)
        out.append(f"{yr} {(vy[f'f{lv}']-1)*100:+7.0f}%" if vy and vy.get(f"f{lv}") else f"{yr}   파산")
    P(f"  연도별({lv}x): " + " | ".join(out))

P("\n"+"="*124); P("[4] 비용 민감도 — 전 지표 균등 (상위20%)"); P("="*124)
for fee in (4, 7, 10, 15, 20, 30):
    v = bt(SA, 0.20, fee=fee)
    f_ = lambda k: f"{v[k]:+8.1f}%" if v.get(k) is not None else "     파산"
    P(f"  왕복 {fee:2d}bp: 거래당 {v['bp']:+6.1f}bp  1x {f_('a1')}  3x {f_('a3')}  5x {f_('a5')}")
