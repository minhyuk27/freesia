"""z2 규칙 기반 신호 구성 — 성과를 전혀 보지 않고 정하는 구성 규칙들.
선택편향 없이 전 지표 균등(3배 +36.5%)보다 나을 수 있는가.
각 규칙은 사전에 말로 정의 가능해야 한다(사후 최적화 금지)."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R/"research"))
import z2_lib as L
from z2_final import build_sigs, exp_z


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


SG = build_sigs(S)
names = list(SG.keys())
Z = {n: exp_z(np.asarray(SG[n], float)) for n in names}
mlen = min(min(len(Z[n]) for n in names), len(d["c"])//STEP)
FAM = {}
for n in names:
    base = "".join(ch for ch in n if not ch.isdigit())
    w = int("".join(ch for ch in n if ch.isdigit()))
    FAM.setdefault(base, {})[w] = n
P(f"지표 {len(names)}개 / 계열 {len(FAM)}개: " + ", ".join(f"{k}({len(v)})" for k, v in FAM.items()) + "\n")


def bt(sigv, q, bar=0.015, cool=60, levs=(1, 3, 5, 8), fee=10.0, yr=None):
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
            out[f"a{lev}"] = None; continue
        eq = np.array(cur); yy = Y if yr is None else 1.0
        pk = np.maximum.accumulate(np.concatenate([[1e7], eq]))[1:]
        out[f"a{lev}"] = ((eq[-1]/1e7)**(1/yy)-1)*100
        out[f"m{lev}"] = (eq/pk-1).min()*100
        out[f"f{lev}"] = eq[-1]/1e7
    return out


def comb(ns, w=None):
    M = np.vstack([Z[n][:mlen] for n in ns])
    if w is None: return np.nanmean(M, axis=0)
    w = np.array(w, float)[:, None]
    return np.nansum(M*w, axis=0)/np.nansum(np.where(np.isfinite(M), w, 0), axis=0)


WINS = sorted({int("".join(c for c in n if c.isdigit())) for n in names})
RULES = []
RULES.append(("전 지표 균등(기준)", names))
for w in WINS:
    ns = [FAM[f][w] for f in FAM if w in FAM[f]]
    if len(ns) >= 4: RULES.append((f"모든 계열 × 창{w}만", ns))
mid = WINS[len(WINS)//2]
RULES.append((f"계열별 중간창({mid})", [FAM[f][mid] for f in FAM if mid in FAM[f]]))
for f in FAM:
    pass
RULES.append(("계열별 전 창 평균 후 균등", None))  # 특수 처리
RULES.append(("창 1440~8640만(중기)", [n for n in names
                                   if 1440 <= int("".join(c for c in n if c.isdigit())) <= 8640]))
RULES.append(("창 2880 이상(장기)", [n for n in names
                                if int("".join(c for c in n if c.isdigit())) >= 2880]))

P("="*126); P("[1] 규칙 기반 구성 (상위15%·20%, ±1.5%, 쿨60) — 셀 = 연수익%(1x/3x/5x)"); P("="*126)
P(f"  {'규칙':>24s} {'개수':>4s} {'분위':>5s} {'연거래':>6s} {'승률%':>7s} {'1x':>8s} {'3x':>8s} {'5x':>8s} {'8x':>8s}")
BEST = []
for lab, ns in RULES:
    if ns is None:
        fam_sigs = [np.nanmean(np.vstack([Z[n][:mlen] for n in FAM[f].values()]), axis=0) for f in FAM]
        sv_base = np.nanmean(np.vstack(fam_sigs), axis=0)
    else:
        sv_base = comb(ns)
    cnt = len(ns) if ns else len(FAM)
    for q in (0.15, 0.20):
        v = bt(sv_base, q)
        if not v: continue
        f_ = lambda k: f"{v[k]:+8.1f}" if v.get(k) is not None else "    파산"
        P(f"  {lab:>24s} {cnt:4d} {q*100:4.0f}% {v['n']/Y:6.0f} {v['wr']:7.2f} "
          f"{f_('a1')} {f_('a3')} {f_('a5')} {f_('a8')}")
        if v.get("a3") is not None and v['n']/Y >= 180:
            BEST.append((v["a3"], lab, q, ns, cnt, v))

P("\n"+"="*126); P("[2] 계열 가중 — 창 길이에 비례/반비례 (성과 무관 규칙)"); P("="*126)
P(f"  {'가중':>24s} {'분위':>5s} {'연거래':>6s} {'승률%':>7s} {'1x':>8s} {'3x':>8s} {'5x':>8s}")
for lab, wf in (("균등", lambda w: 1.0), ("창 비례", lambda w: w), ("창 반비례", lambda w: 1.0/w),
                ("log창 비례", lambda w: np.log(w)), ("sqrt창 반비례", lambda w: 1.0/np.sqrt(w))):
    ws = [wf(int("".join(c for c in n if c.isdigit()))) for n in names]
    sv = comb(names, ws)
    for q in (0.15, 0.20):
        v = bt(sv, q, levs=(1, 3, 5))
        if not v: continue
        f_ = lambda k: f"{v[k]:+8.1f}" if v.get(k) is not None else "    파산"
        P(f"  {lab:>24s} {q*100:4.0f}% {v['n']/Y:6.0f} {v['wr']:7.2f} {f_('a1')} {f_('a3')} {f_('a5')}")
        if v.get("a3") is not None and v['n']/Y >= 180: BEST.append((v["a3"], "가중:"+lab, q, names, len(names), v))

P("\n"+"="*126); P("[3] 최고 규칙 구성의 3단 레버리지 + 연도별"); P("="*126)
BEST.sort(reverse=True)
for a3, lab, q, ns, cnt, v in BEST[:2]:
    P(f"\n■ {lab} / 상위{q*100:.0f}% — 연 {v['n']/Y:.0f}건(이틀당 {v['n']/Y/182.5:.2f}), 승률 {v['wr']:.2f}%")
    if lab.startswith("가중:"):
        wf = {"가중:균등": lambda w: 1.0, "가중:창 비례": lambda w: w, "가중:창 반비례": lambda w: 1.0/w,
              "가중:log창 비례": lambda w: np.log(w), "가중:sqrt창 반비례": lambda w: 1.0/np.sqrt(w)}[lab]
        sv = comb(names, [wf(int("".join(c for c in n if c.isdigit()))) for n in names])
    else:
        sv = comb(ns)
    vv = bt(sv, q, levs=(1, 2, 3, 5, 8, 12))
    P(f"  {'구간':>12s} {'배수':>6s} {'연수익%':>10s} {'MDD%':>8s} {'최종배수':>11s}")
    for lb, lev in (("무레버리지", 1), ("중간", 3), ("중간+", 5), ("공격", 8), ("최대공격", 12)):
        if vv.get(f"a{lev}") is None: P(f"  {lb:>12s} {lev:5d}x        파산"); continue
        P(f"  {lb:>12s} {lev:5d}x {vv[f'a{lev}']:10.1f} {vv[f'm{lev}']:8.1f} {vv[f'f{lev}']:11.2f}")
    for lv in (1, 3, 5):
        out = []
        for yr in range(2020, 2027):
            vy = bt(sv, q, levs=(lv,), yr=yr)
            out.append(f"{yr} {(vy[f'f{lv}']-1)*100:+7.0f}%" if vy and vy.get(f"f{lv}") else f"{yr}   파산")
        P(f"  연도별({lv}x): " + " | ".join(out))
    P("  비용: " + " | ".join(
        f"{fee}bp 3x " + (f"{bt(sv,q,fee=fee)['a3']:+.0f}%" if bt(sv, q, fee=fee).get("a3") is not None else "파산")
        for fee in (4, 10, 15, 20)))
