"""z2 격리마진 최대 배율 — 파산 없이 명목을 얼마나 키울 수 있나.

★ 이전 시뮬의 오류를 고친다: 배율 L의 청산거리 = 1/L - mmr.
   이것이 손절폭보다 가까우면 실제 하단 배리어는 청산거리다.
   즉 고배율에서는 '손절에 닿았는지'가 아니라 '청산거리에 닿았는지'로 판정해야 한다.

격리마진: 자본의 frac만 증거금으로 넣는다. 청산되면 그 frac만 잃고 나머지는 보존.
명목 = frac × L. 같은 명목이라도 frac·L 배분에 따라 손실 상한이 달라진다."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R/"research"))
import z2_lib as L
from z2_final import build_sigs, exp_z


def P(*a, end="\n"):
    print(" ".join(str(x) for x in a), end=end, flush=True)


S = "BTCUSDT"; Y = L.YRS; MMR = L.MMR
d = L.D[S]; idx = d["idx"]; rv = d["rv"]
STEP = L.STEP; MAXH = L.MAXH
warm = 200000; _st = 5000
med = np.full(len(rv), np.nan)
for i in range(warm, len(rv), _st):
    x = rv[:i]; x = x[np.isfinite(x)]
    if len(x) >= 1000: med[i:i+_st] = np.median(x)

SG = build_sigs(S)
names = list(SG.keys())
Z = {n: exp_z(np.asarray(SG[n], float)) for n in names}
mlen = min(min(len(Z[n]) for n in names), len(d["c"])//STEP)
PICK = [n for n in names if n.endswith("4320")]
SIG = np.nanmean(np.vstack([Z[n][:mlen] for n in PICK]), axis=0)
SQ = np.full(len(SIG), np.nan)
for i in range(warm//STEP, len(SIG), _st//STEP):
    x = SIG[:i]; x = x[np.isfinite(x)]
    if len(x) < 1000: continue
    SQ[i:i+_st//STEP] = np.searchsorted(np.sort(x), SIG[i:i+_st//STEP])/len(x)
P(f"신호: 5계열 × 창4320 균등 ({', '.join(PICK)})\n")


@numba.njit(cache=True)
def tr_at(o, h, l, c, starts, up, dn, maxh, cool):
    """up/dn 비대칭 배리어. dn에는 '실효 하단'(손절 또는 청산거리)을 넣는다."""
    n = len(c); K = len(starts)
    ei = np.zeros(K, np.int64); ex = np.zeros(K, np.int64); rr = np.zeros(K)
    hit = np.zeros(K, np.int8)     # 1=익절, -1=하단, 0=시간만료
    k = 0; free = 0
    for si in range(K):
        t = starts[si]
        if t < free or t >= n-maxh-2: continue
        f = t+1; px = o[f]; U = px*(1.0+up); D = px*(1.0-dn)
        e = -1; res = 0.0; hh = 0; end = min(f+maxh, n-1)
        for j in range(f, end+1):
            if l[j] <= D: e = j; res = -dn; hh = -1; break
            if h[j] >= U: e = j; res = up; hh = 1; break
        if e < 0: e = end; res = c[e]/px-1.0; hh = 0
        ei[k] = f; ex[k] = e; rr[k] = res; hit[k] = hh; k += 1
        free = e+1+cool
    return ei[:k], ex[:k], rr[:k], hit[:k]


BAR = 0.015; Q = 0.16; COOL = 60
fin = SIG[np.isfinite(SIG)]
TH = np.quantile(fin, 1-Q)
ST = np.where(np.isfinite(SIG) & (SIG >= TH))[0]*STEP
ST = ST[ST >= warm].astype(np.int64)

CACHE = {}


def trades_for(lev):
    """배율 lev의 실효 하단으로 거래를 다시 생성."""
    liq = 1.0/lev - MMR
    dn = min(BAR, liq)
    key = round(dn, 6)
    if key not in CACHE:
        CACHE[key] = tr_at(d["o"], d["h"], d["l"], d["c"], ST, BAR, dn, MAXH, COOL)
    return CACHE[key], dn, liq


def sim(lev, frac, fee=10.0, dd_k=0.0, yr=None, ruin=0.01):
    (ei, ex, rr, hit), dn, liq = trades_for(lev)
    cap = 1e7; peak = cap; cur = []; nliq = 0
    for a, b, res, h in zip(ei, ex, rr, hit):
        if yr is not None and idx[a].year != yr: continue
        mu = 1.0
        r_ = rv[a-1]; mm = med[a]
        if np.isfinite(r_) and np.isfinite(mm) and r_ > 0: mu *= float(np.clip(mm/r_, 0.1, 4.0))
        s_ = SQ[(a-1)//STEP]
        if np.isfinite(s_): mu *= (2.0 if s_ >= 0.97 else (1.0 if s_ >= 0.90 else 0.5))
        if dd_k > 0: mu *= max(0.2, 1.0-dd_k*(1.0-cap/peak))
        lv = lev*mu
        if lv <= 0: cur.append(cap); continue
        # 이 거래의 실효 하단(사이징으로 배율이 변하므로 재계산)
        liq_i = 1.0/lv - MMR
        r = res
        if h == -1 and liq_i <= dn + 1e-12:
            r = -liq_i; nliq += 1      # 청산: 증거금 전액
        margin = cap*frac
        pnl = margin*lv*(r-fee/1e4)
        if pnl < -margin: pnl = -margin     # 격리: 손실은 증거금까지
        cap += pnl
        if cap > peak: peak = cap
        cur.append(cap)
    if not cur: return None
    eq = np.array(cur)
    fin_ = eq[-1]/1e7
    pk = np.maximum.accumulate(np.concatenate([[1e7], eq]))[1:]
    mdd = (eq/pk-1).min()*100
    yy = Y if yr is None else 1.0
    return dict(ann=(max(fin_, 1e-12)**(1/yy)-1)*100, mdd=mdd, fin=fin_,
                nliq=nliq, ruin=fin_ < ruin, n=len(cur), dn=dn,
                wr=np.mean(np.array([x for x in rr]) > 0)*100)


P("="*128); P("[1] 실효 하단 배리어 — 배율이 올라가면 손절이 강제로 좁아진다"); P("="*128)
P(f"  {'배율':>6s} {'청산거리':>9s} {'실효하단':>9s} {'승률%':>7s} {'거래당bp':>9s} {'연거래':>6s} {'손익분기%':>9s} {'여유%p':>7s}")
for lev in (1, 3, 5, 8, 12, 20, 30, 50, 52, 60, 80, 100, 125, 150, 200):
    (ei, ex, rr, hit), dn, liq = trades_for(lev)
    wr = np.mean(rr > 0)*100
    # 손익분기: 이기면 +BAR, 지면 -dn → p*·BAR = (1-p*)·dn + 비용
    be = (dn+10.0/1e4)/(BAR+dn)*100
    P(f"  {lev:5d}x {liq*100:8.2f}% {dn*100:8.2f}% {wr:7.2f} {rr.mean()*1e4:+9.1f} "
      f"{len(ei)/Y:6.0f} {be:9.2f} {wr-be:+7.2f}")

P("\n"+"="*128); P("[2] 배율 × 자본비율 — 파산(자본 1% 미만) 여부와 연수익"); P("="*128)
FR = (1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02)
P(f"  {'배율':>6s} " + " ".join(f"{'자본'+str(int(f*100))+'%':>13s}" for f in FR))
GRID = []
for lev in (3, 5, 8, 12, 20, 30, 50, 75, 100, 125, 150, 200):
    row = []
    for fr in FR:
        v = sim(lev, fr)
        if v is None: row.append("      -      "); continue
        tag = "파산" if v["ruin"] else f"{v['ann']:+7.1f}%"
        row.append(f"{tag:>7s}/{v['fin']:5.1f}x")
        if not v["ruin"]: GRID.append((v["ann"], lev, fr, v))
    P(f"  {lev:5d}x " + " ".join(f"{x:>13s}" for x in row))
P("  (셀 = 연수익% / 최종배수)")

P("\n"+"="*128); P("[3] 명목(배율×자본비율)으로 정리 — 같은 명목이면 어떤 배분이 나은가"); P("="*128)
P(f"  {'명목':>6s} " + " ".join(f"{'L'+str(l)+'x':>16s}" for l in (3, 5, 10, 20, 50, 100)))
for nom in (1, 2, 3, 5, 8, 12, 20):
    row = []
    for lv in (3, 5, 10, 20, 50, 100):
        fr = nom/lv
        if fr > 1.0: row.append("        -       "); continue
        v = sim(lv, fr)
        if v is None: row.append("        -       "); continue
        row.append("           파산 " if v["ruin"] else f"{v['ann']:+7.1f}%/{v['mdd']:6.1f}%")
    P(f"  {nom:5.0f}x " + " ".join(f"{x:>16s}" for x in row))

P("\n"+"="*128); P("[4] 낙폭 비례 축소를 더하면 — 배율 상한이 얼마나 올라가나"); P("="*128)
P(f"  {'배율':>6s} {'자본비':>6s} {'조절없음':>16s} {'k=1.0':>16s} {'k=1.5':>16s} {'k=2.5':>16s}")
BEST = []
for lev, fr in ((8, 1.0), (12, 1.0), (20, 0.5), (30, 0.3), (50, 0.2), (100, 0.1), (150, 0.05), (200, 0.05)):
    row = []
    for k in (0.0, 1.0, 1.5, 2.5):
        v = sim(lev, fr, dd_k=k)
        if v is None or v["ruin"]: row.append("           파산 "); continue
        row.append(f"{v['ann']:+7.1f}%/{v['mdd']:6.1f}%")
        BEST.append((v["ann"], lev, fr, k, v))
    P(f"  {lev:5d}x {fr*100:5.0f}% " + " ".join(f"{x:>16s}" for x in row))

P("\n"+"="*128); P("[5] 파산 없는 최고 수익 구성 상위 12"); P("="*128)
ALL = []
for lev in (3, 5, 8, 12, 16, 20, 25, 30, 40, 50, 60, 75, 100, 125, 150, 200):
    for fr in (1.0, 0.7, 0.5, 0.3, 0.2, 0.15, 0.1, 0.07, 0.05, 0.03, 0.02):
        for k in (0.0, 1.0, 1.5, 2.5):
            v = sim(lev, fr, dd_k=k)
            if v and not v["ruin"] and v["mdd"] > -99.5:
                ALL.append((v["ann"], lev, fr, k, v))
ALL.sort(reverse=True)
P(f"  {'연수익%':>9s} {'MDD%':>8s} {'배율':>6s} {'자본비':>6s} {'명목':>6s} {'낙폭k':>6s} {'청산수':>6s} {'최종배수':>11s}")
for a, lev, fr, k, v in ALL[:12]:
    P(f"  {a:9.1f} {v['mdd']:8.1f} {lev:5d}x {fr*100:5.0f}% {lev*fr:5.1f}x {k:6.1f} {v['nliq']:6d} {v['fin']:11.1f}")

P("\n"+"="*128); P("[6] 최고 구성의 연도별 + 3단 레버리지 재정리"); P("="*128)
if ALL:
    _, lev, fr, k, _ = ALL[0]
    P(f"  최고: {lev}배 × 자본 {fr*100:.0f}% (명목 {lev*fr:.1f}배), 낙폭조절 k={k}")
    out = []
    for yr in range(2020, 2027):
        v = sim(lev, fr, dd_k=k, yr=yr)
        out.append(f"{yr} {(v['fin']-1)*100:+8.0f}%" if v else f"{yr}    n/a")
    P("  연도별: " + " | ".join(out))
    P("")
P(f"  {'구간':>16s} {'구성':>22s} {'연수익%':>10s} {'MDD%':>8s} {'최종배수':>11s} {'청산':>6s}")
TIERS = [("무레버리지", 1, 1.0, 0.0), ("중간", 3, 1.0, 0.0), ("중간+", 5, 1.0, 0.0),
         ("공격", 12, 1.0, 1.5), ("최대공격", 50, 0.2, 1.5), ("극단(격리)", 100, 0.1, 1.5)]
if ALL:
    TIERS.append(("탐색 최고", ALL[0][1], ALL[0][2], ALL[0][3]))
for lb, lev, fr, k in TIERS:
    v = sim(lev, fr, dd_k=k)
    cfg = f"{lev}x × 자본{fr*100:.0f}%" + (f" 조절{k}" if k else "")
    if v is None or v["ruin"]: P(f"  {lb:>16s} {cfg:>22s}        파산"); continue
    P(f"  {lb:>16s} {cfg:>22s} {v['ann']:10.1f} {v['mdd']:8.1f} {v['fin']:11.1f} {v['nliq']:6d}")
