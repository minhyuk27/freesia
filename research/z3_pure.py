"""z3 — 신호 없는 순수 규칙: rv 비례 지정가 + 넓은 손절 + 긴 보유.

발단: z2.5(지정가+z2신호)의 플라시보가 백분위 0.0%였다. 즉 무작위 시점에
같은 지정가 규칙을 적용하면 더 좋다 — 지정가 대기 구조 자체가 수익원이고
z2 신호는 오히려 진입 시점을 제한할 뿐이다.

구조: 매 15분 c[t]×(1 − rv[t]×k) 에 지정가, W분 유효.
      체결되면 익절 +up / 손절 −dn, 최대 maxh 보유. 단일 포지션 + 쿨다운.
z1.4와의 차이: z1.4는 같은 봉 종가 청산(짧은 보유). z3은 배리어까지 길게 보유."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R/"research"))
import z2_lib as L


def P(*a, end="\n"):
    print(" ".join(str(x) for x in a), end=end, flush=True)


Y = L.YRS; MMR = L.MMR; STEP = L.STEP
warm = 200000
DAT = {}
for S in ("BTCUSDT", "ETHUSDT"):
    d = L.D[S]; rv = d["rv"]
    med = np.full(len(rv), np.nan)
    for i in range(warm, len(rv), 5000):
        x = rv[:i]; x = x[np.isfinite(x)]
        if len(x) >= 1000: med[i:i+5000] = np.median(x)
    DAT[S] = dict(o=d["o"], h=d["h"], l=d["l"], c=d["c"], idx=d["idx"], rv=rv, med=med)
P("데이터 준비 완료\n")


@numba.njit(cache=True)
def walk(o, h, l, c, rv, med, k, up, dn, maxh, step, cool, W, eps,
         nom, mmr, fee_in, fee_out, vol_size, start, n_end, ruin, shift):
    """vol_size=1이면 변동성목표 사이징(clip(med/rv,0.1,4)) 적용.
    shift>0이면 진입 후보 시점을 shift분 밀어 플라시보로 쓴다."""
    cap = 1.0; peak = 1.0; mdd = 0.0
    ntr = 0; nwin = 0; nskip = 0; sum_r = 0.0; sum_hold = 0.0
    t = start
    while t < n_end - maxh - W - 2:
        tt = t + shift
        if tt >= n_end - maxh - W - 2: break
        if np.isnan(rv[tt]) or rv[tt] <= 0: t += step; continue
        T = c[tt]*(1.0 - rv[tt]*k)
        f = -1
        for u in range(tt+1, min(tt+1+W, n_end)):
            if l[u] <= T*(1.0-eps): f = u; break
        if f < 0:
            nskip += 1; t += W; continue
        px = T
        m = 1.0
        if vol_size == 1:
            mm = med[tt]
            if np.isfinite(mm) and rv[tt] > 0:
                x = mm/rv[tt]
                m = 0.1 if x < 0.1 else (4.0 if x > 4.0 else x)
        lv = nom*m
        liq = 1.0/lv - mmr
        stop = dn if dn < liq else liq
        U = px*(1.0+up); D = px*(1.0-stop)
        e = -1; res = 0.0
        end = f+maxh
        if end > n_end-1: end = n_end-1
        for j in range(f, end+1):
            if l[j] <= D: e = j; res = -stop; break
            if h[j] >= U: e = j; res = up; nwin += 1; break
        if e < 0: e = end; res = c[e]/px-1.0
        pnl = cap*lv*(res - (fee_in+fee_out)/1e4)
        if pnl < -cap: pnl = -cap
        cap += pnl
        if cap > peak: peak = cap
        dd = cap/peak-1.0
        if dd < mdd: mdd = dd
        ntr += 1; sum_r += res; sum_hold += (e-f)
        t = e+1+cool
        if cap < ruin: break
    return cap, mdd, ntr, nwin, nskip, sum_r, sum_hold


def sim(S, k, up=0.015, dn=0.03, nom=1.0, maxh=8640, cool=60, W=240,
        fee_in=2.0, fee_out=5.0, vol=1, shift=0, start=None, end=None):
    D_ = DAT[S]
    s0 = warm if start is None else start
    e0 = len(D_["c"]) if end is None else end
    cap, mdd, ntr, nwin, nskip, sr, sh = walk(D_["o"], D_["h"], D_["l"], D_["c"], D_["rv"], D_["med"],
                                              k, up, dn, maxh, STEP, cool, W, 0.0002, float(nom),
                                              MMR, fee_in, fee_out, vol, s0, e0, 0.01, shift)
    if ntr < 30: return None
    yy = Y if start is None else max((e0-s0)/525600, 0.1)
    return dict(ann=(max(cap, 1e-12)**(1/yy)-1)*100, mdd=mdd*100, fin=cap, n=ntr,
                wr=nwin/ntr*100, bp=sr/ntr*1e4, hold=sh/ntr, skip=nskip, ruin=cap < 0.01)


def cell(v):
    return "      파산      " if (v is None or v["ruin"]) else f"{v['ann']:+7.1f}%/{v['wr']:5.1f}%"


P("="*128); P("[1] 깊이 k × 손절 (BTC, 익절1.5%, 명목1배, 메2/테5). 셀 = 연수익%/승률%"); P("="*128)
P(f"  {'깊이':>8s} " + " ".join(f"{'손절'+str(x*100)+'%':>16s}" for x in (0.015, 0.02, 0.03, 0.04, 0.06)))
BEST = []
for k in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0):
    row = []
    for dn in (0.015, 0.02, 0.03, 0.04, 0.06):
        v = sim("BTCUSDT", k, dn=dn)
        row.append(cell(v))
        if v and not v["ruin"]: BEST.append((v["ann"], k, 0.015, dn, v))
    P(f"  rv×{k:<5.1f} " + " ".join(f"{x:>16s}" for x in row))

P("\n"+"="*128); P("[2] 익절 × 깊이 (손절 3%)"); P("="*128)
P(f"  {'깊이':>8s} " + " ".join(f"{'익절'+str(x*100)+'%':>16s}" for x in (0.0075, 0.01, 0.015, 0.02, 0.03)))
for k in (1.0, 1.5, 2.0, 3.0, 4.0):
    row = []
    for up in (0.0075, 0.01, 0.015, 0.02, 0.03):
        v = sim("BTCUSDT", k, up=up)
        row.append(cell(v))
        if v and not v["ruin"]: BEST.append((v["ann"], k, up, 0.03, v))
    P(f"  rv×{k:<5.1f} " + " ".join(f"{x:>16s}" for x in row))

BEST.sort(reverse=True)
P("\n  상위 8개:")
P(f"  {'연수익%':>9s} {'MDD%':>8s} {'승률%':>7s} {'연거래':>6s} {'이틀당':>7s} {'거래당bp':>9s} "
  f"{'평균보유':>9s} {'깊이':>8s} {'익절':>7s} {'손절':>7s}")
for a, k, up, dn, v in BEST[:8]:
    P(f"  {a:9.1f} {v['mdd']:8.1f} {v['wr']:7.2f} {v['n']/Y:6.0f} {v['n']/Y/182.5:7.2f} "
      f"{v['bp']:+9.1f} {v['hold']:8.0f}분 rv×{k:<5.1f} {up*100:6.2f}% {dn*100:6.2f}%")

A, K, UP, DN, V = BEST[0]
P(f"\n  → 최고: rv×{K} 익절{UP*100:.2f}% 손절{DN*100:.2f}% (연 {V['n']/Y:.0f}건, "
  f"이틀당 {V['n']/Y/182.5:.2f}건, 승률 {V['wr']:.2f}%)")

P("\n"+"="*128); P("[3] 플라시보 — 진입 후보 시점을 N분 밀기 (같은 구조, 다른 시점)"); P("="*128)
real = sim("BTCUSDT", K, up=UP, dn=DN)
sh = []
for s in (7, 13, 23, 37, 53, 71, 97, 127, 181, 239, 331, 421, 547, 683, 821, 997,
          1201, 1439, 2003, 2879):
    v = sim("BTCUSDT", K, up=UP, dn=DN, shift=s)
    if v and not v["ruin"]: sh.append(v["bp"])
sh = np.array(sh)
P(f"  거래당bp 실제 {real['bp']:+7.2f} | 시점이동 {len(sh)}회 평균 {sh.mean():+6.2f} "
  f"최대 {sh.max():+6.2f} 최소 {sh.min():+6.2f}")
P(f"  → 백분위 {np.mean(sh < real['bp'])*100:.0f}%")
P("  (주의: 지정가 구조는 시점을 밀어도 같은 딥을 잡을 수 있어 이 검정은 보수적이지 않다)")

P("\n  진짜 대조군 — 지정가 없이 시장가로 같은 시점 진입:")
for k in (K,):
    v = sim("BTCUSDT", 0.0001, up=UP, dn=DN, fee_in=5.0)
    P(f"    시장가(깊이≈0): " + ("파산" if (v is None or v["ruin"])
                              else f"연 {v['ann']:+7.1f}%  승률 {v['wr']:5.2f}%  거래당 {v['bp']:+5.1f}bp  연거래 {v['n']/Y:.0f}"))
    P(f"    지정가 rv×{K}:   연 {real['ann']:+7.1f}%  승률 {real['wr']:5.2f}%  거래당 {real['bp']:+5.1f}bp  연거래 {real['n']/Y:.0f}")

P("\n"+"="*128); P("[4] 3단 레버리지 + 연도별 (BTC)"); P("="*128)
P(f"  {'구간':>12s} {'명목':>6s} {'연수익%':>10s} {'MDD%':>8s} {'최종배수':>11s}")
for lb, nom in (("무레버리지", 1.0), ("중간", 2.0), ("중간+", 3.0), ("공격", 4.0), ("최대공격", 6.0)):
    v = sim("BTCUSDT", K, up=UP, dn=DN, nom=nom)
    P(f"  {lb:>12s} {nom:5.1f}x " + ("       파산" if (v is None or v["ruin"])
                                      else f"{v['ann']:10.1f} {v['mdd']:8.1f} {v['fin']:11.1f}"))
for nom in (1.0, 2.0, 3.0):
    out = []
    for yr in range(2020, 2027):
        ii = np.where(DAT["BTCUSDT"]["idx"].year == yr)[0]
        if len(ii) < 50000: out.append(f"{yr}  n/a"); continue
        a0, b0 = max(ii[0], warm), ii[-1]
        if a0 >= b0-8640: out.append(f"{yr}  n/a"); continue
        v = sim("BTCUSDT", K, up=UP, dn=DN, nom=nom, start=a0, end=b0)
        out.append(f"{yr} {(v['fin']-1)*100:+7.0f}%" if v else f"{yr}  n/a")
    P(f"  연도별({nom:.0f}배): " + " | ".join(out))

P("\n"+"="*128); P("[5] ETH도 되나 + 병합"); P("="*128)
P(f"  {'종목':>10s} {'연거래':>6s} {'승률%':>7s} {'거래당bp':>9s} {'1배%':>9s} {'2배%':>9s} {'3배%':>9s}")
for S in ("BTCUSDT", "ETHUSDT"):
    v = sim(S, K, up=UP, dn=DN)
    if v is None: continue
    r = []
    for nom in (1, 2, 3):
        vv = sim(S, K, up=UP, dn=DN, nom=nom)
        r.append("    파산" if (vv is None or vv["ruin"]) else f"{vv['ann']:+9.1f}")
    P(f"  {S:>10s} {v['n']/Y:6.0f} {v['wr']:7.2f} {v['bp']:+9.1f} " + " ".join(r))

P("\n"+"="*128); P("[6] 비용 내성 · 사이징 기여 · 쿨다운 · 보유상한"); P("="*128)
P("  비용(진입/청산 bp):")
for fi, fo, lab in ((0, 4, "메0/테4"), (2, 5, "메2/테5"), (2, 8, "메2/테8"),
                    (5, 10, "테10/테10"), (8, 15, "최악")):
    r = []
    for nom in (1, 2, 3):
        v = sim("BTCUSDT", K, up=UP, dn=DN, nom=nom, fee_in=fi, fee_out=fo)
        r.append("    파산" if (v is None or v["ruin"]) else f"{v['ann']:+8.1f}%")
    P(f"    {lab:>10s}: 1배{r[0]} 2배{r[1]} 3배{r[2]}")
P("\n  사이징 기여:")
for vol, lab in ((1, "변동성목표 적용"), (0, "사이징 없음")):
    r = []
    for nom in (1, 2, 3):
        v = sim("BTCUSDT", K, up=UP, dn=DN, nom=nom, vol=vol)
        r.append("    파산" if (v is None or v["ruin"]) else f"{v['ann']:+8.1f}%")
    P(f"    {lab:>14s}: 1배{r[0]} 2배{r[1]} 3배{r[2]}")
P("\n  쿨다운:")
for cool in (0, 30, 60, 120, 240, 720):
    v = sim("BTCUSDT", K, up=UP, dn=DN, nom=2.0, cool=cool)
    if v is None: continue
    P(f"    {cool:4d}분: 연거래 {v['n']/Y:4.0f}  승률 {v['wr']:5.2f}%  "
      + ("파산" if v["ruin"] else f"연 {v['ann']:+7.1f}%  MDD {v['mdd']:6.1f}%"))
P("\n  보유 상한:")
for mh in (720, 1440, 2880, 4320, 8640, 20160, 43200):
    v = sim("BTCUSDT", K, up=UP, dn=DN, nom=2.0, maxh=mh)
    if v is None: continue
    P(f"    {mh:6d}분: 연거래 {v['n']/Y:4.0f}  평균보유 {v['hold']:5.0f}분  승률 {v['wr']:5.2f}%  "
      + ("파산" if v["ruin"] else f"연 {v['ann']:+7.1f}%"))
