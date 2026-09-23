"""R4-B 동시 보유 고빈도 구조 — 월 +80%를 향한 유일하게 남은 길.

기준표가 말한 것:
  월 +80% 필요조건 = 승률55%×연20,000건 / 승률60%×연1,000건 / 승률65%×연500건
  현재 실측 = 승률69.5% × 연192건 → 승률은 충분, 빈도가 100배 부족
  단일 포지션은 배리어 결판에 평균 973분이 걸려 구조적으로 막혀 있다

여기서: 동시 보유 N슬롯을 허용해 빈도를 끌어올린다.
  - 진입 스캔 간격을 1/5/15분으로 좁힌다
  - 각 슬롯에 총명목/N 을 배분(또는 슬롯당 고정 명목)
  - 위험: 동시 보유 포지션의 MAE가 합산된다 → 실제 최대 동시 역행을 함께 잰다"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R/"research"))
import z0_20_lib as L2
from z0_20_final import build_sigs, exp_z


def P(*a, end="\n"):
    print(" ".join(str(x) for x in a), end=end, flush=True)


S = "BTCUSDT"; Y = L2.YRS; MMR = L2.MMR
d = L2.D[S]; idx = d["idx"]; rv = d["rv"]
O, H, Lo, C = d["o"], d["h"], d["l"], d["c"]
warm = 200000
SGSTEP = L2.STEP
med = np.full(len(rv), np.nan)
for i in range(warm, len(rv), 5000):
    x = rv[:i]; x = x[np.isfinite(x)]
    if len(x) >= 1000: med[i:i+5000] = np.median(x)
SG = build_sigs(S)
names = list(SG.keys())
Z = {n: exp_z(np.asarray(SG[n], float)) for n in names}
mlen = min(min(len(Z[n]) for n in names), len(C)//SGSTEP)
SIG = np.nanmean(np.vstack([Z[n][:mlen] for n in names if n.endswith("4320")]), axis=0)
SQ = np.full(len(SIG), np.nan)
for i in range(warm//SGSTEP, len(SIG), 5000//SGSTEP):
    x = SIG[:i]; x = x[np.isfinite(x)]
    if len(x) < 1000: continue
    SQ[i:i+5000//SGSTEP] = np.searchsorted(np.sort(x), SIG[i:i+5000//SGSTEP])/len(x)
NEND = len(C)
P(f"데이터 {NEND:,}분, 신호 {len(SIG):,}봉\n")


@numba.njit(cache=True)
def run(o, h, l, c, rv, med, sq, scan, nslot, th, up, dn, maxh, tot_nom, mmr,
        fee_bp, use_size, s_lo, s_mid, s_hi, start, n_end, min_gap):
    """동시 보유 nslot개. 각 슬롯 명목 = tot_nom/nslot × 사이징승수.
    실현 손익은 청산 시점에 자본에 반영. 미실현 역행은 별도 추적."""
    cap = 1.0; peak = 1.0; mdd = 0.0
    ent_i = np.full(nslot, -1, np.int64)
    ent_p = np.zeros(nslot); ent_sz = np.zeros(nslot)
    ntr = 0; nwin = 0; sum_r = 0.0; sum_h = 0.0
    nopen = 0; max_open = 0; nmargin = 0
    worst_unreal = 0.0
    last_entry = -10**9
    sum_open = 0.0; nmin = 0
    for t in range(start, n_end-1):
        # --- 청산 점검 ---
        for s in range(nslot):
            if ent_i[s] < 0: continue
            px = ent_p[s]
            U = px*(1.0+up); D = px*(1.0-dn)
            hit = 0; res = 0.0
            if l[t] <= D: hit = 1; res = -dn
            elif h[t] >= U: hit = 1; res = up; nwin += 1
            elif (t - ent_i[s]) >= maxh: hit = 1; res = c[t]/px-1.0
            if hit == 1:
                pnl = cap*ent_sz[s]*(res - fee_bp/1e4)
                if pnl < -cap: pnl = -cap
                cap += pnl
                if cap > peak: peak = cap
                dd = cap/peak-1.0
                if dd < mdd: mdd = dd
                ntr += 1; sum_r += res; sum_h += (t-ent_i[s])
                ent_i[s] = -1; nopen -= 1
                if cap < 0.01: return cap, mdd, ntr, nwin, sum_r, sum_h, max_open, worst_unreal, sum_open, nmin, nmargin
        # --- 증거금 점검: 평가자산이 유지증거금 아래면 계좌 전체 청산 ---
        if nopen > 0:
            un = 0.0; nomsum = 0.0
            for s in range(nslot):
                if ent_i[s] < 0: continue
                un += ent_sz[s]*(l[t]/ent_p[s]-1.0)
                nomsum += ent_sz[s]
            if un < worst_unreal: worst_unreal = un
            equity = cap*(1.0 + un)
            if equity <= cap*nomsum*mmr:
                nmargin += 1
                cap = equity - cap*nomsum*mmr
                if cap < 0.0: cap = 0.0
                for s in range(nslot):
                    ent_i[s] = -1
                nopen = 0
                if cap > peak: peak = cap
                dd = cap/peak-1.0
                if dd < mdd: mdd = dd
                if cap < 0.01:
                    return cap, mdd, ntr, nwin, sum_r, sum_h, max_open, worst_unreal, sum_open, nmin, nmargin
        sum_open += nopen; nmin += 1
        if nopen > max_open: max_open = nopen
        # --- 진입 ---
        if t % scan != 0: continue
        if nopen >= nslot: continue
        if t - last_entry < min_gap: continue
        j = t//SGSTEP
        if j >= len(sq): continue
        s_ = sq[j]
        if th > 0.0:
            if not np.isfinite(s_) or s_ < th: continue
        if np.isnan(rv[t]) or rv[t] <= 0: continue
        m = 1.0
        if use_size == 1:
            mm = med[t]
            if np.isfinite(mm) and rv[t] > 0:
                x = mm/rv[t]
                m *= 0.1 if x < 0.1 else (4.0 if x > 4.0 else x)
            if np.isfinite(s_):
                m *= s_hi if s_ >= 0.97 else (s_mid if s_ >= 0.90 else s_lo)
        sz = tot_nom/nslot*m
        if sz <= 0: continue
        liq = 1.0/sz - mmr if sz > 0 else 9.9
        if liq < dn: continue          # 청산이 손절보다 먼저 오는 크기는 건너뜀
        for s in range(nslot):
            if ent_i[s] < 0:
                ent_i[s] = t; ent_p[s] = c[t]; ent_sz[s] = sz
                nopen += 1; last_entry = t
                break
    return cap, mdd, ntr, nwin, sum_r, sum_h, max_open, worst_unreal, sum_open, nmin, nmargin


def sim(scan=15, nslot=1, th=0.75, up=0.01, dn=0.01, maxh=4320, tot=1.0,
        fee=10.0, use_size=1, s_lo=0.5, s_mid=1.0, s_hi=2.0, min_gap=0):
    cap, mdd, ntr, nwin, sr, sh, mo, wu, so, nm, nmg = run(
        O, H, Lo, C, rv, med, SQ, scan, nslot, th, up, dn, maxh, float(tot), MMR,
        fee, use_size, s_lo, s_mid, s_hi, warm, NEND, min_gap)
    if ntr < 50: return None
    ann = (max(cap, 1e-12)**(1/Y)-1)*100
    mon = ((1+ann/100)**(1/12)-1)*100 if ann > -99.99 else -100
    return dict(ann=ann, mon=mon, mdd=mdd*100, fin=cap, n=ntr, wr=nwin/ntr*100,
                bp=sr/ntr*1e4, hold=sh/ntr, maxopen=mo, worst=wu*100,
                avgopen=so/max(nm, 1), nmargin=nmg, ruin=cap < 0.01)


if __name__ == "__main__":
    P("="*130); P("[1] 슬롯 수 × 스캔 간격 (신호 상위25%, ±1%, 총명목 1배). 셀 = 연거래 / 승률% / 월%")
    P("="*130)
    P(f"  {'슬롯':>5s} " + " ".join(f"{'스캔'+str(s)+'분':>20s}" for s in (1, 5, 15, 60)))
    BEST = []
    for ns in (1, 2, 4, 8, 16, 32):
        row = []
        for sc in (1, 5, 15, 60):
            v = sim(scan=sc, nslot=ns)
            if v is None: row.append("         -          "); continue
            tag = "파산" if v["ruin"] else f"{v['mon']:+6.2f}%"
            row.append(f"{v['n']/Y:5.0f}/{v['wr']:5.1f}%/{tag:>7s}")
            if not v["ruin"]: BEST.append((v["mon"], ns, sc, v))
        P(f"  {ns:4d}개 " + " ".join(f"{x:>20s}" for x in row))
    
    P("\n"+"="*130); P("[2] 총명목 × 슬롯 (스캔 1분, ±1%, 상위25%)"); P("="*130)
    P(f"  {'총명목':>7s} " + " ".join(f"{'슬롯'+str(n):>17s}" for n in (4, 8, 16, 32, 64)))
    for tot in (1, 2, 3, 5, 8, 12, 20):
        row = []
        for ns in (4, 8, 16, 32, 64):
            v = sim(scan=1, nslot=ns, tot=tot)
            if v is None: row.append("        -        "); continue
            row.append("      파산       " if v["ruin"] else f"{v['mon']:+7.2f}%/{v['mdd']:6.1f}%")
            if not v["ruin"]: BEST.append((v["mon"], ns, 1, v))
        P(f"  {tot:6.1f}x " + " ".join(f"{x:>17s}" for x in row))
    
    P("\n"+"="*130); P("[3] 배리어 × 슬롯 (스캔 1분, 총명목 3배, 상위25%)"); P("="*130)
    P(f"  {'배리어':>8s} " + " ".join(f"{'슬롯'+str(n):>20s}" for n in (8, 16, 32, 64)))
    for bar in (0.003, 0.005, 0.0075, 0.01, 0.015, 0.02):
        row = []
        for ns in (8, 16, 32, 64):
            v = sim(scan=1, nslot=ns, tot=3.0, up=bar, dn=bar, maxh=min(4320, int(bar*400000)))
            if v is None: row.append("         -          "); continue
            row.append("        파산        " if v["ruin"]
                       else f"{v['n']/Y:5.0f}/{v['wr']:4.1f}/{v['mon']:+6.2f}%")
            if not v["ruin"]: BEST.append((v["mon"], ns, 1, v))
        P(f"  ±{bar*100:5.2f}% " + " ".join(f"{x:>20s}" for x in row))
    P("  (셀 = 연거래 / 승률% / 월%)")
    
    P("\n"+"="*130); P("[4] 진입 문턱 완화 — 빈도를 더 올릴 수 있나 (스캔1분, 슬롯32, 총3배, ±1%)"); P("="*130)
    P(f"  {'문턱':>10s} {'연거래':>8s} {'하루':>7s} {'승률%':>7s} {'거래당bp':>9s} {'평균보유':>9s} "
      f"{'최대동시':>8s} {'평균동시':>8s} {'월%':>9s} {'MDD%':>8s}")
    for th in (0.95, 0.90, 0.84, 0.75, 0.60, 0.40, 0.0):
        v = sim(scan=1, nslot=32, tot=3.0, th=th)
        if v is None: continue
        P(f"  상위{(1-th)*100:5.0f}% {v['n']/Y:8.0f} {v['n']/Y/365:7.2f} {v['wr']:7.2f} {v['bp']:+9.1f} "
          f"{v['hold']:8.0f}분 {v['maxopen']:8d} {v['avgopen']:8.2f} "
          + ("    파산" if v["ruin"] else f"{v['mon']:+9.2f}") + f" {v['mdd']:8.1f}")
    
    P("\n"+"="*130); P("[5] 최고 구성 상위 10 (월수익 기준)"); P("="*130)
    BEST.sort(key=lambda x: -x[0])
    seen = set(); shown = 0
    P(f"  {'월%':>8s} {'연%':>12s} {'MDD%':>8s} {'연거래':>8s} {'하루':>7s} {'승률%':>7s} "
      f"{'최대동시':>8s} {'최악미실현%':>11s} {'슬롯':>5s}")
    for mon, ns, sc, v in BEST:
        k = (round(mon, 2), ns, v["n"])
        if k in seen: continue
        seen.add(k)
        P(f"  {mon:+8.2f} {v['ann']:12.1f} {v['mdd']:8.1f} {v['n']/Y:8.0f} {v['n']/Y/365:7.2f} "
          f"{v['wr']:7.2f} {v['maxopen']:8d} {v['worst']:11.1f} {ns:5d}")
        shown += 1
        if shown >= 10: break
    
    P("\n"+"="*130); P("[6] 월 +80% 달성 여부 점검"); P("="*130)
    ok = [x for x in BEST if x[0] >= 80.0]
    if ok:
        P(f"  ★ 월 +80% 이상 구성 {len(ok)}개 발견")
        for mon, ns, sc, v in ok[:5]:
            P(f"    월 {mon:+.2f}% / 연 {v['ann']:,.0f}% / MDD {v['mdd']:.1f}% / 연거래 {v['n']/Y:.0f} "
              f"/ 승률 {v['wr']:.2f}% / 슬롯 {ns}")
    else:
        top = BEST[0] if BEST else None
        if top:
            P(f"  최고 월 {top[0]:+.2f}% (목표 +80% 대비 {(1+top[0]/100)/1.8:.4f}배)")
            v = top[3]
            P(f"    연거래 {v['n']/Y:.0f}(하루 {v['n']/Y/365:.2f}), 승률 {v['wr']:.2f}%, "
              f"거래당 {v['bp']:+.1f}bp, 최대동시 {v['maxopen']}개, 최악 미실현 {v['worst']:.1f}%")
            need_n = np.log(1.8)*12/max(np.log(1+v['bp']/1e4), 1e-9)
            P(f"    같은 거래당 엣지로 월 +80%를 내려면 연 {need_n:,.0f}건 필요 "
              f"(현재의 {need_n/(v['n']/Y):,.0f}배)")
    