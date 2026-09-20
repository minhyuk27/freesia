"""z2 고빈도·고레버리지 — 연 180건(이틀 1건) 이상만 후보.
손절이 타이트하면 청산거리 1/L-mmr > 손절폭인 한 청산이 안 난다.
손절 1%면 L<71x까지 청산 없음 → 레버리지 상한은 청산이 아니라 연속손실이 정한다."""
import sys, warnings, itertools
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
import z2_dual as Z
from z2_sim import D, MMR
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
yrs = Z.yrs; S = "BTCUSDT"; MINY = 180

def stats(ev, lev, frac, dn):
    eq, nliq = Z.equity(ev, 1e7, lev, frac, MMR, 2.0, 5.0, dn)
    if eq[-1] <= 1e7*0.02: return None
    ann = (eq[-1]/1e7)**(1/yrs)-1
    peak = np.maximum.accumulate(np.concatenate([[1e7], eq]))[1:]
    mdd = (eq/peak-1).min()
    rr = np.array([e[2] for e in ev])
    # 최대 연속 손실
    mx=0; c=0
    for x in rr:
        c = c+1 if x <= 0 else 0
        mx = max(mx, c)
    return ann*100, mdd*100, np.mean(rr>0)*100, nliq, mx, eq[-1]/1e7

P("="*130)
P(f"[1] 후보 생성 — 연 {MINY}건 이상만 통과")
P("="*130)
BARS = [(0.005,0.005),(0.0075,0.005),(0.01,0.005),(0.01,0.01),(0.015,0.01),(0.02,0.01),
        (0.015,0.015),(0.02,0.015),(0.03,0.02),(0.0075,0.0075)]
POOL = []
for q in (0.10,0.15,0.20,0.30,0.50):
    for ek in (0.0, 0.5):
        for up,dn in BARS:
            for cool in (0,60,240):
                ev = Z.get(S, q, ek, up, dn, cool)
                if len(ev)/yrs < MINY: continue
                POOL.append((q,ek,up,dn,cool,ev))
P(f"  총 {len(POOL)}개 구성 통과 (전체 {5*2*len(BARS)*3}개 중)")

P("\n"+"="*130)
P("[2] 각 구성 × 레버리지 — 연수익 상위 20 (자본 100%)")
P("="*130)
res = []
for q,ek,up,dn,cool,ev in POOL:
    for lev in (1,2,3,5,8,12,20,30,50):
        for frac in (1.0, 0.5):
            st = stats(ev, lev, frac, dn)
            if st is None: continue
            ann, mdd, wr, nliq, mxl, fin = st
            res.append((ann, q, ek, up, dn, cool, lev, frac, len(ev), mdd, wr, nliq, mxl, fin))
res.sort(reverse=True)
P(f"{'연수익%':>9s} {'MDD%':>8s} {'승률%':>6s} {'연거래':>6s} {'배수':>5s} {'자본':>5s} {'분위':>5s} "
  f"{'진입':>5s} {'익절/손절':>10s} {'쿨':>5s} {'연속손':>5s} {'최종배수':>10s}")
for r in res[:20]:
    ann,q,ek,up,dn,cool,lev,frac,n,mdd,wr,nliq,mxl,fin = r
    P(f"{ann:9.1f} {mdd:8.1f} {wr:6.1f} {n/yrs:6.0f} {lev:4d}x {frac*100:4.0f}% {q*100:4.0f}% "
      f"{'지정':>5s}" if ek>0 else f"{ann:9.1f} {mdd:8.1f} {wr:6.1f} {n/yrs:6.0f} {lev:4d}x {frac*100:4.0f}% {q*100:4.0f}% {'시장':>5s}", end="")
    P(f" {up*100:4.1f}/{dn*100:4.1f}% {cool:5d} {mxl:5d} {fin:10.1f}")

P("\n"+"="*130)
P("[3] 승률 우선 — 연 180건 이상 중 승률 상위 10 (각 구성 최적 레버리지)")
P("="*130)
bywr = {}
for r in res:
    key = r[1:8]
    if key not in bywr or r[0] > bywr[key][0]: bywr[key] = r
cand = sorted(bywr.values(), key=lambda x: -x[10])
P(f"{'승률%':>6s} {'연수익%':>9s} {'MDD%':>8s} {'연거래':>6s} {'배수':>5s} {'자본':>5s} {'분위':>5s} {'익절/손절':>10s} {'쿨':>5s}")
for r in cand[:10]:
    ann,q,ek,up,dn,cool,lev,frac,n,mdd,wr,nliq,mxl,fin = r
    P(f"{wr:6.1f} {ann:9.1f} {mdd:8.1f} {n/yrs:6.0f} {lev:4d}x {frac*100:4.0f}% {q*100:4.0f}% {up*100:4.1f}/{dn*100:4.1f}% {cool:5d}")

P("\n"+"="*130)
P("[4] 상위 3개 구성의 3단 레버리지 + 연도별")
P("="*130)
done = set()
for r in res:
    key = r[1:6]
    if key in done: continue
    done.add(key)
    if len(done) > 3: break
    ann,q,ek,up,dn,cool = r[0],r[1],r[2],r[3],r[4],r[5]
    ev = Z.get(S, q, ek, up, dn, cool)
    P(f"\n■ 상위{q*100:.0f}% {'지정가k0.5' if ek>0 else '시장가'} 익절{up*100:.2f}/손절{dn*100:.2f}% 쿨{cool}분 "
      f"— {len(ev)}건 (연 {len(ev)/yrs:.0f}건, 이틀당 {len(ev)/yrs/182.5:.2f}건)")
    P(f"  {'구간':>10s} {'배수':>5s} {'자본':>5s} {'연수익%':>10s} {'MDD%':>8s} {'최종배수':>10s} {'청산':>5s} {'최대연속손':>7s}")
    for nm, lev, frac in (("무레버리지",1,1.0),("중간",3,1.0),("중간+",5,1.0),("공격",10,1.0),
                          ("최대공격",20,1.0),("극단",35,1.0),("50%자본 50x",50,0.5)):
        st = stats(ev, lev, frac, dn)
        if st is None: P(f"  {nm:>10s} {lev:4d}x {frac*100:4.0f}%       파산"); continue
        a,m,w,nl,mx,fin = st
        P(f"  {nm:>10s} {lev:4d}x {frac*100:4.0f}% {a:10.1f} {m:8.1f} {fin:10.2f} {nl:5d} {mx:7d}")
    P("  연도별 (10배):")
    out=[]
    for yr in range(2020, 2027):
        sub = [e for e in ev if e[0].year == yr]
        if len(sub) < 3: out.append(f"{yr} n/a"); continue
        eq,_ = Z.equity(sub, 1e7, 10, 1.0, MMR, 2.0, 5.0, dn)
        out.append(f"{yr} {(eq[-1]/1e7-1)*100:+7.0f}%")
    P("    " + " | ".join(out))
