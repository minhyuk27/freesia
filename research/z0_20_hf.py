"""z0.20 고빈도·고레버리지 — 연 180건(이틀 1건) 이상만 후보.
손절이 타이트하면 청산거리(1/L-mmr)가 손절폭보다 멀어 청산이 안 난다.
손절 1%면 L<71배까지 청산 없음 → 레버리지 상한은 청산이 아니라 연속손실이 정한다."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R/"research"))
import z0_20_lib as L
def P(*a, end="\n"): print(" ".join(str(x) for x in a), end=end, flush=True)
S = "BTCUSDT"; MINY = 180; Y = L.YRS

BARS = [(0.005,0.005),(0.0075,0.005),(0.01,0.005),(0.01,0.01),(0.015,0.01),(0.02,0.01),
        (0.015,0.015),(0.02,0.015),(0.03,0.02),(0.0075,0.0075),(0.03,0.01),(0.025,0.015)]
QS = (0.10,0.15,0.20,0.30,0.50); EKS = (0.0,0.5); COOLS = (0,60,240)
LEVS = (1,2,3,5,8,12,20,30,50,71); FRACS = (1.0,0.5,0.3)

P("="*132); P(f"[1] 후보 — 연 {MINY}건 이상(이틀 1건)만 통과"); P("="*132)
POOL = []
for q in QS:
    for ek in EKS:
        for up,dn in BARS:
            for cool in COOLS:
                ev = L.get(S,q,ek,up,dn,cool)
                if len(ev)/Y >= MINY: POOL.append((q,ek,up,dn,cool,ev))
P(f"  {len(POOL)} / {len(QS)*len(EKS)*len(BARS)*len(COOLS)} 구성 통과")

res=[]
for q,ek,up,dn,cool,ev in POOL:
    for lev in LEVS:
        for frac in FRACS:
            st = L.stats(ev,lev,frac,dn)
            if st: res.append((st["ann"],q,ek,up,dn,cool,lev,frac,st))
res.sort(key=lambda x:-x[0])

def line(r):
    ann,q,ek,up,dn,cool,lev,frac,st = r
    return (f"{ann:9.1f} {st['mdd']:8.1f} {st['wr']:6.1f} {st['n']/Y:6.0f} {lev:3d}x {frac*100:4.0f}% "
            f"{q*100:4.0f}% {'지정' if ek>0 else '시장':>4s} {up*100:4.2f}/{dn*100:4.2f}% {cool:4d} "
            f"{st['maxloss']:4d} {st['nliq']:4d} {st['fin']:9.1f}")
HDR = (f"{'연수익%':>9s} {'MDD%':>8s} {'승률%':>6s} {'연거래':>6s} {'배수':>4s} {'자본':>5s} "
       f"{'분위':>5s} {'진입':>4s} {'익절/손절':>11s} {'쿨':>4s} {'연속손':>4s} {'청산':>4s} {'최종배':>9s}")

P("\n"+"="*132); P("[2] 연수익 상위 25"); P("="*132); P(HDR)
for r in res[:25]: P(line(r))

P("\n"+"="*132); P("[3] MDD −60% 이내 중 연수익 상위 15 (현실적 공격 구간)"); P("="*132); P(HDR)
for r in [x for x in res if x[8]["mdd"] > -60][:15]: P(line(r))

P("\n"+"="*132); P("[4] 승률 상위 15 (구성별 최적 레버리지)"); P("="*132); P(HDR)
seen={}
for r in res:
    k=r[1:6]
    if k not in seen or r[0]>seen[k][0]: seen[k]=r
for r in sorted(seen.values(), key=lambda x:-x[8]["wr"])[:15]: P(line(r))

P("\n"+"="*132); P("[5] 상위 4개 구성 — 3단 레버리지 + 연도별"); P("="*132)
done=set()
for r in res:
    k=r[1:6]
    if k in done: continue
    done.add(k)
    if len(done)>4: break
    _,q,ek,up,dn,cool,_,_,_ = r
    ev = L.get(S,q,ek,up,dn,cool)
    P(f"\n■ 상위{q*100:.0f}% {'지정가k0.5' if ek>0 else '시장가'} 익절{up*100:.2f}/손절{dn*100:.2f}% 쿨{cool}분")
    P(f"  {len(ev)}건 = 연 {len(ev)/Y:.0f}건 = 이틀당 {len(ev)/Y/182.5:.2f}건")
    P(f"  {'구간':>12s} {'배수':>5s} {'자본':>5s} {'연수익%':>10s} {'MDD%':>8s} {'최종배수':>11s} {'청산':>5s} {'최대연속손':>8s}")
    for nm,lev,frac in (("무레버리지",1,1.0),("중간",3,1.0),("중간+",5,1.0),("공격",10,1.0),
                        ("최대공격",20,1.0),("극단",35,1.0),("50%자본×50x",50,0.5),("30%자본×71x",71,0.3)):
        st = L.stats(ev,lev,frac,dn)
        if not st: P(f"  {nm:>12s} {lev:4d}x {frac*100:4.0f}%        파산"); continue
        P(f"  {nm:>12s} {lev:4d}x {frac*100:4.0f}% {st['ann']:10.1f} {st['mdd']:8.1f} {st['fin']:11.2f} {st['nliq']:5d} {st['maxloss']:8d}")
    for lv in (5,10):
        out=[]
        for yr in range(2020,2027):
            sub=[e for e in ev if e[0].year==yr]
            if len(sub)<3: out.append(f"{yr}  n/a"); continue
            eq,_=L.equity(sub,lv,1.0,dn)
            out.append(f"{yr} {(eq[-1]/1e7-1)*100:+6.0f}%")
        P(f"  연도별({lv}x): " + " | ".join(out))
