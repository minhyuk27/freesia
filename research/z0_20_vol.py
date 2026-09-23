"""z0.20 변동성 목표 레버리지 — 룩어헤드 제거판.
기준 변동성을 '시점 t까지의 확장창 중앙값'으로 계산한다(전체 표본 중앙값 금지).
배수 = clip(rv기준/rv_t, floor, cap). 저변동일수록 크게 태운다."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R/"research"))
import z0_20_lib as L
def P(*a, end="\n"): print(" ".join(str(x) for x in a), end=end, flush=True)
S="BTCUSDT"; Y=L.YRS; d=L.D[S]; idx=d["idx"]

# 확장창 rv 중앙값 (warm 이후, step마다 갱신 — 시점 t 이전 정보만)
rv = d["rv"]
med = np.full(len(rv), np.nan)
warm = 200000; step = 5000
for i in range(warm, len(rv), step):
    x = rv[:i]; x = x[np.isfinite(x)]
    if len(x) < 1000: continue
    med[i:i+step] = np.median(x)
P(f"확장창 rv중앙값: 시작 {np.nanmin(med)*1e4:.1f}bp → 끝 {med[~np.isnan(med)][-1]*1e4:.1f}bp "
  f"(전체중앙 {np.nanmedian(rv)*1e4:.1f}bp)\n")

def run(ev, lev, frac, dn, cap, floor, use_vol=True):
    capital=1e7; cur=[]
    for (f,e,res,m,ii) in ev:
        mult=1.0
        if use_vol:
            r_=rv[ii]; mm=med[ii]
            if not (np.isfinite(r_) and np.isfinite(mm) and r_>0): mult=1.0
            else: mult=float(np.clip(mm/r_, floor, cap))
        lv=lev*mult
        liq=1.0/lv-L.MMR
        r=res
        if r<=-dn and liq<dn: r=-1.0/lv
        fi=2.0 if (m//2)==1 else 5.0
        fo=2.0 if (m%2)==1 else 5.0
        pnl=capital*frac*lv*(r-(fi+fo)/1e4)
        if pnl<-capital*frac: pnl=-capital*frac
        capital+=pnl; cur.append(capital)
        if capital<=1e5: break
    eq=np.array(cur)
    if eq[-1]<=2e5: return None
    peak=np.maximum.accumulate(np.concatenate([[1e7],eq]))[1:]
    return dict(ann=((eq[-1]/1e7)**(1/Y)-1)*100, mdd=(eq/peak-1).min()*100, fin=eq[-1]/1e7, eq=eq)

def getev(q,ek,up,dn,cool):
    th=np.quantile(d["fin"],1-q)
    ei,ex,rr,mk,_ = L._tr(d["o"],d["h"],d["l"],d["c"],d["sig"],th,up,dn,L.MAXH,L.STEP,
                          float(ek),d["rv"],240,0.0002,cool,300000)
    keep = ei >= warm
    return [(idx[a],idx[b],r,m,a) for a,b,r,m in zip(ei[keep],ex[keep],rr[keep],mk[keep])]

BASE=dict(q=0.15,ek=0.0,up=0.015,dn=0.015,cool=60)
ev=getev(**BASE)
P(f"기준 구성 상위15% 시장가 ±1.5% 쿨60 — {len(ev)}건 (연 {len(ev)/Y:.0f}건, 워밍업 이후)\n")

P("="*120); P("[1] cap × 배수 (floor 0.3, 룩어헤드 없음) — 연수익%/MDD%"); P("="*120)
P(f"{'배수':>5s} {'고정':>16s} " + " ".join(f"{'cap'+str(c):>16s}" for c in (2,3,4,6,10)))
for lev in (1,2,3,5,8,12,20):
    row=[f"{run(ev,lev,1.0,BASE['dn'],1,1,False)['ann']:+7.1f}%/{run(ev,lev,1.0,BASE['dn'],1,1,False)['mdd']:6.1f}%"
         if run(ev,lev,1.0,BASE['dn'],1,1,False) else "         파산   "]
    for cp in (2,3,4,6,10):
        r=run(ev,lev,1.0,BASE["dn"],cp,0.3)
        row.append(f"{r['ann']:+7.1f}%/{r['mdd']:6.1f}%" if r else "         파산   ")
    P(f"{lev:4d}x " + " ".join(f"{x:>16s}" for x in row))

P("\n"+"="*120); P("[2] floor 영향 (cap 4)"); P("="*120)
P(f"{'배수':>5s} " + " ".join(f"{'floor'+str(f):>16s}" for f in (0.1,0.2,0.3,0.5,1.0)))
for lev in (3,5,8,12):
    row=[]
    for fl in (0.1,0.2,0.3,0.5,1.0):
        r=run(ev,lev,1.0,BASE["dn"],4,fl)
        row.append(f"{r['ann']:+7.1f}%/{r['mdd']:6.1f}%" if r else "         파산   ")
    P(f"{lev:4d}x " + " ".join(f"{x:>16s}" for x in row))

P("\n"+"="*120); P("[3] 다른 구성에도 적용되나 (cap4 floor0.3, 5x)"); P("="*120)
P(f"{'구성':>34s} {'연거래':>6s} {'승률%':>6s} {'고정5x':>17s} {'voltgt5x':>17s}")
for q,ek,up,dn,cool in [(0.15,0.0,0.015,0.015,60),(0.10,0.0,0.015,0.015,0),(0.20,0.0,0.015,0.015,60),
                        (0.15,0.5,0.02,0.015,0),(0.20,0.5,0.02,0.015,60),(0.10,0.5,0.01,0.01,0),
                        (0.30,0.0,0.015,0.015,60),(0.15,0.0,0.02,0.015,60)]:
    e2=getev(q,ek,up,dn,cool)
    if len(e2)/Y<150: continue
    a=run(e2,5,1.0,dn,1,1,False); b=run(e2,5,1.0,dn,4,0.3)
    wr=np.mean([x[2]>0 for x in e2])*100
    nm=f"상위{q*100:.0f}% {'지정' if ek>0 else '시장'} {up*100:.1f}/{dn*100:.1f}% 쿨{cool}"
    P(f"{nm:>34s} {len(e2)/Y:6.0f} {wr:6.1f} "
      f"{(f'{a[chr(97)+chr(110)+chr(110)]:+7.1f}%/{a[chr(109)+chr(100)+chr(100)]:6.1f}%' if a else '      파산      '):>17s} "
      f"{(f'{b[chr(97)+chr(110)+chr(110)]:+7.1f}%/{b[chr(109)+chr(100)+chr(100)]:6.1f}%' if b else '      파산      '):>17s}")

P("\n"+"="*120); P("[4] 최종 3단 레버리지 (cap4 floor0.3) + 연도별 + 실효배수 분포"); P("="*120)
mults=[]
for (f,e,res,m,ii) in ev:
    r_=rv[ii]; mm=med[ii]
    mults.append(float(np.clip(mm/r_,0.3,4)) if (np.isfinite(r_) and np.isfinite(mm) and r_>0) else 1.0)
mults=np.array(mults)
P(f"  실효배수 승수: 중앙 {np.median(mults):.2f} 평균 {mults.mean():.2f} "
  f"10%{np.percentile(mults,10):.2f} 90%{np.percentile(mults,90):.2f} 최대 {mults.max():.2f}")
P(f"\n  {'구간':>16s} {'기본배수':>8s} {'실효평균':>8s} {'연수익%':>9s} {'MDD%':>8s} {'최종배수':>10s}")
for nm,lev in (("무레버리지",1),("중간",3),("중간+",5),("공격",8),("최대공격",12),("극단",20)):
    r=run(ev,lev,1.0,BASE["dn"],4,0.3)
    if not r: P(f"  {nm:>16s} {lev:7d}x        파산"); continue
    P(f"  {nm:>16s} {lev:7d}x {lev*mults.mean():8.1f} {r['ann']:9.1f} {r['mdd']:8.1f} {r['fin']:10.2f}")
for lv in (3,5,8):
    out=[]
    for yr in range(2020,2027):
        sub=[x for x in ev if x[0].year==yr]
        if len(sub)<3: out.append(f"{yr}  n/a"); continue
        r=run(sub,lv,1.0,BASE["dn"],4,0.3)
        out.append(f"{yr} {(r['fin']-1)*100:+7.0f}%" if r else f"{yr}   파산")
    P(f"  연도별({lv}x): " + " | ".join(out))
