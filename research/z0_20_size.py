"""z0.20 베팅 크기 2축 — ①변동성(확정) ②신호 강도. 둘을 곱해 배분한다.
신호 강도 배수는 확장창 분위로만 계산(룩어헤드 없음)."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R/"research"))
import z0_20_lib as L
def P(*a, end="\n"): print(" ".join(str(x) for x in a), end=end, flush=True)
S="BTCUSDT"; Y=L.YRS; d=L.D[S]; idx=d["idx"]; rv=d["rv"]; sig=d["sig"]
warm=200000; step=5000

med=np.full(len(rv), np.nan)
for i in range(warm, len(rv), step):
    x=rv[:i]; x=x[np.isfinite(x)]
    if len(x)>=1000: med[i:i+step]=np.median(x)

# 신호의 확장창 분위 — 시점 t의 신호가 과거 분포에서 몇 분위인지
sq=np.full(len(sig), np.nan)
sw=warm//L.STEP; ss=step//L.STEP
for i in range(sw, len(sig), ss):
    x=sig[:i]; x=x[np.isfinite(x)]
    if len(x)<1000: continue
    seg=sig[i:i+ss]
    sq[i:i+ss]=np.searchsorted(np.sort(x), seg)/len(x)

def getev(q,ek,up,dn,cool):
    th=np.quantile(d["fin"],1-q)
    ei,ex,rr,mk,_=L._tr(d["o"],d["h"],d["l"],d["c"],sig,th,up,dn,L.MAXH,L.STEP,
                        float(ek),rv,240,0.0002,cool,300000)
    k=ei>=warm
    return [(idx[a],idx[b],r,m,a) for a,b,r,m in zip(ei[k],ex[k],rr[k],mk[k])]

def run(ev,lev,dn,vcap=4.0,vfl=0.1,smode="none",scap=2.0,sfl=0.5,frac=1.0,yr=None):
    cap=1e7; cur=[]
    for (f,e,res,m,ii) in ev:
        if yr and f.year!=yr: continue
        mult=1.0
        r_=rv[ii]; mm=med[ii]
        if np.isfinite(r_) and np.isfinite(mm) and r_>0: mult*=float(np.clip(mm/r_,vfl,vcap))
        if smode!="none":
            s_=sq[ii//L.STEP] if ii//L.STEP < len(sq) else np.nan
            if np.isfinite(s_):
                if smode=="lin":   mult*=float(np.clip(sfl+(scap-sfl)*(s_-0.5)/0.5, sfl, scap))
                elif smode=="step":mult*=(scap if s_>=0.97 else (1.0 if s_>=0.90 else sfl))
                elif smode=="pow": mult*=float(np.clip((max(s_,0.5)/0.85)**3, sfl, scap))
        lv=lev*mult; liq=1.0/lv-L.MMR; r=res
        if r<=-dn and liq<dn: r=-1.0/lv
        fi=2.0 if (m//2)==1 else 5.0
        fo=2.0 if (m%2)==1 else 5.0
        pnl=cap*frac*lv*(r-(fi+fo)/1e4)
        if pnl<-cap*frac: pnl=-cap*frac
        cap+=pnl; cur.append(cap)
        if cap<=1e5: break
    if not cur: return None
    eq=np.array(cur)
    if eq[-1]<=2e5: return None
    n=len(cur); yy=Y if yr is None else 1.0
    peak=np.maximum.accumulate(np.concatenate([[1e7],eq]))[1:]
    return dict(ann=((eq[-1]/1e7)**(1/yy)-1)*100, mdd=(eq/peak-1).min()*100, fin=eq[-1]/1e7)

CFG=[("상위15% 시장 1.5/1.5 쿨60",0.15,0.0,0.015,0.015,60),
     ("상위10% 시장 1.5/1.5 쿨0", 0.10,0.0,0.015,0.015,0),
     ("상위15% 지정 2.0/1.5 쿨0", 0.15,0.5,0.02,0.015,0)]
EV={nm:getev(q,ek,up,dn,cool) for nm,q,ek,up,dn,cool in CFG}
DN={nm:dn for nm,q,ek,up,dn,cool in CFG}
for nm in EV: P(f"{nm}: {len(EV[nm])}건 (연 {len(EV[nm])/Y:.0f})")

P("\n"+"="*124); P("[1] 신호강도 배수 방식 × 배수 (변동성목표 cap4 위에 곱함)"); P("="*124)
for nm in EV:
    P(f"\n  ── {nm} ──")
    P(f"  {'배수':>5s} {'vol만':>17s} {'+선형2x':>17s} {'+계단2x':>17s} {'+3제곱2x':>17s} {'+선형3x':>17s}")
    for lev in (1,2,3,5,8,12):
        row=[]
        for sm,sc in (("none",1),("lin",2.0),("step",2.0),("pow",2.0),("lin",3.0)):
            r=run(EV[nm],lev,DN[nm],smode=sm,scap=sc)
            row.append(f"{r['ann']:+8.1f}%/{r['mdd']:6.1f}%" if r else "          파산   ")
        P(f"  {lev:4d}x " + " ".join(f"{x:>17s}" for x in row))

P("\n"+"="*124); P("[2] 최종 후보 3단 레버리지 + 연도별 (vol cap4 + 선형 신호강도 2x)"); P("="*124)
best=None
for nm in EV:
    P(f"\n■ {nm} — 연 {len(EV[nm])/Y:.0f}건, 이틀당 {len(EV[nm])/Y/182.5:.2f}건, "
      f"승률 {np.mean([x[2]>0 for x in EV[nm]])*100:.1f}%")
    P(f"  {'구간':>12s} {'배수':>5s} {'연수익%':>10s} {'MDD%':>8s} {'최종배수':>11s}")
    for lb,lev in (("무레버리지",1),("중간",3),("중간+",5),("공격",8),("최대공격",12)):
        r=run(EV[nm],lev,DN[nm],smode="lin",scap=2.0)
        if not r: P(f"  {lb:>12s} {lev:4d}x        파산"); continue
        P(f"  {lb:>12s} {lev:4d}x {r['ann']:10.1f} {r['mdd']:8.1f} {r['fin']:11.2f}")
        if lev==3 and (best is None or r['ann']>best[0]): best=(r['ann'],nm)
    for lv in (3,5):
        out=[]
        for yr in range(2020,2027):
            r=run(EV[nm],lv,DN[nm],smode="lin",scap=2.0,yr=yr)
            out.append(f"{yr} {(r['fin']-1)*100:+7.0f}%" if r else f"{yr}   파산")
        P(f"  연도별({lv}x): " + " | ".join(out))

P("\n"+"="*124); P("[3] 최고 구성의 거래 간격 재확인"); P("="*124)
nm=best[1] if best else CFG[0][0]
ev=EV[nm]
hold=np.array([(e[1]-e[0]).total_seconds()/60 for e in ev])
gap=np.array([(ev[i+1][0]-ev[i][1]).total_seconds()/60 for i in range(len(ev)-1)])
P(f"  {nm}")
P(f"  보유: 중앙 {np.median(hold):.0f}분 평균 {hold.mean():.0f}분 90% {np.percentile(hold,90):.0f}분")
P(f"  공백: 중앙 {np.median(gap):.0f}분 평균 {gap.mean():.0f}분 90% {np.percentile(gap,90):.0f}분 최대 {gap.max()/1440:.1f}일")
P(f"  노출: {hold.sum()/((idx[-1]-idx[warm]).total_seconds()/60)*100:.1f}%")
ts=pd.DatetimeIndex([e[0] for e in ev])
mm=pd.Series(1,index=ts).resample("ME").sum()
P(f"  월 거래: 중앙 {mm.median():.0f} 평균 {mm.mean():.1f} 최소 {mm.min():.0f} | 거래 0건 월 {int((mm==0).sum())}/{len(mm)}")
qq=pd.Series(1,index=ts).resample("QE").sum()
P(f"  분기 거래: 최소 {qq.min():.0f} (전 분기 최소 {qq.min()/91*2:.2f}건/이틀)")
