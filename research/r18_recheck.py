"""R18 — 증거금 정규화 정정 후 R12 계열 전면 재측정.

오류: er = tot/wsum (체결층 가중합으로 나눔) → 부분 체결을 2.5~10배 부풀림.
      1층만 체결 시 실제 포지션은 자본의 10%인데 명목 1배로 취급했다.
      그리고 1층만 체결 = 얕은 딥 = 이기는 거래이므로 정확히 승자만 증폭.
정정: er = tot/wtot (총 준비 자본 기준). 미체결분은 현금(수익 0).

추가로 '단층 전액' 구조도 잰다 — 층을 하나만 두고 전액을 거는 방식은 구현 가능하다.
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R/"research"))
import z0_20_lib as L2
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
Y = L2.YRS
EPS, SLIP, FEE = 5.0, 10.0, 10.0
warm = 200000


@numba.njit(cache=True)
def dip(o,h,l,c,rvv,barmin,k0,k1,k2,w0,w1,w2,eps,slip,fee,start,n_end,mode):
    """mode 0=총자본기준(정직) 1=체결층정규화(옛 오류) 2=단층전액(k0만, 전액)"""
    K=200000
    ea=np.zeros(K,np.int64); er=np.zeros(K); nomv=np.zeros(K); k=0
    wtot=w0+w1+w2
    nb=(n_end-start)//barmin
    for b in range(nb):
        t0=start+b*barmin; t1=t0+barmin-1
        if t1>=n_end-1: break
        r=rvv[t0]
        if np.isnan(r) or r<=0: continue
        px0=o[t0]
        if mode==2:
            T=px0*(1.0-r*k0); trig=T*(1.0-eps/1e4); got=-1
            for j in range(t0,t1+1):
                if l[j]<=trig: got=j; break
            if got<0: continue
            fill=T*(1.0+slip/1e4)
            ea[k]=got; er[k]=c[t1]/fill-1.0-fee/1e4; nomv[k]=1.0; k+=1
            if k>=K-2: break
            continue
        tot=0.0; wsum=0.0; ft=-1
        for li in range(3):
            kk=k0 if li==0 else (k1 if li==1 else k2)
            ww=w0 if li==0 else (w1 if li==1 else w2)
            T=px0*(1.0-r*kk); trig=T*(1.0-eps/1e4); got=-1
            for j in range(t0,t1+1):
                if l[j]<=trig: got=j; break
            if got>=0:
                fill=T*(1.0+slip/1e4)
                tot+=ww*(c[t1]/fill-1.0-fee/1e4); wsum+=ww
                if ft<0: ft=got
        if wsum>0:
            ea[k]=ft; nomv[k]=wsum/wtot
            er[k]= tot/wsum if mode==1 else tot/wtot
            k+=1
            if k>=K-2: break
    return ea[:k], er[:k], nomv[:k]


def curve(ea,er,idx,nom=1.0):
    if len(er)<30: return None
    m=pd.Series(nom*er,index=pd.DatetimeIndex(idx[ea])).resample("ME").sum()
    eq=(1+m).cumprod()
    if (eq<=0.01).any(): return None
    a=(eq.iloc[-1]**(12/len(m))-1)*100
    return dict(ann=a, mon=((1+a/100)**(1/12)-1)*100, mdd=(eq/eq.cummax()-1).min()*100)


P("="*124); P("[1] 정규화 방식이 만든 차이 — 720분봉 rv×15"); P("="*124)
P(f"  {'종목':>9s} {'방식':>18s} {'연거래':>6s} {'승률%':>7s} {'거래당bp':>9s} {'실효명목':>8s} {'무레버 연%':>11s}")
for S in ("BTCUSDT","ETHUSDT"):
    d=L2.D[S]; idx=d["idx"]; N=len(d["c"])
    for mode,lab in ((1,"체결층정규화(옛)"),(0,"총자본기준(정직)")):
        ea,er,nv=dip(d["o"],d["h"],d["l"],d["c"],d["rv"],720,15.,21.,30.,1.,3.,6.,
                     EPS,SLIP,FEE,warm,N,mode)
        r=curve(ea,er,idx)
        P(f"  {S[:3]:>9s} {lab:>18s} {len(er)/Y:6.0f} {np.mean(er>0)*100:7.2f} "
          f"{er.mean()*1e4:+9.1f} {nv.mean():8.2f} "
          + ("       파산" if r is None else f"{r['ann']:11.1f}"))

P("\n"+"="*124); P("[2] 단층 전액 구조 — 구현 가능한 방식. 깊이 × 봉길이 (총자본기준)"); P("="*124)
for S in ("BTCUSDT","ETHUSDT"):
    d=L2.D[S]; idx=d["idx"]; N=len(d["c"])
    P(f"\n  ══ {S} ══")
    P(f"  {'깊이':>8s} " + " ".join(f"{str(b)+'분':>15s}" for b in (240,720,1440,2880,5760)))
    for mult in (5,10,15,20,28,40,60):
        row=[]
        for bm in (240,720,1440,2880,5760):
            ea,er,nv=dip(d["o"],d["h"],d["l"],d["c"],d["rv"],bm,float(mult),0.,0.,1.,0.,0.,
                         EPS,SLIP,FEE,warm,N,2)
            if len(er)<40: row.append("       -       "); continue
            r=curve(ea,er,idx)
            row.append("     파산      " if r is None
                       else f"{r['ann']:+7.1f}%/{np.mean(er>0)*100:4.1f}")
        P(f"  rv×{mult:<4d} " + " ".join(f"{x:>15s}" for x in row))
    P("  (셀 = 연수익% / 승률%)")

P("\n"+"="*124); P("[3] 단층 전액 최선안 3단 레버리지"); P("="*124)
BEST=None
for S in ("BTCUSDT","ETHUSDT"):
    d=L2.D[S]; idx=d["idx"]; N=len(d["c"])
    for bm in (240,720,1440,2880,5760):
        for mult in (5,10,15,20,28,40,60):
            ea,er,nv=dip(d["o"],d["h"],d["l"],d["c"],d["rv"],bm,float(mult),0.,0.,1.,0.,0.,
                         EPS,SLIP,FEE,warm,N,2)
            if len(er)<60: continue
            r=curve(ea,er,idx)
            if r and (BEST is None or r["ann"]>BEST[0]): BEST=(r["ann"],S,bm,mult,ea,er,idx)
if BEST:
    a,S,bm,mult,ea,er,idx=BEST
    P(f"  최선: {S} {bm}분봉 rv×{mult} 단층전액 — 연 {len(er)/Y:.0f}건, "
      f"승률 {np.mean(er>0)*100:.2f}%, 거래당 {er.mean()*1e4:+.1f}bp")
    P(f"\n  {'구간':>12s} {'명목':>6s} {'연수익%':>12s} {'월%':>9s} {'MDD%':>8s}")
    for lb,nom in (("무레버리지",1),("중간",2),("중간+",3),("공격",5),("최대공격",8)):
        r=curve(ea,er,idx,nom)
        P(f"  {lb:>12s} {nom:5d}x " + ("        파산" if r is None
                                        else f"{r['ann']:12.1f} {r['mon']:9.2f} {r['mdd']:8.1f}"))
    m=pd.Series(er,index=pd.DatetimeIndex(idx[ea])).resample("ME").sum()
    out=[]
    for yr in range(2020,2027):
        sub=m[m.index.year==yr]
        out.append(f"{yr}  n/a" if len(sub)==0 else f"{yr} {((1+sub).prod()-1)*100:+7.0f}%")
    P("  연도별(무레버): " + " | ".join(out))
    for y0 in (2023,2025):
        mm=np.array([idx[x].year>=y0 for x in ea])
        if mm.sum()<40: continue
        r=curve(ea[mm],er[mm],idx)
        P(f"  {y0}년 이후: {mm.sum()}건 승률 {np.mean(er[mm]>0)*100:.2f}% 거래당 {er[mm].mean()*1e4:+.1f}bp "
          + ("파산" if r is None else f"연 {r['ann']:+.1f}%"))
