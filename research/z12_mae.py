"""z12 봉 중간 역행(MAE) 점검 — 고레버리지 수치가 청산을 무시하고 있는지 확인."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R/"research"))
import z2_lib as L2
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
S="BTCUSDT"; Y=L2.YRS; MMR=L2.MMR
d=L2.D[S]; idx=d["idx"]; rv=d["rv"]; O,H,Lo,C=d["o"],d["h"],d["l"],d["c"]
N=len(C); warm=200000


@numba.njit(cache=True)
def dip_mae(o,h,l,c,rvv,barmin,k0,k1,k2,w0,w1,w2,eps,slip,fee,lev,mmr,start,n_end):
    """체결 후 봉 종가까지의 최대 역행(MAE)을 추적하고, 청산거리를 넘으면 청산 처리."""
    K=200000
    er=np.zeros(K); mae=np.zeros(K); ea=np.zeros(K,np.int64); liq=np.zeros(K,np.int8)
    k=0
    nb=(n_end-start)//barmin
    liq_d = 1.0/lev - mmr
    for b in range(nb):
        t0=start+b*barmin; t1=t0+barmin-1
        if t1>=n_end-1: break
        if np.isnan(rvv[t0]) or rvv[t0]<=0: continue
        px0=o[t0]
        tot=0.0; wsum=0.0; ft=-1; worst=0.0
        for li in range(3):
            kk=k0 if li==0 else (k1 if li==1 else k2)
            ww=w0 if li==0 else (w1 if li==1 else w2)
            T=px0*(1.0-rvv[t0]*kk); trig=T*(1.0-eps/1e4)
            got=-1
            for j in range(t0,t1+1):
                if l[j]<=trig: got=j; break
            if got>=0:
                fill=T*(1.0+slip/1e4)
                tot+=ww*(c[t1]/fill-1.0-fee/1e4); wsum+=ww
                if ft<0: ft=got
                # 이 층의 체결 후 최저가까지의 역행
                lo=fill
                for j in range(got,t1+1):
                    if l[j]<lo: lo=l[j]
                dd=lo/fill-1.0
                if dd<worst: worst=dd
        if wsum>0:
            ea[k]=ft; er[k]=tot/wsum; mae[k]=worst
            if -worst >= liq_d: liq[k]=1
            k+=1
            if k>=K-2: break
    return ea[:k], er[:k], mae[:k], liq[:k]


for BAR,KS in ((720,(15.0,21.0,30.0)),(240,(20.0,28.0,40.0))):
    P("="*112); P(f"봉 {BAR}분, 깊이 분rv×{KS}"); P("="*112)
    ea,er,mae,_=dip_mae(O,H,Lo,C,rv,BAR,*KS,1.0,3.0,6.0,5.0,10.0,10.0,1.0,MMR,warm,N)
    P(f"  거래 {len(er):,}건  거래당 {er.mean()*1e4:+.1f}bp  승률 {np.mean(er>0)*100:.2f}%")
    P(f"  체결 후 봉 종가까지 최대 역행(MAE):")
    for q in (50,75,90,95,99,99.9):
        P(f"    {q:5.1f}분위 {np.percentile(-mae,q)*100:6.2f}%")
    P(f"    최악 {(-mae).max()*100:.2f}%")
    P(f"\n  {'명목':>6s} {'청산거리':>9s} {'청산건':>7s} {'청산률%':>8s} {'청산반영 연수익%':>16s} {'청산무시 연수익%':>16s}")
    for lev in (1,2,3,5,8,12):
        ea2,er2,mae2,lq=dip_mae(O,H,Lo,C,rv,BAR,*KS,1.0,3.0,6.0,5.0,10.0,10.0,float(lev),MMR,warm,N)
        liq_d=1.0/lev-MMR
        r_liq=np.where(lq==1, -1.0/lev, er2)
        def ann(v):
            m=pd.Series(lev*v, index=pd.DatetimeIndex(idx[ea2])).resample("ME").sum()
            eq=(1+m).cumprod()
            if (eq<=0.01).any(): return None
            return (eq.iloc[-1]**(12/len(m))-1)*100
        a1=ann(r_liq); a0=ann(er2)
        f=lambda x: "        파산    " if x is None else f"{x:16.1f}"
        P(f"  {lev:5d}x {liq_d*100:8.2f}% {int(lq.sum()):7d} {lq.mean()*100:8.2f} {f(a1)} {f(a0)}")
    P("")
