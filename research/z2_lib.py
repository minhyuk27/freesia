"""z2 공통 엔진 — 캐시 로드 + 거래 생성 + 자본 곡선. 출력 없음."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, numba
R = Path(__file__).resolve().parents[1]
_z = np.load(R/"data"/"z2_cache.npz", allow_pickle=True)
STEP = int(_z["STEP"]); MAXH = int(_z["MAXH"]); MMR = float(_z["MMR"])
SYM = [str(x) for x in _z["SYM"]]
D = {s: {k: _z[f"{s}_{k}"] for k in ("o","h","l","c","sig","rv")} for s in SYM}
for s in SYM:
    D[s]["idx"] = pd.DatetimeIndex(_z[f"{s}_idx"]).tz_localize("UTC")
    v = D[s]["sig"]; D[s]["fin"] = v[np.isfinite(v)]
YRS = (D["BTCUSDT"]["idx"][-1]-D["BTCUSDT"]["idx"][0]).days/365


@numba.njit(cache=True)
def _tr(o,h,l,c,sig,th,up,dn,maxh,step,entry_k,rv,W,eps,cool,start):
    n=len(c); K=600000
    ei=np.zeros(K,np.int64); ex=np.zeros(K,np.int64); rr=np.zeros(K); mk=np.zeros(K,np.int64)
    k=0; t=start; nskip=0
    while t < n-maxh-W-2:
        i=t//step
        if i>=len(sig): break
        if not np.isfinite(sig[i]) or sig[i]<th: t+=step; continue
        if entry_k>0.0:
            if np.isnan(rv[t]) or rv[t]<=0: t+=step; continue
            T=c[t]*(1.0-rv[t]*entry_k); f=-1
            for u in range(t+1,min(t+1+W,n)):
                if l[u]<=T*(1.0-eps): f=u; break
            if f<0: t+=W; nskip+=1; continue
            px=T; m=1
        else:
            f=t+1; px=o[f]; m=0
        U=px*(1.0+up); Dn=px*(1.0-dn); e=-1; res=0.0; mo=0
        end=min(f+maxh,n-1)
        for j in range(f,end+1):
            if l[j]<=Dn: e=j; res=-dn; mo=0; break
            if h[j]>=U: e=j; res=up; mo=1; break
        if e<0: e=end; res=c[e]/px-1.0; mo=0
        ei[k]=f; ex[k]=e; rr[k]=res; mk[k]=m*2+mo; k+=1
        t=e+1+cool
        if k>=K-10: break
    return ei[:k], ex[:k], rr[:k], mk[:k], nskip


def get(s, q, ek, up, dn, cool=0):
    d = D[s]; th = np.quantile(d["fin"], 1-q)
    ei,ex,rr,mk,_ = _tr(d["o"],d["h"],d["l"],d["c"],d["sig"],th,up,dn,MAXH,STEP,
                        float(ek),d["rv"],240,0.0002,cool,300000)
    ix = d["idx"]
    return list(zip(ix[ei], ix[ex], rr, mk))


def equity(ev, lev, frac, dn, cap0=1e7, fee_m=2.0, fee_t=5.0):
    cap=cap0; cur=[]; nliq=0; liq_d=1.0/lev-MMR
    for (f,e,res,m) in ev:
        r=res
        if r <= -dn and liq_d < dn: r=-1.0/lev; nliq+=1
        fin_ = fee_m if (m//2)==1 else fee_t
        fout = fee_m if (m%2)==1 else fee_t
        pnl = cap*frac*lev*(r-(fin_+fout)/1e4)
        if pnl < -cap*frac: pnl = -cap*frac
        cap += pnl; cur.append(cap)
        if cap <= cap0*0.01: break
    return np.array(cur), nliq


def stats(ev, lev, frac, dn, cap0=1e7):
    if len(ev) < 20: return None
    eq, nliq = equity(ev, lev, frac, dn, cap0)
    if eq[-1] <= cap0*0.02: return None
    ann=(eq[-1]/cap0)**(1/YRS)-1
    peak=np.maximum.accumulate(np.concatenate([[cap0],eq]))[1:]
    mdd=(eq/peak-1).min()
    rr=np.array([e[2] for e in ev])
    mx=0; c=0
    for x in rr:
        c = c+1 if x<=0 else 0
        if c>mx: mx=c
    return dict(ann=ann*100, mdd=mdd*100, wr=np.mean(rr>0)*100, nliq=nliq,
                maxloss=mx, fin=eq[-1]/cap0, n=len(ev))
