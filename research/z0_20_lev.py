"""z0.20 동적 레버리지 — 연속손실이 레버리지 천장을 정한다는 진단에 대한 처방.
①연속손실 감축(anti-martingale) ②변동성 목표 ③변동성 적응 배리어.
전부 시점 t까지의 정보만 사용."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R/"research"))
import z0_20_lib as L
def P(*a, end="\n"): print(" ".join(str(x) for x in a), end=end, flush=True)
S="BTCUSDT"; Y=L.YRS
BASE = dict(q=0.15, ek=0.0, up=0.015, dn=0.015, cool=60)

def eq_dyn(ev, lev, frac, dn, mode="fixed", k=0.5, floor=0.2, nback=3, capf=1.0, rvser=None, rvmed=None):
    """mode: fixed / anti(연속손실 감축) / voltgt(변동성 목표) / both"""
    cap=1e7; cur=[]; streak=0; mmr=L.MMR
    for (f,e,res,m) in ev:
        mult=1.0
        if mode in ("anti","both") and streak>0:
            mult *= max(floor, 1.0 - k*min(streak, 6)/6.0)
        if mode in ("voltgt","both") and rvser is not None:
            t=f
            rv = rvser.get(t, np.nan)
            if np.isfinite(rv) and rv>0: mult *= float(np.clip(rvmed/rv, 0.3, capf))
        lv = lev*mult
        liq_d = 1.0/lv - mmr if lv>0 else 9.9
        r=res
        if r <= -dn and liq_d < dn: r = -1.0/lv
        fi = 2.0 if (m//2)==1 else 5.0
        fo = 2.0 if (m%2)==1 else 5.0
        pnl = cap*frac*lv*(r-(fi+fo)/1e4)
        if pnl < -cap*frac: pnl = -cap*frac
        cap += pnl; cur.append(cap)
        streak = streak+1 if res<=0 else 0
        if cap <= 1e5: break
    eq=np.array(cur)
    if eq[-1] <= 2e5: return None
    ann=(eq[-1]/1e7)**(1/Y)-1
    peak=np.maximum.accumulate(np.concatenate([[1e7],eq]))[1:]
    return dict(ann=ann*100, mdd=(eq/peak-1).min()*100, fin=eq[-1]/1e7)

ev = L.get(S, BASE["q"], BASE["ek"], BASE["up"], BASE["dn"], BASE["cool"])
d = L.D[S]; idx = d["idx"]
rvmap = pd.Series(d["rv"], index=idx)
rvser = {t: rvmap.get(t, np.nan) for t,_,_,_ in ev}
rvmed = float(np.nanmedian(d["rv"]))
P(f"기준: 상위15% 시장가 ±1.5% 쿨60 — {len(ev)}건 (연 {len(ev)/Y:.0f}건)\n")

P("="*118); P("[1] 연속손실 감축 (anti-martingale) — 연속 패배 n회 시 배수 축소"); P("="*118)
P(f"{'배수':>5s} {'고정':>16s} {'감축k0.5':>16s} {'감축k0.8':>16s} {'감축k1.0(하한0.1)':>18s}")
for lev in (3,5,8,12,20,30):
    row=[]
    for mode,k,fl in (("fixed",0,1),("anti",0.5,0.2),("anti",0.8,0.2),("anti",1.0,0.1)):
        r = eq_dyn(ev, lev, 1.0, BASE["dn"], mode, k, fl)
        row.append(f"{r['ann']:+7.1f}%/{r['mdd']:6.1f}%" if r else "         파산   ")
    P(f"{lev:4d}x " + " ".join(f"{x:>16s}" for x in row))

P("\n"+"="*118); P("[2] 변동성 목표 — 변동성 높을 때 배수 축소 (rv중앙/rv, 상한 capf)"); P("="*118)
P(f"{'배수':>5s} {'고정':>16s} {'voltgt cap1':>16s} {'voltgt cap2':>16s} {'voltgt cap3':>16s} {'anti+vol cap2':>16s}")
for lev in (3,5,8,12,20,30):
    row=[]
    for mode,cp in (("fixed",1),("voltgt",1.0),("voltgt",2.0),("voltgt",3.0),("both",2.0)):
        r = eq_dyn(ev, lev, 1.0, BASE["dn"], mode, 0.8, 0.2, capf=cp, rvser=rvser, rvmed=rvmed)
        row.append(f"{r['ann']:+7.1f}%/{r['mdd']:6.1f}%" if r else "         파산   ")
    P(f"{lev:4d}x " + " ".join(f"{x:>16s}" for x in row))

P("\n"+"="*118); P("[3] 변동성 적응 배리어 — 고정 1.5% 대신 rv × 배수 (적응형 접근)"); P("="*118)

@numba.njit(cache=True)
def tr_adapt(o,h,l,c,sig,th,kb,rv,maxh,step,cool,start,minb,maxb):
    n=len(c); K=600000
    ei=np.zeros(K,np.int64); ex=np.zeros(K,np.int64); rr=np.zeros(K); bb=np.zeros(K)
    k=0; t=start
    while t < n-maxh-2:
        i=t//step
        if i>=len(sig): break
        if not np.isfinite(sig[i]) or sig[i]<th: t+=step; continue
        if np.isnan(rv[t]) or rv[t]<=0: t+=step; continue
        b = rv[t]*kb
        if b<minb: b=minb
        if b>maxb: b=maxb
        f=t+1; px=o[f]; U=px*(1.0+b); Dn=px*(1.0-b); e=-1; res=0.0
        end=min(f+maxh,n-1)
        for j in range(f,end+1):
            if l[j]<=Dn: e=j; res=-b; break
            if h[j]>=U: e=j; res=b; break
        if e<0: e=end; res=c[e]/px-1.0
        ei[k]=f; ex[k]=e; rr[k]=res; bb[k]=b; k+=1
        t=e+1+cool
        if k>=K-10: break
    return ei[:k], ex[:k], rr[:k], bb[:k]

P(f"{'rv배수':>7s} {'연거래':>6s} {'승률%':>6s} {'평균배리어':>10s} {'1x 연%':>9s} {'3x 연%':>9s} {'5x 연%':>9s} {'5x MDD':>9s}")
th = np.quantile(d["fin"], 1-BASE["q"])
ADP=[]
for kb in (4,6,8,10,12,16,20):
    ei,ex,rr,bb = tr_adapt(d["o"],d["h"],d["l"],d["c"],d["sig"],th,float(kb),d["rv"],
                           L.MAXH,L.STEP,60,300000,0.003,0.05)
    if len(ei)/Y < 150: continue
    evA = list(zip(idx[ei], idx[ex], rr, np.zeros(len(ei),int)))
    out=[]
    for lev in (1,3,5):
        cap=1e7; cur=[]
        for (f,e,res,_),b in zip(evA, bb):
            pnl = cap*lev*(res-10.0/1e4)
            if pnl < -cap: pnl=-cap
            cap += pnl; cur.append(cap)
            if cap<=1e5: break
        eq=np.array(cur)
        if eq[-1]<=2e5: out.append(None); continue
        peak=np.maximum.accumulate(np.concatenate([[1e7],eq]))[1:]
        out.append(((eq[-1]/1e7)**(1/Y)-1)*100)
        if lev==5: mdd5=(eq/peak-1).min()*100
    if out[2] is None: mdd5=float("nan")
    fmt = lambda v: f"{v:+8.1f}" if v is not None else "    파산"
    P(f"rv×{kb:<4d} {len(ei)/Y:6.0f} {np.mean(rr>0)*100:6.1f} {np.mean(bb)*100:9.2f}% "
      f"{fmt(out[0])} {fmt(out[1])} {fmt(out[2])} {mdd5:8.1f}%")
    ADP.append((out[1] if out[1] else -99, kb, len(ei)/Y, np.mean(rr>0)*100))

P("\n"+"="*118); P("[4] 최종 후보 3단 레버리지 (감축 적용) + 연도별"); P("="*118)
for lev, mode, k, fl, nm in ((1,"fixed",0,1,"무레버리지"),(3,"fixed",0,1,"중간 고정3x"),
                             (5,"fixed",0,1,"중간+ 고정5x"),(8,"anti",0.8,0.2,"공격 8x+감축"),
                             (12,"anti",0.8,0.2,"최대공격 12x+감축"),(20,"anti",1.0,0.1,"극단 20x+강감축")):
    r = eq_dyn(ev, lev, 1.0, BASE["dn"], mode, k, fl)
    if not r: P(f"  {nm:>18s}  파산"); continue
    P(f"  {nm:>18s}  연 {r['ann']:+7.1f}%  MDD {r['mdd']:7.1f}%  최종 {r['fin']:8.2f}배")
P("\n  연도별 (8x+감축):")
out=[]
for yr in range(2020,2027):
    sub=[e for e in ev if e[0].year==yr]
    if len(sub)<3: out.append(f"{yr} n/a"); continue
    r = eq_dyn(sub, 8, 1.0, BASE["dn"], "anti", 0.8, 0.2)
    out.append(f"{yr} {(r['fin']-1)*100:+6.0f}%" if r else f"{yr} 파산")
P("    " + " | ".join(out))
