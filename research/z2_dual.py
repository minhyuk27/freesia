"""z2 확장 — ①BTC+ETH 동시 운용 ②군집 내 연속 진입 제한(쿨다운) ③국면 필터.
노출이 2.4%뿐이라는 관찰에서 출발: 기회를 늘리되 군집 손실을 억제한다."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from z2_sim import SIG, D, SYM, STEP, MAXH, MMR
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)

for s in SYM:
    d = D[s]
    lr = pd.Series(np.log(d["c"])).diff()
    d["rv"] = lr.rolling(1440).std().shift(1).values
    d["rvl"] = pd.Series(d["rv"]).rolling(43200).rank(pct=True).shift(1).values   # 변동성 국면(확장 근사)


@numba.njit(cache=True)
def trades(o,h,l,c,sig,th,up,dn,maxh,step,entry_k,rv,W,eps,cool,rvl,rv_lo,rv_hi,start):
    """진입/청산 인덱스와 수익률(수수료 전) 반환. cool: 청산 후 최소 대기 분."""
    n=len(c); K=400000
    ei=np.zeros(K,np.int64); ex=np.zeros(K,np.int64); rr=np.zeros(K); mk=np.zeros(K,np.int64)
    k=0; t=start; nskip=0
    while t < n-maxh-W-2:
        i=t//step
        if i>=len(sig): break
        if not np.isfinite(sig[i]) or sig[i]<th: t+=step; continue
        if rv_lo>0.0 or rv_hi<1.0:
            if np.isnan(rvl[t]) or rvl[t]<rv_lo or rvl[t]>rv_hi: t+=step; continue
        if entry_k>0.0:
            if np.isnan(rv[t]) or rv[t]<=0: t+=step; continue
            T=c[t]*(1.0-rv[t]*entry_k); f=-1
            for u in range(t+1,min(t+1+W,n)):
                if l[u]<=T*(1.0-eps): f=u; break
            if f<0: t+=W; nskip+=1; continue
            px=T; m=1
        else:
            f=t+1; px=o[f]; m=0
        U=px*(1.0+up); Dn=px*(1.0-dn); e=-1; res=0.0
        end=min(f+maxh,n-1); mo=0
        for j in range(f,end+1):
            if l[j]<=Dn: e=j; res=-dn; mo=0; break
            if h[j]>=U: e=j; res=up; mo=1; break
        if e<0: e=end; res=c[e]/px-1.0; mo=0
        ei[k]=f; ex[k]=e; rr[k]=res; mk[k]=m*2+mo; k+=1
        t=e+1+cool
        if k>=K-10: break
    return ei[:k], ex[:k], rr[:k], mk[:k], nskip


def equity(ev, cap0, lev, frac, mmr, fee_m, fee_t, dn):
    """시간순 정렬된 (t_enter, t_exit, res, mk) 목록으로 자본 곡선. 단일 포지션 전제."""
    cap = cap0; cur = []; nliq = 0
    liq_d = 1.0/lev - mmr
    for (f, e, res, m) in ev:
        r = res
        if r <= -dn and liq_d < dn:        # 손절 전에 청산
            r = -1.0/lev; nliq += 1
        fin = fee_m if (m//2) == 1 else fee_t
        fout = fee_m if (m % 2) == 1 else fee_t
        pnl = cap*frac*lev*(r - (fin+fout)/1e4)
        if pnl < -cap*frac: pnl = -cap*frac
        cap += pnl
        cur.append(cap)
        if cap <= cap0*0.01: break
    return np.array(cur), nliq


def rep(name, ev, yrs, lev, frac, dn, fee_m=2.0, fee_t=5.0, cap0=1e7):
    if len(ev) < 20: return None
    eq, nliq = equity(ev, cap0, lev, frac, MMR, fee_m, fee_t, dn)
    fin = eq[-1]
    if fin <= cap0*0.02: return (name, len(ev), -99.9, -100.0, 0, nliq)
    ann = (fin/cap0)**(1/yrs)-1
    mdd = (eq/np.maximum.accumulate(np.concatenate([[cap0],eq]))[1:]-1).min()
    wr = np.mean([r > 0 for _,_,r,_ in ev])
    return (name, len(ev), ann*100, mdd*100, wr*100, nliq)


yrs = (D["BTCUSDT"]["idx"][-1]-D["BTCUSDT"]["idx"][0]).days/365
FIN = {s: SIG[s][np.isfinite(SIG[s])] for s in SYM}

def get(s, q, ek, up, dn, cool=0, rv_lo=0.0, rv_hi=1.0):
    d = D[s]; th = np.quantile(FIN[s], 1-q)
    ei,ex,rr,mk,nsk = trades(d["o"],d["h"],d["l"],d["c"],SIG[s],th,up,dn,MAXH,STEP,float(ek),
                             np.asarray(d["rv"],float),240,0.0002,cool,
                             np.asarray(d["rvl"],float),rv_lo,rv_hi,300000)
    ix = d["idx"]
    return [(ix[a], ix[b], r, m) for a,b,r,m in zip(ei,ex,rr,mk)]

def merge(evs):
    """여러 종목 이벤트를 시간순 병합하되 단일 포지션 유지(겹치면 건너뜀)."""
    al = sorted([e for L in evs for e in L], key=lambda x: x[0])
    out = []; last = None
    for e in al:
        if last is not None and e[0] <= last: continue
        out.append(e); last = e[1]
    return out

P("="*120)
P("[1] BTC 단독 vs ETH 단독 vs BTC+ETH 병합 (단일 포지션 유지)")
P("="*120)
P(f"{'구성':34s} {'거래':>6s} {'연':>5s} {'연수익%':>8s} {'MDD%':>8s} {'승률%':>6s} {'청산':>5s}")
for q, ek, up, dn, lv in [(0.02,0.5,0.01,0.01,5),(0.05,0.5,0.01,0.01,5),(0.10,0.5,0.01,0.01,3),(0.05,0.0,0.02,0.01,5)]:
    tag = f"q{int(q*100)}% k{ek} {int(up*100)}/{int(dn*100)} {lv}x"
    B = get("BTCUSDT", q, ek, up, dn); E = get("ETHUSDT", q, ek, up, dn)
    for nm, ev in (("BTC "+tag, B), ("ETH "+tag, E), ("BTC+ETH "+tag, merge([B,E]))):
        r = rep(nm, ev, yrs, lv, 1.0, dn)
        if r: P(f"{r[0]:34s} {r[1]:6d} {r[1]/yrs:5.0f} {r[2]:8.1f} {r[3]:8.1f} {r[4]:6.1f} {r[5]:5d}")
    P("")

P("="*120)
P("[2] 군집 억제 — 청산 후 쿨다운 (BTC+ETH, q2% k0.5 1/1, 5x)")
P("="*120)
P(f"{'쿨다운':>8s} {'거래':>6s} {'연':>5s} {'연수익%':>8s} {'MDD%':>8s} {'승률%':>6s}")
for cool in (0, 60, 240, 1440, 4320, 10080):
    B = get("BTCUSDT",0.02,0.5,0.01,0.01,cool); E = get("ETHUSDT",0.02,0.5,0.01,0.01,cool)
    r = rep("", merge([B,E]), yrs, 5, 1.0, 0.01)
    if r: P(f"{cool:7d}분 {r[1]:6d} {r[1]/yrs:5.0f} {r[2]:8.1f} {r[3]:8.1f} {r[4]:6.1f}")

P("\n"+"="*120)
P("[3] 변동성 국면 필터 (BTC+ETH, q5% k0.5 1/1, 5x)")
P("="*120)
P(f"{'국면':>16s} {'거래':>6s} {'연':>5s} {'연수익%':>8s} {'MDD%':>8s} {'승률%':>6s}")
for lo, hi, nm in [(0.0,1.0,"전체"),(0.0,0.3,"저변동 하위30%"),(0.0,0.5,"하위50%"),
                   (0.3,0.7,"중간30~70%"),(0.5,1.0,"상위50%"),(0.7,1.0,"고변동 상위30%")]:
    B = get("BTCUSDT",0.05,0.5,0.01,0.01,0,lo,hi); E = get("ETHUSDT",0.05,0.5,0.01,0.01,0,lo,hi)
    r = rep("", merge([B,E]), yrs, 5, 1.0, 0.01)
    if r: P(f"{nm:>16s} {r[1]:6d} {r[1]/yrs:5.0f} {r[2]:8.1f} {r[3]:8.1f} {r[4]:6.1f}")

P("\n"+"="*120)
P("[4] 최적 후보 3단 레버리지 + 연도별")
P("="*120)
B = get("BTCUSDT",0.02,0.5,0.01,0.01); E = get("ETHUSDT",0.02,0.5,0.01,0.01)
EV = merge([B,E])
P(f"BTC+ETH q2% k0.5 ±1% — 거래 {len(EV)}건 (연 {len(EV)/yrs:.0f}건)")
P(f"{'구간':>10s} {'배수':>5s} {'연수익%':>8s} {'MDD%':>8s} {'청산':>5s}")
for nm, lv in (("무레버리지",1),("중간",3),("중간+",5),("공격",10),("최대공격",20)):
    r = rep("", EV, yrs, lv, 1.0, 0.01)
    if r: P(f"{nm:>10s} {lv:4d}x {r[2]:8.1f} {r[3]:8.1f} {r[5]:5d}")
P("\n연도별 (5배):")
for yr in range(2020, 2027):
    sub = [e for e in EV if e[0].year == yr]
    if len(sub) < 5: P(f"  {yr}: 거래 {len(sub)}건 (부족)"); continue
    eq,_ = equity(sub, 1e7, 5, 1.0, MMR, 2.0, 5.0, 0.01)
    P(f"  {yr}: {len(sub):4d}건  {(eq[-1]/1e7-1)*100:+8.1f}%")
