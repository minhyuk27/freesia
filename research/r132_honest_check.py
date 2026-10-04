"""R132-3 정직 점검: D1(창 시작) 방향 생존 후보를 결판 안 난 창까지 포함한 실제 매매로 재평가.
side2pct 라벨은 ±2%가 실제로 난 창에만 붙어 있어 사후 조건화다 → 진입 close[t−1], ±X 대칭 장벽(H분, 미결판은 시간 청산), 비용 승 9bp·패 14bp.
사용: python research/r132_honest_check.py"""
import sys; sys.path.insert(0, "."); import pandas as pd, numpy as np
from freesia import store
D=pd.read_parquet("data/r132_D1.parquet", columns=["sym","t","yr","p","tier","side2pct","bar10","TA_15m_ema9v21","TA_5m_ema21v50","TA_15m_ichi_cloud_pos","EXT_cbp15","EXT_cbp60","LVL_lo24h_dist","TA_5m_rsi6"])
tr=D[D.yr<=2024]
K={s:store.load("klines_1m",s) for s in ("BTCUSDT","ETHUSDT")}
def trade(row, sd, X, H):
    k=K[row.sym]; o,h,l,c=k.open.values,k.high.values,k.low.values,k.close.values
    i=int(row.t); px=c[i-1]; up=px*(1+X); dn=px*(1-X)
    for j in range(i, i+H):
        hu=h[j]>=up; hd=l[j]<=dn
        if hu and hd: return -X-14e-4  # 모호 → 보수적으로 손실
        if hu: return (X-9e-4) if sd>0 else (-X-14e-4)
        if hd: return (X-9e-4) if sd<0 else (-X-14e-4)
    return sd*(c[i+H-1]/px-1)-14e-4
for f in ("TA_15m_ema9v21","TA_5m_ema21v50","TA_15m_ichi_cloud_pos","TA_5m_rsi6","EXT_cbp60"):
    med=np.nanmedian(tr[f]); 
    for X,H in ((0.02,240),(0.01,240),(0.01,480)):
        out=[]
        for r in D.itertuples():
            v=getattr(r,f)
            if not np.isfinite(v): continue
            if D[f].nunique()<=3:
                if v==0: continue
                sd=int(np.sign(v))
            else: sd=-1 if v<med else 1
            out.append((r.yr, sd, r.tier, trade(r,sd,X,H)))
        O=pd.DataFrame(out,columns=["yr","sd","tier","r"])
        s=" ".join(f"{y%100}:{g.r.mean()*1e4:+.0f}" for y,g in O.groupby("yr"))
        te=O[O.yr>=2025]; dn=te[te.sd<0]; T10=te[te.tier=="T10"]
        print(f"{f:22s} X{X*100:.0f}% H{H}: 전체 {O.r.mean()*1e4:+.1f}bp | 시험25-26 {te.r.mean()*1e4:+.1f}bp(n{len(te)}) 숏만 {dn.r.mean()*1e4:+.1f}bp T10 {T10.r.mean()*1e4:+.1f}bp | 연 {s}")

for f,q in (("EXT_cbp15",.1),("EXT_cbp60",.1),("LVL_lo24h_dist",.1),("TA_5m_rsi6",.2)):
    e=np.nanquantile(tr[f],q)
    for X,H in ((0.02,240),(0.01,480)):
        O=pd.DataFrame([(r.yr,trade(r,-1,X,H)) for r in D.itertuples() if getattr(r,f)<=e],columns=["yr","r"])
        te=O[O.yr>=2025]
        print(f"{f:16s} 하위{int(q*100)}% 숏 X{X*100:.0f}%: 학습 {O[O.yr<=2024].r.mean()*1e4:+.1f}bp(n{(O.yr<=2024).sum()}) 시험 {te.r.mean()*1e4:+.1f}bp(n{len(te)}) 승률 {(te.r>0).mean():.2f} | "+" ".join(f"{y%100}:{g.r.mean()*1e4:+.0f}" for y,g in O.groupby("yr")))
