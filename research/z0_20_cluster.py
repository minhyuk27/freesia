"""z0.20 군집 구조 — 군집 내 순번별 승률, 쿨다운 최적점, 3단 레버리지.
가설: 신호 군집의 첫 거래만 엣지가 있고 후속은 같은 국면 재탕이라 손실."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
import z0_20_dual as Z
from z0_20_sim import D, SYM, MMR
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
yrs = Z.yrs

P("="*120)
P("[A] 군집 내 순번별 승률 — 직전 청산 후 GAP분 이내 진입이면 같은 군집")
P("="*120)
for s in ("BTCUSDT","ETHUSDT"):
    for q in (0.02, 0.05):
        ev = Z.get(s, q, 0.5, 0.01, 0.01)
        for GAP in (1440,):
            seq = []; k = 0; last = None
            for e in ev:
                if last is None or (e[0]-last).total_seconds()/60 > GAP: k = 1
                else: k += 1
                seq.append(k); last = e[1]
            seq = np.array(seq); rr = np.array([e[2] for e in ev])
            P(f"  {s} 상위{int(q*100)}% (군집기준 {GAP}분, 총 {len(ev)}건, 군집 {int((seq==1).sum())}개)")
            for lo, hi, nm in ((1,1,"1번째(군집 첫)"),(2,2,"2번째"),(3,3,"3번째"),(4,5,"4~5번째"),(6,999,"6번째 이상")):
                m = (seq>=lo)&(seq<=hi)
                if m.sum() < 5: continue
                P(f"    {nm:14s} n={m.sum():4d}  승률 {np.mean(rr[m]>0)*100:5.2f}%  평균수익 {np.mean(rr[m])*1e4:+7.1f}bp")
        P("")

P("="*120)
P("[B] BTC 단독 — 쿨다운 × 진입분위 (5배, 지정가k0.5 ±1%)")
P("="*120)
P(f"{'분위':>6s} " + " ".join(f"{c:>13s}" for c in ("쿨0","쿨60","쿨240","쿨1440","쿨4320","쿨10080")))
best = []
for q in (0.01, 0.02, 0.05, 0.10, 0.20):
    row = []
    for cool in (0,60,240,1440,4320,10080):
        ev = Z.get("BTCUSDT", q, 0.5, 0.01, 0.01, cool)
        r = Z.rep("", ev, yrs, 5, 1.0, 0.01)
        if r is None: row.append("     -       "); continue
        row.append(f"{r[2]:+6.1f}%/{r[4]:4.1f}%")
        best.append((r[2], q, cool, r[1], r[3], r[4]))
    P(f"{q*100:5.0f}% " + " ".join(f"{x:>13s}" for x in row))
P("  (셀 = 연수익% / 승률%)")

best.sort(reverse=True)
P("\n  상위 5개 구성:")
for a,q,cool,n,mdd,wr in best[:5]:
    P(f"    상위{q*100:.0f}% 쿨{cool}분: 연 {a:+.1f}%  MDD {mdd:.1f}%  승률 {wr:.1f}%  거래 {n}건(연 {n/yrs:.0f})")

P("\n"+"="*120)
P("[C] 비대칭 배리어 + 쿨다운 (BTC, 익절2%/손절1%, 시장가)")
P("="*120)
P(f"{'분위':>6s} " + " ".join(f"{c:>13s}" for c in ("쿨0","쿨240","쿨1440","쿨4320")))
for q in (0.02, 0.05, 0.10):
    row = []
    for cool in (0,240,1440,4320):
        ev = Z.get("BTCUSDT", q, 0.0, 0.02, 0.01, cool)
        r = Z.rep("", ev, yrs, 5, 1.0, 0.01)
        row.append(f"{r[2]:+6.1f}%/{r[4]:4.1f}%" if r else "      -      ")
    P(f"{q*100:5.0f}% " + " ".join(f"{x:>13s}" for x in row))

P("\n"+"="*120)
P("[D] 최우수 구성 3단 레버리지 + 연도별 + 플라시보")
P("="*120)
if best:
    _, q, cool, _, _, _ = best[0]
    ev = Z.get("BTCUSDT", q, 0.5, 0.01, 0.01, cool)
    P(f"BTC 지정가k0.5 ±1% 상위{q*100:.0f}% 쿨다운{cool}분 — {len(ev)}건 (연 {len(ev)/yrs:.0f}건)")
    P(f"{'구간':>10s} {'배수':>5s} {'연수익%':>9s} {'MDD%':>8s} {'최종배수':>9s} {'청산':>5s}")
    for nm, lv in (("무레버리지",1),("중간",3),("중간+",5),("공격",8),("최대공격",12),("극단",20)):
        r = Z.rep("", ev, yrs, lv, 1.0, 0.01)
        if r:
            eq,_ = Z.equity(ev, 1e7, lv, 1.0, MMR, 2.0, 5.0, 0.01)
            P(f"{nm:>10s} {lv:4d}x {r[2]:9.1f} {r[3]:8.1f} {eq[-1]/1e7:9.2f} {r[5]:5d}")
    P("\n  연도별 (5배):")
    for yr in range(2020, 2027):
        sub = [e for e in ev if e[0].year == yr]
        if len(sub) < 3: P(f"    {yr}: {len(sub)}건 (부족)"); continue
        eq,_ = Z.equity(sub, 1e7, 5, 1.0, MMR, 2.0, 5.0, 0.01)
        wr = np.mean([e[2]>0 for e in sub])*100
        P(f"    {yr}: {len(sub):4d}건  {(eq[-1]/1e7-1)*100:+8.1f}%  승률 {wr:5.1f}%")
    rr = np.array([e[2] for e in ev])
    P(f"\n  전체 승률 {np.mean(rr>0)*100:.2f}%  (±1% 배리어, 표본 {len(rr)})")
    se = np.sqrt(0.25/len(rr))*100
    P(f"  50% 대비 표준오차 {se:.2f}%p → z = {(np.mean(rr>0)*100-50)/se:.2f}")
