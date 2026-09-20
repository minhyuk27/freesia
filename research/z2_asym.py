"""z2 비대칭 배리어 정밀 검증 + 2026 붕괴 원인.
앞선 표의 +106.9% 한 셀이 진짜인지, 이웃 셀과 연도별로 확인한다."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
import z2_dual as Z
from z2_sim import D, SYM, MMR
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
yrs = Z.yrs; S = "BTCUSDT"

P("="*126)
P("[A] 익절/손절 비율 × 쿨다운 전면 (BTC 상위10%, 시장가, 5배) — 셀 = 연수익%/승률%/거래수")
P("="*126)
COOLS = (0, 60, 240, 720, 1440, 4320)
P(f"{'익절/손절':>11s} " + " ".join(f"{'쿨'+str(c):>17s}" for c in COOLS))
grid = []
for up, dn in [(0.01,0.01),(0.015,0.01),(0.02,0.01),(0.03,0.01),(0.02,0.015),(0.03,0.02),(0.01,0.005)]:
    row = []
    for cool in COOLS:
        ev = Z.get(S, 0.10, 0.0, up, dn, cool)
        r = Z.rep("", ev, yrs, 5, 1.0, dn)
        if r is None: row.append("        -        "); continue
        row.append(f"{r[2]:+7.1f}/{r[4]:4.1f}/{r[1]:4d}")
        grid.append((r[2], up, dn, cool, r[1], r[3], r[4]))
    P(f"{up*100:5.1f}/{dn*100:4.1f}% " + " ".join(f"{x:>17s}" for x in row))

P("\n[B] 같은 표, 상위5%")
P(f"{'익절/손절':>11s} " + " ".join(f"{'쿨'+str(c):>17s}" for c in COOLS))
for up, dn in [(0.01,0.01),(0.015,0.01),(0.02,0.01),(0.03,0.01),(0.03,0.02)]:
    row = []
    for cool in COOLS:
        ev = Z.get(S, 0.05, 0.0, up, dn, cool)
        r = Z.rep("", ev, yrs, 5, 1.0, dn)
        row.append(f"{r[2]:+7.1f}/{r[4]:4.1f}/{r[1]:4d}" if r else "        -        ")
    P(f"{up*100:5.1f}/{dn*100:4.1f}% " + " ".join(f"{x:>17s}" for x in row))

P("\n"+"="*126)
P("[C] 상위 후보들의 연도별 — 한 해에 몰렸는지 확인 (5배)")
P("="*126)
grid.sort(reverse=True)
seen = []
for a, up, dn, cool, n, mdd, wr in grid[:6]:
    ev = Z.get(S, 0.10, 0.0, up, dn, cool)
    ys = []
    for yr in range(2020, 2027):
        sub = [e for e in ev if e[0].year == yr]
        if len(sub) < 3: ys.append("  n/a"); continue
        eq,_ = Z.equity(sub, 1e7, 5, 1.0, MMR, 2.0, 5.0, dn)
        ys.append(f"{(eq[-1]/1e7-1)*100:+5.0f}")
    pos = sum(1 for x in ys if x.strip() not in ("n/a",) and float(x) > 0)
    P(f"  익절{up*100:.1f}/손절{dn*100:.1f} 쿨{cool:5d} 연{a:+7.1f}% MDD{mdd:6.1f}% 승률{wr:4.1f}% n={n:4d} | "
      + " ".join(ys) + f" | 양수 {pos}/7")

P("\n"+"="*126)
P("[D] 2026년 붕괴 — 신호 분포가 변했나 (상위5% 지정가k0.5 쿨60, 5배 기준)")
P("="*126)
ev = Z.get(S, 0.05, 0.5, 0.01, 0.01, 60)
df = pd.DataFrame({"t":[e[0] for e in ev], "r":[e[2] for e in ev]})
df["y"] = df.t.dt.year
P(f"{'연도':>6s} {'거래':>5s} {'승률%':>7s} {'평균bp':>8s} {'최대연속손실':>12s}")
for yr, g in df.groupby("y"):
    rr = g.r.values
    mx = 0; c = 0
    for x in rr:
        c = c+1 if x <= 0 else 0
        mx = max(mx, c)
    P(f"{yr:6d} {len(g):5d} {np.mean(rr>0)*100:7.2f} {np.mean(rr)*1e4:+8.1f} {mx:12d}")

P("\n  2026 월별:")
d26 = df[df.y==2026].copy(); d26["m"] = d26.t.dt.month
for m, g in d26.groupby("m"):
    P(f"    {m:2d}월: {len(g):3d}건 승률 {np.mean(g.r>0)*100:5.1f}% 평균 {np.mean(g.r)*1e4:+7.1f}bp")

P("\n  ±1% 어느 쪽이 먼저 닿는지의 기저율(무조건 롱, 15분 격자 표본):")
d = D[S]
import numba
@numba.njit(cache=True)
def base(h,l,c,step,maxh):
    n=len(c); w=0; t=0; tot=0
    out_w=np.zeros(20,np.int64); out_t=np.zeros(20,np.int64)
    for i in range(0, n-maxh-1, step):
        px=c[i]; U=px*1.01; Dn=px*0.99
        for j in range(i+1, i+maxh):
            if l[j]<=Dn: t+=1; tot+=1; break
            if h[j]>=U: w+=1; tot+=1; break
    return w, tot
for yr in range(2020, 2027):
    m = (d["idx"].year == yr)
    ii = np.where(m)[0]
    if len(ii) < 100000: continue
    a,b = ii[0], ii[-1]
    w,tot = base(d["h"][a:b], d["l"][a:b], d["c"][a:b], 60, 4320)
    P(f"    {yr}: 상승 먼저 {w/max(tot,1)*100:5.2f}%  (n={tot})")
