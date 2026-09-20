"""그리드 진짜 검증 — 자본·청산 시뮬, 매수보유 대비 초과수익, 추세 제거 플라시보.
사용: python research/grid_valid.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
SYM = ("BTCUSDT", "ETHUSDT")
TE = pd.Timestamp("2024-01-01", tz="UTC"); REC = pd.Timestamp("2025-01-01", tz="UTC")
MMR = 0.004
D = {}
for s in SYM:
    k = store.load("klines_1m", s).set_index("ts")
    D[s] = dict(o=k.open.values, h=k.high.values, l=k.low.values, c=k.close.values, idx=k.index)


@numba.njit(cache=True)
def grid_cap(o, h, l, c, g, maxinv, fee_bp, N, mmr):
    """자본 1에서 시작. 사다리 전체 명목 = 자본×N (각 단 N/maxinv).
    보유 재고의 미실현 손실이 자본을 넘으면 청산. 실현손익은 자본에 복리."""
    n = len(c)
    cap = 1.0
    entry = np.zeros(maxinv)
    qty = np.zeros(maxinv)     # 각 단의 명목(자본 대비 비율 아님, 절대 금액)
    ni = 0
    nextbuy = c[0]*(1.0-g)
    eq = np.zeros(n); eq[0] = 1.0
    unreal_min = 0.0
    liq = 0
    nb = 0; ns = 0
    for t in range(1, n):
        if cap <= 0.0:
            eq[t] = 0.0
            continue
        # 청산(이익실현)
        i = 0
        while i < ni:
            tgt = entry[i]*(1.0+g)
            if h[t] >= tgt:
                cap += qty[i]*(g - fee_bp/1e4)
                ns += 1
                for j in range(i, ni-1):
                    entry[j] = entry[j+1]; qty[j] = qty[j+1]
                ni -= 1
            else:
                i += 1
        # 신규 매수
        while ni < maxinv and l[t] <= nextbuy:
            entry[ni] = nextbuy
            qty[ni] = cap*N/maxinv
            ni += 1; nb += 1
            nextbuy = nextbuy*(1.0-g)
        if ni == 0:
            nextbuy = c[t]*(1.0-g)
        else:
            top = entry[0]
            for j in range(1, ni):
                if entry[j] < top: top = entry[j]
            if nextbuy > top*(1.0-g):
                nextbuy = top*(1.0-g)
        # 미실현 평가 (저가 기준 최악)
        uw = 0.0; um = 0.0
        for j in range(ni):
            uw += qty[j]*(c[t]/entry[j] - 1.0)
            um += qty[j]*(l[t]/entry[j] - 1.0)
        if um < unreal_min: unreal_min = um
        if cap + um <= cap*mmr:
            cap = 0.0; liq += 1
            eq[t] = 0.0
            continue
        eq[t] = cap + uw
    return eq, liq, nb, ns, unreal_min


def stat(eq, idx, lab, liq, nb, ns):
    s = pd.Series(eq, index=idx).resample("D").last().ffill()
    if s.iloc[-1] <= 1e-6:
        P(f"  {lab:34s} 파산 (청산 {liq})"); return None
    y = (s.index[-1]-s.index[0]).days/365
    cg = (s.iloc[-1]**(1/y)-1)*100
    mdd = float((s/s.cummax()-1).min())*100
    sg = lambda t0: (lambda x: ((x.iloc[-1]/x.iloc[0])**(365/max((x.index[-1]-x.index[0]).days,1))-1)*100 if len(x) > 60 else np.nan)(s[s.index >= t0])
    mo = (s.resample("ME").last()/s.resample("ME").first()-1).dropna()
    P(f"  {lab:34s} 연 {cg:+8.1f}% MDD {mdd:6.1f}% | 24~ {sg(TE):+7.1f}% | 25~ {sg(REC):+7.1f}% | "
      f"왕복 {ns:6d} | 월승률 {float((mo>0).mean()):.2f} 월중앙 {mo.median()*100:+5.1f}%")
    return cg, mdd, s, mo


P("[1] 자본·청산 반영 그리드 (메이커 왕복 4bp)")
for s in SYM:
    P(f"  ── {s} ──")
    d = D[s]
    for gbp, mi in ((10, 20), (20, 20), (30, 20), (50, 20), (30, 50), (50, 50), (100, 20)):
        for N in (1.0, 2.0, 3.0, 5.0):
            eq, liq, nb, ns, um = grid_cap(d["o"], d["h"], d["l"], d["c"], gbp/1e4, mi, 4.0, N, MMR)
            stat(eq, d["idx"], f"{gbp}bp/{mi}단 명목{N:.0f}x", liq, nb, ns)

P("\n[2] 매수보유 대비 — 같은 평균 노출로 맞추면?")
for s in SYM:
    d = D[s]; c = d["c"]
    y = (d["idx"][-1]-d["idx"][0]).days/365
    bh = (c[-1]/c[0])**(1/y)-1
    P(f"  {s} 매수보유 연 {bh*100:+.1f}% (노출 100%)")
    for gbp, mi in ((20, 20), (30, 20), (50, 20)):
        eq, liq, nb, ns, um = grid_cap(d["o"], d["h"], d["l"], d["c"], gbp/1e4, mi, 4.0, 1.0, MMR)
        r = stat(eq, d["idx"], f"   그리드 {gbp}bp/{mi}단 1x", liq, nb, ns)
        if r: P(f"      → 매수보유 대비 {r[0]-bh*100:+.1f}%p")

P("\n[3] 추세 제거 플라시보 — 드리프트를 0으로 만든 가격에서도 돈을 버나")
P("   (같은 분봉 수익률에서 전체 평균을 빼서 드리프트 제거. 변동성·자기상관 구조는 보존)")
for s in SYM:
    d = D[s]
    c = d["c"].astype(np.float64)
    lr = np.diff(np.log(c))
    lr0 = lr - lr.mean()
    c0 = np.empty_like(c); c0[0] = c[0]
    c0[1:] = c[0]*np.exp(np.cumsum(lr0))
    sc = c0/c
    o0 = d["o"]*sc; h0 = d["h"]*sc; l0 = d["l"]*sc
    yv = (d["idx"][-1]-d["idx"][0]).days/365
    P(f"  ── {s} (원본 드리프트 연 {(np.exp(lr.mean()*525600)-1)*100:+.1f}% → 0%) ──")
    for gbp, mi in ((20, 20), (30, 20), (50, 20)):
        eq, liq, nb, ns, um = grid_cap(o0, h0, l0, c0, gbp/1e4, mi, 4.0, 1.0, MMR)
        stat(eq, d["idx"], f"   추세제거 {gbp}bp/{mi}단 1x", liq, nb, ns)

P("\n[4] 무작위 재배열 플라시보 — 분봉 순서를 섞으면 (자기상관 파괴)")
rng = np.random.default_rng(3)
for s in SYM:
    d = D[s]
    c = d["c"].astype(np.float64)
    lr = np.diff(np.log(c))
    for it in range(2):
        pm = rng.permutation(len(lr))
        lrs = lr[pm]
        cs = np.empty_like(c); cs[0] = c[0]; cs[1:] = c[0]*np.exp(np.cumsum(lrs))
        rng_hl = (d["h"]/d["c"])[1:][pm]; rl = (d["l"]/d["c"])[1:][pm]
        hs = np.empty_like(c); ls = np.empty_like(c)
        hs[0] = d["h"][0]; ls[0] = d["l"][0]
        hs[1:] = cs[1:]*rng_hl; ls[1:] = cs[1:]*rl
        os_ = np.empty_like(c); os_[0] = d["o"][0]; os_[1:] = cs[:-1]
        eq, liq, nb, ns, um = grid_cap(os_, hs, ls, cs, 30/1e4, 20, 4.0, 1.0, MMR)
        stat(eq, d["idx"], f"  {s[:3]} 섞음 #{it+1} 30bp/20단 1x", liq, nb, ns)
