"""완전히 다른 계열 — 롱 전용 그리드 / 마켓메이킹. 매일 거래하는 고빈도 구조.
체결 규칙(보수): 매수는 저가가 지정가를 관통해야, 청산은 매수 이후 분에서 고가가 목표를 관통해야.
같은 분 내 왕복은 인정하지 않는다(역선택 방지).
사용: python research/grid_mm.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
SYM = ("BTCUSDT", "ETHUSDT")
TE = pd.Timestamp("2024-01-01", tz="UTC"); REC = pd.Timestamp("2025-01-01", tz="UTC")
D = {}
for s in SYM:
    k = store.load("klines_1m", s).set_index("ts")
    D[s] = dict(o=k.open.values, h=k.high.values, l=k.low.values, c=k.close.values, idx=k.index)
    P(f"{s}: {len(k):,}분봉")


@numba.njit(cache=True)
def grid(o, h, l, c, g, maxinv, fee_bp, eps):
    """롱 전용 그리드. 다음 매수가 아래로 g 간격. 각 매수는 진입가×(1+g)에 지정가 청산.
    반환: 분별 실현손익(명목 1단위 기준), 분별 보유수량, 진입/청산 건수"""
    n = len(c)
    pnl = np.zeros(n)          # 실현 손익(수량 1단위 명목 기준)
    inv = np.zeros(n)          # 보유 수량
    entry = np.zeros(maxinv)   # 보유 진입가
    ni = 0
    nb = 0; ns = 0
    nextbuy = c[0]*(1.0-g)
    for t in range(1, n):
        # 청산 먼저 판정하지 않는다(보수): 같은 분에 산 것은 그 분에 못 판다
        # 1) 기존 보유분 청산
        i = 0
        while i < ni:
            tgt = entry[i]*(1.0+g)
            if h[t] >= tgt*(1.0+eps):
                pnl[t] += g - fee_bp/1e4
                ns += 1
                for j in range(i, ni-1):
                    entry[j] = entry[j+1]
                ni -= 1
            else:
                i += 1
        # 2) 신규 매수
        while ni < maxinv and l[t] <= nextbuy*(1.0-eps):
            entry[ni] = nextbuy
            ni += 1; nb += 1
            nextbuy = nextbuy*(1.0-g)
        # 3) 재기준: 보유 없으면 현재가 기준으로 사다리 재설정
        if ni == 0:
            nextbuy = c[t]*(1.0-g)
        else:
            top = entry[0]
            for j in range(1, ni):
                if entry[j] < top: top = entry[j]
            if nextbuy > top*(1.0-g):
                nextbuy = top*(1.0-g)
        inv[t] = ni
    return pnl, inv, nb, ns


@numba.njit(cache=True)
def unreal(c, inv, g, maxinv):
    return inv


def run(s, g, maxinv, fee_bp=4.0, eps=0.0):
    d = D[s]
    pnl, inv, nb, ns = grid(d["o"], d["h"], d["l"], d["c"], g, maxinv, fee_bp, eps)
    return pnl, inv, nb, ns, d


P("\n[1] 그리드 간격 × 최대 보유 — 롱 전용, 메이커 왕복 4bp")
P(f"  {'종목':8s} {'간격':>7s} {'최대보유':>7s} {'왕복':>8s} {'연 왕복':>8s} {'총수익%':>9s} {'연수익%':>8s} {'평균보유':>8s} {'최대보유':>8s}")
res = {}
for s in SYM:
    for gbp in (5, 10, 20, 30, 50, 100):
        for mi in (5, 10, 20, 50):
            g = gbp/1e4
            pnl, inv, nb, ns, d = run(s, g, mi)
            yrs = (d["idx"][-1]-d["idx"][0]).days/365
            tot = pnl.sum()*100        # 수량 1단위 명목 기준 누적 수익률(%)
            res[(s, gbp, mi)] = (pnl, inv, nb, ns, d)
            P(f"  {s[:3]:8s} {gbp:6d}bp {mi:7d} {ns:8d} {ns/yrs:8.0f} {tot:+8.1f}% {tot/yrs:+7.1f}% {inv.mean():8.2f} {inv.max():8.0f}")

P("\n[2] 핵심 문제 진단 — 미실현 손실(물린 재고)의 크기")
for s in SYM:
    for gbp, mi in ((10, 20), (30, 20), (50, 20), (100, 20)):
        pnl, inv, nb, ns, d = res[(s, gbp, mi)]
        g = gbp/1e4
        # 재고가 가득 찬 동안의 가격 하락폭 = 미실현 손실 근사
        c = d["c"]
        full = inv >= mi
        P(f"  {s[:3]} {gbp}bp/{mi}단: 재고 만재 시간 {full.mean()*100:5.1f}% | "
          f"사다리 전체 깊이 {(1-(1-g)**mi)*100:5.2f}% | 실현 연 {pnl.sum()/((d['idx'][-1]-d['idx'][0]).days/365)*100:+6.1f}%")

P("\n[3] 자본 기준 환산 — 명목 상한을 지키면 실제 수익률은?")
P("  (사다리 전체가 채워졌을 때의 명목 = 자본 × 명목배수. 각 단계 명목 = 자본×배수/단수)")
P(f"  {'종목':6s} {'간격':>6s} {'단수':>5s} {'사다리깊이':>9s} {'명목배수1x':>11s} {'명목배수3x':>11s} {'최대미실현손실':>13s}")
for s in SYM:
    d = D[s]; c = d["c"]
    for gbp, mi in ((10, 20), (20, 20), (30, 20), (50, 20), (100, 20), (30, 50), (50, 50)):
        pnl, inv, nb, ns, _ = res[(s, gbp, mi)] if (s, gbp, mi) in res else run(s, gbp/1e4, mi)[:4]+(None,)
        g = gbp/1e4
        yrs = (d["idx"][-1]-d["idx"][0]).days/365
        depth = 1-(1-g)**mi
        # 각 단계 명목 = 1/mi (총 명목 1x). 실현 수익 = ns × g / mi
        r1 = ns*(g - 4e-4)/mi
        P(f"  {s[:3]:6s} {gbp:5d}bp {mi:5d} {depth*100:8.2f}% {r1/yrs*100:+10.1f}% {r1*3/yrs*100:+10.1f}% {depth/2*100:12.2f}%")
