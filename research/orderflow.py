"""고빈도 새 방향 — 주문흐름(taker_buy_volume)과 거래량 조건부 되돌림.
이론: Llorente-Michaely-Saar-Wang(2002). 적은 거래량에 크게 밀린 가격 = 유동성 충격 → 되돌림.
      많은 거래량에 밀린 가격 = 정보 → 지속.
신호는 t분 종가까지 정보만. 진입 t+1 시가. 룩어헤드 없음.
사용: python research/orderflow.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
SYM = ("BTCUSDT", "ETHUSDT")
D = {}
for s in SYM:
    k = store.load("klines_1m", s).set_index("ts")
    d = pd.DataFrame(index=k.index)
    d["o"], d["h"], d["l"], d["c"] = k.open, k.high, k.low, k.close
    d["qv"] = k.quote_volume
    d["tb"] = k.taker_buy_quote_volume
    d["n"] = k.trades
    d["ofi"] = (2*d.tb/d.qv.replace(0, np.nan) - 1.0)          # -1(전량 매도) ~ +1(전량 매수)
    d["r"] = d.c.pct_change()
    d["qvz"] = (d.qv/d.qv.rolling(1440).mean()).shift(0)        # 최근 하루 평균 대비 거래대금
    d["amihud"] = d.r.abs()/d.qv.replace(0, np.nan)             # 가격충격(비유동성)
    d["avgsz"] = d.qv/d.n.replace(0, np.nan)                    # 평균 체결 크기
    D[s] = d
    P(f"{s}: {len(d):,}분 | OFI 중앙 {d.ofi.median():+.4f} | 거래대금 중앙 {d.qv.median():,.0f}")

HOR = (1, 3, 5, 15, 30, 60)


def fw(c, h): return c.shift(-h)/c - 1.0


P("\n" + "="*104)
P("[1] 거래량 조건부 되돌림 — LMSW 핵심 검정")
P("    직전 5분 수익률을 5분위로, 그 구간의 거래대금을 3분위로 나눠 이후 수익(bp)")
P("="*104)
for s in SYM:
    d = D[s]
    r5 = d.c.pct_change(5)
    qz = d.qv.rolling(5).sum()/d.qv.rolling(1440).mean()/5
    F = {h: fw(d.c, h) for h in HOR}
    P(f"\n  ── {s} ──")
    P(f"    {'5분수익':>10s} {'거래량':>8s} {'n':>8s} " + " ".join(f"{h:>6d}분" for h in HOR))
    rq = r5.rank(pct=True); vq = qz.rank(pct=True)
    for rlo, rhi, rlab in ((0, .05, "최하위5%"), (.05, .2, "하위20%"), (.8, .95, "상위20%"), (.95, 1.01, "최상위5%")):
        for vlo, vhi, vlab in ((0, .33, "적음"), (.33, .67, "보통"), (.67, 1.01, "많음")):
            sel = (rq >= rlo) & (rq < rhi) & (vq >= vlo) & (vq < vhi)
            n = int(sel.sum())
            if n < 500: continue
            row = " ".join(f"{F[h][sel].mean()*1e4:+6.2f}" for h in HOR)
            P(f"    {rlab:>10s} {vlab:>8s} {n:8d} {row}")
    row = " ".join(f"{F[h].mean()*1e4:+6.2f}" for h in HOR)
    P(f"    {'전체':>10s} {'':>8s} {len(d):8d} {row}")

P("\n" + "="*104)
P("[2] 주문흐름 불균형(OFI) 극단 — 공격적 매도가 쏟아진 직후")
P("="*104)
for s in SYM:
    d = D[s]
    of5 = (d.tb.rolling(5).sum()*2/d.qv.rolling(5).sum() - 1.0)
    F = {h: fw(d.c, h) for h in HOR}
    P(f"\n  ── {s} ── (5분 누적 OFI)")
    P(f"    {'OFI':>12s} {'n':>8s} " + " ".join(f"{h:>6d}분" for h in HOR))
    q = of5.rank(pct=True)
    for lo, hi, lab in ((0, .01, "최하위1%"), (.01, .05, "하위5%"), (.05, .2, "하위20%"),
                        (.8, .95, "상위20%"), (.95, .99, "상위5%"), (.99, 1.01, "최상위1%")):
        sel = (q >= lo) & (q < hi)
        n = int(sel.sum())
        if n < 500: continue
        P(f"    {lab:>12s} {n:8d} " + " ".join(f"{F[h][sel].mean()*1e4:+6.2f}" for h in HOR))

P("\n" + "="*104)
P("[3] 조합 — 급락 + 저거래량 + 매도 OFI (순수 유동성 충격 가설)")
P("="*104)
for s in SYM:
    d = D[s]
    r5 = d.c.pct_change(5); rq = r5.rank(pct=True)
    qz = (d.qv.rolling(5).sum()/d.qv.rolling(1440).mean()/5); vq = qz.rank(pct=True)
    of5 = (d.tb.rolling(5).sum()*2/d.qv.rolling(5).sum() - 1.0); oq = of5.rank(pct=True)
    am = d.amihud.rolling(5).mean(); aq = am.rank(pct=True)
    F = {h: fw(d.c, h) for h in HOR}
    P(f"\n  ── {s} ──")
    P(f"    {'조건':>34s} {'n':>7s} " + " ".join(f"{h:>6d}분" for h in HOR))
    CONDS = [
        ("급락5% + 거래량하위33%", (rq < .05) & (vq < .33)),
        ("급락5% + 거래량상위33%", (rq < .05) & (vq > .67)),
        ("급락5% + OFI하위20%", (rq < .05) & (oq < .2)),
        ("급락5% + 거래량하위33% + OFI하위20%", (rq < .05) & (vq < .33) & (oq < .2)),
        ("급락1% + 거래량하위50%", (rq < .01) & (vq < .5)),
        ("급락1% + 비유동성상위10%", (rq < .01) & (aq > .9)),
        ("비유동성 상위1% (충격 큼)", aq > .99),
        ("비유동성 상위1% & 하락", (aq > .99) & (r5 < 0)),
        ("급등5% + 거래량하위33%", (rq > .95) & (vq < .33)),
        ("급등5% + 거래량상위33%", (rq > .95) & (vq > .67)),
    ]
    for lab, sel in CONDS:
        n = int(sel.sum())
        if n < 300: continue
        P(f"    {lab:>34s} {n:7d} " + " ".join(f"{F[h][sel].mean()*1e4:+6.2f}" for h in HOR))

P("\n" + "="*104)
P("[4] 평균 체결 크기 · 체결 건수 — 고래 vs 개미")
P("="*104)
for s in SYM:
    d = D[s]
    szq = (d.avgsz/d.avgsz.rolling(1440).mean()).rank(pct=True)
    r5 = d.c.pct_change(5); rq = r5.rank(pct=True)
    F = {h: fw(d.c, h) for h in HOR}
    P(f"\n  ── {s} ──")
    for lab, sel in (("급락5% + 평균체결 큼(상위20%)", (rq < .05) & (szq > .8)),
                     ("급락5% + 평균체결 작음(하위20%)", (rq < .05) & (szq < .2)),
                     ("평균체결 최상위1%", szq > .99)):
        n = int(sel.sum())
        if n < 300: continue
        P(f"    {lab:>34s} {n:7d} " + " ".join(f"{F[h][sel].mean()*1e4:+6.2f}" for h in HOR))

P("\n[5] 비용 기준선: 메이커 왕복 4bp, 메이커+테이커 7bp, 테이커 왕복 10bp")
