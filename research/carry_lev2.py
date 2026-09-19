"""결합 캐리(무기한+분기)의 레버리지 한계와 실제 운영 시뮬레이션 (Binance BTC·ETH만).
핵심 질문: 델타중립인데 현물 지갑과 선물 지갑이 분리돼 있어 급등 시 선물 증거금이 먼저 터진다.
5분마다 현물→선물 증거금 이체 규칙을 넣고 레버리지별 청산 여부를 실측한다.
사용: python research/carry_lev2.py"""
import sys
import warnings
from pathlib import Path

import numba
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from freesia import store  # noqa: E402

MMR = 0.004
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)


@numba.njit(cache=True)
def sim(mark, fund_h, L, m0, thr, add, lag):
    """현물 롱 N + 무기한 숏 N (N = L × 자본). 선물 지갑 증거금 m(자본 비율).
    급등 시 선물 손실 → 증거금 비율이 thr 아래면 lag봉 뒤 add만큼 현물에서 이체.
    반환: (청산여부, 최소증거금비율, 총 이체 횟수, 최종 자본)"""
    n = len(mark)
    eq = 1.0            # 총 자본
    m = m0              # 선물 지갑 잔고 (자본 비율)
    spot = eq - m       # 현물 지갑 (자본 비율) — 명목은 L
    entry = mark[0]
    pending = -1
    minr = 1e9
    ntr = 0
    liq = 0
    for i in range(1, n):
        move = mark[i] / entry - 1.0          # 진입 대비 가격 변화
        pnl = -L * move                        # 숏 손익(자본 비율)
        bal = m + pnl                          # 선물 지갑 잔고
        need = L * (1 + move) * MMR            # 유지증거금
        if bal <= need:
            liq = 1
            break
        r = bal / (L * (1 + move))             # 증거금 비율
        if r < minr:
            minr = r
        if pending >= 0 and i >= pending:
            avail = spot * (1 + move)          # 현물 평가액
            t = add if add < avail * 0.5 else avail * 0.5
            m += t; spot -= t / (1 + move); ntr += 1
            pending = -1
        elif r < thr and pending < 0:
            pending = i + lag
        eq += fund_h[i] * L                    # 펀딩 수취(숏)
    return liq, minr, ntr, eq


for sym in ("BTCUSDT", "ETHUSDT"):
    mk = store.load("mark_1m", sym).set_index("ts").close
    f = store.load("funding", sym).set_index("ts").rate
    fh = f.reindex(mk.index, method="ffill").fillna(0)
    fh = fh.where(fh.index.isin(f.index), 0.0).values     # 정산 시점에만 계상
    m = mk.values
    P("=" * 90 + f"\n### {sym} (마크가격 1분봉 {len(m):,}개)")
    P("30일 보유 구간을 겹치지 않게 나눠 시뮬레이션. 증거금 초기 m0, 하한 thr 미만이면 5분 뒤 add 이체")
    starts = np.arange(0, len(m) - 43200, 43200)          # 30일 = 43200분
    rows = []
    for L in (1, 2, 3, 5):
        for m0, thr, add in ((0.5, 0.35, 0.15), (0.35, 0.25, 0.10), (0.25, 0.18, 0.08), (0.15, 0.12, 0.05)):
            liqs = 0; mins = []; trs = []
            for s in starts:
                seg = m[s:s + 43200]; fseg = fh[s:s + 43200]
                l, mr, nt, eq = sim(seg, fseg, L, m0, thr, add, 5)
                liqs += l
                if not l:
                    mins.append(mr); trs.append(nt)
            rows.append((L, m0, thr, liqs, len(starts), np.min(mins) * 100 if mins else np.nan,
                         np.mean(trs) if trs else np.nan))
    d = pd.DataFrame(rows, columns=["레버리지", "초기증거금", "하한", "청산", "구간수", "최저증거금비율%", "평균이체횟수"])
    P(d.round(2).to_string(index=False))
    # 최악 상승 구간
    up30 = (pd.Series(m).rolling(43200).max()[::-1] / pd.Series(m)[::-1] - 1)[::-1]
    P(f"  참고: 30일 내 최대 상승폭 중앙 {up30.median()*100:.1f}% / 99분위 {up30.quantile(.99)*100:.1f}% / 최대 {up30.max()*100:.1f}%")
    P(f"       5분 내 최대 상승폭 99.9분위 {(pd.Series(m).shift(-5)/pd.Series(m)-1).quantile(.999)*100:.2f}% / 최대 {(pd.Series(m).shift(-5)/pd.Series(m)-1).max()*100:.2f}%")
