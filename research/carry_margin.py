"""델타중립 캐리의 증거금 최소화 — 명목 = 자본 − 선물증거금 이므로 증거금을 줄일수록 수익이 커진다.
현물(코인)을 선물 지갑으로 이체(멀티에셋 담보, 헤어컷 5%)하는 규칙을 넣고 청산 여부를 실측.
사용: python research/carry_margin.py"""
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
HAIR = 0.95          # 코인 담보 헤어컷
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)


@numba.njit(cache=True)
def sim(mark, m0, thr, lag, topup):
    """자본 1. 현물 코인 N=1-m0 매수, 무기한 숏 명목 N. 선물 지갑 현금 m0.
    증거금비율 < thr 이면 lag분 뒤 현물에서 topup(자본 비율)만큼 이체(헤어컷 적용).
    반환 (청산, 최저증거금비율, 이체횟수, 잔여현물비율)"""
    N = 1.0 - m0
    cash = m0
    spot = N                       # 코인 수량(진입가 기준 가치)
    pend = -1
    minr = 1e9
    ntr = 0
    p0 = mark[0]
    for i in range(1, len(mark)):
        mv = mark[i] / p0 - 1.0
        pnl = -N * mv                       # 숏 미실현 손익
        bal = cash + pnl + (N - spot) * 0.0  # 현금 + 손익
        # 이체된 코인 담보 반영
        coll = (N - spot) * (1.0 + mv) * HAIR
        bal = cash + pnl + coll
        maint = N * (1.0 + mv) * MMR
        if bal <= maint:
            return 1, minr, ntr, spot
        r = bal / (N * (1.0 + mv))
        if r < minr:
            minr = r
        if pend >= 0 and i >= pend:
            t = topup if topup < spot * 0.8 else spot * 0.8
            spot -= t; ntr += 1; pend = -1
        elif r < thr and pend < 0:
            pend = i + lag
    return 0, minr, ntr, spot


for sym in ("BTCUSDT", "ETHUSDT"):
    mk = store.load("mark_1m", sym).set_index("ts").close.values
    P("=" * 88 + f"\n### {sym}  (마크가격 1분봉 {len(mk):,}개, 30일 비중복 구간 {len(mk)//43200}개)")
    starts = np.arange(0, len(mk) - 43200, 43200)
    rows = []
    for m0 in (0.50, 0.35, 0.25, 0.18, 0.12, 0.08, 0.05):
        for lag, nm in ((5, "5분"), (60, "60분")):
            liq = 0; mins = []; trs = []
            for s in starts:
                l, mr, nt, sp = sim(mk[s:s + 43200], m0, max(m0 * 0.6, 0.06), lag, m0 * 0.4)
                liq += l
                if not l:
                    mins.append(mr); trs.append(nt)
            # 청산 없는 경우의 명목·수익(캐리 연 7% 가정)
            rows.append((m0, nm, 1 - m0, liq, len(starts), np.min(mins) * 100 if mins else np.nan,
                         np.mean(trs) if trs else np.nan, (1 - m0) * 7.0))
    d = pd.DataFrame(rows, columns=["초기증거금", "이체지연", "명목", "청산", "구간수", "최저증거금%", "평균이체", "캐리7% 기준 연%"])
    P(d.round(2).to_string(index=False))
    up = pd.Series(mk)
    P(f"  30일 내 최대 상승폭: 중앙 {(up.rolling(43200).max()[::-1]/up[::-1]-1)[::-1].median()*100:.1f}% / 99분위 {(up.rolling(43200).max()[::-1]/up[::-1]-1)[::-1].quantile(.99)*100:.1f}%")
    P(f"  연속 급등(이체 불가 가정) 5분 최대 {(up.shift(-5)/up-1).max()*100:.2f}%, 60분 최대 {(up.shift(-60)/up-1).max()*100:.2f}%")
