"""양방향 구조의 핵심 조건 직접 측정: 가격이 d(=0.6%)만큼 간 뒤, 되돌려 반대쪽 d를 치기 전에 d를 더 갈 확률.
이 확률 p가 손익분기 조건을 결정한다. 트리거별·배율별로 잰다."""
import sys, warnings
from pathlib import Path
import numba, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R))
from freesia import store
from freesia.bt import ind
TE = pd.Timestamp("2024-01-01", tz="UTC")
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)


@numba.njit(cache=True)
def cont(hi, lo, cl, ent, d, H):
    """첫 다리가 죽은(=d 도달) 뒤, 살아남은 다리가 반대쪽 d를 맞기 전에 추가 d를 가는가?"""
    n = len(cl); won = 0; lost = 0; none = 0
    nxt = 0
    for i in range(n - 2):
        if (not ent[i]) or i < nxt:
            continue
        p0 = cl[i]; first = 0; anchor = 0.0
        end = min(i + 1 + H, n)
        for j in range(i + 1, end):
            if first == 0:
                if hi[j] / p0 - 1.0 >= d:
                    first = 1; anchor = p0 * (1 + d)
                elif 1.0 - lo[j] / p0 >= d:
                    first = -1; anchor = p0 * (1 - d)
            else:
                if first == 1:
                    if hi[j] / anchor - 1.0 >= d:
                        won += 1; nxt = j; break
                    if 1.0 - lo[j] / anchor >= d:
                        lost += 1; nxt = j; break
                else:
                    if 1.0 - lo[j] / anchor >= d:
                        won += 1; nxt = j; break
                    if hi[j] / anchor - 1.0 >= d:
                        lost += 1; nxt = j; break
        else:
            if first == 0:
                none += 1
            nxt = end
    return won, lost, none


for sym in ("BTCUSDT", "ETHUSDT"):
    k = store.load("klines_1m", sym).set_index("ts")
    idx = k.index; hi, lo, cl, n_ = k.high.values, k.low.values, k.close.values, k.trades.values
    lp = np.log(cl); r1 = pd.Series(lp).diff()
    rv60 = r1.rolling(60).std(); q = rv60.rolling(10080).rank(pct=True)
    c5 = pd.Series(cl, index=idx).resample("5min").last(); h5 = pd.Series(hi, index=idx).resample("5min").max(); l5 = pd.Series(lo, index=idx).resample("5min").min()
    bm, bu, bl = ind.bbands(c5, 20, 2.0); km, ku, kl = ind.keltner(h5, l5, c5, 20, 1.5)
    sq = ((bu < ku) & (bl > kl)); fired = (sq.shift(1) & ~sq).reindex(idx, method="ffill").fillna(False).values
    tz = pd.Series(n_).rolling(60).mean() / pd.Series(n_).rolling(1440).mean()
    trig = {"무조건": np.ones(len(k), bool), "저변동 하위20%": (q < 0.2).values, "고변동 상위20%": (q > 0.8).values,
            "스퀴즈 해제": fired, "체결급증>2": (tz > 2).values,
            "미국장 13~16": ((idx.hour >= 13) & (idx.hour < 16))}
    P("=" * 84 + f"\n### {sym}  '계속 갈 확률' p — 손익분기 p* 와 비교")
    P(f"{'트리거':18s} {'배율':>4} {'d%':>6} {'계속':>7} {'되돌림':>7} {'p':>7} {'p*':>7} {'기대%':>8}")
    for L, H in ((100, 240), (50, 720), (25, 1440)):
        d = 1.0 / L - 0.004
        fee = 3.0 * L * 0.0005          # 진입2 + 청산1 다리 수수료(증거금 배수)
        # 이기면 살아남은 다리 +2Ld-1-fee, 지면 -1-1-fee (양쪽 다 죽음)
        winv = (2 * L * d - 1 - fee) / 2.0
        losv = (-2 - fee) / 2.0
        pstar = -losv / (winv - losv)
        for nm, m in trig.items():
            mm = np.nan_to_num(m, nan=0).astype(bool).copy(); mm[:11000] = False
            w, l, nn = cont(hi, lo, cl, mm, d, H)
            if w + l < 100:
                continue
            p = w / (w + l)
            ev = (p * winv + (1 - p) * losv) * 100
            P(f"{nm:18s} {L:>4} {d*100:6.2f} {w:>7} {l:>7} {p:7.3f} {pstar:7.3f} {ev:8.2f}")
