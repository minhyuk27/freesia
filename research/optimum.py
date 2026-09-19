"""최적점 탐색 — 저변동 역행 되돌림 단일 전략, 캐리 제외.
변수: 종목별 명목 배수, 결합 방식(동시 진입 시 합산/분할), 체결 가정.
출력: 로그성장 최적, MDD 제약별 최적, 그리고 최적점의 최근 12개월 월별 성과.
사용: python research/optimum.py"""
import sys, warnings
from pathlib import Path
import numba, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R))
from freesia import store
TE = pd.Timestamp("2024-01-01", tz="UTC")
MMR = 0.004
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
B = 240


@numba.njit(cache=True)
def trades(o, h, l, c, B, xarr, cond, buf):
    nb = len(c) // B
    ret = np.full(nb, np.nan); mae = np.full(nb, np.nan)
    for b in range(nb):
        if not cond[b] or np.isnan(xarr[b]) or xarr[b] <= 0:
            continue
        s = b * B; e = s + B
        O = o[s]; x = xarr[b]
        trig = O * (1.0 - x); fill = trig * (1.0 - buf)
        j = -1
        for t in range(s, e):
            if l[t] <= fill:
                j = t; break
        if j < 0:
            continue
        lo_ = trig
        for t in range(j, e):
            if l[t] < lo_:
                lo_ = l[t]
        ret[b] = (c[e - 1] / trig - 1.0) * 1e4 - 7.0
        mae[b] = (1.0 - lo_ / trig) * 1e4
    return ret, mae


DATA = {}
for sym in ("BTCUSDT", "ETHUSDT"):
    k = store.load("klines_1m", sym).set_index("ts")
    o, h, l, c = k.open.values, k.high.values, k.low.values, k.close.values
    idx = k.index; nb = len(c) // B
    bo = o[::B][:nb]; bc = c[B - 1::B][:nb]
    bh = np.array([h[i * B:(i + 1) * B].max() for i in range(nb)])
    bl = np.array([l[i * B:(i + 1) * B].min() for i in range(nb)])
    bi = idx[::B][:nb]
    s = pd.Series
    xa = (s((bh - bl) / bo).shift(1) * 1.0).values
    rv = s(np.log(bc)).diff().rolling(24).std()
    cd = (rv < rv.rolling(720).quantile(0.3)).fillna(False).values
    DATA[sym] = {}
    for buf, bn in ((0.0, "낙관"), (0.0005, "보수")):
        ret, mae = trades(o, h, l, c, B, xa, cd, buf)
        DATA[sym][bn] = (ret, mae, bi)


def sim(nb_, wB, wE, fill, cap=None):
    """자본 1. 봉별로 BTC 명목 wB, ETH 명목 wE. 청산거리는 (자본/총명목)-mmr (교차마진)."""
    rB, mB, bi = DATA["BTCUSDT"][fill]
    rE, mE, _ = DATA["ETHUSDT"][fill]
    n = min(len(rB), len(rE))
    eq = 1.0
    out = np.zeros(n); dates = bi[:n]
    for i in range(n):
        hasB = not np.isnan(rB[i]); hasE = not np.isnan(rE[i])
        if not hasB and not hasE:
            continue
        nB = wB if hasB else 0.0; nE = wE if hasE else 0.0
        tot = nB + nE
        if cap is not None and tot > cap:
            sc = cap / tot; nB *= sc; nE *= sc; tot = cap
        d = (1.0 / tot - MMR) * 1e4 if tot > 0 else 1e9      # 교차마진 청산거리(bp)
        liq = (hasB and mB[i] >= d) or (hasE and mE[i] >= d)
        if liq:
            pnl = -1.0
        else:
            pnl = (nB * (rB[i] if hasB else 0) + nE * (rE[i] if hasE else 0)) / 1e4
        out[i] = pnl
        eq *= (1 + pnl)
        if eq <= 0.01:
            return out[:i + 1], dates[:i + 1], True
    return out, dates, False


P("=" * 100 + "\n[1] 단일 종목 명목 배수별 (낙관 체결, 전체 기간)")
for sym in ("BTCUSDT", "ETHUSDT"):
    r, ma, bi = DATA[sym]["낙관"]
    ok = ~np.isnan(r)
    rows = []
    for N in (1, 2, 3, 4, 5, 6, 8, 10, 12, 15):
        d = (1.0 / N - MMR) * 1e4
        liq = ma[ok] >= d
        pnl = np.where(liq, -1.0, N * r[ok] / 1e4)
        eq = np.cumprod(1 + pnl)
        ruin = bool((eq <= 0.01).any())
        te = bi[ok] >= TE
        eq2 = np.cumprod(1 + pnl[te])
        yrs = (bi[ok][-1] - TE).days / 365
        rows.append((N, d, int(liq.sum()), "파산" if ruin else f"{eq[-1]:.1f}배",
                     float((eq / np.maximum.accumulate(eq) - 1).min()) * 100,
                     "파산" if (eq2 <= 0.01).any() else f"{(eq2[-1]**(1/yrs)-1)*100:+.1f}%",
                     float((eq2 / np.maximum.accumulate(eq2) - 1).min()) * 100,
                     np.mean(np.log1p(pnl)) * 1e4))
    P(f"\n  {sym}")
    P(pd.DataFrame(rows, columns=["명목N", "청산거리bp", "청산", "전체누적", "전체MDD%", "테스트 연환산", "테스트MDD%", "로그성장bp"]).round(2).to_string(index=False))

P("\n" + "=" * 100 + "\n[2] 결합 — BTC·ETH 명목 배분 격자 (낙관, 총명목 상한 없음, 교차마진)")
rows = []
for wB in (0, 1, 2, 3, 4, 5):
    for wE in (0, 1, 2, 3, 4, 5, 6):
        if wB + wE == 0:
            continue
        pnl, dates, ruin = sim(None, wB, wE, "낙관")
        if ruin:
            rows.append((wB, wE, "파산", np.nan, np.nan, np.nan)); continue
        eq = np.cumprod(1 + pnl)
        te = dates >= TE
        eq2 = np.cumprod(1 + pnl[te]); yrs = (dates[-1] - TE).days / 365
        rows.append((wB, wE, f"{eq[-1]:.1f}배", (eq2[-1] ** (1 / yrs) - 1) * 100,
                     float((eq2 / np.maximum.accumulate(eq2) - 1).min()) * 100,
                     np.mean(np.log1p(pnl)) * 1e4))
d = pd.DataFrame(rows, columns=["BTC명목", "ETH명목", "전체누적", "테스트연%", "테스트MDD%", "로그성장bp"])
P(d.dropna().sort_values("로그성장bp", ascending=False).head(12).round(2).to_string(index=False))
P("\n  MDD -30% 이내:")
P(d[(d["테스트MDD%"] > -30)].sort_values("로그성장bp", ascending=False).head(6).round(2).to_string(index=False))
P("  MDD -50% 이내:")
P(d[(d["테스트MDD%"] > -50)].sort_values("로그성장bp", ascending=False).head(6).round(2).to_string(index=False))

P("\n" + "=" * 100 + "\n[3] 보수 체결(5bp 관통 필요) 가정에서 같은 격자")
rows = []
for wB in (0, 1, 2, 3, 4, 5):
    for wE in (0, 1, 2, 3, 4, 5, 6):
        if wB + wE == 0:
            continue
        pnl, dates, ruin = sim(None, wB, wE, "보수")
        if ruin:
            continue
        eq = np.cumprod(1 + pnl); te = dates >= TE
        eq2 = np.cumprod(1 + pnl[te]); yrs = (dates[-1] - TE).days / 365
        rows.append((wB, wE, (eq2[-1] ** (1 / yrs) - 1) * 100,
                     float((eq2 / np.maximum.accumulate(eq2) - 1).min()) * 100, np.mean(np.log1p(pnl)) * 1e4))
d2 = pd.DataFrame(rows, columns=["BTC명목", "ETH명목", "테스트연%", "테스트MDD%", "로그성장bp"])
P(d2.sort_values("로그성장bp", ascending=False).head(8).round(2).to_string(index=False))

P("\n" + "=" * 100 + "\n[4] 권장 조합의 최근 12개월 월별 성과")
for lab, wB, wE, fill in (("공격(MDD-50 허용) BTC2·ETH5", 2, 5, "낙관"),
                          ("권장(MDD-30 이내) BTC1·ETH3", 1, 3, "낙관"),
                          ("권장·보수체결 BTC1·ETH3", 1, 3, "보수"),
                          ("안전 ETH3 단독", 0, 3, "낙관")):
    pnl, dates, ruin = sim(None, wB, wE, fill)
    ser = pd.Series(pnl, index=dates)
    ser = ser[ser != 0]
    mon = ser.resample("ME").apply(lambda x: np.prod(1 + x) - 1) * 100
    last12 = mon.tail(12)
    eq = np.cumprod(1 + ser.values)
    P(f"\n  ■ {lab}  (전체 {eq[-1]:.1f}배, MDD {float((eq/np.maximum.accumulate(eq)-1).min())*100:.1f}%)")
    P("    " + " | ".join(f"{i.strftime('%Y-%m')} {v:+6.2f}%" for i, v in last12.items()))
    P(f"    최근 12개월: 합계 {(np.prod(1+last12/100)-1)*100:+.2f}% | 월평균 {last12.mean():+.2f}% | 양수 {int((last12>0).sum())}/12 | 최악 {last12.min():+.2f}%")
