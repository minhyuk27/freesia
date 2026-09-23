"""R45 — 차티스트 방식 그대로: 상위 추세 → 핵심 레벨 → 하위 트리거 → 구조 손절 + 비대칭 목표.

이전 차트 연구(937회 실패)는 패턴을 '고정 대칭 배리어·단일 시간대'로 평가했다. 실제 차티스트는 그렇게 매매하지 않는다.
여기서는 과정 전체를 규칙화한다 (전부 인과적: 4시간 피벗은 k봉 뒤 확정되어야 사용).
  상위(4시간): 프랙탈 피벗(k), 구조 추세(HH·HL = 상승), EMA50, ATR14
  S1 돌파 후 리테스트: 4시간 종가가 직전 스윙 고점 Hs 돌파(EMA50 위) → 이후 15분 저가가 Hs+tol·ATR 안으로 되돌아오면 '터치'
     → 15분 양봉이 Hs 위로 마감하면 다음 15분 시가 진입. 손절 = 터치 이후 최저 − 0.1ATR
  S2 피보나치 되돌림: 구조 상승 중 마지막 확정 스윙 저점 L2 → 이후 최고 Hmax (다리 ≥ 2ATR).
     15분 저가가 0.5(tol=0.382면 0.382) 되돌림 안으로 들어오면 터치 → 15분 양봉이 되돌림 구간 중앙 위로 마감하면 진입.
     손절 = min(터치 후 최저, 0.786 되돌림) − 0.1ATR. L2 이탈 시 무효.
  S3 깃발 돌파: 구조 상승 중 직전 6봉 범위 ≤ 2ATR, 그 전 6봉 상승 ≥ 2ATR(깃대)
     → 15분 종가가 횡보 고점 돌파 시 진입. 손절 = 횡보 저점 − 0.1ATR
  공통: 손절거리 0.15~1.5 ATR만 허용, 목표 = 진입 + R×손절거리(지정가) 또는 추적(+1R 본전 → 최고가−1ATR).
       시간대 두 벌: 4시간/15분(72시간 제한), 일봉/1시간(10일 제한).
       같은 15분봉에서 손절·목표 모두 닿으면 손절로 처리(보수). 손절은 봉 시가가 이미 넘었으면 시가 체결.
  숏 = 가격을 역수로 뒤집어 같은 롱 엔진에 넣는다(고저 교환).
비용: 진입 테이커 5+2bp, 손절·시간청산 테이커 5+2bp, 목표 지정가 메이커 2bp.
판정: 파라미터(k, tol, R)는 2020~2022에서만 고르고 2023~2026 고정 평가.
위약: 같은 추세 조건의 무작위 15분 진입 + 실제 거래와 같은 손절거리·R 목표 → '레벨+트리거'의 순수 기여 분리.
사용: python research/r45_chartist.py"""
import sys, warnings, itertools
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R))
from freesia import store, costs
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
CT = costs.TAKER + 2e-4; CM = costs.MAKER
SPLIT = pd.Timestamp("2023-01-01", tz="UTC")
TLIM = 72 * 4


@numba.njit(cache=True)
def exit_trail(o, h, l, c, i0, stop, dist, trail, tlim, ct):
    """+1R 도달 시 본전 손절, 이후 최고가 − trail 추적. 목표 없음(이익은 달리게)."""
    e = o[i0]; worst = l[i0]; best = h[i0]; n = len(c)
    for j in range(i0, min(i0 + tlim, n)):
        if j > i0 and o[j] <= stop:
            return j, o[j] / e - 1 - 2 * ct, 1 - min(worst, o[j]) / e
        if l[j] <= stop:
            return j, stop / e - 1 - 2 * ct, 1 - min(worst, stop) / e
        if l[j] < worst: worst = l[j]
        if h[j] > best: best = h[j]
        if best >= e + dist: stop = max(stop, e)
        if best - trail > stop and best >= e + dist: stop = best - trail
    j = min(i0 + tlim, n) - 1
    return j, c[j] / e - 1 - 2 * ct, 1 - worst / e


@numba.njit(cache=True)
def exit_sim(o, h, l, c, i0, stop, tgt, tlim, ct, cm):
    """15분 인덱스 i0 시가 진입. 반환 (청산 인덱스, 순수익, 최대역행)."""
    e = o[i0]; worst = l[i0]
    n = len(c)
    for j in range(i0, min(i0 + tlim, n)):
        if j > i0 and o[j] <= stop:
            return j, o[j] / e - 1 - 2 * ct, 1 - min(worst, o[j]) / e
        if l[j] <= stop:
            return j, stop / e - 1 - 2 * ct, 1 - min(worst, stop) / e
        if l[j] < worst: worst = l[j]
        if h[j] >= tgt:
            return j, tgt / e - 1 - ct - cm, 1 - worst / e
    j = min(i0 + tlim, n) - 1
    return j, c[j] / e - 1 - 2 * ct, 1 - worst / e


def prep(s, inv, lo="15min", hi="4h"):
    k = store.load("klines_1m", s).set_index("ts")
    m = k.resample(lo).agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    if inv:
        m = pd.DataFrame({"open": 1 / m.open, "high": 1 / m.low, "low": 1 / m.high, "close": 1 / m.close}, index=m.index)
    g = m.resample(hi).agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    tr = pd.concat([g.high - g.low, (g.high - g.close.shift()).abs(), (g.low - g.close.shift()).abs()], axis=1).max(axis=1)
    g["atr"] = tr.rolling(14).mean(); g["ema"] = g.close.ewm(span=50, adjust=False).mean()
    # 15분 → 속한 4시간 봉 번호
    pos = g.index.searchsorted(m.index, side="right") - 1
    return m, g, pos


def pivots(g, k):
    H, L = g.high.values, g.low.values; n = len(g)
    ph = np.zeros(n, bool); pl = np.zeros(n, bool)
    for i in range(k, n - k):
        if H[i] == H[i - k:i + k + 1].max(): ph[i] = True
        if L[i] == L[i - k:i + k + 1].min(): pl[i] = True
    return ph, pl


def run(m, g, pos, setup, k, tol, Rm, tlim=TLIM):
    o, h, l, c = (m[x].values for x in ("open", "high", "low", "close"))
    G_h, G_l, G_c, atr, ema = g.high.values, g.low.values, g.close.values, g.atr.values, g.ema.values
    ph, pl = pivots(g, k)
    idx15 = np.arange(len(m))
    starts = np.searchsorted(pos, np.arange(len(g)))            # 4시간 봉 j의 첫 15분 인덱스
    sh, sl = [], []                                              # 확정 피벗 (값, 봉)
    trades = []; busy_until = -1
    armed = None                                                 # S1/S2 상태 dict
    for j in range(len(g) - 1):
        # 4시간 봉 j 마감 → 피벗 확정(j-k), 이후 j+1 구간에서 의사결정
        if j - k >= 0:
            if ph[j - k]: sh.append((G_h[j - k], j - k))
            if pl[j - k]: sl.append((G_l[j - k], j - k))
        A = atr[j]
        if np.isnan(A) or len(sh) < 2 or len(sl) < 2: continue
        up = sh[-1][0] > sh[-2][0] and sl[-1][0] > sl[-2][0]
        a0, a1 = starts[j + 1], (starts[j + 2] if j + 2 < len(starts) else len(m))
        if setup == "S1":
            Hs = sh[-1][0]
            if armed is None and G_c[j] > Hs and G_c[j] > ema[j]:
                armed = dict(lv=Hs, atr=A, touched=False, low=np.inf, born=j)
            if armed is not None and (j - armed["born"] > 30 or G_c[j] < armed["lv"] - armed["atr"]):
                armed = None
        elif setup == "S2":
            if armed is None and up:
                L2, b2 = sl[-1]
                Hmax = G_h[b2:j + 1].max()
                leg = Hmax - L2
                if leg >= 2 * A:
                    ztop = Hmax - (0.382 if tol >= 0.5 else 0.5) * leg
                    armed = dict(lv=ztop, mid=Hmax - 0.55 * leg, stop0=Hmax - 0.786 * leg, inval=L2, atr=A,
                                 touched=False, low=np.inf, born=j, key=(b2, Hmax))
            if armed is not None and (j - armed["born"] > 30 or G_c[j] < armed["inval"]):
                armed = None
        for i in range(a0, a1):
            if i <= busy_until or i + 1 >= len(m): continue
            ent = None
            if setup in ("S1", "S2") and armed is not None:
                zone = armed["lv"] + (tol * armed["atr"] if setup == "S1" else 0.0)
                if l[i] <= zone:
                    armed["touched"] = True
                if armed["touched"]:
                    armed["low"] = min(armed["low"], l[i])
                    trig = c[i] > o[i] and (c[i] > armed["lv"] if setup == "S1" else c[i] > armed["mid"])
                    if trig:
                        st = armed["low"] - 0.1 * armed["atr"]
                        if setup == "S2": st = min(st, armed["stop0"] - 0.1 * armed["atr"])
                        ent = (i + 1, st, armed["atr"]); armed = None
            elif setup == "S3" and up and j >= 12:
                rng_hi = G_h[j - 5:j + 1].max(); rng_lo = G_l[j - 5:j + 1].min()
                pole = G_c[j - 6] - G_l[j - 11:j - 5].min()
                if rng_hi - rng_lo <= 2.0 * A and pole >= 2.0 * A and c[i] > rng_hi:
                    ent = (i + 1, rng_lo - 0.1 * A, A)
            if ent is None: continue
            i0, st, A0 = ent
            e = o[i0]; dist = e - st
            if not (0.15 * A0 <= dist <= 1.5 * A0): continue
            if Rm > 0:
                jx, r, mae = exit_sim(o, h, l, c, i0, st, e + Rm * dist, tlim, CT, CM)
            else:
                jx, r, mae = exit_trail(o, h, l, c, i0, st, dist, A0, tlim, CT)
            risk = dist / e + 2 * CT
            trades.append((m.index[i0], m.index[jx], r, r / risk, risk, mae, up, A0 / e))
            busy_until = jx
            if setup == "S3": break
    return pd.DataFrame(trades, columns=["t_in", "t_out", "ret", "R", "risk", "mae", "up", "atrf"])


def placebo(m, T, Rm, rng, tlim=TLIM):
    """실제 거래와 같은 달·같은 손절거리 비율, 무작위 15분 진입."""
    o, h, l, c = (m[x].values for x in ("open", "high", "low", "close"))
    out = []
    ix = m.index
    for _, t in T.iterrows():
        mon = ix.searchsorted(t.t_in - pd.Timedelta(days=15)), ix.searchsorted(t.t_in + pd.Timedelta(days=15))
        i0 = int(rng.integers(mon[0], max(mon[0] + 1, mon[1] - tlim)))
        e = o[i0]; dist = (t.risk - 2 * CT) * e
        if Rm > 0:
            jx, r, _ = exit_sim(o, h, l, c, i0, e - dist, e + Rm * dist, tlim, CT, CM)
        else:
            jx, r, _ = exit_trail(o, h, l, c, i0, e - dist, dist, t.atrf * e, tlim, CT)
        out.append(r / t.risk)
    return np.mean(out)


if __name__ == "__main__":
    RS = (1.5, 2.0, 3.0, 0.0)                                   # 0 = 추적 손절(이익은 달리게)
    GRID = {"S1": list(itertools.product((2, 3), (0.25, 0.5), RS)),
            "S2": list(itertools.product((2, 3), (0.25, 0.5), RS)),
            "S3": list(itertools.product((2, 3), (0.0,), RS))}
    TFS = (("15min", "4h", 72 * 4), ("1h", "1D", 24 * 10))
    P(f"총 구성 {sum(len(v) for v in GRID.values()) * 2 * len(TFS)}개 (셋업×파라미터×방향×시간대), 선택은 셋업·방향·시간대별 학습 최상")
    rng = np.random.default_rng(0)
    allbest = []
    for lo, hi, tl in TFS:
        D = {(s_, inv): prep(s_, inv, lo, hi) for s_ in ("BTCUSDT", "ETHUSDT") for inv in (False, True)}
        for setup, grid in GRID.items():
            for inv in (False, True):
                side = "숏" if inv else "롱"
                res = []
                for p in grid:
                    T = pd.concat([run(*D[(s_, inv)], setup, *p, tlim=tl).assign(sym=s_) for s_ in ("BTCUSDT", "ETHUSDT")])
                    if (T.t_in < SPLIT).sum() < 30: continue
                    res.append((T[T.t_in < SPLIT].R.mean(), p, T))
                if not res:
                    P(f"\n[{hi}/{lo} {setup} {side}] 학습 표본 부족"); continue
                res.sort(key=lambda z: z[0], reverse=True)
                isR, p, T = res[0]
                te = T[T.t_in >= SPLIT]
                by = T.groupby(T.t_in.dt.year).R.mean()
                pl = np.nanmean([np.nanmean([placebo(D[(s_, inv)][0], te[te.sym == s_], p[2], rng, tl)
                                             for s_ in ("BTCUSDT", "ETHUSDT") if (te.sym == s_).any()]) for _ in range(5)])
                P(f"\n[{hi}/{lo} {setup} {side}] 학습 최상 k={p[0]} tol={p[1]} {'추적' if p[2]==0 else f'R={p[2]}'}: "
                  f"학습 {isR:+.3f}R ({(T.t_in<SPLIT).sum()}건) → 평가 {te.R.mean():+.3f}R ({len(te)}건, 하루 {len(te)/(3.72*365):.2f}) "
                  f"승률 {(te.ret>0).mean():.2f} 상위1% {te.R.quantile(.99):+.1f}R | 위약 {pl:+.3f}R | 연도별 " +
                  " ".join(f"{y%100}{v:+.2f}" for y, v in by.items()))
                P("   학습 상위 5의 평가 R: " + " ".join(f"{q[1]}→{q[2][q[2].t_in>=SPLIT].R.mean():+.3f}" for q in res[:5]))
                allbest.append(T.assign(setup=setup, side=side, par=str(p), tf=hi))
    pd.concat(allbest).to_parquet(R / "data" / "r45_trades.parquet")
