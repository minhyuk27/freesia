"""R47 — 30배 방향 맞히기 구조 (사용자 규칙, 2026-09-24) + 승률 → 수익 환산표.

구조:
  진입마다 현재 자본의 50%를 증거금, 30배 → 명목 = 자본 × 15. 한 번에 한 포지션.
  청산: 대칭 익절/손절 ±d (가격 기준) 또는 시간 제한 T. 강제청산 거리 = 1/30 − MMR(0.4%) = 2.93% (마크가격).
  체결: 진입 시장가(테이커 5bp+슬리피지 2bp), 익절 지정가(메이커 2bp), 손절·시간 청산 시장가(5+2bp).
        같은 분에 익절·손절 모두 닿으면 손절로 처리(보수).
  수익은 오직 '방향을 맞히는 것'에서 나온다: 맞으면 +15d, 틀리면 −15d (수수료 별도).
진입 신호는 교체 가능한 부품(entries: ts, sym, side). 이번에는 신호 품질을 미루고,
  '정답을 확률 p로 고르는 신호'(오라클)를 무작위 시각에 넣어 승률 p → 계좌 성과를 실제 가격 경로로 환산한다.
  정답 = 그 시각에 롱·숏 중 익절에 먼저 닿는 쪽(둘 다 안 닿으면 시간 청산 수익이 큰 쪽).
사용: python research/r47_x30_frame.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R))
from freesia import store, costs
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
LEV = 30.0; MARGIN = 0.5; NOT = LEV * MARGIN
LIQ = 1 / LEV - costs.MMR_TIER1
CT = costs.TAKER + 2e-4; CM = costs.MAKER
DS = (0.005, 0.01, 0.015, 0.02)
HS = (720, 1440)
STEP = 15
T0 = pd.Timestamp("2021-01-01", tz="UTC")


@numba.njit(cache=True)
def outcome(o, h, l, c, mh, ml, idx, side, d, liq, H, ct, cm):
    """반환: 가격 수익(수수료 포함), 종류(1 익절 / -1 손절 / -2 강제청산 / 0 시간), 청산까지 분."""
    n = len(idx); r = np.empty(n); kind = np.zeros(n, np.int8); tt = np.full(n, H, np.int32)
    for q in range(n):
        t = idx[q]; e = o[t] * (1 + side * 2e-4)                 # 진입 슬리피지
        tp = e * (1 + side * d); sl = e * (1 - side * d); lq = e * (1 - side * liq)
        done = False
        for j in range(t, t + H):
            slhit = (l[j] <= sl) if side > 0 else (h[j] >= sl)
            liqhit = (ml[j] <= lq) if side > 0 else (mh[j] >= lq)
            tphit = (h[j] >= tp) if side > 0 else (l[j] <= tp)
            if slhit:
                px = sl
                if j > t and ((side > 0 and o[j] < sl) or (side < 0 and o[j] > sl)): px = o[j]   # 갭
                r[q] = side * (px / e - 1) - 2 * ct; kind[q] = -1; tt[q] = j - t; done = True; break
            if liqhit:
                r[q] = -liq; kind[q] = -2; tt[q] = j - t; done = True; break
            if tphit:
                r[q] = d - ct - cm; kind[q] = 1; tt[q] = j - t; done = True; break
        if not done:
            r[q] = side * (c[t + H - 1] / e - 1) - 2 * ct
    return r, kind, tt


def load():
    out = {}
    for s in ("BTCUSDT", "ETHUSDT"):
        k = store.load("klines_1m", s).set_index("ts")
        mk = store.load("mark_1m", s).set_index("ts").reindex(k.index).ffill()
        idx = np.arange(0, len(k) - max(HS) - 1, STEP)
        idx = idx[k.index[idx] >= T0]
        res = {"ts": k.index[idx]}
        for d in DS:
            for H in HS:
                for side in (1, -1):
                    res[(d, H, side)] = outcome(k.open.values, k.high.values, k.low.values, k.close.values,
                                                mk.high.values, mk.low.values, idx, side, d, LIQ, H, CT, CM)
        out[s] = res
        P(f"{s} 후보 {len(idx):,}")
    return out


def run(O, d, H, p, per_week, rng):
    """오라클 정확도 p, 주 per_week 회 무작위 진입. 한 번에 한 포지션. 자본 경로 반환."""
    rows = []
    for s, res in O.items():
        n = len(res["ts"])
        rl, kl, tl = res[(d, H, 1)]; rs_, ks, tsh = res[(d, H, -1)]
        # 정답 방향: 익절에 먼저 닿은 쪽, 둘 다 못 닿으면 시간 청산 수익이 큰 쪽
        right_long = np.where((kl == 1) & (ks == 1), tl <= tsh,
                              np.where(kl == 1, True, np.where(ks == 1, False, rl > rs_)))
        rows.append((s, res["ts"], rl, rs_, kl, ks, right_long))
    weeks = (rows[0][1][-1] - rows[0][1][0]).days / 7
    ntr = int(per_week * weeks * 1.6)                  # 겹침으로 버려질 몫 여유
    picks = []
    for _ in range(ntr):
        si = rng.integers(2); s, ts, rl, rs_, kl, ks, rlg = rows[si]
        q = rng.integers(len(ts))
        correct = rng.random() < p
        side = 1 if (rlg[q] == correct) else -1
        r = rl[q] if side > 0 else rs_[q]; kd = kl[q] if side > 0 else ks[q]
        picks.append((ts[q], r, kd))
    picks.sort(key=lambda z: z[0])
    cap = 1.0; busy = None; eq = []; nt = 0; kinds = []
    target_gap = pd.Timedelta(days=7 / per_week)
    for ts, r, kd in picks:
        if busy is not None and ts < busy: continue
        m = max(1 + NOT * r, 1 - MARGIN) if kd != -2 else 1 - MARGIN - NOT * CT
        cap *= m; nt += 1; kinds.append(kd)
        busy = ts + max(pd.Timedelta(minutes=H), target_gap * 0.5)
        eq.append((ts, cap))
        if cap < 1e-6: break
    e = pd.Series([x[1] for x in eq], index=pd.DatetimeIndex([x[0] for x in eq]))
    return e, nt / weeks, np.array(kinds)


def stat(e):
    d = e.resample("D").last().ffill()
    yrs = (d.index[-1] - d.index[0]).days / 365.25
    fin = d.iloc[-1]
    cg = fin ** (1 / yrs) - 1 if fin > 0 else -1.0
    mon = d.resample("ME").last().pct_change().dropna()
    return cg, (1 + cg) ** (1 / 12) - 1 if cg > -1 else -1.0, mon.median(), float((d / d.cummax() - 1).min()), fin


if __name__ == "__main__":
    O = load()
    P(f"\n규칙: 증거금 {MARGIN:.0%} × {LEV:.0f}배 = 명목 자본 × {NOT:.0f}, 강제청산 거리 {LIQ*100:.2f}%")
    P("이론 손익분기 승률(대칭 ±d, 시간청산 무시): 산술 p* = 0.5 + 비용/(2d), 로그성장 기준은 더 높다")
    for d in DS:
        c = 2 * CT - (CT - CM) / 2
        w, lo = NOT * (d - CT - CM), NOT * (d + 2 * CT)
        pg = np.log(1 + lo) / (np.log(1 + w) + np.log(1 + lo)) if lo < 1 else np.nan
        P(f"  ±{d*100:.1f}%: 맞으면 자본 +{w*100:.1f}%, 틀리면 −{lo*100:.1f}% | 산술 본전 {lo/(w+lo)*100:.1f}% · 로그 본전 {pg*100:.1f}%")
    rng = np.random.default_rng(0)
    for H in HS:
        for d in DS:
            P(f"\n── ±{d*100:.1f}%, 시간제한 {H//60}시간 (2021~2026, 무작위 시각 × 정확도 p 오라클, 20회 반복 중앙값)")
            for per_week in (3, 7):
                for p in (0.50, 0.55, 0.60, 0.65, 0.70):
                    S = []; K = []
                    for rep in range(20):
                        e, fw, kinds = run(O, d, H, p, per_week, rng)
                        S.append((*stat(e), fw)); K.append(kinds)
                    S = np.array(S, dtype=float)
                    kk = np.concatenate(K)
                    ruin = np.mean(S[:, 4] < 0.01)
                    P(f"  주{per_week}회 p={p:.2f} | 실제 주 {np.median(S[:,5]):.1f}회 | 익절 {np.mean(kk==1):.2f} 손절 {np.mean(kk==-1):.2f} "
                      f"시간 {np.mean(kk==0):.2f} 청산 {np.mean(kk==-2):.3f} | 월 복리 중앙 {np.median(S[:,1])*100:+6.1f}% "
                      f"(10% {np.percentile(S[:,1],10)*100:+6.1f}%) 월중앙값 {np.median(S[:,2])*100:+6.1f}% "
                      f"MDD 중앙 {np.median(S[:,3])*100:6.1f}% | 자본 99% 손실 비율 {ruin:.2f}")
