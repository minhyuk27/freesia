"""목표가 조기청산 + 변동성 타겟팅. 자본 회전을 높여 복리 확대.
TP = 체결단계 평균가 × (1 + rv×kg). 지정가(메이커)로 체결. 미달 시 봉 종가 테이커 청산.
단계가 추가 체결되면 평균가가 낮아지므로 TP를 재계산(취소·재주문) — 실거래 가능.
사용: python research/tp_exit.py
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from lib_dip import bars
TE = pd.Timestamp("2024-01-01", tz="UTC")
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
MMR = 0.004
ALL = [(s, b) for s in ("BTCUSDT", "ETHUSDT") for b in (60, 120, 240, 480)]
D = {}
for key in ALL:
    d = bars(key[0], key[1])
    d["rv"] = (pd.Series(np.log(d["bc"])).diff().rolling(20).std()).shift(1).values
    D[key] = d


@numba.njit(cache=True)
def ev_tp(o, h, l, c, B, x1, x2, x3, kg, rv, fee_m, fee_t):
    """kg<=0 이면 목표가 없음(종가 청산)."""
    n = len(c) // B
    ei = np.full(n, -1, np.int64); ex = np.full(n, -1, np.int64)
    rr = np.zeros(n); mm = np.zeros(n); tp = np.zeros(n); hold = np.zeros(n)
    k = 0
    for b in range(n):
        if np.isnan(x1[b]) or x1[b] <= 0:
            continue
        s = b * B; e = s + B
        O = o[s]; C = c[e - 1]
        t1 = O*(1-x1[b]); t2 = O*(1-x2[b]); t3 = O*(1-x3[b])
        g = rv[b]*kg if kg > 0 else 0.0
        cnt = 0.0; wsum = 0.0; f1 = -1
        lo_ = 1e18; hit = -1; px = 0.0
        for t in range(s, e):
            # 이 분에서 먼저 지정가 매수 체결 판정
            if cnt < 1.0 and l[t] <= t1:
                cnt += 1.0; wsum += t1
                if f1 < 0: f1 = t
            if cnt < 2.0 and l[t] <= t2 and t2 < t1:
                cnt += 1.0; wsum += t2
            if cnt < 3.0 and l[t] <= t3 and t3 < t2:
                cnt += 1.0; wsum += t3
            if cnt == 0.0:
                continue
            if l[t] < lo_: lo_ = l[t]
            if g > 0.0:
                avg = wsum/cnt
                T = avg*(1.0+g)
                # 진입한 분에는 고가가 진입 전일 수 있으므로 다음 분부터 TP 허용(보수)
                if t > f1 and h[t] >= T:
                    hit = t; px = T; break
        if f1 < 0:
            continue
        if hit < 0:
            hit = e-1; px = C; fee = fee_m + fee_t
        else:
            fee = fee_m + fee_m
        avg = wsum/cnt
        ei[k] = f1; ex[k] = hit
        rr[k] = ((px/avg - 1.0) - fee/1e4) * (cnt/3.0)
        mm[k] = (1.0 - lo_/avg) * (cnt/3.0)
        tp[k] = 1.0 if px != C or hit != e-1 else 0.0
        hold[k] = hit - f1
        k += 1
    return ei[:k], ex[:k], rr[:k], mm[:k], tp[:k], hold[:k]


def build(legs, split, kg, fee_m=2.0, fee_t=5.0):
    E = []; ntp = 0; hs = []
    for key in legs:
        d = D[key]; k1, k2, k3 = split; rv = d["rv"]; B = d["B"]
        ei, ex, rr, mm, tpf, hold = ev_tp(d["o"], d["h"], d["l"], d["c"], B,
                                          np.asarray(rv*k1, float), np.asarray(rv*k2, float),
                                          np.asarray(rv*k3, float), float(kg),
                                          np.asarray(rv, float), fee_m, fee_t)
        ts = d["idx"]; ntp += int(tpf.sum()); hs.append(hold)
        for a, bb, r, m in zip(ei, ex, rr, mm):
            E.append((int(a), int(bb), float(r), float(m), ts[int(a)], key, float(rv[int(a)//B])))
    E.sort(key=lambda z: z[0])
    return E, ntp, np.concatenate(hs) if hs else np.array([0.0])


def sim(E, w, Nmax, vt=True, lo=0.2, hi=3.0, warm=40, mmr=MMR):
    cap = 1.0; open_ = []; eqs = []; times = []; liq = 0; rvh = {}
    for a, b, r, m, ts, key, rvb in E:
        for p in [p for p in open_ if p[0] <= a]:
            cap += p[1]*p[2]; open_.remove(p)
        if cap <= 1e-9: break
        if not np.isfinite(rvb) or rvb <= 0: continue
        rh = rvh.setdefault(key, [])
        notional = cap*w*(float(np.clip(np.median(rh)/rvb, lo, hi)) if (vt and len(rh) >= warm) else (lo if vt else 1.0))
        notional = min(notional, max(cap*Nmax - sum(p[1] for p in open_), 0.0))
        rh.append(rvb)
        if notional <= 1e-9: continue
        if notional*m + sum(p[1]*p[3] for p in open_) >= cap*(1-mmr):
            cap = 0.0; liq += 1; break
        open_.append((b, notional, r, m, key))
        eqs.append(cap); times.append(ts)
    for p in open_: cap += p[1]*p[2]
    eqs.append(cap); times.append(E[-1][4])
    return pd.Series(eqs, index=pd.DatetimeIndex(times)), liq


def stat(eq, liq, label, quiet=False):
    if eq.iloc[-1] <= 1e-6:
        if not quiet: P(f"  {label:28s} 파산 (청산 {liq})")
        return None
    yrs = (eq.index[-1]-eq.index[0]).days/365
    te = eq[eq.index >= TE]; t = np.nan
    if len(te) > 30:
        t = ((te.iloc[-1]/te.iloc[0])**(1/((te.index[-1]-te.index[0]).days/365))-1)*100
    cagr = (eq.iloc[-1]**(1/yrs)-1)*100; mdd = float((eq/eq.cummax()-1).min())*100
    if not quiet: P(f"  {label:28s} 전체 연 {cagr:+9.0f}% MDD {mdd:6.1f}% | 테스트 {t:+8.0f}% | 청산 {liq}")
    return cagr, mdd, t, eq


BASE = [(s, b) for s in ("BTCUSDT", "ETHUSDT") for b in (60, 240, 480)]
SP = (5.0, 6.5, 8.0)

P("[1] 목표가 배수 kg별 — 거래 통계")
for kg in (0, 1, 2, 3, 4, 6, 8, 12):
    E, ntp, hold = build(BASE, SP, kg)
    r = np.array([e[2] for e in E])
    P(f"  kg={kg:2d}  n={len(E):5d} 목표달성 {ntp:5d}({ntp/max(len(E),1)*100:5.1f}%) "
      f"평균 {r.mean()*1e4:+6.1f}bp 승률 {(r>0).mean():.3f} | 보유 중앙 {np.median(hold):5.0f}분 평균 {hold.mean():6.0f}분")

P("\n[2] kg × 명목 (변동성타겟 적용)")
best = None
for kg in (0, 2, 3, 4, 6, 8):
    E, ntp, hold = build(BASE, SP, kg)
    P(f"  -- kg={kg} --")
    for w in (2.0, 3.0, 5.0, 8.0):
        for Nm in (8.0, 12.0, 20.0, 40.0):
            eq, liq = sim(E, w, Nm)
            r = stat(eq, liq, f"w={w:.0f} Nmax={Nm:.0f}", quiet=(eq.iloc[-1] <= 1e-6))
            if r and (best is None or r[0] > best[0]): best = (r[0], r[1], r[2], kg, w, Nm, r[3], liq)
if best:
    P(f"\n★ 최고: kg={best[3]} w={best[4]:.0f} Nmax={best[5]:.0f} → 연 {best[0]:+.0f}% MDD {best[1]:.1f}% 테스트 {best[2]:+.0f}%")
    eq = best[6]
    yr = eq.resample("YE").last()/eq.resample("YE").first()-1
    mon = eq.resample("ME").last()/eq.resample("ME").first()-1; l12 = mon.tail(12)
    P("  연도별: " + " | ".join(f"{i.year} {v*100:+.0f}%" for i, v in yr.items()))
    P(f"  최근 12개월 {(np.prod(1+l12)-1)*100:+.1f}% 양수 {int((l12>0).sum())}/12 | 월승률 {float((mon>0).mean()):.3f}")

P("\n[3] 최근 약세 진단 — 연도별 거래 단위 지표 (kg=0, 명목1배)")
E, _, _ = build(BASE, SP, 0)
df = pd.DataFrame([(e[4], e[2], e[3], e[6]) for e in E], columns=["ts", "r", "mae", "rv"]).set_index("ts")
g = df.groupby(df.index.year)
P(pd.DataFrame({
    "진입수": g.size(), "평균bp": g.r.mean()*1e4, "중앙bp": g.r.median()*1e4,
    "승률": g.r.apply(lambda s: (s > 0).mean()), "MAE평균%": g.mae.mean()*100, "rv중앙%": g.rv.median()*100,
}).round(3).to_string())
