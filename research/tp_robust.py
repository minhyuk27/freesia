"""kg=2 목표가 전략 강건성: 체결 가정·절벽·워크포워드·비용·최근약세 정규화.
체결 보수화: 지정가는 '닿음'이 아니라 '관통'을 요구(eps 만큼 더 가야 체결).
사용: python research/tp_robust.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from lib_dip import bars
TE = pd.Timestamp("2024-01-01", tz="UTC")
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
MMR = 0.004
ALL = [(s, b) for s in ("BTCUSDT", "ETHUSDT") for b in (60, 240, 480)]
D = {}
for key in ALL:
    d = bars(key[0], key[1])
    d["rv"] = (pd.Series(np.log(d["bc"])).diff().rolling(20).std()).shift(1).values
    D[key] = d


@numba.njit(cache=True)
def ev(o, h, l, c, B, x1, x2, x3, kg, rv, fee_m, fee_t, eps):
    """eps: 지정가 관통 요구폭(비율). 매수는 가격이 t*(1-eps) 이하로, 매도는 T*(1+eps) 이상으로 가야 체결."""
    n = len(c)//B
    ei = np.full(n, -1, np.int64); ex = np.full(n, -1, np.int64)
    rr = np.zeros(n); mm = np.zeros(n); tpf = np.zeros(n); hold = np.zeros(n)
    k = 0
    for b in range(n):
        if np.isnan(x1[b]) or x1[b] <= 0: continue
        s = b*B; e = s+B
        O = o[s]; C = c[e-1]
        t1 = O*(1-x1[b]); t2 = O*(1-x2[b]); t3 = O*(1-x3[b])
        g = rv[b]*kg if kg > 0 else 0.0
        cnt = 0.0; wsum = 0.0; f1 = -1; lo_ = 1e18; hit = -1; px = 0.0
        for t in range(s, e):
            if cnt < 1.0 and l[t] <= t1*(1-eps):
                cnt += 1.0; wsum += t1
                if f1 < 0: f1 = t
            if cnt < 2.0 and l[t] <= t2*(1-eps) and t2 < t1:
                cnt += 1.0; wsum += t2
            if cnt < 3.0 and l[t] <= t3*(1-eps) and t3 < t2:
                cnt += 1.0; wsum += t3
            if cnt == 0.0: continue
            if l[t] < lo_: lo_ = l[t]
            if g > 0.0 and t > f1:
                T = (wsum/cnt)*(1.0+g)
                if h[t] >= T*(1+eps):
                    hit = t; px = T; break
        if f1 < 0: continue
        if hit < 0:
            hit = e-1; px = C; fee = fee_m + fee_t
        else:
            fee = fee_m + fee_m
        avg = wsum/cnt
        ei[k] = f1; ex[k] = hit
        rr[k] = ((px/avg-1.0) - fee/1e4)*(cnt/3.0)
        mm[k] = (1.0-lo_/avg)*(cnt/3.0)
        tpf[k] = 1.0 if hit != e-1 else 0.0
        hold[k] = hit-f1
        k += 1
    return ei[:k], ex[:k], rr[:k], mm[:k], tpf[:k], hold[:k]


def build(legs, split, kg, fee_m=2.0, fee_t=5.0, eps=0.0):
    E = []; ntp = 0
    for key in legs:
        d = D[key]; k1, k2, k3 = split; rv = d["rv"]; B = d["B"]
        a_, b_, r_, m_, f_, h_ = ev(d["o"], d["h"], d["l"], d["c"], B,
                                    np.asarray(rv*k1, float), np.asarray(rv*k2, float), np.asarray(rv*k3, float),
                                    float(kg), np.asarray(rv, float), fee_m, fee_t, eps)
        ts = d["idx"]; ntp += int(f_.sum())
        for a, bb, r, m in zip(a_, b_, r_, m_):
            E.append((int(a), int(bb), float(r), float(m), ts[int(a)], key, float(rv[int(a)//B])))
    E.sort(key=lambda z: z[0])
    return E, ntp


def sim(E, w, Nmax, lo=0.2, hi=3.0, warm=40, mmr=MMR, start=None, end=None):
    cap = 1.0; open_ = []; eqs = []; times = []; liq = 0; rvh = {}
    for a, b, r, m, ts, key, rvb in E:
        for p in [p for p in open_ if p[0] <= a]:
            cap += p[1]*p[2]; open_.remove(p)
        if cap <= 1e-9: break
        if not np.isfinite(rvb) or rvb <= 0: continue
        rh = rvh.setdefault(key, [])
        f = float(np.clip(np.median(rh)/rvb, lo, hi)) if len(rh) >= warm else lo
        rh.append(rvb)
        if start is not None and ts < start: continue
        if end is not None and ts >= end: break
        notional = min(cap*w*f, max(cap*Nmax - sum(p[1] for p in open_), 0.0))
        if notional <= 1e-9: continue
        if notional*m + sum(p[1]*p[3] for p in open_) >= cap*(1-mmr):
            cap = 0.0; liq += 1; break
        open_.append((b, notional, r, m, key))
        eqs.append(cap); times.append(ts)
    for p in open_: cap += p[1]*p[2]
    eqs.append(cap); times.append(times[-1] if times else E[-1][4])
    return pd.Series(eqs, index=pd.DatetimeIndex(times)), liq


def cagr(eq):
    if eq.iloc[-1] <= 1e-6 or len(eq) < 3: return -100.0
    y = (eq.index[-1]-eq.index[0]).days/365
    return (eq.iloc[-1]**(1/max(y, .01))-1)*100


def stat(eq, liq, label, quiet=False):
    if eq.iloc[-1] <= 1e-6:
        if not quiet: P(f"  {label:26s} 파산 (청산 {liq})")
        return None
    te = eq[eq.index >= TE]; t = np.nan
    if len(te) > 30: t = cagr(te)
    mdd = float((eq/eq.cummax()-1).min())*100
    if not quiet: P(f"  {label:26s} 전체 연 {cagr(eq):+9.0f}% MDD {mdd:6.1f}% | 테스트 {t:+8.0f}% | 청산 {liq}")
    return cagr(eq), mdd, t, eq


BASE = ALL; SP = (5.0, 6.5, 8.0); KG = 2.0

P("[1] 체결 보수화 — 지정가 관통 요구폭 eps")
for eps in (0.0, 0.0001, 0.0002, 0.0005, 0.001):
    E, ntp = build(BASE, SP, KG, eps=eps)
    r = np.array([e[2] for e in E])
    P(f"  eps={eps*1e4:4.0f}bp  n={len(E):5d} 목표달성 {ntp/max(len(E),1)*100:5.1f}% 평균 {r.mean()*1e4:+6.1f}bp 승률 {(r>0).mean():.3f}")
    for w, Nm in ((3.0, 8.0), (5.0, 8.0), (8.0, 8.0)):
        eq, liq = sim(E, w, Nm); stat(eq, liq, f"    w={w:.0f} Nmax={Nm:.0f}")

P("\n[2] 절벽 확인 — w×Nmax 세밀 (eps=2bp)")
E, _ = build(BASE, SP, KG, eps=0.0002)
rows = []
for w in (3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0):
    for Nm in (6.0, 8.0, 10.0, 12.0, 16.0):
        if Nm < w: continue
        eq, liq = sim(E, w, Nm)
        rows.append((w, Nm, "파산" if liq else f"{cagr(eq):+.0f}%",
                     "" if liq else f"{float((eq/eq.cummax()-1).min())*100:.0f}%"))
P(pd.DataFrame(rows, columns=["w", "Nmax", "전체연", "MDD"]).pivot(index="w", columns="Nmax", values="전체연").to_string())

P("\n[3] 워크포워드 (2년 학습 → 1년 검증, 학습기간 최고 w·Nmax 선택)")
yrs = pd.date_range("2022-01-01", "2026-01-01", freq="YS", tz="UTC")
tot = []
for y in yrs:
    tr0, tr1, te1 = y - pd.DateOffset(years=2), y, y + pd.DateOffset(years=1)
    bw = None
    for w in (2.0, 3.0, 5.0, 8.0):
        for Nm in (6.0, 8.0, 10.0):
            if Nm < w: continue
            eq, liq = sim(E, w, Nm, start=tr0, end=tr1)
            if liq: continue
            cc = cagr(eq)
            if bw is None or cc > bw[0]: bw = (cc, w, Nm)
    if bw is None: continue
    eq, liq = sim(E, bw[1], bw[2], start=tr1, end=te1)
    c = cagr(eq) if not liq else -100.0
    tot.append(c)
    P(f"  {tr0.date()}~{tr1.date()} 학습 → w={bw[1]:.0f} Nmax={bw[2]:.0f} | {tr1.date()}~{te1.date()} 검증 연 {c:+8.0f}% {'청산' if liq else ''}")
if tot: P(f"  → 검증 평균 {np.mean(tot):+.0f}% 양수 {sum(1 for x in tot if x>0)}/{len(tot)}")

P("\n[4] 비용 민감도 (w=5 Nmax=8, eps=2bp)")
for fm, ft in ((2.0, 5.0), (3.0, 6.0), (4.0, 7.0), (5.0, 8.0), (7.0, 10.0), (10.0, 13.0)):
    E2, _ = build(BASE, SP, KG, fee_m=fm, fee_t=ft, eps=0.0002)
    eq, liq = sim(E2, 5.0, 8.0); stat(eq, liq, f"메이커{fm:.0f}bp 테이커{ft:.0f}bp")

P("\n[5] 최근 약세 진단 — 변동성 정규화 (eps=2bp, 명목1배)")
df = pd.DataFrame([(e[4], e[2], e[6]) for e in E], columns=["ts", "r", "rv"]).set_index("ts")
df["norm"] = df.r/df.rv
g = df.groupby(df.index.year)
P(pd.DataFrame({"진입수": g.size(), "평균bp": g.r.mean()*1e4, "중앙bp": g.r.median()*1e4,
                "승률": g.r.apply(lambda s: (s > 0).mean()), "rv중앙%": g.rv.median()*100,
                "수익/rv 중앙": g.norm.median(), "수익/rv 평균": g.norm.mean()}).round(3).to_string())
P("\n  최근 12개월 월별 (w=5 Nmax=8):")
eq, _ = sim(E, 5.0, 8.0)
mon = eq.resample("ME").last()/eq.resample("ME").first()-1
P("  " + " | ".join(f"{i.strftime('%y-%m')} {v*100:+.0f}%" for i, v in mon.tail(12).items()))
