"""ETH/BTC 스프레드 딥 — 시장중립 보완 수익원 탐색.
논리: 같은 유동성공급 프리미엄을 '가격'이 아니라 '상대가격'에 적용.
ETH가 BTC 대비 비정상적으로 급락하면 ETH 롱 + BTC 숏(명목 동일) → 봉 종가 청산.
폭락 상관이 쌓이지 않으므로 z0.3과 결합 가능한지가 핵심.
사용: python research/pair_dip.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
import exit_tier as X
from absfloor import build_f, met
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
TE = pd.Timestamp("2024-01-01", tz="UTC"); REC = pd.Timestamp("2025-01-01", tz="UTC")

b = store.load("klines_1m", "BTCUSDT").set_index("ts")
e = store.load("klines_1m", "ETHUSDT").set_index("ts")
ix = b.index.intersection(e.index)
b, e = b.loc[ix], e.loc[ix]
RB = dict(o=b.open.values, h=b.high.values, l=b.low.values, c=b.close.values)
RE = dict(o=e.open.values, h=e.high.values, l=e.low.values, c=e.close.values)
S = RE["c"] / RB["c"]          # ETH/BTC 비율 (분 단위 종가 기준)
IDX = ix
P(f"공통 분봉 {len(S):,}개  {IDX[0].date()} ~ {IDX[-1].date()}")


@numba.njit(cache=True)
def ev_pair(sb, se, cb, ce, st, B, x, fee_bp, eps):
    """스프레드 s=ce/cb. 봉 시가 s0 대비 x 아래로 내려가면 진입(ETH롱+BTC숏), 봉 종가 청산.
    수익 = (ce_out/ce_in - 1) - (cb_out/cb_in - 1) - 비용. MAE는 스프레드 기준."""
    n = len(st)
    ei = np.full(n, -1, np.int64); ex = np.full(n, -1, np.int64)
    rr = np.zeros(n); mm = np.zeros(n); k = 0
    for j in range(n):
        if np.isnan(x[j]) or x[j] <= 0: continue
        s0 = st[j]; end = s0 + B
        if end > len(cb): break
        S0 = se[s0] / sb[s0]
        T = S0 * (1 - x[j])
        f = -1; lo_ = 1e18
        for t in range(s0, end):
            cur = se[t] / sb[t]
            if f < 0:
                if cur <= T * (1 - eps): f = t
                else: continue
            if cur < lo_: lo_ = cur
        if f < 0: continue
        re_ = ce[end-1]/ce[f] - 1.0
        rb_ = cb[end-1]/cb[f] - 1.0
        ei[k] = f; ex[k] = end-1
        rr[k] = (re_ - rb_) - fee_bp/1e4
        mm[k] = 1.0 - lo_/(se[f]/sb[f])
        k += 1
    return ei[:k], ex[:k], rr[:k], mm[:k]


def build_pair(bars, depths, floors, fee=14.0, eps=0.0002):
    """fee 14bp = 양다리 각 7bp(메이커진입+테이커청산)"""
    E = []
    for B in bars:
        n = len(S)//B
        st = (np.arange(n)*B).astype(np.int64)
        sc = S[st+B-1]
        rv = pd.Series(np.log(sc)).diff().rolling(20).std().shift(1).values
        for k, wgt in depths.items():
            x = np.maximum(rv*k, floors.get(k, 0.0))
            x = np.where(np.isnan(rv), np.nan, x)
            a_, b_, r_, m_ = ev_pair(RB["c"], RE["c"], RB["c"], RE["c"], st, B, np.asarray(x, float), fee, eps)
            for a, bb, r, m in zip(a_, b_, r_, m_):
                E.append((int(a), int(bb), float(r), float(m), IDX[int(a)], ("PAIR", B, k),
                          float(rv[int(a)//B]), wgt))
    E.sort(key=lambda z: z[0])
    return E


P("\n[1] 스프레드 기초 — ETH/BTC 변동성은 개별 종목의 몇 분의 1인가")
for B in (240, 480, 1440):
    n = len(S)//B
    st = np.arange(n)*B
    rs = pd.Series(np.log(S[st+B-1])).diff().std()
    rb = pd.Series(np.log(RB["c"][st+B-1])).diff().std()
    re = pd.Series(np.log(RE["c"][st+B-1])).diff().std()
    P(f"  {B:5d}분봉: 스프레드 rv {rs*100:.2f}% | BTC {rb*100:.2f}% | ETH {re*100:.2f}% → 비율 {rs/rb:.2f}")

P("\n[2] 스프레드 딥 — 깊이별 거래당 수익 (비용 14bp)")
BLP = (240, 480, 720, 1440)
P(f"  {'k':>5s} {'n':>6s} {'평균bp':>8s} {'중앙bp':>8s} {'승률':>6s} {'거래월':>7s}")
for k in (2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 14.0):
    E = build_pair(BLP, {k: 1.0}, {})
    if len(E) < 20: continue
    r = np.array([x[2] for x in E]); ts = pd.DatetimeIndex([x[4] for x in E])
    P(f"  {k:5.1f} {len(r):6d} {r.mean()*1e4:+8.1f} {np.median(r)*1e4:+8.1f} {(r>0).mean():6.3f} {len(set(ts.to_period('M'))):5d}/81")

P("\n[3] 반대 방향(ETH가 BTC 대비 급등 시 숏)도 대칭인가")
@numba.njit(cache=True)
def ev_pair_up(cb, ce, st, B, x, fee_bp, eps):
    n = len(st); ei = np.full(n, -1, np.int64); rr = np.zeros(n); k = 0
    for j in range(n):
        if np.isnan(x[j]) or x[j] <= 0: continue
        s0 = st[j]; end = s0+B
        if end > len(cb): break
        S0 = ce[s0]/cb[s0]; T = S0*(1+x[j]); f = -1
        for t in range(s0, end):
            if ce[t]/cb[t] >= T*(1+eps): f = t; break
        if f < 0: continue
        rr[k] = -((ce[end-1]/ce[f]-1.0) - (cb[end-1]/cb[f]-1.0)) - fee_bp/1e4
        ei[k] = f; k += 1
    return ei[:k], rr[:k]
for k in (3.0, 5.0, 7.0, 10.0):
    acc = []
    for B in BLP:
        n = len(S)//B; st = (np.arange(n)*B).astype(np.int64)
        rv = pd.Series(np.log(S[st+B-1])).diff().rolling(20).std().shift(1).values
        _, r_ = ev_pair_up(RB["c"], RE["c"], st, B, np.asarray(rv*k, float), 14.0, 0.0002)
        acc.append(r_)
    r = np.concatenate(acc)
    P(f"  k={k:4.1f} (급등시 숏): n={len(r):5d} 평균 {r.mean()*1e4:+7.1f}bp 승률 {(r>0).mean():.3f}")

P("\n[4] 유망 구성의 3단 레버리지 성과 (스프레드 단독)")
for dep, lab in (({5.0: 1.0, 7.0: 2.0, 10.0: 4.0}, "5·7·10 (1:2:4)"),
                 ({3.0: 1.0, 5.0: 2.0, 7.0: 4.0}, "3·5·7 (1:2:4)"),
                 ({7.0: 1.0, 10.0: 3.0, 14.0: 6.0}, "7·10·14 (1:3:6)")):
    E = build_pair(BLP, dep, {})
    r = np.array([x[2] for x in E]); ts = pd.DatetimeIndex([x[4] for x in E])
    P(f"  ── {lab} n={len(r)} 평균 {r.mean()*1e4:+.0f}bp 승률 {(r>0).mean():.3f} 거래월 {len(set(ts.to_period('M')))}/81 ──")
    for w, Nm, tag in ((1.0, 1.0, "무레버"), (3.0, 5.0, "중간"), (5.0, 8.0, "공격")):
        eq, fl, liq = X.sim(E, w, Nm); m = met(eq, fl, liq)
        if m is None: P(f"     {tag:6s} 파산"); continue
        P(f"     {tag:6s} 연 {m[0]:+7.0f}% MDD {m[1]:6.1f}% 최악 {m[2]:6.1f}% | 24~ {m[3]:+7.0f}% | 25~ {m[4]:+7.0f}%")

P("\n[5] z0.3과의 상관 — 같은 달에 같이 터지나")
Z = build_f({10.0: 1.0, 14.0: 3.0, 20.0: 6.0}, {10.0: 0.05, 14.0: 0.07, 20.0: 0.10},
            bars=(240, 360, 480, 720, 960, 1440))
zd = pd.DataFrame([(x[4], x[2]) for x in Z], columns=["ts", "r"]).set_index("ts").resample("ME").sum()
PE = build_pair(BLP, {5.0: 1.0, 7.0: 2.0, 10.0: 4.0}, {})
pdz = pd.DataFrame([(x[4], x[2]) for x in PE], columns=["ts", "r"]).set_index("ts").resample("ME").sum()
j = pd.concat([zd.r.rename("z03"), pdz.r.rename("pair")], axis=1).fillna(0)
P(f"  월별 수익 상관 {j.z03.corr(j.pair):+.3f}")
P(f"  z0.3 거래 없는 달 {int((j.z03==0).sum())}개 중 페어가 양수인 달 {int(((j.z03==0)&(j.pair>0)).sum())}개, "
  f"그 달들 페어 평균 {j[j.z03==0].pair.mean()*1e4:+.0f}bp")
