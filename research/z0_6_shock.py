"""z0.6 — 5분 유동성 충격 되돌림 (고빈도). 임계는 확장창 분위수 → 룩어헤드 없음.
신호: t분까지의 5분 수익률이 극단 하위 + 거래량/비유동성 조건. 진입 t+1 시가, N분 후 종가 청산.
사용: python research/z0_6_shock.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
import exit_tier as X
from absfloor import met
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
SYM = ("BTCUSDT", "ETHUSDT")
TE = pd.Timestamp("2024-01-01", tz="UTC"); REC = pd.Timestamp("2025-01-01", tz="UTC")
D = {}
for s in SYM:
    k = store.load("klines_1m", s).set_index("ts")
    c = k.close.values
    qv = k.quote_volume.values
    r5 = pd.Series(c).pct_change(5).values
    qv5 = pd.Series(qv).rolling(5).sum().values
    qvm = pd.Series(qv).rolling(1440).mean().values
    D[s] = dict(o=k.open.values, h=k.high.values, l=k.low.values, c=c, idx=k.index,
                r5=r5, qvz=qv5/(qvm*5), amh=np.abs(r5)/np.maximum(qv5, 1.0))
    P(f"{s}: {len(c):,}분")

_TH = {}
def eq_th(v, q, key, warm=200000, step=10080):
    ck = (key, q)
    if ck in _TH: return _TH[ck]
    n = len(v); th = np.full(n, np.nan)
    for i in range(warm, n, step):
        th[i:i+step] = np.nanquantile(v[:i], q)
    _TH[ck] = th
    return th


@numba.njit(cache=True)
def run(o, h, l, c, sig, th, filt, fth, above, hold, cool, fee_bp):
    """sig<=th 이고 (above? filt>=fth : filt<=fth) 이면 t+1 시가 매수, hold분 후 종가 청산."""
    n = len(c)
    ei = np.full(n, -1, np.int64); ex = np.full(n, -1, np.int64)
    rr = np.zeros(n); mm = np.zeros(n); k = 0; last = -10**9
    for t in range(n-1):
        if np.isnan(th[t]) or np.isnan(sig[t]): continue
        if sig[t] > th[t]: continue
        if not np.isnan(fth[t]):
            if above:
                if np.isnan(filt[t]) or filt[t] < fth[t]: continue
            else:
                if np.isnan(filt[t]) or filt[t] > fth[t]: continue
        if t - last < cool: continue
        f = t+1; e = f+hold
        if e >= n: break
        px = o[f]; lo_ = 1e18
        for u in range(f, e+1):
            if l[u] < lo_: lo_ = l[u]
        ei[k] = f; ex[k] = e
        rr[k] = (c[e]/px - 1.0) - fee_bp/1e4
        mm[k] = 1.0 - lo_/px
        k += 1; last = t
    return ei[:k], ex[:k], rr[:k], mm[:k]


def build(q, mode, fq, hold, cool, fee=10.0, wgt=1.0):
    """mode: 'none' 필터없음 / 'vol_low' 거래량 하위 fq / 'illiq_hi' 비유동성 상위 fq"""
    E = []
    for s in SYM:
        d = D[s]
        th = eq_th(d["r5"], q, key="r5"+s)
        if mode == "none":
            filt = np.zeros(len(d["c"])); fth = np.full(len(d["c"]), np.nan); above = False
        elif mode == "vol_low":
            filt = d["qvz"]; fth = eq_th(d["qvz"], fq, key="qv"+s); above = False
        else:
            filt = d["amh"]; fth = eq_th(d["amh"], 1-fq, key="am"+s); above = True
        a_, b_, r_, m_ = run(d["o"], d["h"], d["l"], d["c"], d["r5"], th, filt, fth, above, hold, cool, fee)
        for a, bb, r, m in zip(a_, b_, r_, m_):
            E.append((int(a), int(bb), float(r), float(m), d["idx"][int(a)], (s, mode, hold), 0.01, wgt))
    E.sort(key=lambda z: z[0])
    return E


P("\n[1] 격자 — 진입 분위 × 필터 × 보유시간 (테이커 왕복 10bp)")
P(f"  {'분위':>8s} {'필터':>10s} {'필터값':>7s} {'보유':>5s} {'쿨':>5s} {'n':>6s} {'연건수':>7s} {'평균bp':>8s} {'승률':>6s} {'연합계bp':>9s} {'거래월':>7s}")
rows = []
for q in (0.002, 0.005, 0.01):
    for mode, fq in (("none", 0), ("vol_low", 0.5), ("vol_low", 0.33), ("illiq_hi", 0.1), ("illiq_hi", 0.25)):
        for hold in (30, 60, 120):
            for cool in (30, 120):
                E = build(q, mode, fq, hold, cool)
                if len(E) < 100: continue
                r = np.array([x[2] for x in E]); ts = pd.DatetimeIndex([x[4] for x in E])
                yrs = 6.7
                rows.append((q, mode, fq, hold, cool, len(r), len(r)/yrs, r.mean()*1e4,
                             (r > 0).mean(), r.sum()/yrs*1e4, len(set(ts.to_period("M")))))
t = pd.DataFrame(rows, columns=["분위", "필터", "필터값", "보유", "쿨", "n", "연건수", "평균bp", "승률", "연합계bp", "거래월"])
P(t.sort_values("연합계bp", ascending=False).head(20).round(3).to_string(index=False))

P("\n[2] 상위 후보 연도별 + 3단 레버리지")
for _, rw in t.sort_values("연합계bp", ascending=False).head(3).iterrows():
    E = build(rw.분위, rw.필터, rw.필터값, int(rw.보유), int(rw.쿨))
    r = np.array([x[2] for x in E])
    df = pd.DataFrame([(x[4], x[2]) for x in E], columns=["ts", "r"]).set_index("ts")
    g = df.groupby(df.index.year).r
    P(f"\n  ── q={rw.분위} {rw.필터}{rw.필터값} 보유{int(rw.보유)}분 쿨{int(rw.쿨)}분 n={len(r)} 평균 {r.mean()*1e4:+.1f}bp 승률 {(r>0).mean():.3f}")
    P("     연도별: " + " ".join(f"{y}{v*1e4:+5.1f}×{n}" for (y, v), (_, n) in zip(g.mean().items(), g.size().items())))
    for w, Nm, tag in ((1.0, 1.0, "무레버"), (3.0, 5.0, "중간"), (5.0, 8.0, "공격"), (8.0, 12.0, "초공격")):
        eq, fl, liq = X.sim(E, w, Nm); m = met(eq, fl, liq)
        if m is None: P(f"     {tag:6s} 파산 (청산 {liq})"); continue
        mo = (1+m[0]/100)**(1/12)-1
        mon = (m[5].resample("ME").last()/m[5].resample("ME").first()-1).dropna()
        P(f"     {tag:6s} 연 {m[0]:+8.1f}% (월 {mo*100:+5.1f}%) MDD {m[1]:6.1f}% 최악 {m[2]:6.1f}% | 24~ {m[3]:+8.1f}% | 25~ {m[4]:+8.1f}% | 월승률 {float((mon>0).mean()):.2f}")

P("\n[3] 비용 민감도 (최적 후보)")
bq = t.sort_values("연합계bp", ascending=False).iloc[0]
for fee in (4.0, 7.0, 10.0, 14.0, 20.0):
    E = build(bq.분위, bq.필터, bq.필터값, int(bq.보유), int(bq.쿨), fee=fee)
    r = np.array([x[2] for x in E])
    P(f"  왕복 {fee:4.0f}bp: 평균 {r.mean()*1e4:+6.2f}bp 승률 {(r>0).mean():.3f} 연합계 {r.sum()/6.7*1e4:+8.0f}bp")

P("\n[4] 플라시보 — 같은 보유시간, 무작위 분 진입 500회")
@numba.njit(cache=True)
def rnd(o, c, st, hd, fee):
    out = np.zeros(len(st))
    for i in range(len(st)):
        e = st[i]+hd[i]
        if e >= len(c): e = len(c)-1
        out[i] = (c[e]/o[st[i]]-1.0) - fee/1e4
    return out
E = build(bq.분위, bq.필터, bq.필터값, int(bq.보유), int(bq.쿨))
r = np.array([x[2] for x in E]); hd = np.array([x[1]-x[0] for x in E])
rng = np.random.default_rng(13); nul = []
for _ in range(500):
    acc = []
    for s in SYM:
        d = D[s]; N = len(d["c"]); cnt = len(E)//2
        acc.append(rnd(d["o"], d["c"], rng.integers(0, N-500, cnt).astype(np.int64),
                       rng.choice(hd, cnt).astype(np.int64), 10.0))
    nul.append(np.concatenate(acc).mean()*1e4)
nul = np.array(nul)
P(f"  위약 평균 {nul.mean():+.2f}bp 99분위 {np.quantile(nul,0.99):+.2f} 최대 {nul.max():+.2f} | "
  f"실제 {r.mean()*1e4:+.2f}bp → 백분위 {(nul < r.mean()*1e4).mean()*100:.1f}%")
