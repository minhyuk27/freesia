"""z0.4 — 프리미엄 괴리 전략. 무기한이 지수 대비 극단적으로 싸게 팔릴 때 매수.
임계값은 확장창 분위수(직전까지의 정보만) → 룩어헤드 없음. 신호 t분, 진입 t+1 시가.
사용: python research/z04_prem.py"""
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
    pr = store.load("premium_1m", s).set_index("ts")
    ix = k.index.intersection(pr.index)
    D[s] = dict(o=k.open[ix].values, h=k.high[ix].values, l=k.low[ix].values, c=k.close[ix].values,
                prem=pr.close[ix].values, idx=ix)
    P(f"{s}: {len(ix):,}행")


_TH = {}
def exp_thresh(v, q, key=None, warm=200000, step=10080):
    """확장창 분위수를 주 단위로 갱신. 직전까지의 정보만 사용. 정렬 한 번으로 O(n log n)."""
    ck = (key, q, warm, step)
    if ck in _TH: return _TH[ck]
    n = len(v); th = np.full(n, np.nan)
    import bisect
    srt = []
    prev = 0
    for i in range(warm, n, step):
        seg = v[prev:i]; seg = seg[np.isfinite(seg)]
        for x in seg: bisect.insort(srt, float(x))
        prev = i
        if srt:
            pos = int(q*(len(srt)-1))
            th[i:i+step] = srt[pos]
    _TH[ck] = th
    return th


@numba.njit(cache=True)
def run(o, h, l, c, prem, th, hold, cool, fee_bp, tp_prem):
    """prem[t] <= th[t] 이면 t+1 시가 매수. hold분 후 종가 청산.
    tp_prem>0 이면 프리미엄이 tp_prem 이상 회복하면 조기 청산."""
    n = len(c)
    ei = np.full(n, -1, np.int64); ex = np.full(n, -1, np.int64)
    rr = np.zeros(n); mm = np.zeros(n)
    k = 0; last = -10**9
    for t in range(n-1):
        if np.isnan(th[t]) or np.isnan(prem[t]): continue
        if prem[t] > th[t]: continue
        if t - last < cool: continue
        f = t+1
        e = f + hold
        if e >= n: break
        px = o[f]
        lo_ = 1e18; end = e
        for u in range(f, e+1):
            if l[u] < lo_: lo_ = l[u]
            if tp_prem < 0.0 and prem[u] >= tp_prem:
                end = u; break
        ei[k] = f; ex[k] = end
        rr[k] = (c[end]/px - 1.0) - fee_bp/1e4
        mm[k] = 1.0 - lo_/px
        k += 1
        last = t
    return ei[:k], ex[:k], rr[:k], mm[:k]


def build(q, hold, cool, fee=10.0, tp=0.0, wgt=1.0):
    E = []
    for s in SYM:
        d = D[s]
        th = exp_thresh(d["prem"], q, key=s)
        a_, b_, r_, m_ = run(d["o"], d["h"], d["l"], d["c"], d["prem"], th, hold, cool, fee, tp)
        for a, bb, r, m in zip(a_, b_, r_, m_):
            E.append((int(a), int(bb), float(r), float(m), d["idx"][int(a)], (s, q, hold), 0.01, wgt))
    E.sort(key=lambda z: z[0])
    return E


P("\n[1] 분위 × 보유시간 × 쿨다운 격자 (왕복 테이커 10bp)")
P(f"  {'분위':>8s} {'보유':>6s} {'쿨다운':>6s} {'n':>6s} {'평균bp':>8s} {'중앙bp':>8s} {'승률':>6s} {'연수익bp':>10s} {'거래월':>7s}")
rows = []
for q in (0.0005, 0.001, 0.002, 0.005):
    for hold in (30, 60, 120, 240, 480):
        for cool in (60, 240, 1440):
            E = build(q, hold, cool)
            if len(E) < 30: continue
            r = np.array([x[2] for x in E]); ts = pd.DatetimeIndex([x[4] for x in E])
            nm = len(set(ts.to_period("M")))
            ann = r.sum()/6.7*1e4
            rows.append((q, hold, cool, len(r), r.mean()*1e4, np.median(r)*1e4, (r > 0).mean(), ann, nm))
t = pd.DataFrame(rows, columns=["분위", "보유", "쿨다운", "n", "평균bp", "중앙bp", "승률", "연합계bp", "거래월"])
P(t.sort_values("연합계bp", ascending=False).head(20).round(3).to_string(index=False))

P("\n[2] 상위 후보의 연도별")
top = t.sort_values("연합계bp", ascending=False).head(4)
for _, rw in top.iterrows():
    E = build(rw.분위, int(rw.보유), int(rw.쿨다운))
    df = pd.DataFrame([(x[4], x[2]) for x in E], columns=["ts", "r"]).set_index("ts")
    g = df.groupby(df.index.year).r
    P(f"  q={rw.분위} 보유{int(rw.보유)}분 쿨{int(rw.쿨다운)}분 (n={len(E)}): " +
      " ".join(f"{y} {v*1e4:+5.0f}bp×{n}" for (y, v), (_, n) in zip(g.mean().items(), g.size().items())))

P("\n[3] 비용 민감도 (최적 후보)")
bq, bh, bc = float(top.iloc[0].분위), int(top.iloc[0].보유), int(top.iloc[0].쿨다운)
for fee in (4.0, 7.0, 10.0, 14.0, 20.0, 30.0):
    E = build(bq, bh, bc, fee=fee)
    r = np.array([x[2] for x in E])
    P(f"  왕복 {fee:4.0f}bp: 평균 {r.mean()*1e4:+7.1f}bp 승률 {(r>0).mean():.3f} 연합계 {r.sum()/6.7*1e4:+8.0f}bp")

P("\n[4] 3단 레버리지 성과 (최적 후보)")
E = build(bq, bh, bc)
r = np.array([x[2] for x in E]); ts = pd.DatetimeIndex([x[4] for x in E])
P(f"  구성: 프리미엄 확장창 하위 {bq*100:g}%, 보유 {bh}분, 쿨다운 {bc}분, n={len(E)}, 거래월 {len(set(ts.to_period('M')))}/81")
for w, Nm, tag in ((1.0, 1.0, "무레버"), (3.0, 5.0, "중간"), (5.0, 8.0, "공격"), (8.0, 12.0, "초공격")):
    eq, fl, liq = X.sim(E, w, Nm); m = met(eq, fl, liq)
    if m is None: P(f"     {tag:6s} 파산 (청산 {liq})"); continue
    P(f"     {tag:6s} 연 {m[0]:+8.0f}% MDD {m[1]:6.1f}% 최악 {m[2]:6.1f}% | 24~ {m[3]:+8.0f}% | 25~ {m[4]:+8.0f}%")

P("\n[5] 플라시보 — 같은 보유시간, 무작위 분 진입 500회")
@numba.njit(cache=True)
def rnd(o, c, st, hd, fee):
    out = np.zeros(len(st))
    for i in range(len(st)):
        e = st[i]+hd[i]
        if e >= len(c): e = len(c)-1
        out[i] = (c[e]/o[st[i]]-1.0) - fee/1e4
    return out
rng = np.random.default_rng(5); hd = np.array([x[1]-x[0] for x in E]); nul = []
for _ in range(500):
    acc = []
    for s in SYM:
        d = D[s]; N = len(d["c"]); cnt = len(E)//2
        acc.append(rnd(d["o"], d["c"], rng.integers(0, N-1000, cnt).astype(np.int64),
                       rng.choice(hd, cnt).astype(np.int64), 10.0))
    nul.append(np.concatenate(acc).mean()*1e4)
nul = np.array(nul)
P(f"  위약 평균 {nul.mean():+.1f}bp 99분위 {np.quantile(nul,0.99):+.1f} 최대 {nul.max():+.1f} | "
  f"실제 {r.mean()*1e4:+.1f}bp → 백분위 {(nul < r.mean()*1e4).mean()*100:.1f}%")

P("\n[6] 반대편 — 프리미엄 상위(무기한이 비쌀 때) 숏은?")
for q in (0.999, 0.995, 0.99):
    acc = []
    for s in SYM:
        d = D[s]
        th = exp_thresh(-d["prem"], 1-q, key="n"+s)
        a_, b_, r_, m_ = run(d["o"], d["h"], d["l"], d["c"], -d["prem"], th, bh, bc, 10.0, 0.0)
        acc.append(-r_ - 20.0/1e4 + 10.0/1e4)   # 숏 수익 = -롱수익, 비용 재적용
    rr = np.concatenate(acc)
    P(f"  상위 {(1-q)*100:g}% 숏: n={len(rr)} 평균 {rr.mean()*1e4:+7.1f}bp 승률 {(rr>0).mean():.3f}")
