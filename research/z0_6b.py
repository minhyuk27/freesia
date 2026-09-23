"""z0.6b — 충격 감지 후 '시장가 추격'이 아니라 '더 아래 지정가'. z0.3의 구조를 5분 스케일로.
신호 t분 → c[t]×(1−x) 에 지정가 매수, 유효시간 W분. 체결되면 H분 후 종가 청산.
사용: python research/z0_6b.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from z0_6_shock import D, SYM, eq_th
import exit_tier as X
from absfloor import met
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
TE = pd.Timestamp("2024-01-01", tz="UTC"); REC = pd.Timestamp("2025-01-01", tz="UTC")
# 5분 실현변동성(직전까지)으로 지정가 깊이 정규화
for s in SYM:
    d = D[s]
    lr = pd.Series(np.log(d["c"])).diff()
    d["rv5"] = lr.rolling(300).std().shift(1).values*np.sqrt(5)   # 5분 스케일 변동성


@numba.njit(cache=True)
def run_lim(o, h, l, c, sig, th, rv, kx, W, H, cool, fm, ft, eps):
    """sig<=th 이면 c[t]*(1-rv*kx) 에 지정가. W분 내 체결되면 H분 후 종가 청산."""
    n = len(c)
    ei = np.full(n, -1, np.int64); ex = np.full(n, -1, np.int64)
    rr = np.zeros(n); mm = np.zeros(n); k = 0; last = -10**9
    for t in range(n-1):
        if np.isnan(th[t]) or np.isnan(sig[t]) or np.isnan(rv[t]) or rv[t] <= 0: continue
        if sig[t] > th[t]: continue
        if t - last < cool: continue
        T = c[t]*(1.0 - rv[t]*kx)
        f = -1
        end_w = t+1+W
        if end_w >= n: break
        for u in range(t+1, end_w):
            if l[u] <= T*(1.0-eps):
                f = u; break
        last = t
        if f < 0: continue
        e = f+H
        if e >= n: break
        lo_ = 1e18
        for u in range(f, e+1):
            if l[u] < lo_: lo_ = l[u]
        ei[k] = f; ex[k] = e
        rr[k] = (c[e]/T - 1.0) - (fm+ft)/1e4
        mm[k] = 1.0 - lo_/T
        k += 1
    return ei[:k], ex[:k], rr[:k], mm[:k]


def build(q, kx, W, H, cool, fm=2.0, ft=5.0, eps=0.0002, wgt=1.0):
    E = []; tries = 0
    for s in SYM:
        d = D[s]
        th = eq_th(d["r5"], q, key="r5"+s)
        a_, b_, r_, m_ = run_lim(d["o"], d["h"], d["l"], d["c"], d["r5"], th,
                                 np.asarray(d["rv5"], float), kx, W, H, cool, fm, ft, eps)
        for a, bb, r, m in zip(a_, b_, r_, m_):
            E.append((int(a), int(bb), float(r), float(m), d["idx"][int(a)], (s, kx, H), 0.01, wgt))
    E.sort(key=lambda z: z[0])
    return E


P("[1] 지정가 깊이 × 유효시간 × 보유시간 (메이커진입+테이커청산 7bp)")
P(f"  {'분위':>7s} {'깊이k':>6s} {'유효':>5s} {'보유':>5s} {'n':>6s} {'연건수':>7s} {'평균bp':>8s} {'중앙bp':>8s} {'승률':>6s} {'연합계bp':>9s}")
rows = []
for q in (0.002, 0.005, 0.01):
    for kx in (1.0, 2.0, 3.0, 5.0):
        for W in (10, 30):
            for H in (60, 120, 240):
                E = build(q, kx, W, H, 30)
                if len(E) < 100: continue
                r = np.array([x[2] for x in E])
                rows.append((q, kx, W, H, len(r), len(r)/6.7, r.mean()*1e4, np.median(r)*1e4,
                             (r > 0).mean(), r.sum()/6.7*1e4))
t = pd.DataFrame(rows, columns=["분위", "깊이k", "유효", "보유", "n", "연건수", "평균bp", "중앙bp", "승률", "연합계bp"])
P(t.sort_values("연합계bp", ascending=False).head(18).round(2).to_string(index=False))

P("\n[2] 상위 후보 3단 레버리지 + 연도별")
for _, rw in t.sort_values("연합계bp", ascending=False).head(4).iterrows():
    E = build(rw.분위, rw.깊이k, int(rw.유효), int(rw.보유), 30)
    r = np.array([x[2] for x in E]); mae = np.array([x[3] for x in E])
    df = pd.DataFrame([(x[4], x[2]) for x in E], columns=["ts", "r"]).set_index("ts")
    g = df.groupby(df.index.year).r
    P(f"\n  ── q={rw.분위} k={rw.깊이k} 유효{int(rw.유효)}분 보유{int(rw.보유)}분 n={len(r)}({len(r)/6.7:.0f}/년) "
      f"평균 {r.mean()*1e4:+.1f}bp 승률 {(r>0).mean():.3f} MAE평균 {mae.mean()*100:.2f}% 최대 {mae.max()*100:.1f}%")
    P("     연도별: " + " ".join(f"{y}{v*1e4:+5.1f}×{n}" for (y, v), (_, n) in zip(g.mean().items(), g.size().items())))
    for w, Nm, tag in ((1.0, 1.0, "무레버"), (3.0, 5.0, "중간"), (5.0, 8.0, "공격"), (8.0, 12.0, "초공격")):
        eq, fl, liq = X.sim(E, w, Nm); m = met(eq, fl, liq)
        if m is None: P(f"     {tag:6s} 파산 (청산 {liq})"); continue
        mo = (1+m[0]/100)**(1/12)-1
        mon = (m[5].resample("ME").last()/m[5].resample("ME").first()-1).dropna()
        P(f"     {tag:6s} 연 {m[0]:+8.1f}% (월 {mo*100:+5.1f}%) MDD {m[1]:6.1f}% 최악 {m[2]:6.1f}% | 24~ {m[3]:+8.1f}% | 25~ {m[4]:+8.1f}% | 월승률 {float((mon>0).mean()):.2f}")

P("\n[3] 비용·체결 민감도 (최적)")
b = t.sort_values("연합계bp", ascending=False).iloc[0]
for fm, ft, eps, lab in ((2, 5, 0.0002, "기준 메2/테5 관통2bp"), (3, 6, 0.0005, "보수 메3/테6 관통5bp"),
                          (4, 7, 0.001, "비관 메4/테7 관통10bp"), (5, 8, 0.002, "최악 메5/테8 관통20bp")):
    E = build(b.분위, b.깊이k, int(b.유효), int(b.보유), 30, fm=fm, ft=ft, eps=eps)
    r = np.array([x[2] for x in E])
    eq, fl, liq = X.sim(E, 1.0, 1.0); m = met(eq, fl, liq)
    P(f"  {lab:26s} n={len(r):5d} 평균 {r.mean()*1e4:+6.2f}bp | 무레버 연 {m[0] if m else float('nan'):+6.1f}%")

P("\n[4] 플라시보 500회")
@numba.njit(cache=True)
def rnd(o, c, st, hd, fee):
    out = np.zeros(len(st))
    for i in range(len(st)):
        e = st[i]+hd[i]
        if e >= len(c): e = len(c)-1
        out[i] = (c[e]/o[st[i]]-1.0) - fee/1e4
    return out
E = build(b.분위, b.깊이k, int(b.유효), int(b.보유), 30)
r = np.array([x[2] for x in E]); hd = np.array([x[1]-x[0] for x in E])
rng = np.random.default_rng(29); nul = []
for _ in range(500):
    acc = []
    for s in SYM:
        d = D[s]; N = len(d["c"]); cnt = len(E)//2
        acc.append(rnd(d["o"], d["c"], rng.integers(0, N-500, cnt).astype(np.int64),
                       rng.choice(hd, cnt).astype(np.int64), 7.0))
    nul.append(np.concatenate(acc).mean()*1e4)
nul = np.array(nul)
P(f"  위약 평균 {nul.mean():+.2f}bp 99분위 {np.quantile(nul,0.99):+.2f} 최대 {nul.max():+.2f} | "
  f"실제 {r.mean()*1e4:+.2f}bp → 백분위 {(nul < r.mean()*1e4).mean()*100:.1f}%")
