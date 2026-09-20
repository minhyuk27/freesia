"""z0.4b — 프리미엄 극단 + 정상화 청산. 그리고 z0.3과의 결합(프리미엄을 증폭기로).
사용: python research/z04b.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from z04_prem import D, SYM, exp_thresh
import exit_tier as X
from absfloor import met
from anchor import build_a
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
TE = pd.Timestamp("2024-01-01", tz="UTC"); REC = pd.Timestamp("2025-01-01", tz="UTC")


@numba.njit(cache=True)
def run2(o, h, l, c, prem, th, thx, maxhold, cool, fee_bp):
    """prem<=th 진입(t+1 시가). prem이 thx 이상 회복하면 청산, 아니면 maxhold에서 청산."""
    n = len(c)
    ei = np.full(n, -1, np.int64); ex = np.full(n, -1, np.int64)
    rr = np.zeros(n); mm = np.zeros(n); k = 0; last = -10**9
    for t in range(n-1):
        if np.isnan(th[t]) or np.isnan(prem[t]) or np.isnan(thx[t]): continue
        if prem[t] > th[t]: continue
        if t - last < cool: continue
        f = t+1
        lim = min(f+maxhold, n-1)
        px = o[f]; lo_ = 1e18; end = lim
        for u in range(f, lim+1):
            if l[u] < lo_: lo_ = l[u]
            if prem[u] >= thx[u]: end = u; break
        ei[k] = f; ex[k] = end
        rr[k] = (c[end]/px - 1.0) - fee_bp/1e4
        mm[k] = 1.0 - lo_/px
        k += 1; last = t
    return ei[:k], ex[:k], rr[:k], mm[:k]


def build2(q, qx, maxhold, cool, fee=10.0, wgt=1.0):
    E = []
    for s in SYM:
        d = D[s]
        th = exp_thresh(d["prem"], q, key=s)
        thx = exp_thresh(d["prem"], qx, key=s)
        a_, b_, r_, m_ = run2(d["o"], d["h"], d["l"], d["c"], d["prem"], th, thx, maxhold, cool, fee)
        for a, bb, r, m in zip(a_, b_, r_, m_):
            E.append((int(a), int(bb), float(r), float(m), d["idx"][int(a)], (s, q, maxhold), 0.01, wgt))
    E.sort(key=lambda z: z[0])
    return E


P("[1] 극단 분위 × 정상화 청산 기준")
P(f"  {'진입q':>8s} {'청산q':>7s} {'최대보유':>8s} {'n':>6s} {'평균bp':>8s} {'중앙bp':>8s} {'승률':>6s} {'보유중앙':>8s} {'연합계bp':>9s}")
rows = []
for q in (0.0002, 0.0005, 0.001, 0.002):
    for qx in (0.05, 0.2, 0.5):
        for mh in (60, 240, 480, 1440):
            E = build2(q, qx, mh, 60)
            if len(E) < 40: continue
            r = np.array([x[2] for x in E]); hd = np.array([x[1]-x[0] for x in E])
            rows.append((q, qx, mh, len(r), r.mean()*1e4, np.median(r)*1e4, (r > 0).mean(),
                         np.median(hd), r.sum()/6.7*1e4))
t = pd.DataFrame(rows, columns=["진입q", "청산q", "최대보유", "n", "평균bp", "중앙bp", "승률", "보유중앙", "연합계bp"])
P(t.sort_values("연합계bp", ascending=False).head(18).round(3).to_string(index=False))

P("\n[2] 상위 후보 3단 레버리지 + 연도별")
for _, rw in t.sort_values("연합계bp", ascending=False).head(4).iterrows():
    E = build2(rw.진입q, rw.청산q, int(rw.최대보유), 60)
    r = np.array([x[2] for x in E])
    df = pd.DataFrame([(x[4], x[2]) for x in E], columns=["ts", "r"]).set_index("ts")
    g = df.groupby(df.index.year).r
    P(f"  ── q={rw.진입q} 청산q={rw.청산q} 최대{int(rw.최대보유)}분 n={len(r)} 평균 {r.mean()*1e4:+.0f}bp 승률 {(r>0).mean():.3f}")
    P("     연도별: " + " ".join(f"{y}{v*1e4:+5.0f}×{n}" for (y, v), (_, n) in zip(g.mean().items(), g.size().items())))
    for w, Nm, tag in ((1.0, 1.0, "무레버"), (3.0, 5.0, "중간"), (5.0, 8.0, "공격")):
        eq, fl, liq = X.sim(E, w, Nm); m = met(eq, fl, liq)
        if m is None: P(f"     {tag:6s} 파산"); continue
        P(f"     {tag:6s} 연 {m[0]:+7.0f}% MDD {m[1]:6.1f}% 최악 {m[2]:6.1f}% | 24~ {m[3]:+7.0f}% | 25~ {m[4]:+7.0f}%")

P("\n[3] 프리미엄을 z0.3의 증폭기로 — 딥 체결 시점의 프리미엄 수준별 성과")
Z = build_a("open", ext=0.10)
PM = {s: pd.Series(D[s]["prem"], index=D[s]["idx"]) for s in SYM}
recs = []
for a, b, r, m, ts, key, rv, wg in Z:
    s = key[0]
    ser = PM[s]
    i = ser.index.searchsorted(ts)
    if i <= 0 or i >= len(ser): continue
    recs.append((ts, r, float(ser.iloc[i-1]), s))
zp = pd.DataFrame(recs, columns=["ts", "r", "prem", "sym"])
zp["q"] = zp.groupby("sym").prem.rank(pct=True)
P(f"  z0.3 체결 시점 프리미엄 중앙 {zp.prem.median()*1e4:+.1f}bp (전체 분봉 중앙 -3.4bp)")
for lo, hi, lab in ((0.0, 0.2, "프리미엄 하위20%"), (0.2, 0.5, "20~50%"), (0.5, 0.8, "50~80%"), (0.8, 1.01, "상위20%")):
    sel = zp[(zp.q >= lo) & (zp.q < hi)]
    if len(sel) < 5: continue
    P(f"    {lab:16s} n={len(sel):4d} 평균 {sel.r.mean()*1e4:+7.0f}bp 중앙 {sel.r.median()*1e4:+6.0f}bp 승률 {(sel.r>0).mean():.3f}")

P("\n[4] z0.3 + 프리미엄 가중 (프리미엄이 낮을수록 명목 증폭)")
for amp in (1.0, 1.5, 2.0, 3.0):
    E2 = []
    for a, b, r, m, ts, key, rv, wg in Z:
        s = key[0]; ser = PM[s]
        i = ser.index.searchsorted(ts)
        if i <= 0 or i >= len(ser): E2.append((a, b, r, m, ts, key, rv, wg)); continue
        pv = float(ser.iloc[i-1])
        f = amp if pv < -0.001 else 1.0
        E2.append((a, b, r, m, ts, key, rv, wg*f))
    row = f"  증폭 {amp:.1f}x |"
    for w, Nm, tag in ((1.0, 1.0, "무레버"), (3.0, 5.0, "중간"), (5.0, 8.0, "공격")):
        eq, fl, liq = X.sim(E2, w, Nm); m = met(eq, fl, liq)
        row += f" {tag} {'파산' if m is None else f'{m[0]:+5.0f}%(최악{m[2]:5.1f}%)'}"
    P(row)

P("\n[5] z0.4 단독과 z0.3의 월별 상관")
best = t.sort_values("연합계bp", ascending=False).iloc[0]
E4 = build2(best.진입q, best.청산q, int(best.최대보유), 60)
m4 = pd.DataFrame([(x[4], x[2]) for x in E4], columns=["ts", "r"]).set_index("ts").resample("ME").sum()
m3 = pd.DataFrame([(x[4], x[2]) for x in Z], columns=["ts", "r"]).set_index("ts").resample("ME").sum()
j = pd.concat([m3.r.rename("z03"), m4.r.rename("z04")], axis=1).fillna(0)
P(f"  월별 상관 {j.z03.corr(j.z04):+.3f}")
P(f"  z0.3 무거래 달 {int((j.z03==0).sum())}개 중 z0.4 양수 {int(((j.z03==0)&(j.z04>0)).sum())}개, "
  f"평균 {j[j.z03==0].z04.mean()*1e4:+.0f}bp")
