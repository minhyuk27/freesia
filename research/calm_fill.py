"""조용한 장의 유휴 자본 — 보완 수익원 탐색.
z0.3이 거래하지 않는 달(59%)에 쓸 수 있는 독립 수익원을 찾는다. 캐리 제외(사용자 지시).
후보: (A) 얕은 딥을 저변동 국면에서만 (B) 일중 시간대 편향 (C) 요일/월말 (D) 변동성 국면별 딥 깊이 적응
   (E) 되돌림 반대편: 급등 뒤 지정가 매도(롱 전용이므로 미실행, 측정만) (F) 롱 편향 단순 보유 대비
사용: python research/calm_fill.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from exit_tier import build, sim, RAW, SYM, BL, prep, CACHE, ev, EPS
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
DEP = {10.0: 1.0, 14.0: 3.0, 20.0: 6.0}

E = build(DEP, 0)
d = pd.DataFrame([(e[4], e[2]) for e in E], columns=["ts", "r"]).set_index("ts").sort_index()
busy = set(d.resample("ME").size()[lambda s: s > 0].index.to_period("M"))
P(f"z0.3 거래월 {len(busy)}개 / 전체 71개. 나머지 42개월이 유휴")

P("\n[A] 얕은 딥이 '조용한 달'에는 살아있나 — 국면별 분리")
for s in SYM:
    dd = RAW[s]
    for B in (240, 480):
        if (s, B) not in CACHE: CACHE[(s, B)] = prep(s, B)
        st, rv = CACHE[(s, B)]
        for k in (3.0, 4.0, 5.0, 6.5):
            a_, b_, r_, m_ = ev(dd["o"], dd["h"], dd["l"], dd["c"], st.astype(np.int64), B,
                                np.asarray(rv*k, float), 0, np.zeros(len(rv)), 0, EPS, 2.0, 5.0)
            if len(r_) < 50: continue
            ts = dd["idx"][a_.astype(int)]
            df = pd.DataFrame({"r": r_}, index=ts)
            inb = df.index.to_period("M").isin(list(busy))
            q = df[~inb].r; b2 = df[inb].r
            if s == "BTCUSDT" and B == 240 and k == 3.0:
                P(f"  {'심볼':8s}{'봉':>5s}{'k':>5s} {'조용한달 n':>9s} {'평균bp':>8s} {'승률':>6s} | {'거래달 n':>8s} {'평균bp':>8s}")
            P(f"  {s[:3]:8s}{B:5d}{k:5.1f} {len(q):9d} {q.mean()*1e4:+8.1f} {(q>0).mean():6.3f} | {len(b2):8d} {b2.mean()*1e4:+8.1f}")

P("\n[B] 일중 시간대 편향 — 4시간봉 시작 시각(UTC)별 깊은딥 성과")
dd2 = pd.DataFrame([(e[4], e[2], e[5][1]) for e in E], columns=["ts", "r", "B"]).set_index("ts")
g = dd2.groupby(dd2.index.hour // 4 * 4).r
P("  " + " | ".join(f"{h:02d}시 {v*1e4:+.0f}bp(n={n})" for (h, v), (_, n) in zip(g.mean().items(), g.size().items())))

P("\n[C] 요일 편향")
g = dd2.groupby(dd2.index.dayofweek).r
nm = ["월", "화", "수", "목", "금", "토", "일"]
P("  " + " | ".join(f"{nm[i]} {v*1e4:+.0f}bp(n={n})" for (i, v), (_, n) in zip(g.mean().items(), g.size().items())))

P("\n[D] 변동성 국면별 최적 깊이 — 조용할 때 더 얕게 가도 되나")
rows = []
for s in SYM:
    dd = RAW[s]
    for B in (240, 480, 720):
        if (s, B) not in CACHE: CACHE[(s, B)] = prep(s, B)
        st, rv = CACHE[(s, B)]
        rvq = pd.Series(rv).expanding(200).rank(pct=True).values   # 확장창 분위 → 룩어헤드 없음
        for k in (3.0, 5.0, 7.0, 10.0, 14.0):
            a_, b_, r_, m_ = ev(dd["o"], dd["h"], dd["l"], dd["c"], st.astype(np.int64), B,
                                np.asarray(rv*k, float), 0, np.zeros(len(rv)), 0, EPS, 2.0, 5.0)
            if len(r_) < 30: continue
            bi = (a_ // B).astype(int)
            qq = rvq[bi]
            for lo, hi, lab in ((0.0, 0.33, "저변동"), (0.33, 0.67, "중변동"), (0.67, 1.01, "고변동")):
                sel = (qq >= lo) & (qq < hi)
                if sel.sum() < 15: continue
                rows.append((lab, k, int(sel.sum()), r_[sel].mean()*1e4, (r_[sel] > 0).mean()))
t = pd.DataFrame(rows, columns=["국면", "깊이k", "n", "평균bp", "승률"])
P(t.groupby(["국면", "깊이k"]).apply(lambda x: pd.Series({
    "n": x.n.sum(), "평균bp": (x.평균bp*x.n).sum()/x.n.sum(), "승률": (x.승률*x.n).sum()/x.n.sum()})).round(2).to_string())

P("\n[E] 상대 비교 — 매수보유 및 z0.3 무레버리지")
for s in SYM:
    c = RAW[s]["c"]; idx = RAW[s]["idx"]
    y = (idx[-1]-idx[0]).days/365
    P(f"  {s} 매수보유: 전체 연 {((c[-1]/c[0])**(1/y)-1)*100:+.0f}%  "
      f"2024~ {((c[-1]/c[np.searchsorted(idx, pd.Timestamp('2024-01-01', tz='UTC'))])**(365/(idx[-1]-pd.Timestamp('2024-01-01', tz='UTC')).days)-1)*100:+.0f}%")
eq, fl, liq = sim(E, 1.0, 1.0)
yv = (eq.index[-1]-eq.index[0]).days/365
P(f"  z0.3 무레버리지: 전체 연 {(eq.iloc[-1]**(1/yv)-1)*100:+.0f}% (시장 노출 시간 극히 낮음)")
expo = sum(e[1]-e[0] for e in E) / (len(RAW['BTCUSDT']['c']) * 1.0) * 100
P(f"  z0.3 시장 노출: 전체 분의 {expo:.2f}% (포지션 보유 시간 합 / 전체 시간)")
