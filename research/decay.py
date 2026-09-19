"""엣지 감쇠 점검 — 명목 1배 거래당 bp의 시간 추이, 빈도·승률·진입깊이 변화, 추세 유의성.
사용: python research/decay.py"""
import sys, warnings
from pathlib import Path
import numba, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R))
from freesia import store
from scipy import stats
TE = pd.Timestamp("2024-01-01", tz="UTC")
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
B = 240


@numba.njit(cache=True)
def tr(o, h, l, c, B, xarr, cond):
    nb = len(c) // B
    ret = np.full(nb, np.nan); xs = np.full(nb, np.nan); oc = np.full(nb, np.nan)
    for b in range(nb):
        if not cond[b] or np.isnan(xarr[b]) or xarr[b] <= 0:
            continue
        s = b * B; e = s + B
        O = o[s]; x = xarr[b]; trig = O * (1.0 - x)
        j = -1
        for t in range(s, e):
            if l[t] <= trig:
                j = t; break
        if j < 0:
            continue
        ret[b] = (c[e - 1] / trig - 1.0) * 1e4 - 7.0
        xs[b] = x * 1e4
        oc[b] = (c[e - 1] / O - 1.0) * 1e4
    return ret, xs, oc


S = {}
for sym in ("BTCUSDT", "ETHUSDT"):
    k = store.load("klines_1m", sym).set_index("ts")
    o, h, l, c = k.open.values, k.high.values, k.low.values, k.close.values
    idx = k.index; nb = len(c) // B
    bo = o[::B][:nb]; bc = c[B - 1::B][:nb]
    bh = np.array([h[i * B:(i + 1) * B].max() for i in range(nb)])
    bl = np.array([l[i * B:(i + 1) * B].min() for i in range(nb)])
    bi = idx[::B][:nb]
    s = pd.Series
    rp = s((bh - bl) / bo).shift(1)
    xa = (rp * 1.0).values
    rv = s(np.log(bc)).diff().rolling(24).std()
    cd = (rv < rv.rolling(720).quantile(0.3)).fillna(False).values
    ret, xs, oc = tr(o, h, l, c, B, xa, cd)
    ok = ~np.isnan(ret)
    d = pd.DataFrame({"r": ret[ok], "x": xs[ok], "oc": oc[ok]}, index=bi[ok])
    S[sym] = d
    P("=" * 92 + f"\n### {sym}  (명목 1배, 거래당 순bp)")
    P("\n[1] 반년 단위 추이")
    g = d.r.resample("2QE").agg(["mean", "count", lambda x: (x > 0).mean()])
    g.columns = ["거래당bp", "거래수", "승률"]
    g["진입깊이bp"] = d.x.resample("2QE").mean()
    g["같은봉 시가→종가bp"] = d.oc.resample("2QE").mean()
    P(g.tail(10).round(2).to_string())
    P("\n[2] 기간 비교")
    for nm, a, b_ in (("학습 2020~2023", "2020-01-01", "2024-01-01"),
                      ("테스트 전체 2024~", "2024-01-01", "2026-09-20"),
                      ("2024", "2024-01-01", "2025-01-01"),
                      ("2025", "2025-01-01", "2026-01-01"),
                      ("2026", "2026-01-01", "2026-09-20"),
                      ("최근 12개월", "2025-09-20", "2026-09-20"),
                      ("최근 6개월", "2026-03-20", "2026-09-20")):
        m = (d.index >= pd.Timestamp(a, tz="UTC")) & (d.index < pd.Timestamp(b_, tz="UTC"))
        if m.sum() < 10:
            continue
        x = d.r[m]
        P(f"  {nm:16s} 거래당 {x.mean():7.2f}bp | 중앙 {x.median():6.2f} | 승률 {float((x>0).mean()):.3f} | "
          f"n {int(m.sum()):4d} | 주당 {m.sum()/((min(pd.Timestamp(b_,tz='UTC'),d.index[-1])-pd.Timestamp(a,tz='UTC')).days/7):.2f} | "
          f"깊이 {d.x[m].mean():6.1f}bp | t {x.mean()/x.std()*np.sqrt(len(x)):5.2f}")
    P("\n[3] 감쇠 추세 유의성 (거래 순번 대비 회귀)")
    for nm, m in (("전체", d.index >= d.index[0]), ("테스트만", d.index >= TE)):
        y = d.r[m].values; xx = np.arange(len(y))
        sl, ic, rr, pv, se = stats.linregress(xx, y)
        P(f"  {nm}: 기울기 {sl*100:+.4f}bp/100거래 | p값 {pv:.3f} | {'감쇠 유의' if pv<0.05 and sl<0 else '유의한 추세 없음'}")
    P("\n[4] 12개월 롤링 평균 (최근 8개 분기말)")
    roll = d.r.rolling("365D").mean()
    P("  " + " | ".join(f"{i.strftime('%y-%m')} {v:.1f}bp" for i, v in roll.resample("QE").last().tail(8).items()))

P("\n" + "=" * 92 + "\n[5] 시장 환경 변화 — 4시간봉 범위(진입 깊이의 원천)")
for sym in ("BTCUSDT", "ETHUSDT"):
    k = store.load("klines_1m", sym).set_index("ts")
    nb = len(k) // B
    bo = k.open.values[::B][:nb]
    bh = np.array([k.high.values[i*B:(i+1)*B].max() for i in range(nb)])
    bl = np.array([k.low.values[i*B:(i+1)*B].min() for i in range(nb)])
    bi = k.index[::B][:nb]
    rng_ = pd.Series((bh - bl) / bo * 1e4, index=bi)
    g = rng_.resample("2QE").mean()
    P(f"  {sym} 4시간봉 평균 범위(bp): " + " | ".join(f"{i.strftime('%y-%m')} {v:.0f}" for i, v in g.tail(8).items()))
