"""깊이 단독 전략 혹독한 검증 — 위약·MAE·배율·워크포워드·비용·체결버퍼·감쇠.
전략: 봉 시가 대비 고정 x% 하락 시 지정가 매수 → 그 봉 종가 시장가 청산. 조건 없음.
사용: python research/depth_valid.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from lib_dip import bars
TE = pd.Timestamp("2024-01-01", tz="UTC")
rng = np.random.default_rng(263)
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
LEGS = [("BTCUSDT", 240), ("BTCUSDT", 480), ("ETHUSDT", 240), ("ETHUSDT", 480)]


@numba.njit(cache=True)
def run4(o, h, l, c, B, x, buf, fee):
    n = len(c) // B
    ret = np.full(n, np.nan); mae = np.full(n, np.nan); tin = np.zeros(n, np.int32); hold = np.zeros(n, np.int32)
    for b in range(n):
        s = b * B; e = s + B
        O = o[s]; trig = O * (1.0 - x); fill = trig * (1.0 - buf)
        j = -1
        for t in range(s, e):
            if l[t] <= fill:
                j = t; break
        if j < 0:
            continue
        lo_ = trig
        for t in range(j, e):
            if l[t] < lo_:
                lo_ = l[t]
        ret[b] = (c[e - 1] / trig - 1.0) * 1e4 - fee
        mae[b] = (1.0 - lo_ / trig) * 1e4
        tin[b] = j - s; hold[b] = e - 1 - j
    return ret, mae, tin, hold


@numba.njit(cache=True)
def rand_entry(o, c, B, seedmod, fee):
    """대조군: 같은 봉에서 무작위 분 진입 → 종가 청산"""
    n = len(c) // B
    out = np.full(n, np.nan)
    for b in range(n):
        s = b * B; e = s + B
        j = s + (b * 7919 + seedmod) % B
        out[b] = (c[e - 1] / c[j] - 1.0) * 1e4 - fee
    return out


D = {k: bars(k[0], k[1]) for k in LEGS}

P("[1] 위약 — 같은 봉에서 무작위 시점 진입(종가 청산). 진입 건수 매칭")
for xbp in (300, 400, 500):
    real_all = []; pl_all = []
    for (sym, B) in LEGS:
        d = D[(sym, B)]
        ret, mae, tin, hold = run4(d["o"], d["h"], d["l"], d["c"], B, xbp / 1e4, 0.0, 7.0)
        ok = ~np.isnan(ret); te = ok & (d["bi"] >= TE)
        real_all.append(pd.Series(ret[te], index=d["bi"][te]))
        idxs = np.where(te)[0]
        for r_ in range(200):
            rr = rand_entry(d["o"], d["c"], B, r_ * 13 + 1, 7.0)
            sel = rng.choice(np.where(d["bi"] >= TE)[0], len(idxs), replace=False)
            pl_all.append((r_, pd.Series(rr[sel], index=d["bi"][sel])))
    real = pd.concat(real_all).mean()
    agg = {}
    for r_, s_ in pl_all:
        agg.setdefault(r_, []).append(s_)
    pl = np.array([pd.concat(v).mean() for v in agg.values()])
    P(f"  x={xbp}bp 실제 {real:7.2f}bp | 위약 평균 {pl.mean():7.2f} 95% {np.percentile(pl,95):7.2f} 최대 {pl.max():7.2f} → 백분위 {(pl<real).mean()*100:5.1f}%")

P("\n[2] 진입 후 최대 역행(MAE) 분포 — 배율 상한 결정")
for xbp in (300, 400, 500):
    for (sym, B) in LEGS:
        d = D[(sym, B)]
        ret, mae, tin, hold = run4(d["o"], d["h"], d["l"], d["c"], B, xbp / 1e4, 0.0, 7.0)
        ok = ~np.isnan(mae)
        m = mae[ok]
        P(f"  x={xbp} {sym[:3]}{B//60}h: 중앙 {np.median(m):6.1f} 90% {np.percentile(m,90):7.1f} 99% {np.percentile(m,99):7.1f} 최대 {m.max():7.1f}bp | 진입까지 중앙 {np.median(tin[ok]):.0f}분 보유 중앙 {np.median(hold[ok]):.0f}분")

P("\n[3] 배율별 통합 성과 (교차마진, 청산거리=1/N-0.4%)")
for xbp in (300, 400, 500):
    ss = []
    for (sym, B) in LEGS:
        d = D[(sym, B)]
        ret, mae, tin, hold = run4(d["o"], d["h"], d["l"], d["c"], B, xbp / 1e4, 0.0, 7.0)
        ok = ~np.isnan(ret)
        ss.append(pd.DataFrame({"r": ret[ok], "m": mae[ok]}, index=d["bi"][ok]))
    A = pd.concat(ss).sort_index()
    te = A[A.index >= TE]
    P(f"  ── x={xbp}bp (테스트 n={len(te)}, 주{len(te)/((te.index[-1]-te.index[0]).days/7):.2f}) ──")
    for N in (1, 2, 3, 5, 8, 12):
        dist = (1 / N - 0.004) * 1e4
        liq = te.m >= dist
        pnl = np.where(liq, -1.0, N * te.r / 1e4)
        eq = np.cumprod(1 + pnl)
        yrs = (te.index[-1] - te.index[0]).days / 365
        ruin = bool((eq <= 0.01).any())
        P(f"    {N:2d}배 청산거리 {dist:6.0f}bp 청산 {int(liq.sum()):3d}건 | "
          f"{'파산' if ruin else f'연 {(eq[-1]**(1/yrs)-1)*100:+7.1f}%'} | MDD {float((eq/np.maximum.accumulate(eq)-1).min())*100:6.1f}%")

P("\n[4] 앵커드 워크포워드 — 매년 직전까지 학습해 x 선택")
ALL = {}
for xbp in (100, 150, 200, 300, 400, 500, 700):
    ss = []
    for (sym, B) in LEGS:
        d = D[(sym, B)]
        ret, mae, tin, hold = run4(d["o"], d["h"], d["l"], d["c"], B, xbp / 1e4, 0.0, 7.0)
        ok = ~np.isnan(ret)
        ss.append(pd.Series(ret[ok], index=d["bi"][ok]))
    ALL[xbp] = pd.concat(ss).sort_index()
wf = []
for y in range(2022, 2027):
    sD = pd.Timestamp(f"{y}-01-01", tz="UTC"); eD = pd.Timestamp(f"{y+1}-01-01", tz="UTC")
    best = None
    for xbp, s_ in ALL.items():
        past = s_[s_.index < sD]
        if len(past) < 200:
            continue
        if best is None or past.mean() > best[0]:
            best = (past.mean(), xbp, s_)
    if best is None:
        continue
    _, xbp, s_ = best
    yr = s_[(s_.index >= sD) & (s_.index < eD)]
    if len(yr) < 20:
        continue
    wf.append((y, xbp, yr.mean(), float((yr > 0).mean()), len(yr)))
P("  " + pd.DataFrame(wf, columns=["연", "선택 x", "그해bp", "승률", "거래"]).round(2).to_string(index=False).replace("\n", "\n  "))

P("\n[5] 비용·체결 민감도 (x=400bp 통합)")
for fee, bn in ((7, "메이커7bp"), (12, "테이커12bp"), (20, "보수20bp")):
    for buf, bufn in ((0.0, "터치"), (0.0005, "5bp더"), (0.0010, "10bp더")):
        ss = []
        for (sym, B) in LEGS:
            d = D[(sym, B)]
            ret, _, _, _ = run4(d["o"], d["h"], d["l"], d["c"], B, 0.04, buf, float(fee))
            ok = ~np.isnan(ret)
            ss.append(pd.Series(ret[ok], index=d["bi"][ok]))
        A = pd.concat(ss).sort_index(); te = A[A.index >= TE]
        P(f"  {bn:10s} {bufn:6s} 테스트 {te.mean():7.2f}bp (n{len(te)}, t{te.mean()/te.std()*np.sqrt(len(te)):5.2f})")

P("\n[6] 반기별 감쇠 점검 (x=400bp 통합)")
s_ = ALL[400]
g = s_.resample("2QE").agg(["mean", "count"])
P("  " + " | ".join(f"{i.strftime('%y-%m')} {r['mean']:.0f}bp({int(r['count'])})" for i, r in g.tail(10).iterrows()))
