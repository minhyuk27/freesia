"""최적화: 실현변동성 정규화 + 3단 분할 진입. 세부 k 탐색, 연도별·월별·워크포워드·위약.
사용: python research/depth_best.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from lib_dip import bars
from depth_norm import runv, runpyr
TE = pd.Timestamp("2024-01-01", tz="UTC")
rng = np.random.default_rng(277)
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
LEGS = [("BTCUSDT", 240), ("BTCUSDT", 480), ("ETHUSDT", 240), ("ETHUSDT", 480)]
D = {}
for (sym, B) in LEGS:
    d = bars(sym, B); s = pd.Series
    d["rv"] = (s(np.log(d["bc"])).diff().rolling(20).std()).shift(1).values
    D[(sym, B)] = d


def agg(k, split=None, fee=7.0):
    ss = []
    for key in LEGS:
        d = D[key]
        if split is None:
            xa = d["rv"] * k
            ret, mae, xs = runv(d["o"], d["h"], d["l"], d["c"], d["B"], np.asarray(xa, float), fee)
        else:
            k1, k2, k3 = split
            ret, mae = runpyr(d["o"], d["h"], d["l"], d["c"], d["B"],
                              np.asarray(d["rv"] * k1, float), np.asarray(d["rv"] * k2, float),
                              np.asarray(d["rv"] * k3, float), fee)
            xs = d["rv"] * k1 * 1e4
        ok = ~np.isnan(ret)
        ss.append(pd.DataFrame({"r": ret[ok], "m": mae[ok]}, index=d["bi"][ok]))
    return pd.concat(ss).sort_index()


def stat(A, N):
    dist = (1 / N - 0.004) * 1e4
    liq = A.m >= dist
    pnl = np.where(liq, -1.0, N * A.r / 1e4)
    eq = np.cumprod(1 + pnl)
    if (eq <= 0.01).any():
        return None
    yrs = (A.index[-1] - A.index[0]).days / 365
    te = (A.index >= TE).values
    eq2 = np.cumprod(1 + pnl[te]); yrs2 = (A.index[-1] - TE).days / 365
    if (eq2 <= 0.01).any():
        return None
    wk = len(A) / ((A.index[-1] - A.index[0]).days / 7)
    return dict(liq=int(liq.sum()), full=(eq[-1] ** (1 / yrs) - 1) * 100,
                fmdd=float((eq / np.maximum.accumulate(eq) - 1).min()) * 100,
                test=(eq2[-1] ** (1 / yrs2) - 1) * 100,
                tmdd=float((eq2 / np.maximum.accumulate(eq2) - 1).min()) * 100,
                n=len(A), wk=wk, per=A.r.mean(), pnl=pd.Series(pnl, index=A.index))


P("[1] 단일 진입 — 실현변동성 배수 k 세부 탐색")
rows = []
for k in (3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0):
    A = agg(k)
    if len(A) < 150:
        continue
    for N in (1, 2, 3):
        st = stat(A, N)
        if st is None:
            rows.append((k, N, len(A), A.m.max(), "파산", np.nan, np.nan, np.nan, np.nan)); continue
        rows.append((k, N, st["n"], A.m.max(), f"{st['full']:+.0f}%", st["fmdd"], f"{st['test']:+.0f}%", st["tmdd"], st["wk"]))
P(pd.DataFrame(rows, columns=["rv배수", "명목", "거래", "MAE최대", "전체연", "전체MDD%", "테스트연", "테스트MDD%", "주당"]).round(1).to_string(index=False))

P("\n[2] 3단 분할 진입 — 실현변동성 정규화")
rows = []
for sp in ((3.0, 4.5, 6.0), (4.0, 5.5, 7.0), (2.5, 4.0, 5.5), (3.5, 5.0, 6.5), (2.0, 3.5, 5.0)):
    A = agg(None, split=sp)
    if len(A) < 150:
        continue
    for N in (1, 2, 3, 4):
        st = stat(A, N)
        if st is None:
            rows.append((str(sp), N, len(A), A.m.max(), "파산", np.nan, np.nan, np.nan, np.nan, np.nan)); continue
        rows.append((str(sp), N, st["n"], A.m.max(), f"{st['full']:+.0f}%", st["fmdd"], f"{st['test']:+.0f}%", st["tmdd"], st["wk"], st["per"]))
P(pd.DataFrame(rows, columns=["분할(rv배수)", "명목", "거래", "MAE최대", "전체연", "전체MDD%", "테스트연", "테스트MDD%", "주당", "거래당bp"]).round(1).to_string(index=False))

P("\n[3] 최적 후보 정밀 — 분할 (3.0,4.5,6.0), 명목 2배")
A = agg(None, split=(3.0, 4.5, 6.0)); st = stat(A, 2.0)
if st:
    pnl = st["pnl"]
    P(f"  전체 연 {st['full']:+.1f}% MDD {st['fmdd']:.1f}% | 테스트 연 {st['test']:+.1f}% MDD {st['tmdd']:.1f}% | 거래 {st['n']} 주 {st['wk']:.2f}회 | 거래당 {st['per']:.1f}bp | 청산 {st['liq']}건")
    yr = pnl.groupby(pnl.index.year).agg([lambda x: (np.prod(1 + x) - 1) * 100, "count", lambda x: (x > 0).mean()])
    yr.columns = ["연수익%", "거래", "승률"]
    P("  연도별\n  " + yr.round(2).to_string().replace("\n", "\n  "))
    mon = pnl.resample("ME").apply(lambda x: np.prod(1 + x) - 1) * 100
    l12 = mon.tail(12)
    P("  최근 12개월: " + " | ".join(f"{i.strftime('%y-%m')} {v:+.1f}%" for i, v in l12.items()))
    P(f"  최근 12개월 합계 {(np.prod(1+l12/100)-1)*100:+.1f}% | 월평균 {l12.mean():+.2f}% | 양수 {int((l12>0).sum())}/12")

P("\n[4] 워크포워드 — 매년 직전까지 학습해 (분할, 명목) 선택")
CANDS = [((3.0, 4.5, 6.0), 2), ((4.0, 5.5, 7.0), 2), ((2.5, 4.0, 5.5), 2), ((3.5, 5.0, 6.5), 2),
         ((3.0, 4.5, 6.0), 1), ((4.0, 5.5, 7.0), 3), ((2.0, 3.5, 5.0), 1)]
CA = {str(c): agg(None, split=c[0]) for c in CANDS}
wf = []
for y in range(2022, 2027):
    sD = pd.Timestamp(f"{y}-01-01", tz="UTC"); eD = pd.Timestamp(f"{y+1}-01-01", tz="UTC")
    best = None
    for sp, N in CANDS:
        A = CA[str((sp, N))]
        past = A[A.index < sD]
        if len(past) < 120:
            continue
        dist = (1 / N - 0.004) * 1e4
        p = np.where(past.m >= dist, -1.0, N * past.r / 1e4)
        eq = np.cumprod(1 + p)
        if (eq <= 0.01).any():
            continue
        g = np.mean(np.log1p(p))
        if best is None or g > best[0]:
            best = (g, sp, N, A)
    if best is None:
        continue
    _, sp, N, A = best
    yrA = A[(A.index >= sD) & (A.index < eD)]
    if len(yrA) < 15:
        continue
    dist = (1 / N - 0.004) * 1e4
    p = np.where(yrA.m >= dist, -1.0, N * yrA.r / 1e4)
    wf.append((y, str(sp), N, (np.prod(1 + p) - 1) * 100, float((p > 0).mean()), len(p), int((yrA.m >= dist).sum())))
P("  " + pd.DataFrame(wf, columns=["연", "분할", "명목", "그해%", "승률", "거래", "청산"]).round(2).to_string(index=False).replace("\n", "\n  "))

P("\n[5] 위약 — 같은 봉 무작위 시점 진입 200회 (분할 3.0/4.5/6.0)")
real = agg(None, split=(3.0, 4.5, 6.0))
realm = real[real.index >= TE].r.mean()
pl = []
for _ in range(200):
    tot = []
    for key in LEGS:
        d = D[key]; B = d["B"]; n = d["n"]
        o_, c_ = d["o"], d["c"]
        j = rng.integers(0, B, n)
        ent = c_[np.arange(n) * B + j]
        ex = c_[np.arange(n) * B + B - 1]
        r_ = (ex / ent - 1) * 1e4 - 7
        m_ = (d["bi"] >= TE)
        tot.append(pd.Series(r_[m_], index=d["bi"][m_]))
    pl.append(pd.concat(tot).mean())
pl = np.array(pl)
P(f"  실제 {realm:.2f}bp | 위약 평균 {pl.mean():.2f} 95% {np.percentile(pl,95):.2f} 최대 {pl.max():.2f} → 백분위 {(pl<realm).mean()*100:.1f}%")
