"""통합 운용: BTC·ETH × 4h·8h를 '직전 봉 활발도 상위' 조건으로 동시 운용.
표본·빈도를 늘려 통계력 확보. 깊이 파라미터 동시 탐색. 연도별·워크포워드·위약.
사용: python research/combo_port.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from lib_dip import bars
from grid_clean import run2
TE = pd.Timestamp("2024-01-01", tz="UTC")
rng = np.random.default_rng(241)
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
LEGS = [("BTCUSDT", 240), ("BTCUSDT", 480), ("ETHUSDT", 240), ("ETHUSDT", 480)]
CACHE = {}
for sym, B in LEGS:
    D = bars(sym, B); s = pd.Series
    bo, bc, bh, bl = D["bo"], D["bc"], D["bh"], D["bl"]
    rngL = s((bh - bl) / bo).shift(1)
    uwL = s((bh - np.maximum(bo, bc)) / bo).shift(1)
    CACHE[(sym, B)] = dict(D=D, rngL=rngL, uwL=uwL,
                           qr=rngL.rolling(504).rank(pct=True), qu=uwL.rolling(504).rank(pct=True))


def leg_ret(sym, B, axis, thr, mult, fee=7.0):
    c = CACHE[(sym, B)]; D = c["D"]
    q = c["qr"] if axis == "범위" else (c["qu"] if axis == "위꼬리" else
                                      pd.concat([c["qr"], c["qu"]], axis=1).max(axis=1))
    cd = (q >= thr).fillna(False).values
    xa = (c["rngL"] * mult).values
    ret = run2(D["o"], D["h"], D["l"], D["c"], B, np.asarray(xa, float), np.asarray(cd, np.bool_), 1, fee)
    ok = ~np.isnan(ret)
    return pd.Series(ret[ok], index=D["bi"][ok])


P("[1] 축·문턱·깊이 격자 — 4다리 통합 (각 다리 동일 명목, 단순 평균)")
rows = []
for axis in ("범위", "위꼬리", "둘중큰값"):
    for thr in (0.7, 0.8, 0.9):
        for mult in (0.5, 0.75, 1.0, 1.25):
            ss = [leg_ret(s_, b_, axis, thr, mult) for s_, b_ in LEGS]
            all_ = pd.concat(ss).sort_index()
            tr = all_[all_.index < TE]; te = all_[all_.index >= TE]
            if len(tr) < 100 or len(te) < 80:
                continue
            wk = len(te) / max((te.index[-1] - te.index[0]).days / 7, 1)
            rows.append((axis, thr, mult, tr.mean(), len(tr), te.mean(), float((te > 0).mean()), len(te),
                         te.mean() / te.std() * np.sqrt(len(te)), wk))
d = pd.DataFrame(rows, columns=["축", "문턱", "깊이배수", "학습bp", "학습n", "테스트bp", "승률", "테스트n", "t", "주당"])
P(d.sort_values("t", ascending=False).head(18).round(2).to_string(index=False))
P(f"\n학습·테스트 모두 양수 {int(((d.학습bp>0)&(d.테스트bp>0)).sum())}/{len(d)} | 테스트 t>2 {int((d.t>2).sum())}개")

best = d.sort_values("t", ascending=False).iloc[0]
axis, thr, mult = best.축, best.문턱, best.깊이배수
P(f"\n[2] 최적 조합 정밀: 축={axis} 문턱={thr} 깊이={mult}")
ss = {f"{s_[:3]}{b_//60}h": leg_ret(s_, b_, axis, thr, mult) for s_, b_ in LEGS}
all_ = pd.concat(ss.values()).sort_index()
P("  다리별 테스트 성과")
for nm, v in ss.items():
    te = v[v.index >= TE]
    P(f"    {nm:8s} 학습 {v[v.index<TE].mean():7.2f}bp(n{len(v[v.index<TE])}) | 테스트 {te.mean():7.2f}bp(n{len(te)}, 승률 {float((te>0).mean()):.3f}, 주{len(te)/((te.index[-1]-te.index[0]).days/7):.2f})")
P("\n  연도별(통합)")
g = all_.groupby(all_.index.year).agg(["mean", "count", lambda x: (x > 0).mean()])
g.columns = ["평균bp", "거래수", "승률"]
P("  " + g.round(2).to_string().replace("\n", "\n  "))
P("\n  [3] 앵커드 워크포워드 — 매년 직전까지 학습해 (축,문턱,깊이) 선택")
wf = []
for y in range(2022, 2027):
    sD = pd.Timestamp(f"{y}-01-01", tz="UTC"); eD = pd.Timestamp(f"{y+1}-01-01", tz="UTC")
    bestv = None
    for ax in ("범위", "위꼬리", "둘중큰값"):
        for th in (0.7, 0.8, 0.9):
            for mu in (0.5, 0.75, 1.0, 1.25):
                a_ = pd.concat([leg_ret(s_, b_, ax, th, mu) for s_, b_ in LEGS]).sort_index()
                past = a_[a_.index < sD]
                if len(past) < 150:
                    continue
                if bestv is None or past.mean() > bestv[0]:
                    bestv = (past.mean(), ax, th, mu, a_)
    if bestv is None:
        continue
    _, ax, th, mu, a_ = bestv
    yr = a_[(a_.index >= sD) & (a_.index < eD)]
    if len(yr) < 20:
        continue
    wf.append((y, ax, th, mu, yr.mean(), float((yr > 0).mean()), len(yr)))
P("  " + pd.DataFrame(wf, columns=["연", "축", "문턱", "깊이", "그해bp", "승률", "거래"]).round(2).to_string(index=False).replace("\n", "\n  "))
P("\n  [4] 위약 — 각 다리 조건을 무작위 배치 200회")
pl = []
for _ in range(200):
    tot = []
    for s_, b_ in LEGS:
        c = CACHE[(s_, b_)]; D = c["D"]
        q = c["qr"] if axis == "범위" else (c["qu"] if axis == "위꼬리" else
                                          pd.concat([c["qr"], c["qu"]], axis=1).max(axis=1))
        k_ = int((q >= thr).fillna(False).sum())
        cr = np.zeros(D["n"], bool); cr[rng.choice(D["n"], k_, replace=False)] = True
        r2 = run2(D["o"], D["h"], D["l"], D["c"], b_, np.asarray((c["rngL"] * mult).values, float), cr, 1, 7.0)
        m2 = ~np.isnan(r2) & (D["bi"] >= TE)
        if m2.sum() > 20:
            tot.append(pd.Series(r2[m2], index=D["bi"][m2]))
    if tot:
        pl.append(pd.concat(tot).mean())
pl = np.array(pl); real = all_[all_.index >= TE].mean()
P(f"    위약 평균 {pl.mean():.2f}bp 95% {np.percentile(pl,95):.2f} 99% {np.percentile(pl,99):.2f} 최대 {pl.max():.2f} | 실제 {real:.2f} → 백분위 {(pl<real).mean()*100:.1f}%")
