"""R27-2 — 차트 특징으로 얕은 딥을 걸러 고빈도에서 비용선을 넘길 수 있는가.

[1] 단일 특징 5분위: 경계는 2020~2022에서만 정하고 2023~2026에서 거래당 bp를 본다.
[2] 워크포워드 부스팅: 연도 Y 평가 = Y 이전 전 연도로 학습(확장창). 예측 순수익(비용 포함) > 문턱만 거래.
[3] 플라시보: 학습 라벨을 연도 안에서 섞어 같은 절차 — 선택 행위만으로 생기는 가짜 엣지 크기.
사용: python research/r27_filter.py [접미사 예: _e5]"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
SUF = sys.argv[1] if len(sys.argv) > 1 else ""
E = pd.read_parquet(R / "data" / f"r27_events{SUF}.parquet")
E["yr"] = pd.DatetimeIndex(E.ts).year
E["isbtc"] = (E.sym == "BTCUSDT").astype(float)
YEARS = (pd.Timestamp(E.ts.max()) - pd.Timestamp(E.ts.min())).days / 365.25
META = {"sym", "f", "x", "ret", "mae", "ts", "yr"}
FEAT = [c for c in E.columns if c not in META]
P(f"이벤트 {len(E):,} 특징 {len(FEAT)}개: {FEAT}")

P("\n[0] 기준선 — (봉, 깊이)별 거래당 bp·연간 건수 (두 종목)")
for (B, k), g in E.groupby(["B", "k"]):
    by = g.groupby("yr").ret.mean() * 1e4
    P(f"  B{B:2d} k{k:.0f} n/년 {len(g)/YEARS:6.0f} {g.ret.mean()*1e4:+6.2f}bp | " +
      " ".join(f"{str(y)[2:]}{v:+5.1f}" for y, v in by.items()))

P("\n[1] 단일 특징 5분위 (경계=2020~22, 평가=2023~26) — 풀: B10/15/30 × k3·4")
pool = E[E.B.isin([10, 15, 30]) & E.k.isin([3.0, 4.0])]
tr, te = pool[pool.yr <= 2022], pool[pool.yr >= 2023]
rows = []
for c in FEAT:
    if c in ("B", "k", "isbtc"): continue
    x = tr[c].dropna()
    if x.nunique() < 5: continue
    qs = np.unique(np.nanquantile(x, [0.2, 0.4, 0.6, 0.8]))
    bt = np.digitize(tr[c], qs); be = np.digitize(te[c], qs)
    mt = [tr.ret[bt == i].mean() * 1e4 for i in range(len(qs) + 1)]
    me = [te.ret[be == i].mean() * 1e4 for i in range(len(qs) + 1)]
    sp = me[-1] - me[0]
    rows.append((abs(sp), c, mt, me))
rows.sort(reverse=True)
for _, c, mt, me in rows[:18]:
    P(f"  {c:9s} 학습 " + " ".join(f"{v:+6.1f}" for v in mt) + " | 평가 " + " ".join(f"{v:+6.1f}" for v in me))


def walk(E, label_shuffle=False, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    out = []
    for Y in range(2022, 2027):
        tr = E[E.yr < Y]; te = E[E.yr == Y]
        if len(te) == 0: continue
        y = np.clip(tr.ret.values, -0.02, 0.02)
        if label_shuffle:
            y = y.copy()
            for yy in tr.yr.unique():
                m = (tr.yr == yy).values
                y[m] = rng.permutation(y[m])
        mdl = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, max_leaf_nodes=15,
                                            min_samples_leaf=400, l2_regularization=1.0, random_state=seed)
        mdl.fit(tr[FEAT].values, y)
        te = te.assign(pred=mdl.predict(te[FEAT].values))
        out.append(te)
    return pd.concat(out)


def summary(W, lab):
    P(f"\n  ── {lab}")
    for th in (0.0, 0.0005, 0.001, 0.002, 0.004):
        s = W[W.pred > th]
        if len(s) < 30:
            P(f"   예측>{th*1e4:4.0f}bp 표본부족({len(s)})"); continue
        yrs = (pd.Timestamp(W.ts.max()) - pd.Timestamp(W.ts.min())).days / 365.25
        epi = s.assign(h=pd.DatetimeIndex(s.ts).floor("15min")).drop_duplicates(["sym", "h"])
        by = s.groupby("yr").ret.agg(["mean", "size"])
        P(f"   예측>{th*1e4:4.0f}bp n/년 {len(s)/yrs:6.0f}(사건 {len(epi)/yrs:5.0f}/년 = 하루 {len(epi)/yrs/365:4.1f}) "
          f"{s.ret.mean()*1e4:+6.2f}bp 승률{(s.ret>0).mean():.3f} | " +
          " ".join(f"{str(y)[2:]}{r['mean']*1e4:+5.1f}×{int(r['size'])}" for y, r in by.iterrows()))


P("\n[2] 워크포워드 부스팅 — 전 풀(B5~60 × k3~8)")
W = walk(E)
summary(W, "실제 라벨")
P("\n[2b] 빈도 풀만(B5~30 × k3~5) — 하루 여러 건 조건")
Wf = W[W.B.isin([5, 10, 15, 30]) & W.k.isin([3.0, 4.0, 5.0])]
summary(Wf, "빈도 풀")

P("\n[3] 플라시보 — 라벨 연도내 셔플 3회")
for sd in range(3):
    summary(walk(E, label_shuffle=True, seed=sd), f"셔플 {sd}")

W.to_parquet(R / "data" / f"r27_wf{SUF}.parquet", index=False)
P(f"\n저장 data/r27_wf{SUF}.parquet")
