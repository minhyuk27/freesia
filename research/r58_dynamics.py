"""R58 — 방향 B·C 심화: 수준이 아니라 '변화' (호가벽의 생성·소멸, 포지션 이동과 가격의 괴리).

조사 근거: 호가는 깊은 단계·매도 우위 쪽이 더 예측적, 벽은 허수일 수 있어 지속·소멸이 중요.
          롱숏비는 수준보다 가격 움직임 중의 변화가 유용(실무).
호가(bookDepth 분 스냅샷, 2023~, 스냅샷 분이 끝난 뒤 사용):
  dsl15/60/240 = log(n−5/n−1) 변화, dbid1_15/60 = log(n−1) 변화(가까운 매수벽), dask1_15/60 = log(n+1) 변화,
  deep = (n−5 − n+5)/(n−5 + n+5), ddeep60 = 그 60분 변화, tot = Σ매수(1~5) / Σ매도(1~5) 로그, dtot60
포지셔닝(5분 지표, +5분 지연):
  dlsr60/240/1440 = 개인 롱숏 계정비 변화(로그), dtl240 = 상위 트레이더 포지션비 변화, gap = log(tlsr/lsr) (고수 − 개인),
  dgap240, taker60 = 테이커 롱숏 거래량비 60분 평균(로그), div240 = sign(4시간 가격 변화) × dlsr240 (가격과 반대로 쌓이는 포지션)
평가: ① R49 방식 단독 꼬리(학습 2023 경계 → 2024~ 평가, 포지셔닝은 2020~22 → 2023~)
     ② R50 분류기(시드 3 평균)에 추가 vs 미추가 비교, 양 꼬리 선택, 30배 계좌
사용: python research/r58_dynamics.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
from r50_direction_model import D as D0, FEAT as F0, select
from r47_x30_frame import simulate_entries, stat
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)

rows = []
for s in ("BTCUSDT", "ETHUSDT"):
    k = store.load("klines_1m", s).set_index("ts")
    idx = k.index; c = k.close
    bd = store.load("bookdepth", s)
    bd["m"] = bd.ts.dt.floor("min") + pd.Timedelta(minutes=1)
    bd = bd.groupby("m").last().reindex(idx, method="ffill", limit=10)
    lg = lambda x: np.log(x.clip(lower=1.0))
    sl = lg(bd["n-5"]) - lg(bd["n-1"])
    X = pd.DataFrame(index=idx)
    for w in (15, 60, 240): X[f"dsl{w}"] = sl - sl.shift(w)
    for w in (15, 60):
        X[f"dbid1_{w}"] = lg(bd["n-1"]) - lg(bd["n-1"]).shift(w)
        X[f"dask1_{w}"] = lg(bd["n+1"]) - lg(bd["n+1"]).shift(w)
    X["deep"] = (bd["n-5"] - bd["n+5"]) / (bd["n-5"] + bd["n+5"])
    X["ddeep60"] = X.deep - X.deep.shift(60)
    tb = sum(bd[f"n-{i}"] for i in range(1, 6)); ta = sum(bd[f"n+{i}"] for i in range(1, 6))
    X["tot"] = lg(tb) - lg(ta); X["dtot60"] = X.tot - X.tot.shift(60)
    m = store.load("metrics_5m", s).set_index("ts"); m.index = m.index + pd.Timedelta(minutes=5)
    m = m.reindex(idx, method="ffill")
    lsr = np.log(m.count_long_short_ratio); tl = np.log(m.sum_toptrader_long_short_ratio)
    for w in (60, 240, 1440): X[f"dlsr{w}"] = lsr - lsr.shift(w)
    X["dtl240"] = tl - tl.shift(240)
    X["gap"] = tl - lsr; X["dgap240"] = X.gap - X.gap.shift(240)
    X["taker60"] = np.log(m.sum_taker_long_short_vol_ratio).rolling(60).mean()
    X["div240"] = np.sign(np.log(c / c.shift(240))) * X.dlsr240
    g = D0[D0.sym == s][["sym", "t"]].copy()
    tt = g.t.values - 1
    for col in X.columns:
        g[col] = X[col].values[tt].astype(np.float32)
    rows.append(g)
    P(f"{s} 변화 특징 {X.shape[1]}개")
XF = pd.concat(rows).replace([np.inf, -np.inf], np.nan)
NEW = [c for c in XF.columns if c not in ("sym", "t")]
D = D0.merge(XF, on=["sym", "t"], how="left")

P("\n[1] 단독 꼬리 (결판 표본 '위 먼저' 비율) — 호가: 2023 경계→2024~, 포지셔닝: 2020~22 경계→2023~")
for y in ("y10", "y15"):
    res = []
    for f in NEW:
        book = f.startswith(("dsl", "dbid", "dask", "deep", "ddeep", "tot", "dtot"))
        tr = D[D.yr == 2023] if book else D[D.yr < 2023]
        te = D[(D.yr >= (2024 if book else 2023)) & (D[y] != 0)]
        x = tr[f].dropna()
        if len(x) < 1000 or te[f].notna().sum() < 1000: continue
        for q in (0.01, 0.003):
            hi, lo = x.quantile(1 - q), x.quantile(q)
            for tail, sel in (("상단", te[te[f] >= hi]), ("하단", te[te[f] <= lo])):
                if len(sel) < 40: continue
                up = (sel[y] == 1).mean()
                yrs = sel.groupby("yr")[y].apply(lambda v: (v == 1).mean())
                best = max(up, 1 - up); worst_yr = (yrs if up >= 0.5 else 1 - yrs).min()
                res.append((best, f, q, tail, "상승" if up >= 0.5 else "하락", len(sel), worst_yr))
    res.sort(reverse=True)
    P(f"  {y}: 상위 12")
    for r in res[:12]:
        P(f"    {r[0]:.3f} {r[1]:9s} q{r[2]*100:.1f}% {r[3]} → {r[4]} n={r[5]} 최저해 {r[6]:.3f}")


def walk(feats, y):
    out = []
    for Y in range(2022, 2027):
        tr = D[(D.yr < Y) & (D[y] != 0)]; te = D[D.yr == Y].copy()
        fu = [c for c in feats if tr[c].notna().sum() > 5000]
        ps = [HistGradientBoostingClassifier(max_iter=400, learning_rate=0.03, max_leaf_nodes=31, min_samples_leaf=1000,
                                             l2_regularization=1.0, random_state=sd).fit(tr[fu].values, (tr[y].values == 1).astype(int))
              .predict_proba(te[fu].values)[:, 1] for sd in (0, 1, 2)]
        te["p"] = np.mean(ps, axis=0); out.append(te)
    return pd.concat(out).sort_values(["t", "sym"]).reset_index(drop=True)


P("\n[2] R50 분류기(시드 3 평균)에 변화 특징 추가")
for y, d in (("y15", 0.015), ("y10", 0.01)):
    for lab, fs in (("기존", F0), ("기존+변화", F0 + NEW)):
        W = walk(fs, y); wk = (W.ts.max() - W.ts.min()).days / 7
        for q in (0.003, 0.001):
            S = select(W, q); dec = S[S[y] != 0]
            acc = (dec[y] == dec.side).mean()
            eq, T = simulate_entries(S[["ts", "sym", "side"]], d, 720)
            cg, mo, mmed, mdd, fin = stat(eq)
            by = dec.groupby("yr").apply(lambda g: (g[y] == g.side).mean())
            P(f"  ±{d*100:.1f}% {lab:8s} q={q*100:.1f}%: 주 {len(S)/wk:4.1f}회 결판 적중 {acc:.3f} (n={len(dec)}) 무결판 {(S[y]==0).mean():.2f} "
              f"| 계좌 월 {mo*100:+6.1f}% | " + " ".join(f"{yy%100}:{v:.2f}" for yy, v in by.items()))
