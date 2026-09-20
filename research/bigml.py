"""대규모 특징 × 롤링 재학습 — 평가는 순손익(bp) 기준.
학습창 {1개월, 3개월, 1년, 확장창} × 모델 {로지스틱, GBM} × 특징수 {전체145, 상위선택}
사용: python research/bigml.py"""
import sys, warnings, gc
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from bigfeat import build_features, SYM
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
HOR = 60
STEP = 15
TH = {"BTCUSDT": 55.2, "ETHUSDT": 53.9}
FEE = 7.0

DATA = {}
for s in SYM:
    X, c = build_features(s, step=STEP)
    y = (pd.Series(c).shift(-HOR).values/c.values - 1.0)[::STEP]
    y = y[:len(X)]
    m = np.isfinite(X.values).all(axis=1) & np.isfinite(y)
    DATA[s] = dict(X=X.values[m].astype(np.float32), y=y[m].astype(np.float32),
                   idx=X.index[m], cols=list(X.columns))
    P(f"{s}: 유효 표본 {m.sum():,} / 특징 {X.shape[1]}")
    del X; gc.collect()


def rolling(s, win_days, kind, ncol=None, refit_days=14, start="2021-01-01"):
    d = DATA[s]; Xa = d["X"]; ret = d["y"]; idx = d["idx"]
    ybin = (ret > 0).astype(int)
    ends = pd.date_range(pd.Timestamp(start, tz="UTC"), idx[-1], freq=f"{refit_days}D", tz="UTC")
    pr = []; rr = []; tt = []; atr = []
    for e in ends:
        s0 = e - pd.Timedelta(days=win_days)
        tr = (idx >= s0) & (idx < e)
        te = (idx >= e) & (idx < e + pd.Timedelta(days=refit_days))
        if tr.sum() < 200 or te.sum() < 10: continue
        Xtr, Xte = Xa[tr], Xa[te]
        cols = np.arange(Xa.shape[1])
        if ncol:   # 학습 구간에서만 상관 기준 특징 선택 (룩어헤드 없음)
            cr = np.abs(np.array([np.corrcoef(Xtr[:, j], ret[tr])[0, 1] if Xtr[:, j].std() > 0 else 0
                                  for j in range(Xa.shape[1])]))
            cr[~np.isfinite(cr)] = 0
            cols = np.argsort(-cr)[:ncol]
            Xtr = Xtr[:, cols]; Xte = Xte[:, cols]
        try:
            if kind == "logit":
                sc = StandardScaler().fit(Xtr)
                mdl = LogisticRegression(max_iter=200, C=0.05).fit(sc.transform(Xtr), ybin[tr])
                p = mdl.predict_proba(sc.transform(Xte))[:, 1]
                ptr = mdl.predict_proba(sc.transform(Xtr))[:, 1]
            elif kind == "gbm":
                mdl = HistGradientBoostingClassifier(max_iter=80, max_depth=3, learning_rate=0.06,
                                                     min_samples_leaf=50, l2_regularization=1.0).fit(Xtr, ybin[tr])
                p = mdl.predict_proba(Xte)[:, 1]
                ptr = mdl.predict_proba(Xtr)[:, 1]
            else:  # 회귀 — 수익률 자체를 예측 (비대칭 문제 대응)
                sc = StandardScaler().fit(Xtr)
                mdl = Ridge(alpha=100.0).fit(sc.transform(Xtr), ret[tr])
                p = mdl.predict(sc.transform(Xte))
                ptr = mdl.predict(sc.transform(Xtr))
        except Exception:
            continue
        pr.append(p); rr.append(ret[te]); tt.append(idx[te])
        atr.append(((ptr > (0.5 if kind != "ridge" else 0)) == (ybin[tr] > .5)).mean())
    if not pr: return None
    return dict(p=np.concatenate(pr), r=np.concatenate(rr),
                ts=pd.DatetimeIndex(np.concatenate(tt)), acc_tr=float(np.mean(atr)))


def rep(res, s, lab):
    if res is None: P(f"  {lab:38s} 표본 부족"); return
    p, r, ts = res["p"], res["r"], res["ts"]
    acc = ((p > np.median(p)) == (r > 0)).mean()
    out = f"  {lab:38s} 학습 {res['acc_tr']*100:5.2f}% |"
    for q, nm in ((0.9, "상위10%"), (0.95, "상위5%"), (0.99, "상위1%")):
        sel = p >= np.quantile(p, q)
        a = (r[sel] > 0).mean()*100
        net = r[sel].mean()*1e4 - FEE
        out += f" {nm} {a:5.2f}%/{net:+6.2f}bp"
    P(out)
    return res


P("\n" + "="*130)
P("[1] 145특징 전체 — 학습창 × 모델 (순손익 bp 기준, 비용 7bp 차감)")
P("="*130)
for s in SYM:
    P(f"  ── {s} (문턱 {TH[s]:.1f}%) ──")
    for wd, lab in ((30, "1개월"), (90, "3개월"), (365, "1년"), (3650, "확장창")):
        for kind in ("logit", "gbm", "ridge"):
            rep(rolling(s, wd, kind), s, f"{lab} {kind} 145특징")

P("\n" + "="*130)
P("[2] 특징 선택 — 학습 구간 상관 상위 N개만")
P("="*130)
for s in SYM:
    P(f"  ── {s} ──")
    for wd in (90, 365):
        for nc in (10, 30):
            rep(rolling(s, wd, "logit", ncol=nc), s, f"{wd}일 logit 상위{nc}특징")
            rep(rolling(s, wd, "gbm", ncol=nc), s, f"{wd}일 gbm 상위{nc}특징")

P("\n" + "="*130)
P("[3] 최선 조합의 연도별 순손익")
P("="*130)
for s in SYM:
    for wd, kind, nc, lab in ((365, "logit", None, "1년 logit 전체"), (365, "gbm", 30, "1년 gbm 상위30"),
                              (3650, "ridge", None, "확장창 ridge")):
        res = rolling(s, wd, kind, ncol=nc)
        if res is None: continue
        df = pd.DataFrame({"p": res["p"], "r": res["r"]}, index=res["ts"])
        row = f"  {s[:3]} {lab:20s}: "
        for yr, g in df.groupby(df.index.year):
            sel = g.p >= g.p.quantile(0.9)
            row += f"{yr} {(g.r[sel].mean()*1e4-FEE):+6.1f}bp "
        P(row)
