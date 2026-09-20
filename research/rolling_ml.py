"""롤링 재학습 실험 — 학습창 길이 × 모델 × 특징 수. 72시간부터 1년·확장창까지.
매 주 재학습 → 다음 1주 예측. 전부 인과적, 학습 구간은 예측 시점 이전만.
사용: python research/rolling_ml.py"""
import sys, warnings, time
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from smooth_dir2 import feats
from smooth_dir import D, SYM
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
STEP = 15          # 15분마다 한 표본 (겹침 완화)
HOR = 60
TH = {"BTCUSDT": 55.2, "ETHUSDT": 53.9}    # 60분 메이커 문턱
FEW = ["px_ema15", "px_ema30", "ema10_sl", "kal0.01_sl", "med15", "rv_ratio", "qvz", "er60"]

P("특징 생성 중...")
DATA = {}
for s in SYM:
    d = D[s]; c = d["c"]
    X = feats(d)
    y = (pd.Series(c).shift(-HOR).values/c - 1.0)
    idx = d["idx"]
    m = np.isfinite(X.values).all(axis=1) & np.isfinite(y)
    sel = np.zeros(len(c), bool); sel[::STEP] = True
    m = m & sel
    DATA[s] = dict(X=X.values[m], Xf=X[FEW].values[m], y=y[m], idx=idx[m], cols=list(X.columns))
    P(f"  {s}: 표본 {m.sum():,}개 ({STEP}분 간격), 특징 {X.shape[1]}개 / 축소 {len(FEW)}개")


def rolling(s, win_days, model, use_few, refit_days=7, start="2021-01-01"):
    d = DATA[s]
    Xa = d["Xf"] if use_few else d["X"]
    y = (d["y"] > 0).astype(int)
    idx = d["idx"]
    t0 = pd.Timestamp(start, tz="UTC")
    ends = pd.date_range(t0, idx[-1], freq=f"{refit_days}D", tz="UTC")
    preds = []; trues = []; rets = []; times = []
    acc_tr = []
    for e in ends:
        s0 = e - pd.Timedelta(days=win_days)
        tr = (idx >= s0) & (idx < e)
        te = (idx >= e) & (idx < e + pd.Timedelta(days=refit_days))
        if tr.sum() < 60 or te.sum() < 5: continue
        if len(np.unique(y[tr])) < 2: continue
        try:
            if model == "logit":
                sc = StandardScaler().fit(Xa[tr])
                mdl = LogisticRegression(max_iter=200, C=0.1).fit(sc.transform(Xa[tr]), y[tr])
                p = mdl.predict_proba(sc.transform(Xa[te]))[:, 1]
                ptr = mdl.predict_proba(sc.transform(Xa[tr]))[:, 1]
            else:
                mdl = HistGradientBoostingClassifier(max_iter=60, max_depth=3, learning_rate=0.08,
                                                     min_samples_leaf=30).fit(Xa[tr], y[tr])
                p = mdl.predict_proba(Xa[te])[:, 1]
                ptr = mdl.predict_proba(Xa[tr])[:, 1]
        except Exception:
            continue
        preds.append(p); trues.append(y[te]); rets.append(d["y"][te]); times.append(idx[te])
        acc_tr.append(((ptr > .5) == (y[tr] > .5)).mean())
    if not preds: return None
    p = np.concatenate(preds); t_ = np.concatenate(trues); r_ = np.concatenate(rets)
    ts = pd.DatetimeIndex(np.concatenate(times))
    return dict(p=p, y=t_, r=r_, ts=ts, acc_tr=float(np.mean(acc_tr)))


def report(res, s, lab):
    if res is None: P(f"  {lab:34s} 표본 부족"); return None
    p, y, r, ts = res["p"], res["y"], res["r"], res["ts"]
    acc = ((p > .5) == (y > .5)).mean()
    th = TH[s]
    # 확신 상위/하위 10%
    hi = p >= np.quantile(p, 0.9); lo = p <= np.quantile(p, 0.1)
    acc_hi = (y[hi] > .5).mean(); acc_lo = (y[lo] < .5).mean()
    # 순손익: 롱 확신 상위 10%만 진입, 비용 7bp
    net = (r[hi].mean() - 7e-4)*1e4
    net_all = ((2*(p > .5)-1)*r).mean()*1e4 - 7
    P(f"  {lab:34s} 학습 {res['acc_tr']*100:5.2f}% 표본밖 {acc*100:5.2f}% | 상위10% {acc_hi*100:5.2f}% "
      f"하위10% {acc_lo*100:5.2f}% | 문턱 {th:.1f}% {'통과' if acc_hi*100 > th else f'{acc_hi*100-th:+.1f}p'} | "
      f"상위10% 순 {net:+6.2f}bp n={int(hi.sum())}")
    return acc, acc_hi, net, res


P("\n" + "="*116)
P("[1] 학습창 길이별 — 60분 방향, 주 1회 재학습, 로지스틱(축소 8특징)")
P("="*116)
for s in SYM:
    P(f"  ── {s} (문턱 {TH[s]:.1f}%) ──")
    for wd, lab in ((3, "72시간"), (7, "1주"), (30, "1개월"), (90, "3개월"), (365, "1년"), (3650, "확장창(전체)")):
        t0 = time.time()
        res = rolling(s, wd, "logit", True)
        report(res, s, f"{lab} 로지스틱 8특징")

P("\n" + "="*116)
P("[2] 같은 조건, 전체 44특징 — 특징이 많으면 짧은 창에서 무너지나")
P("="*116)
for s in SYM:
    P(f"  ── {s} ──")
    for wd, lab in ((3, "72시간"), (30, "1개월"), (365, "1년"), (3650, "확장창")):
        res = rolling(s, wd, "logit", False)
        report(res, s, f"{lab} 로지스틱 44특징")

P("\n" + "="*116)
P("[3] GBM (비선형) — 창 길이별")
P("="*116)
for s in SYM:
    P(f"  ── {s} ──")
    for wd, lab in ((7, "1주"), (30, "1개월"), (90, "3개월"), (365, "1년"), (3650, "확장창")):
        res = rolling(s, wd, "gbm", True)
        report(res, s, f"{lab} GBM 8특징")

P("\n" + "="*116)
P("[4] 연도별 안정성 — 최적 조합")
P("="*116)
for s in SYM:
    for wd, mdl, few, lab in ((365, "logit", True, "1년 로지스틱8"), (30, "logit", True, "1개월 로지스틱8"),
                              (3, "logit", True, "72시간 로지스틱8")):
        res = rolling(s, wd, mdl, few)
        if res is None: continue
        df = pd.DataFrame({"p": res["p"], "y": res["y"], "r": res["r"]}, index=res["ts"])
        g = df.groupby(df.index.year)
        row = f"  {s[:3]} {lab:18s}: "
        for yr, gg in g:
            hi = gg.p >= gg.p.quantile(0.9)
            row += f"{yr} {(gg.y[hi] > .5).mean()*100:4.1f}%/{(gg.r[hi].mean()-7e-4)*1e4:+5.1f}bp  "
        P(row)
