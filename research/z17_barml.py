"""z1.7 — 145특징 롤링 학습으로 ±1% 우선도달 방향 예측. 목표: 확신 구간 60%+.
룩어헤드 방어: 학습에는 '예측 시점 이전에 결판난' 사례만. 특징선택·임계도 학습구간에서만.
사용: python research/z17_barml.py"""
import sys, warnings, gc
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from bigfeat import build_features, SYM
from freesia import store
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
STEP = 15; BAR = 0.01; MAXH = 4320


@numba.njit(cache=True)
def touch(h, l, c, bar, maxh, step):
    n = len(c); m = n//step
    out = np.zeros(m, np.int8); res = np.zeros(m, np.int64); tt = np.zeros(m, np.int32)
    for k in range(m):
        i = k*step
        p = c[i]; U = p*(1.0+bar); Dn = p*(1.0-bar)
        r = 0; rt = i+maxh
        end = min(i+maxh, n-1)
        for j in range(i+1, end+1):
            hi = h[j] >= U; lo = l[j] <= Dn
            if lo: r = -1; rt = j; break
            if hi: r = 1; rt = j; break
        out[k] = r; res[k] = rt; tt[k] = rt-i
    return out, res, tt


P("데이터 준비...")
DATA = {}
for s in SYM:
    k = store.load("klines_1m", s).set_index("ts")
    Xf, c = build_features(s, step=STEP)
    o_, r_, t_ = touch(k.high.values, k.low.values, k.close.values, BAR, MAXH, STEP)
    n = min(len(Xf), len(o_))
    m = np.isfinite(Xf.values[:n]).all(axis=1) & (o_[:n] != 0)
    DATA[s] = dict(X=Xf.values[:n][m].astype(np.float32), y=(o_[:n][m] == 1).astype(int),
                   res=r_[:n][m], idx=Xf.index[:n][m], tt=t_[:n][m], cols=list(Xf.columns))
    P(f"  {s}: 표본 {m.sum():,} | 위먼저 {DATA[s]['y'].mean()*100:.2f}% | 중앙 결판 {np.median(DATA[s]['tt']):.0f}분")
    del Xf; gc.collect()


def rolling(s, win_days=365, refit_days=30, ncol=20, kind="logit", start="2021-06-01", C=0.05):
    d = DATA[s]; X = d["X"]; y = d["y"]; idx = d["idx"]; res = d["res"]
    # 결판 시각을 분 인덱스 → 타임스탬프로
    base = idx[0]
    res_ts = pd.to_datetime(idx[0]) + pd.to_timedelta((res - res[0]) + 0, unit="m")
    # 더 단순하게: 결판까지 걸린 분(tt)을 진입시각에 더한다
    res_ts = idx + pd.to_timedelta(d["tt"], unit="m")
    pr = []; yy = []; tt = []; atr = []
    for e in pd.date_range(pd.Timestamp(start, tz="UTC"), idx[-1], freq=f"{refit_days}D", tz="UTC"):
        s0 = e - pd.Timedelta(days=win_days)
        tr = (idx >= s0) & (res_ts < e)             # ★ 예측 시점 이전에 '결판난' 것만
        te = (idx >= e) & (idx < e+pd.Timedelta(days=refit_days))
        if tr.sum() < 500 or te.sum() < 20: continue
        if len(np.unique(y[tr])) < 2: continue
        Xtr = X[tr]; ytr = y[tr]
        cr = np.abs(np.array([np.corrcoef(Xtr[:, j], ytr)[0, 1] if Xtr[:, j].std() > 0 else 0
                              for j in range(X.shape[1])]))
        cr[~np.isfinite(cr)] = 0
        cc = np.argsort(-cr)[:ncol]
        try:
            if kind == "logit":
                sc = StandardScaler().fit(Xtr[:, cc])
                mdl = LogisticRegression(max_iter=400, C=C).fit(sc.transform(Xtr[:, cc]), ytr)
                p = mdl.predict_proba(sc.transform(X[te][:, cc]))[:, 1]
                ptr = mdl.predict_proba(sc.transform(Xtr[:, cc]))[:, 1]
            else:
                mdl = HistGradientBoostingClassifier(max_iter=100, max_depth=3, learning_rate=0.06,
                                                     min_samples_leaf=50, l2_regularization=1.0).fit(Xtr[:, cc], ytr)
                p = mdl.predict_proba(X[te][:, cc])[:, 1]
                ptr = mdl.predict_proba(Xtr[:, cc])[:, 1]
        except Exception:
            continue
        pr.append(p); yy.append(y[te]); tt.append(idx[te])
        atr.append(((ptr > .5) == (ytr > .5)).mean())
    if not pr: return None
    return dict(p=np.concatenate(pr), y=np.concatenate(yy),
                ts=pd.DatetimeIndex(np.concatenate(tt)), acc_tr=float(np.mean(atr)))


def rep(r, s, lab):
    if r is None: P(f"  {lab:38s} 표본부족"); return None
    p, y = r["p"], r["y"]
    acc = ((p > .5) == (y > .5)).mean()
    out = f"  {lab:38s} 학습{r['acc_tr']*100:5.2f}% 전체{acc*100:5.2f}% |"
    best = 0
    for q, nm in ((0.8, "상위20%"), (0.9, "상위10%"), (0.95, "상위5%"), (0.99, "상위1%")):
        hi = p >= np.quantile(p, q); lo = p <= np.quantile(p, 1-q)
        ah = (y[hi] == 1).mean()*100; al = (y[lo] == 0).mean()*100
        both = (np.concatenate([y[hi] == 1, y[lo] == 0])).mean()*100
        out += f" {nm} 롱{ah:5.2f}/숏{al:5.2f}/합{both:5.2f}%"
        best = max(best, both)
    P(out)
    return best, r


P("\n" + "="*150)
P("[1] 학습창 × 특징수 × 모델 — 확신 구간 정확도 (문턱: 시장가 55.0% / 지정가 53.5%)")
P("="*150)
best_all = []
for s in SYM:
    P(f"\n  ══ {s} (기저 {DATA[s]['y'].mean()*100:.2f}%) ══")
    for wd in (180, 365, 730):
        for nc in (10, 20, 40):
            r = rolling(s, win_days=wd, ncol=nc, kind="logit")
            b = rep(r, s, f"{wd}일 logit 특징{nc}")
            if b: best_all.append((b[0], s, wd, nc, "logit"))
    for wd in (365, 730):
        for nc in (20, 40):
            r = rolling(s, win_days=wd, ncol=nc, kind="gbm")
            b = rep(r, s, f"{wd}일 GBM 특징{nc}")
            if b: best_all.append((b[0], s, wd, nc, "gbm"))

P("\n" + "="*150)
P("[2] 최고 구성의 확신 구간별 정확도 곡선")
P("="*150)
best_all.sort(reverse=True)
for sc_, s, wd, nc, kind in best_all[:3]:
    r = rolling(s, win_days=wd, ncol=nc, kind=kind)
    if r is None: continue
    p, y = r["p"], r["y"]
    P(f"\n  ── {s} {wd}일 {kind} 특징{nc} (표본 {len(p):,}) ──")
    P(f"    {'확신구간':>10s} {'n':>7s} {'롱정확도':>9s} {'숏정확도':>9s} {'합':>7s} {'시장가55%':>9s} {'지정가53.5%':>11s}")
    for q in (0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995):
        hi = p >= np.quantile(p, q); lo = p <= np.quantile(p, 1-q)
        ah = (y[hi] == 1).mean()*100; al = (y[lo] == 0).mean()*100
        both = np.concatenate([y[hi] == 1, y[lo] == 0]).mean()*100
        nn = int(hi.sum()+lo.sum())
        P(f"    {'상위'+str(int((1-q)*100))+'%':>10s} {nn:7d} {ah:8.2f}% {al:8.2f}% {both:6.2f}% "
          f"{'통과' if both > 55 else f'{both-55:+.1f}':>9s} {'통과' if both > 53.5 else f'{both-53.5:+.1f}':>11s}")
    # 연도별
    df = pd.DataFrame({"p": p, "y": y}, index=r["ts"])
    P("    연도별 상위10% 정확도: " + " ".join(
        f"{yr} {np.concatenate([g.y[g.p >= g.p.quantile(0.9)] == 1, g.y[g.p <= g.p.quantile(0.1)] == 0]).mean()*100:5.1f}%"
        for yr, g in df.groupby(df.index.year) if len(g) > 200))
