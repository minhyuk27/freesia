"""주름 펴기 2차 — 극단 분위·다중 필터 정렬·곡률·조건부·학습 모델.
목표: 30~60분 구간에서 방향 정확도 54~57% 문턱 돌파 가능한가.
모든 특징은 인과적(t까지의 정보만). 학습/검증 시간 분할.
사용: python research/smooth_dir2.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
from smooth_dir import ema, wma, kama, kalman, sg_causal, med_filt, D, SYM
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
HOR = (15, 30, 60)
TH = {("BTCUSDT", 15): 60.2, ("BTCUSDT", 30): 57.3, ("BTCUSDT", 60): 55.2,
      ("ETHUSDT", 15): 57.7, ("ETHUSDT", 30): 55.5, ("ETHUSDT", 60): 53.9}


def fw(c, h): return pd.Series(c).shift(-h).values/c - 1.0


def feats(d):
    c = d["c"]; qv = d["qv"]
    F = {}
    for n in (3, 5, 10, 15, 30, 60):
        e = ema(c, n)
        F[f"ema{n}_sl"] = np.diff(e, prepend=e[0])/c
        F[f"px_ema{n}"] = (c-e)/c
        v, sl = sg_causal(c, n, 1)
        F[f"sg{n}_sl"] = sl/c
    for n in (10, 20, 30):
        km = kama(c, n, 2, 30)
        F[f"kama{n}_sl"] = np.diff(km, prepend=km[0])/c
    for q_ in (0.001, 0.01, 0.1):
        kf = kalman(c, q_, 1.0)
        F[f"kal{q_}_sl"] = np.diff(kf, prepend=kf[0])/c
    for n in (5, 15, 30):
        F[f"med{n}"] = (c-np.nan_to_num(med_filt(c, n), nan=c[0]))/c
    # 곡률
    e10 = ema(c, 10); e30 = ema(c, 30)
    F["curv10"] = (np.diff(e10, prepend=e10[0]) - np.roll(np.diff(e10, prepend=e10[0]), 1))/c
    F["ema_cross"] = (e10-e30)/c
    # 효율비 (추세 강도)
    for n in (15, 60):
        chg = np.abs(pd.Series(c).diff(n).values)
        path = pd.Series(np.abs(np.diff(c, prepend=c[0]))).rolling(n).sum().values
        F[f"er{n}"] = np.where(path > 0, chg/path, 0.0)
    # 변동성·거래량 국면
    lr = pd.Series(np.log(c)).diff()
    F["rv30"] = lr.rolling(30).std().values
    F["rv_ratio"] = (lr.rolling(30).std()/lr.rolling(1440).std()).values
    F["qvz"] = (pd.Series(qv).rolling(30).mean()/pd.Series(qv).rolling(1440).mean()).values
    # 수익률 계열
    for n in (3, 5, 15, 30, 60):
        F[f"r{n}"] = pd.Series(c).pct_change(n).values
    return pd.DataFrame(F)


P("[1] 극단 분위에서 역방향(평균회귀) 신호의 크기")
P(f"  {'종목':5s} {'신호':>18s} {'구간':>4s} {'분위':>7s} {'n':>7s} {'역방향 수익bp':>12s} {'정확도':>7s} {'문턱':>6s} {'초과':>6s}")
for s in SYM:
    d = D[s]; c = d["c"]
    X = feats(d)
    for nm in ("px_ema15", "med15", "ema10_sl", "sg5_sl"):
        v = X[nm].values
        q = pd.Series(v).rank(pct=True).values
        for h in HOR:
            F = fw(c, h)
            for lo, hi, lab in ((0, .001, "0.1%"), (0, .01, "1%"), (0, .05, "5%")):
                sel = np.isfinite(F) & np.isfinite(v) & (q >= lo) & (q < hi)
                if sel.sum() < 300: continue
                # 신호 하위 = 가격이 필터보다 크게 아래 → 반등 기대(롱)
                a = F[sel].mean()*1e4; acc = (F[sel] > 0).mean()
                th = TH[(s, h)]
                P(f"  {s[:3]:5s} {nm:>18s} {h:3d}분 하위{lab:>4s} {int(sel.sum()):7d} {a:+11.2f} {acc*100:6.2f}% {th:5.1f}% {acc*100-th:+5.1f}")

P("\n[2] 다중 필터 정렬 — 여러 필터가 동시에 같은 방향일 때")
for s in SYM:
    d = D[s]; c = d["c"]; X = feats(d)
    cols = ["px_ema5", "px_ema15", "px_ema30", "med15", "ema10_sl", "kal0.01_sl"]
    sign = np.sign(X[cols].values)
    agree = sign.sum(axis=1)
    P(f"  ── {s} (6개 필터 중 몇 개가 하락 방향인가) ──")
    for h in HOR:
        F = fw(c, h)
        row = f"    {h:3d}분: "
        for k in (-6, -5, -4, 4, 5, 6):
            sel = np.isfinite(F) & (agree == k)
            if sel.sum() < 500: continue
            acc = (F[sel] > 0).mean() if k < 0 else (F[sel] < 0).mean()
            row += f"{k:+d}:{F[sel].mean()*1e4:+5.1f}bp({acc*100:4.1f}%,n={int(sel.sum())//1000}k) "
        P(row)

P("\n[3] 조건부 — 변동성·추세강도 국면별로 평균회귀가 강해지나")
for s in SYM:
    d = D[s]; c = d["c"]; X = feats(d)
    v = X["px_ema15"].values; q = pd.Series(v).rank(pct=True).values
    P(f"  ── {s} (px-EMA15 하위5% = 급락 후 반등 기대) ──")
    for cond_nm, cond in (("전체", np.ones(len(c), bool)),
                          ("고변동(rv비율>1.5)", X["rv_ratio"].values > 1.5),
                          ("저변동(rv비율<0.7)", X["rv_ratio"].values < 0.7),
                          ("추세약함(ER60<0.15)", X["er60"].values < 0.15),
                          ("추세강함(ER60>0.4)", X["er60"].values > 0.4),
                          ("거래량많음(qvz>1.5)", X["qvz"].values > 1.5)):
        for h in (30, 60):
            F = fw(c, h)
            sel = np.isfinite(F) & (q <= 0.05) & cond
            if sel.sum() < 500: continue
            acc = (F[sel] > 0).mean(); th = TH[(s, h)]
            P(f"    {cond_nm:>20s} {h:3d}분 n={int(sel.sum()):6d} 수익 {F[sel].mean()*1e4:+7.2f}bp 정확도 {acc*100:6.2f}% (문턱 {th:.1f}%, {acc*100-th:+.1f})")

P("\n[4] 학습 모델 — 모든 필터 특징으로 방향 예측 (시간 분할)")
try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    ok = True
except Exception as e:
    P(f"  sklearn 없음: {e}"); ok = False
if ok:
    for s in SYM:
        d = D[s]; c = d["c"]; X = feats(d)
        idx = d["idx"]
        for h in (30, 60):
            y = (fw(c, h) > 0).astype(float)
            m = np.isfinite(X.values).all(axis=1) & np.isfinite(y)
            Xv = X.values[m]; yv = y[m]; tv = idx[m]
            tr = tv < pd.Timestamp("2024-01-01", tz="UTC")
            te = ~tr
            if tr.sum() < 10000 or te.sum() < 5000: continue
            sub = np.arange(0, tr.sum(), 5)          # 겹침 줄이려 5분 간격 표본
            for nm, mdl in (("로지스틱", LogisticRegression(max_iter=300)),
                            ("GBM", HistGradientBoostingClassifier(max_iter=150, max_depth=4, learning_rate=0.06))):
                if nm == "로지스틱":
                    sc = StandardScaler().fit(Xv[tr][sub])
                    mdl.fit(sc.transform(Xv[tr][sub]), yv[tr][sub])
                    p = mdl.predict_proba(sc.transform(Xv[te]))[:, 1]
                    ptr = mdl.predict_proba(sc.transform(Xv[tr][sub]))[:, 1]
                else:
                    mdl.fit(Xv[tr][sub], yv[tr][sub])
                    p = mdl.predict_proba(Xv[te])[:, 1]
                    ptr = mdl.predict_proba(Xv[tr][sub])[:, 1]
                th = TH[(s, h)]
                acc_tr = ((ptr > .5) == (yv[tr][sub] > .5)).mean()
                acc = ((p > .5) == (yv[te] > .5)).mean()
                hi = p >= np.quantile(p, 0.9); lo = p <= np.quantile(p, 0.1)
                acc_hi = (yv[te][hi] > .5).mean(); acc_lo = (yv[te][lo] < .5).mean()
                P(f"  {s[:3]} {h:3d}분 {nm:7s}: 학습 {acc_tr*100:5.2f}% 테스트 {acc*100:5.2f}% | "
                  f"확신상위10% {acc_hi*100:5.2f}% 하위10% {acc_lo*100:5.2f}% | 문턱 {th:.1f}% → "
                  f"{'통과' if max(acc_hi, acc_lo)*100 > th else f'{max(acc_hi,acc_lo)*100-th:+.1f}%p 미달'}")
