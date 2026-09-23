"""방향 2~5 동시 검증 — 분위수 회귀·레짐 분리·로그효용 최적화·실행 학습.
1차 신호는 z0.6(유동성 충격, n=2428). 전부 롤링, 임계는 학습 구간에서만.
사용: python research/dir2345.py"""
import sys, warnings, gc
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from meta_strong import attach, FEAT, SYM
from z0_6b import build as build_z6
from sklearn.linear_model import QuantileRegressor, LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
import exit_tier as X
from absfloor import met
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
T0 = pd.Timestamp("2021-06-01", tz="UTC")
TE = pd.Timestamp("2024-01-01", tz="UTC"); REC = pd.Timestamp("2025-01-01", tz="UTC")

E6 = [x for x in build_z6(0.005, 1.0, 30, 240, 30) if x[4] >= T0]
AV = attach(E6)
TS = pd.DatetimeIndex([a[0][4] for a in AV])
XM = np.array([a[1] for a in AV], np.float32)
RET = np.array([a[0][2] for a in AV])
MAE = np.array([a[0][3] for a in AV])
P(f"1차 신호 {len(AV)}건, 특징 {XM.shape[1]}개")


def base_show(E, lab):
    r = np.array([x[2] for x in E])
    out = f"  {lab:40s} n={len(E):5d} {r.mean()*1e4:+7.1f}bp |"
    for w, Nm, tag in ((1.0, 1.0, "무레버"), (2.0, 3.0, "저레버"), (3.0, 5.0, "중간")):
        eq, fl, liq = X.sim(E, w, Nm); m = met(eq, fl, liq)
        out += f" {tag} {'파산' if m is None else f'{m[0]:+7.1f}%'}"
    P(out)


def rolling_apply(fn, win_days=365, refit_days=30, ncol=10, start="2021-06-01"):
    """fn(Xtr, rtr, mtr, Xte) -> (통과여부 배열, 가중 배열)"""
    keep = []
    ends = pd.date_range(pd.Timestamp(start, tz="UTC"), TS[-1], freq=f"{refit_days}D", tz="UTC")
    for e in ends:
        s0 = e - pd.Timedelta(days=win_days)
        tr = (TS >= s0) & (TS < e); te = (TS >= e) & (TS < e+pd.Timedelta(days=refit_days))
        if tr.sum() < 80 or te.sum() < 1: continue
        Xtr = XM[tr]
        cr = np.abs(np.array([np.corrcoef(Xtr[:, j], RET[tr])[0, 1] if Xtr[:, j].std() > 0 else 0
                              for j in range(XM.shape[1])]))
        cr[~np.isfinite(cr)] = 0
        cols = np.argsort(-cr)[:ncol]
        try:
            ok, wt = fn(Xtr[:, cols], RET[tr], MAE[tr], XM[te][:, cols])
        except Exception:
            continue
        gi = np.where(te)[0]
        for k, g in enumerate(gi):
            if ok[k]:
                x = AV[g][0]
                keep.append((x[0], x[1], x[2], x[3], x[4], x[5], x[6], float(wt[k])))
    keep.sort(key=lambda z: z[0])
    return keep


P("\n[기준] 1차 신호 단독")
base_show(E6, "z0.6 단독")

P("\n" + "="*120)
P("[방향2] 조건부 분위수 예측 → 켈리 사이징 (점 예측 포기)")
P("="*120)
def f_quantile(Xtr, rtr, mtr, Xte, qlo=0.25, qhi=0.75, mode="filter"):
    sc = StandardScaler().fit(Xtr)
    Ztr = sc.transform(Xtr); Zte = sc.transform(Xte)
    mlo = QuantileRegressor(quantile=qlo, alpha=0.01, solver="highs").fit(Ztr, rtr)
    mhi = QuantileRegressor(quantile=qhi, alpha=0.01, solver="highs").fit(Ztr, rtr)
    plo = mlo.predict(Zte); phi = mhi.predict(Zte)
    mid = (plo+phi)/2.0
    width = np.maximum(phi-plo, 1e-6)
    # 켈리 근사: 기대수익 / 분산 ∝ mid / width^2
    kel = mid/(width**2)
    ktr_lo = mlo.predict(Ztr); ktr_hi = mhi.predict(Ztr)
    ktr = ((ktr_lo+ktr_hi)/2)/np.maximum(ktr_hi-ktr_lo, 1e-6)**2
    th = np.quantile(ktr, 0.5)
    ok = kel >= th
    wt = np.clip(kel/max(abs(th), 1e-9), 0.5, 3.0) if mode == "size" else np.ones(len(kel))
    return ok, wt
for mode in ("filter", "size"):
    base_show(rolling_apply(lambda a, b, c, d: f_quantile(a, b, c, d, mode=mode)), f"분위수회귀 켈리 [{mode}]")
for qlo, qhi in ((0.1, 0.9), (0.3, 0.7)):
    base_show(rolling_apply(lambda a, b, c, d: f_quantile(a, b, c, d, qlo=qlo, qhi=qhi)),
              f"분위수회귀 q={qlo}/{qhi}")

P("\n" + "="*120)
P("[방향3] 레짐 분리 — 변동성·추세 레짐별 모델 (단일 모델 포기)")
P("="*120)
# 레짐 특징: rv_ratio, er60 (이미 145특징에 있음) — 인덱스 찾기
cols_all = FEAT[SYM[0]]["cols"]
ir = cols_all.index("rvratio15_240") if "rvratio15_240" in cols_all else 0
ie = cols_all.index("er60") if "er60" in cols_all else 1
REG = np.where(XM[:, ir] > np.nanmedian(XM[:, ir]), 1, 0)   # 고변동/저변동
def f_regime(Xtr, rtr, mtr, Xte, rtr_reg=None, rte_reg=None):
    ok = np.ones(len(Xte), bool)
    return ok, np.ones(len(Xte))
for lab, mask in (("고변동 국면만", REG == 1), ("저변동 국면만", REG == 0)):
    E = [AV[i][0] for i in range(len(AV)) if mask[i]]
    base_show(E, f"z0.6 {lab}")
# 레짐별 메타 모델
def f_meta_regime(Xtr, rtr, mtr, Xte, reg_tr=None, reg_te=None):
    sc = StandardScaler().fit(Xtr)
    y = (rtr > 0).astype(int)
    if len(np.unique(y)) < 2: return np.ones(len(Xte), bool), np.ones(len(Xte))
    m = LogisticRegression(max_iter=300, C=0.05).fit(sc.transform(Xtr), y)
    ptr = m.predict_proba(sc.transform(Xtr))[:, 1]
    pte = m.predict_proba(sc.transform(Xte))[:, 1]
    return pte >= np.quantile(ptr, 0.5), np.ones(len(Xte))
for lab, mask in (("고변동", REG == 1), ("저변동", REG == 0)):
    idxs = np.where(mask)[0]
    if len(idxs) < 300: continue
    sub = [AV[i] for i in idxs]
    # 서브셋에 대해 롤링 메타
    global AV2
    keep = []
    ts2 = TS[mask]; X2 = XM[mask]; r2 = RET[mask]
    ends = pd.date_range(pd.Timestamp("2021-06-01", tz="UTC"), ts2[-1], freq="30D", tz="UTC")
    for e in ends:
        s0 = e - pd.Timedelta(days=365)
        tr = (ts2 >= s0) & (ts2 < e); te = (ts2 >= e) & (ts2 < e+pd.Timedelta(days=30))
        if tr.sum() < 60 or te.sum() < 1: continue
        Xtr = X2[tr]
        cr = np.abs(np.array([np.corrcoef(Xtr[:, j], r2[tr])[0, 1] if Xtr[:, j].std() > 0 else 0
                              for j in range(X2.shape[1])]))
        cr[~np.isfinite(cr)] = 0
        cc = np.argsort(-cr)[:10]
        try:
            ok, _ = f_meta_regime(Xtr[:, cc], r2[tr], None, X2[te][:, cc])
        except Exception:
            continue
        gi = np.where(te)[0]
        for k, g in enumerate(gi):
            if ok[k]: keep.append(sub[g][0])
    keep.sort(key=lambda z: z[0])
    base_show(keep, f"{lab} 국면 + 메타")

P("\n" + "="*120)
P("[방향4] 로그 효용 직접 최적화 (정확도 목적함수 포기)")
P("="*120)
def f_logutil(Xtr, rtr, mtr, Xte, N=3.0):
    """샘플 가중을 log(1+N*r)의 기여도로 주고 분류 → 복리 성장에 기여하는 거래를 배운다"""
    sc = StandardScaler().fit(Xtr)
    u = np.log1p(np.clip(N*rtr, -0.99, None))
    y = (u > 0).astype(int)
    if len(np.unique(y)) < 2: return np.ones(len(Xte), bool), np.ones(len(Xte))
    w = np.abs(u)/np.abs(u).mean()          # 성장률 기여가 큰 표본에 가중
    m = LogisticRegression(max_iter=300, C=0.05).fit(sc.transform(Xtr), y, sample_weight=w)
    ptr = m.predict_proba(sc.transform(Xtr))[:, 1]
    pte = m.predict_proba(sc.transform(Xte))[:, 1]
    return pte >= np.quantile(ptr, 0.5), np.ones(len(Xte))
for N in (1.0, 3.0, 5.0):
    base_show(rolling_apply(lambda a, b, c, d: f_logutil(a, b, c, d, N=N)), f"로그효용 가중 N={N}")

P("\n" + "="*120)
P("[방향5] 실행 학습 — 방향 대신 MAE·보유시간 예측 (방향 예측 포기)")
P("="*120)
def f_mae_filter(Xtr, rtr, mtr, Xte, q=0.5):
    """예측 MAE가 낮은 거래만 취한다 — 방향이 아니라 위험을 예측"""
    sc = StandardScaler().fit(Xtr)
    rg = Ridge(alpha=50.0).fit(sc.transform(Xtr), mtr)
    ptr = rg.predict(sc.transform(Xtr)); pte = rg.predict(sc.transform(Xte))
    th = np.quantile(ptr, q)
    return pte <= th, np.ones(len(Xte))
for q in (0.3, 0.5, 0.7):
    base_show(rolling_apply(lambda a, b, c, d: f_mae_filter(a, b, c, d, q=q)), f"예측MAE 하위{q*100:.0f}%만")
def f_mae_size(Xtr, rtr, mtr, Xte):
    sc = StandardScaler().fit(Xtr)
    rg = Ridge(alpha=50.0).fit(sc.transform(Xtr), mtr)
    pte = np.clip(rg.predict(sc.transform(Xte)), 1e-4, None)
    wt = np.clip(mtr.mean()/pte, 0.3, 3.0)
    return np.ones(len(Xte), bool), wt
base_show(rolling_apply(f_mae_size), "예측MAE 반비례 사이징(전량진입)")
def f_both(Xtr, rtr, mtr, Xte):
    ok1, _ = f_meta_regime(Xtr, rtr, mtr, Xte)
    ok2, _ = f_mae_filter(Xtr, rtr, mtr, Xte, q=0.7)
    return ok1 & ok2, np.ones(len(Xte))
base_show(rolling_apply(f_both), "방향메타 + 예측MAE 필터 결합")
