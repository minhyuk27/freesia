"""±0.1% 중 먼저 닿는 쪽 예측 — 최대 배율 가정.
핵심 산수: 수익도 수수료도 명목 비례 → 레버리지는 무관. 손익분기 정확도 p* = (0.001 + 수수료율)/0.002
  테이커 왕복 0.100% → p* = 100.0% (불가능)   메이커 왕복 0.040% → p* = 70.0%   메이커+BNB 0.036% → p* = 68.0%
측정: 기저율, 도달 시간, 특징 37종 + 그래디언트 부스팅의 실제 정확도(확신 분위별), 학습 2020~2023 / 테스트 2024~.
사용: python research/tick01.py
"""
import sys, warnings
from pathlib import Path
import numba, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R))
from freesia import store
from sklearn.ensemble import HistGradientBoostingClassifier
TE = pd.Timestamp("2024-01-01", tz="UTC")
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)


@numba.njit(cache=True)
def barrier01(hi, lo, cl, x, H):
    n = len(cl); lab = np.zeros(n, np.int8); mins = np.zeros(n, np.int32)
    for i in range(n - 1):
        p0 = cl[i]; up = p0 * (1 + x); dn = p0 * (1 - x)
        for j in range(i + 1, min(i + 1 + H, n)):
            hu = hi[j] >= up; hd = lo[j] <= dn
            if hu and hd:
                lab[i] = 0; mins[i] = j - i; break      # 같은 봉에 양쪽 → 판정 불가
            if hu:
                lab[i] = 1; mins[i] = j - i; break
            if hd:
                lab[i] = -1; mins[i] = j - i; break
        else:
            mins[i] = H
    return lab, mins


P("손익분기 정확도 p* = (0.001 + 수수료율) / 0.002")
for nm, fee in (("테이커 왕복 0.100%", 0.0010), ("테이커+슬리피지 0.120%", 0.0012),
                ("메이커 왕복 0.040%", 0.0004), ("메이커+BNB 0.036%", 0.00036), ("수수료 0", 0.0)):
    P(f"  {nm:24s} p* = {(0.001 + fee) / 0.002 * 100:6.1f}%")

for sym in ("BTCUSDT", "ETHUSDT"):
    k = store.load("klines_1m", sym).set_index("ts")
    idx = k.index
    hi, lo, cl, o = k.high.values, k.low.values, k.close.values, k.open.values
    v, n_, tb = k.quote_volume.values, k.trades.values, k.taker_buy_quote_volume.values
    pr = store.load("premium_1m", sym).set_index("ts").close.reindex(idx).values * 1e4
    lab, mins = barrier01(hi, lo, cl, 0.001, 120)
    ok = lab != 0
    P("\n" + "=" * 90 + f"\n### {sym} 1분봉 {len(k):,}개  (±0.1% 배리어, 최대 120분)")
    P(f"  판정 가능 {ok.mean():.1%} | 기저율 위 {float((lab[ok]==1).mean()):.4f} | 도달 시간 중앙 {np.median(mins[ok]):.0f}분, 평균 {mins[ok].mean():.1f}분")
    for q in (0.25, 0.5, 0.75, 0.9):
        P(f"    도달 {int(q*100)}분위 {np.quantile(mins[ok], q):.0f}분", end="" if q < 0.9 else "\n")
    # ── 특징
    s = pd.Series
    lp = np.log(cl)
    F = pd.DataFrame(index=idx)
    for L in (1, 2, 3, 5, 10, 15, 30, 60):
        F[f"r{L}"] = s(lp).diff(L).values
        F[f"rz{L}"] = (s(lp).diff(L) / s(lp).diff(1).rolling(1440).std() / np.sqrt(L)).values
    imb = 2 * tb / np.where(v == 0, np.nan, v) - 1
    for L in (1, 3, 5, 15, 60):
        F[f"imb{L}"] = s(imb).rolling(L).mean().values
    for L in (5, 15, 60, 240):
        F[f"vol{L}"] = (s(v).rolling(L).sum() / s(v).rolling(1440).sum() * 1440 / L).values
        F[f"trd{L}"] = (s(n_).rolling(L).sum() / s(n_).rolling(1440).sum() * 1440 / L).values
        F[f"rv{L}"] = s(lp).diff().rolling(L).std().values
    F["prem"] = pr
    F["premz"] = ((s(pr) - s(pr).rolling(240).mean()) / s(pr).rolling(240).std()).values
    F["premd"] = s(pr).diff(5).values
    F["body"] = ((cl - o) / np.where(hi - lo == 0, np.nan, hi - lo))
    F["upw"] = ((hi - np.maximum(cl, o)) / np.where(hi - lo == 0, np.nan, hi - lo))
    F["dnw"] = ((np.minimum(cl, o) - lo) / np.where(hi - lo == 0, np.nan, hi - lo))
    F["rng"] = ((hi - lo) / cl)
    F["pos"] = ((cl - s(lo).rolling(60).min()) / (s(hi).rolling(60).max() - s(lo).rolling(60).min())).values
    F["hour"] = idx.hour
    F["dow"] = idx.dayofweek
    F["amih"] = (np.abs(s(lp).diff()) / s(v).replace(0, np.nan)).rolling(15).mean().values
    X = F.values
    good = ok & ~np.isnan(X).any(1)
    y = (lab == 1).astype(int)
    tr = good & (idx < TE); te = good & (idx >= TE)
    P(f"  특징 {X.shape[1]}종 | 학습 {tr.sum():,} / 테스트 {te.sum():,}")
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_depth=6,
                                       min_samples_leaf=200, l2_regularization=1.0, random_state=0)
    m.fit(X[tr], y[tr])
    p_tr = m.predict_proba(X[tr])[:, 1]; p_te = m.predict_proba(X[te])[:, 1]
    P(f"  전체 정확도: 학습 {((p_tr>0.5)==y[tr]).mean():.4f} | 테스트 {((p_te>0.5)==y[te]).mean():.4f}  (기저 {max(y[te].mean(),1-y[te].mean()):.4f})")
    P("  [확신 분위별 테스트 정확도] — 메이커 손익분기 70.0%")
    rows = []
    conf = np.abs(p_te - 0.5)
    for lo_, hi_ in ((0.0, 0.5), (0.5, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 0.99), (0.99, 1.0)):
        a, b = np.quantile(conf, lo_), np.quantile(conf, hi_)
        sel = (conf >= a) & (conf < b) if hi_ < 1.0 else (conf >= a)
        if sel.sum() < 100:
            continue
        acc = ((p_te[sel] > 0.5) == y[sel := np.where(te)[0][sel]] if False else ((p_te[(conf >= a) & (conf <= b if hi_ == 1.0 else conf < b)] > 0.5) == y[te][(conf >= a) & (conf <= b if hi_ == 1.0 else conf < b)]).mean())
        n = int(((conf >= a) & (conf < b)).sum())
        rows.append((f"{int(lo_*100)}~{int(hi_*100)}%", acc, n, acc - 0.70))
    P(pd.DataFrame(rows, columns=["확신분위", "정확도", "n", "손익분기(70%) 대비"]).round(4).to_string(index=False))
    # 최상위 확신 구간 세부
    for topq in (0.999, 0.9999):
        thr = np.quantile(conf, topq)
        sel = conf >= thr
        if sel.sum() > 30:
            P(f"  상위 {(1-topq)*100:.2f}% 확신 (n={int(sel.sum())}): 정확도 {(((p_te[sel]>0.5)==y[te][sel]).mean()):.4f}")
