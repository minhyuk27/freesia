"""R40 — 국면에 따라 공격적으로: 폭발적인 달에만 명목을 크게.

근거: z0.31 월 분포 상위 달(2024-03 +123% 등)은 전부 고변동 국면. 딥 엣지는 변동성과 함께 커진다
      (변동성목표 사이징이 딥에 해로웠던 이유). 그러면 반대로 '변동성이 클 때 더 크게'가 맞는가.
배율 m(거래별, 전부 인과적):
  V1 변동성 국면: m = clip((vol30 / vol365중앙)^α, 0.3, 4)  — vol30 = 전일까지 30일 일간 수익 표준편차
  V2 예측 비례:   m = clip(pred / 10bp, 0.5, 3)
  V3 = V1 × V2 (상한 6)
  V4 자기 성과 모멘텀: 직전 30일 청산 완료 손익 합 > 0 이면 ×2, 아니면 ×0.5 (계좌 수준 안티마틴게일)
계좌: R34 킬스위치(K1 120분 −2% → 720분 중지) 포함, 명목 = 자본 × w × m, 총명목 ≤ 자본 × N.
보고: 연·월 복리, 월 분포(10/50/90%), 월 +100% 이상 달 수, MDD, 최악 여유.
사용: python research/r40_regime_leverage.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R))
from freesia import store
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
MMR = 0.005
W = pd.read_parquet(R / "data" / "r32_wf.parquet")
W["ts"] = pd.DatetimeIndex(W.ts)
W = W[W.pred > 0.001].sort_values(["f", "sym"]).reset_index(drop=True)
W["ret"] = W.ret - 1.2e-4                                      # R36 청산 지연 비용
for s in ("BTCUSDT", "ETHUSDT"):
    k = store.load("klines_1m", s).set_index("ts").close.resample("D").last()
    r = np.log(k).diff()
    v30 = r.rolling(30).std(); vmed = v30.rolling(365, min_periods=120).median()
    reg = (v30 / vmed).shift(1)                                # 전일까지
    m = W.sym == s
    W.loc[m, "vreg"] = reg.reindex(W.loc[m, "ts"].dt.normalize()).values
W["vreg"] = W.vreg.fillna(1.0)
YRS = (W.ts.max() - W.ts.min()).days / 365.25


def sim(S, w, N, mult, eqmom=False, L=120, X=0.02, H=720):
    cap = 1.0; open_ = []; closed = []; halt = -1; days = {}; peak = 1.0; mdd = 0.0; worst = 0.0
    f_, x_, r_, m_, ts_ = S.f.values, S.x.values, S.ret.values, S.mae.values, S.ts.values
    long_hist = []
    for i in range(len(S)):
        a = f_[i]; keep = []
        for p in open_:
            if p[0] <= a:
                cap += p[1] * p[2]; closed.append((p[0], p[2])); long_hist.append((p[0], p[1] * p[2] / max(cap, 1e-12)))
            else: keep.append(p)
        open_ = keep
        if cap <= 1e-9: break
        recent = sum(r for e, r in closed if a - L < e <= a)
        if recent < -X: halt = max(halt, a + H)
        closed = [c for c in closed if c[0] > a - L]
        if a < halt: continue
        mi = mult[i]
        if eqmom:
            long_hist = [h for h in long_hist if h[0] > a - 43200]
            mi *= 2.0 if sum(h[1] for h in long_hist) > 0 else 0.5
        used = sum(p[1] for p in open_)
        n_ = min(cap * w * mi, max(cap * N - used, 0.0))
        if n_ <= 1e-12: continue
        loss = n_ * m_[i] + sum(p[1] * p[3] for p in open_)
        worst = min(worst, -loss / cap)
        if loss >= cap * (1 - MMR): cap = 0.0; break
        open_.append((x_[i], n_, r_[i], m_[i]))
        peak = max(peak, cap); mdd = min(mdd, cap / peak - 1)
        days[pd.Timestamp(ts_[i]).normalize()] = cap
    for p in open_: cap += p[1] * p[2]
    if cap <= 1e-9: return None
    eq = pd.Series(days).sort_index(); eq.iloc[-1] = cap
    return eq, mdd, worst


def show(lab, w, N, mult, eqmom=False):
    res = sim(W, w, N, mult, eqmom)
    if res is None:
        P(f"  {lab:38s} 파산"); return
    eq, mdd, worst = res
    cg = eq.iloc[-1] ** (1 / YRS) - 1
    mon = (eq.resample("ME").last() / eq.resample("ME").last().shift(1).fillna(1.0) - 1) * 100
    P(f"  {lab:38s} 연 {cg*100:+8.0f}% 월 {((1+cg)**(1/12)-1)*100:+6.1f}% | 월 10/50/90% {mon.quantile(.1):+5.0f}/{mon.quantile(.5):+4.0f}/{mon.quantile(.9):+5.0f}"
      f" 최고 {mon.max():+5.0f}% ≥100%달 {int((mon>=100).sum())} | MDD {mdd*100:5.1f}% 최악 {worst*100:5.1f}%")


V1 = {a: np.clip(W.vreg.values ** a, 0.3, 4.0) for a in (1.0, 2.0, 3.0)}
V2 = np.clip(W.pred.values / 0.001, 0.5, 3.0)
ONE = np.ones(len(W))
P(f"거래 {len(W)/YRS:.0f}/년, 거래당 {W.ret.mean()*1e4:+.2f}bp (청산지연 1.2bp 반영), vreg 분포 " +
  " ".join(f"{q:.0%}:{W.vreg.quantile(q):.2f}" for q in (.1, .5, .9)))
for w, N in ((0.5, 3), (1, 3), (1, 5), (2, 8), (3, 12)):
    P(f"\n── w{w} N{N}")
    show("V0 기본", w, N, ONE)
    for a in (1.0, 2.0, 3.0):
        show(f"V1 변동성국면 α{a:.0f}", w, N, V1[a])
    show("V2 예측비례", w, N, V2)
    show("V3 변동성α2 × 예측", w, N, np.minimum(V1[2.0] * V2, 6.0))
    show("V4 자기성과 모멘텀", w, N, ONE, eqmom=True)
    show("V3 + V4", w, N, np.minimum(V1[2.0] * V2, 6.0), eqmom=True)
