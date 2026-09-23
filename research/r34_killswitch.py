"""R34 — 킬스위치(서킷브레이커): 일방향 붕괴에서 딥 매수를 멈춘다.

근거: 조사(R31) — 메이커는 일방향 붕괴에서 역선택으로 망한다(arXiv 2602.00776, 2025-10-10).
      R32에서도 폭락 구간 선택 거래 평균 −69bp.
규칙(전부 인과적):
  K1 손실 연쇄: 최근 L분 안에 '이미 청산된' 거래 손익 합 < −X 이면 H분 신규 진입 중지
  K2 캐스케이드: 체결 직전 accel(매도 가속)·oi15(OI 급감)가 학습구간(2020~2021) 극단이면 그 거래 건너뜀
입력: data/r32_wf.parquet (기존+전문가 모델 워크포워드 예측)
사용: python research/r34_killswitch.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
MMR = 0.005
W = pd.read_parquet(R / "data" / "r32_wf.parquet")
W["ts"] = pd.DatetimeIndex(W.ts)
W = W.sort_values(["f", "sym"]).reset_index(drop=True)
E0 = pd.read_parquet(R / "data" / "r32_events.parquet", columns=["ts", "accel", "oi15"])
E0 = E0[pd.DatetimeIndex(E0.ts).year <= 2021]
ACC = float(np.nanquantile(E0.accel, 0.98)); OIQ = float(np.nanquantile(E0.oi15, 0.02))
YRS = (W.ts.max() - W.ts.min()).days / 365.25


def sim(S, w, Nmax, L=60, X=None, H=240, casc=False, extra=0.0):
    cap = 1.0; open_ = []; closed = []; halt = -1; days = {}; peak = 1.0; mdd = 0.0; worst = 0.0; skipped = 0
    f_, x_, r_, m_ = S.f.values, S.x.values, S.ret.values - extra / 1e4, S.mae.values
    ac, oq, ts_ = S.accel.values, S.oi15.values, S.ts.values
    for i in range(len(S)):
        a = f_[i]; keep = []
        for p in open_:
            if p[0] <= a:
                cap += p[1] * p[2]; closed.append((p[0], p[2]))
            else: keep.append(p)
        open_ = keep
        if cap <= 1e-9: break
        if X is not None:
            recent = sum(r for e, r in closed if a - L < e <= a)
            if recent < -X: halt = max(halt, a + H)
            closed = [c for c in closed if c[0] > a - L]
        if a < halt: skipped += 1; continue
        if casc and ((np.isfinite(ac[i]) and ac[i] > ACC) or (np.isfinite(oq[i]) and oq[i] < OIQ)):
            skipped += 1; continue
        used = sum(p[1] for p in open_)
        n_ = min(cap * w, max(cap * Nmax - used, 0.0))
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
    return eq, mdd, worst, skipped


def show(S, lab, w, Nm, **kw):
    res = sim(S, w, Nm, **kw)
    if res is None: P(f"  {lab:40s} 파산"); return
    eq, mdd, worst, sk = res
    cg = eq.iloc[-1] ** (1 / YRS) - 1
    mon = eq.resample("ME").last().pct_change().dropna()
    yr = eq.resample("YE").last() / eq.resample("YE").last().shift(1).fillna(1.0) - 1
    oct10 = eq[(eq.index >= "2025-10-09") & (eq.index <= "2025-10-12")]
    o10 = (oct10.iloc[-1] / oct10.iloc[0] - 1) * 100 if len(oct10) > 1 else np.nan
    P(f"  {lab:40s} 연 {cg*100:+7.1f}% 월 {((1+cg)**(1/12)-1)*100:+5.2f}% MDD {mdd*100:6.1f}% 최악 {worst*100:6.1f}% "
      f"월승률 {float((mon>0).mean()):.2f} 건너뜀 {sk:5d} 25-10-10 {o10:+5.1f}% | " +
      " ".join(f"{i.year%100}{v*100:+.0f}%" for i, v in yr.items()))


P(f"캐스케이드 문턱(2020~21 학습): accel>{ACC:.2f}, oi15<{OIQ*100:.2f}%")
for th in (0.001, 0.002):
    S = W[W.pred > th].reset_index(drop=True)
    P(f"\n[예측>{th*1e4:.0f}bp] {len(S)/YRS:.0f}건/년, 거래당 {S.ret.mean()*1e4:+.2f}bp")
    for tier, w, Nm in (("무레버", 0.5, 1.0), ("중간", 1.0, 3.0), ("공격", 1.0, 5.0)):
        P(f" ── {tier} (w{w} N{Nm})")
        show(S, "킬스위치 없음", w, Nm)
        for L, X, H in ((60, 0.01, 240), (60, 0.02, 240), (120, 0.02, 720), (60, 0.03, 1440)):
            show(S, f"K1 {L}분 손실합<−{X*100:.0f}% → {H}분 중지", w, Nm, L=L, X=X, H=H)
        show(S, "K2 캐스케이드 극단 건너뜀", w, Nm, casc=True)
        show(S, "K1(60분 −2%→240분)+K2", w, Nm, L=60, X=0.02, H=240, casc=True)
    P(" ── 비용 +3bp 스트레스 (중간, K1+K2)")
    show(S, "K1+K2 추가비용 +3bp", 1.0, 3.0, L=60, X=0.02, H=240, casc=True, extra=3.0)
