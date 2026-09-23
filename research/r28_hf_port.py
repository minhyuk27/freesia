"""R28 — R27 워크포워드 선택 딥의 계좌 시뮬레이션 (공유자본·동시보유·청산 점검).

회계: 거래마다 명목 = 자본 × w 를 먼저 정하고 수익을 곱한다(exit_tier 방식).
      동시 보유 명목 합 ≤ 자본 × Nmax. 같은 종목 다중 층은 하나의 큰 포지션으로 합산된다.
청산 점검: 열린 포지션 전부가 동시에 각자의 MAE를 맞는다고 가정(보수적). 합계 손실 ≥ 자본×(1−MMR)이면 파산.
비용 민감도: 거래당 추가 비용(슬리피지·관통 악화) +3·+6·+10bp.
사용: python research/r28_hf_port.py [접미사 예: _e5]"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
MMR = 0.005
SUF = sys.argv[1] if len(sys.argv) > 1 else ""
W = pd.read_parquet(R / "data" / f"r27_wf{SUF}.parquet")
W["ts"] = pd.DatetimeIndex(W.ts)
W = W.sort_values(["f", "sym"]).reset_index(drop=True)
# 종목별 분 인덱스는 두 종목이 같은 시간축(연속 1분)이라 f를 공통 시계로 쓴다
T0 = W.ts.min()


def sim(S, w, Nmax, extra=0.0, size_by_pred=False):
    cap = 1.0; open_ = []; liq = 0
    days = {}; peak = 1.0; mdd = 0.0; worst = 0.0
    f_ = S.f.values; x_ = S.x.values; r_ = S.ret.values - extra / 1e4; m_ = S.mae.values
    p_ = S.pred.values; ts_ = S.ts.values
    for i in range(len(S)):
        a = f_[i]
        keep = []
        for p in open_:
            if p[0] <= a: cap += p[1] * p[2]
            else: keep.append(p)
        open_ = keep
        if cap <= 1e-9: break
        wi = w * (min(max(p_[i] / 0.001, 0.5), 3.0) if size_by_pred else 1.0)
        used = sum(p[1] for p in open_)
        n_ = min(cap * wi, max(cap * Nmax - used, 0.0))
        if n_ <= 1e-12: continue
        loss = n_ * m_[i] + sum(p[1] * p[3] for p in open_)
        worst = min(worst, -loss / cap)
        if loss >= cap * (1 - MMR):
            liq += 1; cap = 0.0; break
        open_.append((x_[i], n_, r_[i], m_[i]))
        peak = max(peak, cap); mdd = min(mdd, cap / peak - 1)
        days[pd.Timestamp(ts_[i]).normalize()] = cap
    for p in open_: cap += p[1] * p[2]
    if cap <= 1e-9 or not days: return None, liq
    eq = pd.Series(days).sort_index()
    eq.iloc[-1] = cap
    return (eq, mdd, worst), liq


def show(S, lab, w, Nmax, extra=0.0, size_by_pred=False, detail=False):
    res, liq = sim(S, w, Nmax, extra, size_by_pred)
    if res is None:
        P(f"  {lab:34s} 파산 (청산 {liq})"); return None
    eq, mdd, worst = res
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cg = eq.iloc[-1] ** (1 / yrs) - 1
    mon = eq.resample("ME").last().pct_change().dropna()
    mcg = (1 + cg) ** (1 / 12) - 1
    yr = eq.resample("YE").last() / eq.resample("YE").last().shift(1).fillna(1.0) - 1
    P(f"  {lab:34s} 연 {cg*100:+8.1f}% 월 {mcg*100:+6.2f}% MDD {mdd*100:6.1f}% 최악여유 {worst*100:6.1f}% "
      f"월승률 {float((mon>0).mean()):.2f} | " + " ".join(f"{i.year%100}{v*100:+.0f}%" for i, v in yr.items()))
    return cg


if __name__ == "__main__":
    for th in (0.0005, 0.001, 0.002):
        S = W[W.pred > th].reset_index(drop=True)
        yrs = (S.ts.max() - S.ts.min()).days / 365.25
        epi = S.assign(h=S.ts.dt.floor("15min")).drop_duplicates(["sym", "h"])
        P(f"\n[예측>{th*1e4:.0f}bp] 거래 {len(S)/yrs:.0f}/년, 사건 {len(epi)/yrs/365:.1f}/일, 거래당 {S.ret.mean()*1e4:+.2f}bp, "
          f"보유 중앙 {np.median(S.x-S.f):.0f}분, MAE95 {np.percentile(S.mae,95)*100:.2f}%")
        P("  3단 레버리지 (w=거래당 명목/자본, Nmax=총명목 상한)")
        for lab, w, Nm in (("① 무레버  w0.25 N1", 0.25, 1.0), ("① 무레버  w0.5 N1", 0.5, 1.0),
                           ("② 중간    w0.5 N3", 0.5, 3.0), ("② 중간    w1 N3", 1.0, 3.0),
                           ("③ 공격    w1 N5", 1.0, 5.0), ("③ 공격    w2 N8", 2.0, 8.0),
                           ("③ 극공격  w3 N12", 3.0, 12.0)):
            show(S, lab, w, Nm)
        P("  예측 비례 사이징 (w × clip(pred/10bp, 0.5, 3))")
        for lab, w, Nm in (("무레버 w0.25 N1", 0.25, 1.0), ("중간 w0.5 N3", 0.5, 3.0), ("공격 w1 N5", 1.0, 5.0)):
            show(S, lab, w, Nm, size_by_pred=True)
        P("  비용 민감도 (중간 w0.5 N3)")
        for ex in (3.0, 6.0, 10.0):
            show(S, f"추가비용 +{ex:.0f}bp", 0.5, 3.0, extra=ex)
