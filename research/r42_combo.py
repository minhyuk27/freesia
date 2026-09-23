"""R42 — z0.31(딥, V3 국면 레버리지) + R41 돌파 롱(볼록)을 한 계좌에.

돌파 구성(R41 최상, 롱만): N=4320분 최고 돌파, 60분 거래대금 ≥ 2×일평균, 손절 0.5σd, 1단위, 최대 3일.
  ※ 96개 조합 중 최상을 골랐다(선택 편향). 이웃 구성도 +0.1~0.2R이라 고원은 있음.
돌파 사이징: 명목 = 자본 × rf / 손절폭 (거래당 위험 rf = 0·2·5·10%), MAE = 손절폭(갭 포함은 실손익에 반영).
딥 사이징: R40 V3 (변동성국면² × 예측, 상한 6), 킬스위치 K1.
공유 자본, 총명목 ≤ 자본 × N, 열린 포지션 전부 동시 최악 가정 청산 점검.
사용: python research/r42_combo.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from r40_regime_leverage import W as DIP
V1 = {2.0: np.clip(DIP.vreg.values ** 2.0, 0.3, 4.0)}
V2 = np.clip(DIP.pred.values / 0.001, 0.5, 3.0)
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
MMR = 0.005
B = pd.read_parquet(R / "data" / "r41_events.parquet")
B = B[(B.N == 4320) & (B.v == 2.0) & (B.ks == 0.5) & (B.ka == 0.5) & (B.kt == 1.0) & (B.U == 1) & (B.side > 0)].copy()
from freesia import store
for s in ("BTCUSDT", "ETHUSDT"):
    ts = store.load("klines_1m", s).ts
    m = B.sym == s
    B.loc[m, "ts"] = ts.values[B.loc[m, "ent"].values]
B["ts"] = pd.DatetimeIndex(B.ts)
t0 = DIP.ts.min()
B = B[B.ts >= t0]
DIP["kind"] = 0; DIP["mult"] = np.minimum(V1[2.0] * V2, 6.0)
B = B.assign(kind=1, f=B.ent, x=B.ext, mae=B.risk, mult=0.0, pred=0.0)
ALL = pd.concat([DIP[["ts", "f", "x", "ret", "mae", "kind", "mult", "sym"]],
                 B[["ts", "f", "x", "ret", "mae", "kind", "mult", "sym", "risk"]]], ignore_index=True)
ALL = ALL.sort_values(["f", "sym"]).reset_index(drop=True)
YRS = (ALL.ts.max() - ALL.ts.min()).days / 365.25
P(f"딥 {len(DIP)/YRS:.0f}건/년 거래당 {DIP.ret.mean()*1e4:+.1f}bp | 돌파 {len(B)/YRS:.0f}건/년 평균 {B.R.mean():+.3f}R, 명목당 {B.ret.mean()*1e4:+.1f}bp")
md = DIP.groupby(DIP.ts.dt.to_period("M")).ret.sum(); mb = B.groupby(B.ts.dt.to_period("M")).ret.sum()
j = pd.concat([md, mb], axis=1).fillna(0)
P(f"월 합계 수익 상관(딥 vs 돌파): {j.corr().iloc[0,1]:+.3f}")


def sim(S, w, N, rf, L=120, X=0.02, H=720):
    cap = 1.0; open_ = []; closed = []; halt = -1; days = {}; peak = 1.0; mdd = 0.0; worst = 0.0
    f_, x_, r_, m_, k_, mu, ts_ = S.f.values, S.x.values, S.ret.values, S.mae.values, S.kind.values, S.mult.values, S.ts.values
    rk = S.risk.values if "risk" in S else np.zeros(len(S))
    for i in range(len(S)):
        a = f_[i]; keep = []
        for p in open_:
            if p[0] <= a:
                cap += p[1] * p[2]
                if p[4] == 0: closed.append((p[0], p[2]))
            else: keep.append(p)
        open_ = keep
        if cap <= 1e-9: break
        if k_[i] == 0:
            recent = sum(r for e, r in closed if a - L < e <= a)
            if recent < -X: halt = max(halt, a + H)
            closed = [c for c in closed if c[0] > a - L]
            if a < halt or w == 0: continue
            want = cap * w * mu[i]
        else:
            if rf == 0: continue
            want = cap * rf / rk[i]
        used = sum(p[1] for p in open_)
        n_ = min(want, max(cap * N - used, 0.0))
        if n_ <= 1e-12: continue
        loss = n_ * m_[i] + sum(p[1] * p[3] for p in open_)
        worst = min(worst, -loss / cap)
        if loss >= cap * (1 - MMR): cap = 0.0; break
        open_.append((x_[i], n_, r_[i], m_[i], k_[i]))
        peak = max(peak, cap); mdd = min(mdd, cap / peak - 1)
        days[pd.Timestamp(ts_[i]).normalize()] = cap
    for p in open_: cap += p[1] * p[2]
    if cap <= 1e-9: return None
    eq = pd.Series(days).sort_index(); eq.iloc[-1] = cap
    return eq, mdd, worst


def show(lab, w, N, rf):
    res = sim(ALL, w, N, rf)
    if res is None:
        P(f"  {lab:30s} 파산"); return
    eq, mdd, worst = res
    cg = eq.iloc[-1] ** (1 / YRS) - 1
    mon = (eq.resample("ME").last() / eq.resample("ME").last().shift(1).fillna(1.0) - 1) * 100
    yr = eq.resample("YE").last() / eq.resample("YE").last().shift(1).fillna(1.0) - 1
    P(f"  {lab:30s} 연 {cg*100:+7.0f}% 월 {((1+cg)**(1/12)-1)*100:+5.1f}% | 월 10/50/90 {mon.quantile(.1):+4.0f}/{mon.quantile(.5):+3.0f}/{mon.quantile(.9):+4.0f}"
      f" ≥100%달 {int((mon>=100).sum())} | MDD {mdd*100:5.1f}% 최악 {worst*100:5.1f}% | " + " ".join(f"{i.year%100}{v*100:+.0f}" for i, v in yr.items()))


for rf in (0.02, 0.05, 0.10):
    show(f"돌파만 위험 {rf*100:.0f}%", 0, 10, rf)
for w, N in ((1, 3), (1, 5), (2, 8), (3, 12)):
    P(f"\n── 딥 w{w} N{N}")
    for rf in (0.0, 0.02, 0.05, 0.10):
        show(f"딥 + 돌파 위험 {rf*100:.0f}%", w, N, rf)
