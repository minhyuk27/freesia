"""R122-1 — 엣지 공장 1단계: 진입 메커니즘 110개의 거래 목록(롱·숏 양쪽, 조건 없음).

목적: 기존 두 엣지(z0.31 딥 지정가, z0.33 세션 돌파)의 뼈대를 롱·숏 양쪽·여러 시간 규모로 펼친다.
  숏·짧은 세션 등 '평균이 음수'라 버렸던 메커니즘도 조건(2단계)을 붙이면 살아나는 틈새가 있는지 보기 위해 전부 만든다.
A. 딥/급등 지정가(r26 run, 봉 종가 테이커 청산, 메이커 진입): 봉 B ∈ {5, 10, 15, 30, 60}분 × 깊이 k ∈ {3, 4, 5, 6, 8}σ × 방향(롱 = 시가 아래 매수, 숏 = 시가 위 매도) = 50개
B. 세션 돌파(스톱 진입 테이커 + 슬리피지 2bp, 청산 테이커 + 2bp, 펀딩 실제): 세션 Ph ∈ {4, 8, 12, 24, 48}시간(시작 시각 앙상블)
   × 수준 {DT N4 K0.5, LW R3 K1.0, DT N2 K0.7} × 방향 ±1 × 청산 {X0 세션 종가, X4 다음 세션 종가} = 60개
출력: data/r122_trades.parquet — mech, fam, side, sym, e(진입 분), x(청산 분), ret(비용·펀딩 후), mae, px
사용: python research/r122_mechanisms.py"""
import sys, warnings, time
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store, costs
import r26_hf_twosided as H26, r100_lw_deep as B, r102_joint_account as J
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
SES = {4: range(0, 4), 8: range(0, 8), 12: range(0, 12), 24: range(0, 24, 2), 48: range(0, 48, 4)}
LVL = [("DT", 4, 0.5), ("LW", 3, 1.0), ("DT", 2, 0.7)]


@numba.njit(cache=True)
def run_side(s0, s1, o, h, l, c, lv, side, xm):
    """세션 돌파(방향 side): 반환 진입 분 인덱스, 청산 분 인덱스(마지막 봉), 진입 기준가, 청산가, 보유 중 역행 극값."""
    n = len(s0); J0 = np.full(n, -1); J1 = np.full(n, -1); PX = np.full(n, np.nan); EX = np.full(n, np.nan); AD = np.full(n, np.nan)
    for k in range(n - 1):
        if np.isnan(lv[k]): continue
        a = s0[k]; b = s1[k]; endk = s1[k + 1] if xm >= 3 else b
        j0 = -1
        for j in range(a, b):
            if (side == 1 and h[j] >= lv[k]) or (side == -1 and l[j] <= lv[k]): j0 = j; break
        if j0 < 0: continue
        px = max(lv[k], o[j0]) if side == 1 else min(lv[k], o[j0]); ad = px
        for j in range(j0, endk):
            if side == 1 and l[j] < ad: ad = l[j]
            if side == -1 and h[j] > ad: ad = h[j]
        J0[k] = j0; J1[k] = endk - 1; PX[k] = px; EX[k] = c[endk - 1]; AD[k] = ad
    return J0, J1, PX, EX, AD


if __name__ == "__main__":
    t_ = time.time(); out = []
    for sym in ("BTCUSDT", "ETHUSDT"):
        k = store.load("klines_1m", sym); k["ts"] = pd.DatetimeIndex(k.ts); mi = J.mins(k.ts)
        o, h, l, c = k.open.values, k.high.values, k.low.values, k.close.values
        rv1 = pd.Series(np.log(c)).diff().rolling(300).std().shift(1).values
        # A. 딥/급등 지정가
        for Bb in (5, 10, 15, 30, 60):
            for kx in (3.0, 4.0, 5.0, 6.0, 8.0):
                for side in (1, -1):
                    ei, ex, rr, mm = H26.run(o, h, l, c, rv1, Bb, side, kx, 0, 1.0, 0, H26.EPS, H26.FM, H26.FT)
                    if len(ei) == 0: continue
                    s0 = (ei // Bb) * Bb; T = o[s0] * (1 - side * kx * rv1[s0] * np.sqrt(Bb))
                    out.append(pd.DataFrame(dict(mech=f"DIP B{Bb} k{kx:g} {'L' if side > 0 else 'S'}", fam="dip", side=side, sym=sym,
                                                 e=mi[ei], x=mi[ex] + 1, ret=rr.astype("float32"), mae=mm.astype("float32"), px=T)))
        P(f"  {sym} 딥/급등 완료 ({time.time()-t_:.0f}s)")
        # B. 세션 돌파
        for Ph, anchors in SES.items():
            for a in anchors:
                s0, s1, O, Hs, Ls, C, full, sts = B.sessions(k, Ph, a)
                for (fam, N, K) in LVL:
                    d = J.level(fam, N, K, O, Hs, Ls, C)
                    for side in (1, -1):
                        lv = np.where(full, O + side * d, np.nan)
                        for xm in (0, 4):
                            J0, J1, PX, EX, AD = run_side(s0.astype(np.int64), s1.astype(np.int64), o, h, l, c, lv.astype(float), side, xm)
                            m = J0 >= 0
                            if not m.any(): continue
                            px = PX[m] * (1 + side * 2e-4); exv = EX[m] * (1 - side * 2e-4)
                            ret = side * (exv / px - 1) - 2 * costs.TAKER
                            out.append(pd.DataFrame(dict(mech=f"BRK {Ph}h {fam}{N}K{K:g} X{xm} {'L' if side > 0 else 'S'}", fam="brk", side=side, sym=sym,
                                                         e=mi[J0[m]], x=mi[J1[m]] + 1, ret=ret.astype("float32"),
                                                         mae=(side * (1 - AD[m] / px)).clip(min=0).astype("float32") if side > 0 else ((AD[m] / px - 1).clip(min=0)).astype("float32"),
                                                         px=px)))
            P(f"  {sym} 세션 {Ph}h 완료 ({time.time()-t_:.0f}s)")
    T = pd.concat(out, ignore_index=True)
    # 펀딩(돌파만 — 딥은 분 단위 보유): 롱은 지불, 숏은 수취
    br = T.fam == "brk"
    Tb = J.add_funding(T[br][["sym", "e", "x"]].copy())
    T.loc[br, "ret"] = (T.loc[br, "ret"].values - T.loc[br, "side"].values * Tb.fund.values).astype("float32")
    T["e"] = T.e.astype(np.int64); T["x"] = T.x.astype(np.int64)
    T.to_parquet(R / "data" / "r122_trades.parquet", index=False)
    S = T.groupby("mech").agg(n=("ret", "size"), bp=("ret", "mean"))
    S["bp"] *= 1e4
    P(f"\n메커니즘 {T.mech.nunique()}개, 거래 {len(T):,} ({time.time()-t_:.0f}s)")
    P("거래당 bp 분포(조건 없음): 롱 딥 " + f"{S[S.index.str.startswith('DIP') & S.index.str.endswith('L')].bp.median():+.1f}" +
      " · 숏 급등 " + f"{S[S.index.str.startswith('DIP') & S.index.str.endswith('S')].bp.median():+.1f}" +
      " · 롱 돌파 " + f"{S[S.index.str.startswith('BRK') & S.index.str.endswith('L')].bp.median():+.1f}" +
      " · 숏 돌파 " + f"{S[S.index.str.startswith('BRK') & S.index.str.endswith('S')].bp.median():+.1f}")
    P(S.sort_values("bp", ascending=False).round(1).to_string())
