"""R103 — z0.33 비용 줄이기: 스톱(테이커) 진입 대신 되돌림 지정가(메이커), 펀딩 과열 시 진입 생략, BNB 할인.

V0 기준: 돌파선 스톱 진입(테이커 5bp + 슬리피지 2bp), 세션 종가 시장가 청산(테이커 5bp + 2bp), 펀딩 실제 부과 (= R102).
V1 되돌림 지정가: 돌파가 확인된 뒤 돌파 가격(px)에 매수 지정가를 걸고, 같은 진입 세션 안에서 분 저가가 px × (1 − 1bp) 이하가 되면 체결(관통 요구, 메이커 2bp).
   안 내려오면 거래 없음. 청산은 V0와 같다.
V2 펀딩 필터: 진입 직전 마지막 확정 펀딩률 > f* 이면 생략 (f* = 0.01%·0.02%·0.03%, 학습 2020~23 성과로 하나를 고른다).
V3 = V1 + V2, V4 = BNB 할인(수수료 ×0.9)을 V3에 추가.
보고: 거래 수·거래당 bp(학습/평가·연도별), 돌파 단독 계좌 샤프(R102 sim_mtm), z0.31 공유 계좌 3단(최선 L33 근처).
사용: python research/r103_breakout_cost.py"""
import sys, warnings, time
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store, costs
import r102_joint_account as J
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
T0 = int(J.mins([pd.Timestamp("2020-01-01", tz="UTC")])[0])


@numba.njit(cache=True)
def retest(e, se, x, px, s, LO, t0, pen):
    n = len(e); F = np.full(n, -1); L = np.full(n, np.nan)
    for i in range(n):
        if x[i] - t0 > LO.shape[1] or e[i] < t0: continue
        tgt = px[i] * (1 - pen); f = -1
        for m in range(e[i] + 1, se[i]):
            if LO[s[i], m - t0] <= tgt: f = m; break
        if f < 0 or f >= x[i]: continue
        lo = px[i]
        for m in range(f, x[i]):
            v = LO[s[i], m - t0]
            if v < lo: lo = v
        F[i] = f; L[i] = lo
    return F, L


def last_funding(T):
    out = np.full(len(T), np.nan)
    for sym in ("BTCUSDT", "ETHUSDT"):
        f = store.load("funding", sym); fm = J.mins(f.ts); m = (T.sym == sym).values
        i = np.searchsorted(fm, T.e.values[m], "right") - 1; out[m] = np.where(i >= 0, f.rate.values[np.clip(i, 0, None)], np.nan)
    return out


def events(T, maker_in, fee_k=1.0, fmax=None):
    in_fee = (costs.MAKER if maker_in else costs.TAKER) * fee_k; out_fee = costs.TAKER * fee_k
    px = T.px.values * (1 if maker_in else 1 + 2e-4); ex = T.ex.values * (1 - 2e-4)
    ret = ex / px - 1 - in_fee - out_fee - T.fund.values
    mae = np.maximum(1 - T.lo.values / px, 0) + out_fee
    E = pd.DataFrame(dict(e=T.e.values, x=T.x.values, ret=ret, mae=mae, k=1, mult=T.sz.values * T.wt.values,
                          s=(T.sym.values == "ETHUSDT").astype(int), px=px, lf=T.lf.values))
    if fmax is not None: E = E[~(E.lf > fmax)]
    return E.reset_index(drop=True)


def bp_line(nm, E):
    y = (J.D0 + pd.to_timedelta(E.e, unit="min")).dt.year.values
    r = pd.Series(E.ret.values, index=y); tr = r[r.index < 2024]; te = r[r.index >= 2024]
    return (f"  {nm:34s} 거래 {len(E):6,} 학습 {tr.mean()*1e4:+6.1f}bp 평가 {te.mean()*1e4:+6.1f}bp 승률 {(te>0).mean()*100:.0f}% | "
            + " ".join(f"{k%100}:{v*1e4:+.0f}" for k, v in r.groupby(level=0).mean().items()))


if __name__ == "__main__":
    t_ = time.time()
    DIP = J.dip_events()
    LO, CL = J.grid(T0)
    for nm, cf in (("TOP1", J.TOP1), ("TOP8", J.TOP8)):
        T = J.add_funding(J.breakout_trades(cf)); T["lf"] = last_funding(T)
        xm = T.cfg.str[-1].astype(int).values; se = np.where(xm >= 3, T.x.values - 1440, T.x.values)
        F, Lr = retest(T.e.values.astype(np.int64), se.astype(np.int64), T.x.values.astype(np.int64), T.px.values, (T.sym.values == "ETHUSDT").astype(np.int64),
                       LO, T0, 1e-4)
        T1 = T[F >= 0].copy(); T1["e"] = F[F >= 0]; T1["lo"] = Lr[F >= 0]; T1 = J.add_funding(T1)
        P(f"\n===== {nm}: 돌파 {len(T):,}건 중 되돌림 체결 {len(T1):,}건 ({len(T1)/len(T)*100:.0f}%) ({time.time()-t_:.0f}s)")
        V = {"V0 스톱 진입(기준)": events(T, False), "V1 되돌림 지정가": events(T1, True)}
        # V2: 펀딩 문턱 — 학습 거래당 bp로 선택
        best = None
        for fmax in (1e-4, 2e-4, 3e-4):
            E = events(T, False, fmax=fmax); y = (J.D0 + pd.to_timedelta(E.e, unit="min")).dt.year.values
            v = E.ret.values[y < 2024].mean() * (y < 2024).sum()                      # 학습 총수익(거래당 × 건수)
            P(f"  펀딩 문턱 {fmax*1e4:.0f}bp: 학습 총수익 지표 {v:+.2f}, 거래 {len(E):,}")
            if best is None or v > best[1]: best = (fmax, v)
        V[f"V2 펀딩≤{best[0]*1e4:.0f}bp"] = events(T, False, fmax=best[0])
        V[f"V3 되돌림 + 펀딩≤{best[0]*1e4:.0f}bp"] = events(T1, True, fmax=best[0])
        V["V4 = V3 + BNB 할인"] = events(T1, True, fee_k=0.9, fmax=best[0])
        for k, E in V.items(): P(bp_line(k, E))
        start = J.mins([pd.Timestamp("2022-01-01", tz="UTC")])[0]
        Ls = (1, 2, 4) if nm == "TOP1" else (0.5, 1, 2)
        for k, E in V.items():
            E2 = E[E.e >= start].sort_values("e").reset_index(drop=True)
            J.rep(f"{k} 단독 L33={Ls[-1]*2}", E2, 0, 0, Ls[-1] * 2)
            EE = pd.concat([DIP, E2], ignore_index=True).sort_values(["e", "k"]).reset_index(drop=True)
            for tier, (w, N) in (("무레버", (1, 4)), ("공격", (3, 12))):
                for L in Ls: J.rep(f"{k[:2]} {tier} L33={L}", EE, w, N, L * w)
    P(f"완료 ({time.time()-t_:.0f}s)")
