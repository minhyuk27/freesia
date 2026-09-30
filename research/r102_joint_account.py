"""R102 — z0.31 딥 + z0.33 24h 돌파 앙상블을 한 계좌(공유 자본)로: 동시 명목·펀딩·스톱 슬리피지·청산을 함께 반영.

R101 결합(샤프 2.91)은 두 전략의 일간 수익을 더한 근사였다. 여기서는 거래 목록을 시간순으로 한 엔진에 넣는다.
z0.33 거래 목록: R101 학습 1위(DT N4 K0.5 · MA5 · 세션 종가 · 변동성 조절)와 학습 상위 8 동일가중(민감도).
  24개 시작 시각 × BTC·ETH = 48 슬롯, 슬롯 명목 = 자본 × L33/48 × vt(변동성 조절, 0~3; 고정 구성은 1).
  진입 = max(돌파선, 그 분 시가) × (1 + 스톱 슬리피지 s), 청산 = 세션(또는 다음 세션) 종가 시장가(슬리피지 2bp).
  수수료 테이커 5bp×2. s ∈ {2, 5, 10, 20bp} (2bp가 R101의 14bp 왕복과 같다). 펀딩: 보유 중 정산 시각마다 롱이 rate 지불.
z0.31: R80/R101과 같은 딥 프레임(문턱 25bp·V3 배율·킬 3%/12h), 명목 = 자본 × w × m, z0.31 총명목 ≤ 자본 × N.
청산 판정(보수적, R40 방식): 새 진입 시 (열린 모든 포지션 명목 × 각자 보유 중 최대 역행) 합 ≥ 자본 × (1 − MMR) 이면 파산.
보고: 2022~23 / 2024~26(평가) 월 복리·월 중앙·하위 10%·월 ≥+100% 달·MDD, 최대 동시 명목, 펀딩 비용, 3단(w1N4 / w2N8 / w3N12).
사용: python research/r102_joint_account.py"""
import sys, warnings, time
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store, costs
import r100_lw_deep as B
import r80_max_return as R80
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
D0 = pd.Timestamp("2020-01-01", tz="UTC")
MMR = 0.0065
TOP1 = [("DT", 4, 0.5, 5, 0, "변동성조절")]
TOP8 = TOP1 + [("LW", 3, 1.2, 10, 4, "고정"), ("LW", 3, 1.2, 5, 4, "고정"), ("DT", 5, 0.5, 20, 4, "고정"), ("LW", 3, 1.2, 20, 4, "고정"),
               ("DT", 6, 0.5, 20, 4, "고정"), ("LW", 5, 1.0, 20, 4, "고정"), ("LW", 3, 1.0, 20, 4, "고정")]


def mins(ts):
    ts = pd.DatetimeIndex(ts); ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
    return ((ts - D0).total_seconds() // 60).astype(np.int64).values


@numba.njit(cache=True)
def run_long(s0, s1, o, h, l, c, lv, xm):
    """롱 돌파: 반환 (진입 분 인덱스, 청산 분 인덱스(마지막 봉), 진입 기준가, 청산가, 보유 중 최저가)."""
    n = len(s0); J0 = np.full(n, -1); J1 = np.full(n, -1); PX = np.full(n, np.nan); EX = np.full(n, np.nan); LO = np.full(n, np.nan)
    for k in range(n - 1):
        if np.isnan(lv[k]): continue
        a = s0[k]; b = s1[k]; endk = s1[k + 1] if xm >= 3 else b
        j0 = -1
        for j in range(a, b):
            if h[j] >= lv[k]: j0 = j; break
        if j0 < 0: continue
        px = max(lv[k], o[j0]); lo = px
        for j in range(j0, endk):
            if l[j] < lo: lo = l[j]
        J0[k] = j0; J1[k] = endk - 1; PX[k] = px; EX[k] = c[endk - 1]; LO[k] = lo
    return J0, J1, PX, EX, LO


def level(fam, N, K, O, H, L, C):
    Hs, Ls, Cs = pd.Series(H), pd.Series(L), pd.Series(C)
    if fam == "DT":
        HH = Hs.rolling(N).max().shift(1); LL = Ls.rolling(N).min().shift(1); HC = Cs.rolling(N).max().shift(1); LC = Cs.rolling(N).min().shift(1)
        return np.maximum(HH - LC, HC - LL).values * K
    return pd.Series(H - L).rolling(N).mean().shift(1).values * K


def breakout_trades(cfgs):
    """구성 목록의 48슬롯 거래(동일가중: 구성 수로 나눈 비중 wt)."""
    out = []
    for sym in ("BTCUSDT", "ETHUSDT"):
        k = store.load("klines_1m", sym); k["ts"] = pd.DatetimeIndex(k.ts); mi = mins(k.ts)
        o, h, l, c = k.open.values, k.high.values, k.low.values, k.close.values
        for a in range(24):
            s0, s1, O, H, L, C, full, sts = B.sessions(k, 24, a)
            Ro = pd.Series(H - L).shift(1).values; vt = np.clip(0.02 / (Ro / O), 0, 3)
            for (fam, N, K, M, xm, sz) in cfgs:
                d = level(fam, N, K, O, H, L, C); ma = pd.Series(C).rolling(M).mean().shift(1).values
                lv = np.where(full & (O > ma), O + d, np.nan)
                J0, J1, PX, EX, LO = run_long(s0.astype(np.int64), s1.astype(np.int64), o, h, l, c, lv.astype(float), xm)
                m = J0 >= 0
                out.append(pd.DataFrame(dict(sym=sym, a=a, cfg=f"{fam}{N}K{K}M{M}X{xm}", e=mi[J0[m]], x=mi[J1[m]] + 1, px=PX[m], ex=EX[m], lo=LO[m],
                                             sz=(vt[m] if sz == "변동성조절" else 1.0), wt=1.0 / len(cfgs))))
    T = pd.concat(out, ignore_index=True)
    return T[np.isfinite(T.sz) & (T.sz > 0)].reset_index(drop=True)


def add_funding(T):
    fr = []
    for sym in ("BTCUSDT", "ETHUSDT"):
        f = store.load("funding", sym); fm = mins(f.ts); cum = np.r_[0.0, np.cumsum(f.rate.values)]
        m = (T.sym == sym).values
        i0 = np.searchsorted(fm, T.e.values[m], "right"); i1 = np.searchsorted(fm, T.x.values[m], "right")   # (진입, 청산] 정산
        fr.append(pd.Series(cum[i1] - cum[i0], index=T.index[m]))
    T["fund"] = pd.concat(fr).sort_index()
    return T


def lw_events(T, s_in, s_out=2e-4):
    px = T.px.values * (1 + s_in); ex = T.ex.values * (1 - s_out)
    ret = ex / px - 1 - 2 * costs.TAKER - T.fund.values
    mae = np.maximum(1 - T.lo.values / px, 0) + costs.TAKER
    return pd.DataFrame(dict(e=T.e.values, x=T.x.values, ret=ret, mae=mae, k=1, mult=T.sz.values * T.wt.values))


def dip_events():
    W = R80.dip_frame(0.0025)
    mult = np.minimum(np.clip(W.vreg.values ** 1.5, 0.3, 4) * np.clip(W.pred.values / 0.001, 0.5, 3), 6)
    e = mins(W.ts)
    return pd.DataFrame(dict(e=e, x=e + (W.x.values - W.f.values), ret=W.ret.values, mae=W.mae.values, k=0, mult=mult))


def sim(E, w, N, L33, X=0.03, Hh=720, Lk=120):
    """E: 진입 분 순 정렬된 사건(k 0 = 딥, 1 = 돌파). 자본은 청산 시 실현(R40 방식)."""
    cap = 1.0; open_ = []; closed = []; halt = -1; days = {}; peak_not = 0.0; fund = 0.0
    e_, x_, r_, m_, k_, u_ = (E[c].values for c in ("e", "x", "ret", "mae", "k", "mult"))
    for i in range(len(E)):
        a = e_[i]; keep = []
        for p in open_:
            if p[0] <= a:
                cap += p[1] * p[2]
                if p[4] == 0: closed.append((p[0], p[2]))
            else: keep.append(p)
        open_ = keep
        if cap <= 1e-9: return None
        days[a // 1440] = cap
        if k_[i] == 0:
            closed = [q for q in closed if q[0] > a - Lk]
            if sum(r for _, r in closed) < -X: halt = max(halt, a + Hh)
            if a < halt: continue
            used = sum(p[1] for p in open_ if p[4] == 0)
            n_ = min(cap * w * u_[i], max(cap * N - used, 0.0))
        else:
            n_ = cap * L33 / 48 * u_[i]
        if n_ <= 1e-12: continue
        loss = n_ * m_[i] + sum(p[1] * p[3] for p in open_)
        if loss >= cap * (1 - MMR): return None
        open_.append((x_[i], n_, r_[i], m_[i], k_[i]))
        peak_not = max(peak_not, sum(p[1] for p in open_) / cap)
    for p in open_: cap += p[1] * p[2]
    days[max(days) + 1] = cap
    eq = pd.Series(days); eq.index = D0 + pd.to_timedelta(eq.index, unit="D")
    return eq, peak_not


def rep(lab, E, w, N, L33):
    res = sim(E, w, N, L33)
    if res is None:
        P(f"  {lab:34s} 파산"); return None
    eq, pk = res; d = R80.daily_from_eq(eq)
    a = R80.metrics(d, "2022-01-01", "2024-01-01"); b = R80.metrics(d, "2024-01-01", "2026-09-23")
    s = lambda x: x.mean() / x.std() * np.sqrt(365) if x.std() > 0 else np.nan
    P(f"  {lab:34s} 22~23 월 {a['mo']:+6.1f}% MDD {a['mdd']:5.0f}% | 평가 24~26 월 {b['mo']:+6.1f}% 중앙 {b['med']:+5.1f}% 하위10% {b['p10']:+6.1f}%"
      f" ≥100%달 {b['big']} MDD {b['mdd']:5.0f}% 샤프 {s(d['2024':]):.2f} | 최대 명목 {pk:4.1f}배 | {b['yr']}")
    return d


if __name__ == "__main__":
    t_ = time.time()
    DIP = dip_events(); P(f"z0.31 사건 {len(DIP):,} ({time.time()-t_:.0f}s)")
    TT = {}
    for nm, cf in (("TOP1", TOP1), ("TOP8", TOP8)):
        T = add_funding(breakout_trades(cf)); TT[nm] = T
        yrs = (T.e.max() - T.e.min()) / 1440 / 365.25
        P(f"{nm} 돌파 거래 {len(T):,} (하루 {len(T)/yrs/365.25:.1f}건), 평균 보유 {(T.x-T.e).mean()/60:.1f}h, 거래당 펀딩 {T.fund.mean()*1e4:+.2f}bp, "
          f"최대 역행 중앙 {np.median(1-T.lo/T.px)*100:.2f}% 99% {np.quantile(1-T.lo/T.px, .99)*100:.1f}% ({time.time()-t_:.0f}s)")
    # 돌파 단독: 슬리피지 민감도 (거래당 bp, 연도별)
    P("\n===== z0.33 단독 거래당 순수익(bp, 비용·펀딩 후) × 스톱 슬리피지")
    for nm, T in TT.items():
        for s_in in (2e-4, 5e-4, 10e-4, 20e-4):
            L = lw_events(T, s_in); yr = pd.Series(L.ret.values, index=(D0 + pd.to_timedelta(L.e, unit="min")).year)
            wr = yr.groupby(level=0).mean() * 1e4
            P(f"  {nm} s={s_in*1e4:4.0f}bp 전체 {L.ret.mean()*1e4:+6.1f}bp 승률 {(L.ret>0).mean()*100:.0f}% | " + " ".join(f"{y%100}:{v:+.0f}" for y, v in wr.items()))
    # 결합 계좌
    for nm, T in TT.items():
        for s_in in (2e-4, 10e-4):
            E = pd.concat([DIP, lw_events(T, s_in)], ignore_index=True)
            E = E[E.e >= mins([pd.Timestamp("2022-01-01", tz="UTC")])[0]].sort_values(["e", "k"]).reset_index(drop=True)
            P(f"\n===== 공유 계좌: z0.31 + z0.33[{nm}] · 스톱 슬리피지 {s_in*1e4:.0f}bp")
            for tier, (w, N) in (("무레버(w1N4)", (1, 4)), ("중간(w2N8)", (2, 8)), ("공격(w3N12)", (3, 12))):
                for L33 in (0, 6, 12, 24, 48):
                    rep(f"{tier} L33={L33}", E, w * 1.0, N, L33 * w)
            if nm == "TOP8" and s_in > 5e-4:
                for L33 in (12, 24, 48, 96):
                    rep(f"돌파 단독 L33={L33}", E[E.k == 1].reset_index(drop=True), 0, 0, L33)
    P(f"완료 ({time.time()-t_:.0f}s)")
    (R / "logs" / "r102.summary").write_text("done\n")
