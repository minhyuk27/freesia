"""R118a — 횡보(사이드웨이) 구간 페이드: R76 "다음 H시간 횡보" 예측이 맞는 구간에서만 양방향 지정가로 되돌림을 노린다.

가설: freesia의 두 생존 엣지(z0.31 딥매수 지정가, z0.33 24시간 돌파 롱)는 모두 롱 편향이다. R76은 다음 24h/12h이
"횡보"일지를 변동성·거래량 특징만으로 예측할 수 있음을 보였다(확신 상위 10%에서 적중 74~79% vs 기저 ~50%,
HistGB M4 기준). 숏도 낼 수 있는 시장중립적 엣지로 포트폴리오를 분산할 수 있는지 검증한다.

설계(사전 등록, 결과를 보기 전 확정 — 이후 그리드를 추가하지 않는다)
  국면 모델: R76의 특징·라벨 구성을 그대로 재사용(함수 import: r76_regime_next24.extra/label, r68_bar_patterns.build).
    4시간봉마다 다음 H시간(H=12,24) 횡보 확률을 연도별 확장창 워크포워드로 산출(연도 Y 모델은 "Y-01-01 − 1일"
    이전 데이터만 학습, 2021~2026 평가 — R76은 2022부터지만 여기서는 학습(2021~23) 구간 평가를 위해 2021부터).
    모델은 R76 M4(HistGB 3시드 평균, 횡보 확률 top-10%에서 가장 정확했던 모델). 활성 구간 = 확률이 "그 연도
    학습 구간 예측치" 상위 {10%, 20%} 분위(문턱은 학습 예측으로 고정, 평가 연도의 미래 분위를 쓰지 않는다).
  구간 [W0, W0+H) 안: 기준가 A = W0 분봉 시가, 밴드폭 b = k×σ(σ = W0까지 1분봉 로그수익률 24시간 실현변동성 ×
    √(H분), k ∈ {0.5, 0.75, 1.0}). A(1+b)에 매도 지정가, A(1−b)에 매수 지정가. 체결 = 분 고가/저가가 지정가를
    1bp 이상 관통(고가 > 지정가×(1+1bp) 매도, 저가 < 지정가×(1−1bp) 매수). 체결 후 청산 = A 복귀(지정가, 1bp
    관통) / 진입가×(1∓s×b) 스톱(스톱마켓, 슬리피지 3bp, s ∈ {1.0, 2.0, 없음}) / 구간 끝(시장가) 중 먼저 오는 것.
    종목당 동시 1포지션, 체결 후 반대쪽 주문은 구간 끝까지 유지(편당 재진입 1회만 — 두 번째 체결 없음).
  비용: 메이커 2bp(지정가 체결·TP 청산), 테이커 5bp(스톱·구간끝 시장가 청산). 펀딩: 정산 시각을 지나는
    포지션만 실제 펀딩(r102_joint_account.add_funding과 동일 방식, 롱은 양의 요율 지불·숏은 수취).
  평가: 구성별 거래수·거래/일·거래당 순bp(학습 2021~2023 vs 평가 2024~2026-09)·승률·연도별bp·무레버리지
    일간 샤프(진입일별 수익 합산, 동일 명목)·롱/숏 분리·코어(z0.31+z0.33, r118_engine.run_multi) 일간수익과의
    상관(2022+). 선택 규칙(사전 고정): 학습(2021~23) 샤프 최대. 무작위 구간 위약(동일 빈도, 시드 20개) 대비
    평가 샤프가 위약 중앙값을 넘어야 유효로 인정.
  산출: 선택 구성(+학습 샤프 상위 3) 표준 사건을 data/r118a_events.parquet에 저장(k=2, mult=1.0, cfg 열 추가,
    e/x는 r102_joint_account.mins 분 단위). 공유 계좌 테스트 run_multi(core+선택, 3, 12, {1: L/48, 2: Wc}),
    L ∈ {12,16}, Wc ∈ {0.5,1,2,4}를 코어 단독(w3N12 L12 평가 월 +32.9%·샤프2.66, L16 +35.0%·2.45)과 비교.
사용: python research/r118a_sideways_fade.py
"""
import sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
from sklearn.ensemble import HistGradientBoostingClassifier
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
import r68_bar_patterns as R68
import r76_regime_next24 as R76
import r102_joint_account as J
import r118_engine as G

P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
D0 = J.D0
SYMS = ("BTCUSDT", "ETHUSDT")
TRAIN_A, TRAIN_B = "2021-01-01", "2024-01-01"
EVAL_A, EVAL_B = "2024-01-01", "2026-09-23"


# ===================== 1) R76 국면 프레임 재구성(함수는 import, 조립 코드만 재사용) =====================
def build_regime_frame():
    closes = {s: store.load("klines_1m", s).set_index("ts").close.resample("4h").last() for s in SYMS}
    parts = []
    for s in SYMS:
        B = R68.build(s, "4h", 6, closes["ETHUSDT" if s == "BTCUSDT" else "BTCUSDT"])
        Ex = R76.extra(s); B = B.join(Ex, on="t")
        parts.append(B.assign(sym=s))
    D = pd.concat(parts, ignore_index=True).replace([np.inf, -np.inf], np.nan).sort_values(["sym", "t"]).reset_index(drop=True)
    D["yr"] = D.t.dt.year
    D["sig24"] = D.groupby("sym").r24.transform(lambda x: x.rolling(360, min_periods=120).std())
    D["z24"] = D.r24 / D.sig24; D["az24"] = D.z24.abs()
    for j in range(24): D[f"H_{j}"] = D[f"H_{j}"] / (D.sig24 / np.sqrt(24))
    prevz = D.groupby("sym").z24.shift(6)
    D["prevstate"] = np.where(np.sign(prevz) * D.z24 >= 0.5, 0, np.where(np.sign(prevz) * D.z24 <= -0.5, 1, 2))
    FE = [c for c in D.columns if (c[:2] in ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9") and c[2] == "_")] + \
         ["z24", "az24", "er1h", "pos24", "dd24", "ru24", "vtr"] + [f"H_{j}" for j in range(24)]
    return D, FE


def walkforward_sideways(D, FE, h, years=range(2021, 2027)):
    """R76 M4(HistGB 3시드)로 다음 h시간 횡보 확률을 연도별 워크포워드 산출. 학습 구간 예측 분위로 top10/20% 문턱 고정.
    반환: D에 p_side/active10/active20 열 추가, 연도별 확신 상위 10%/20% 정밀도(검증용) 리스트."""
    y, _ = R76.label(D, h); D = D.copy(); D["y"] = y
    D["p_side"] = np.nan; D["active10"] = False; D["active20"] = False
    prec = []
    for Y in years:
        tr = D[(D.t < pd.Timestamp(f"{Y}-01-01", tz="UTC") - pd.Timedelta(days=1)) & (D.y >= 0)]
        te = D[(D.yr == Y) & (D.y >= 0)]
        if len(tr) < 300 or len(te) == 0: continue
        models = [HistGradientBoostingClassifier(max_iter=300, learning_rate=0.04, max_leaf_nodes=31, min_samples_leaf=200,
                                                   l2_regularization=1.0, random_state=sd).fit(tr[FE].values, tr.y.values) for sd in range(3)]
        p_tr = np.mean([m.predict_proba(tr[FE].values) for m in models], axis=0)
        p_te = np.mean([m.predict_proba(te[FE].values) for m in models], axis=0)
        D.loc[te.index, "p_side"] = p_te[:, 2]
        thr10, thr20 = np.quantile(p_tr[:, 2], 0.9), np.quantile(p_tr[:, 2], 0.8)
        D.loc[te.index, "active10"] = p_te[:, 2] >= thr10
        D.loc[te.index, "active20"] = p_te[:, 2] >= thr20
        sel10 = p_te[:, 2] >= thr10; sel20 = p_te[:, 2] >= thr20
        base = (te.y.values == 2).mean()
        prec.append((Y, (te.y.values[sel10] == 2).mean() if sel10.sum() else np.nan, int(sel10.sum()),
                     (te.y.values[sel20] == 2).mean() if sel20.sum() else np.nan, int(sel20.sum()), base, len(te)))
    return D, prec


# ===================== 2) 분봉 페이드 시뮬레이션(numba) =====================
@numba.njit(cache=True)
def sim_fade(w0, b_, H, s_list, o, h, l, c, t0, n_s):
    """활성 구간 w0(분)마다 A=시가, 밴드 b_=k*sigma로 양방향 지정가를 걸고 페이드. s_list 길이 n_s의 스톱 배율별로
    병렬 상태머신을 돌려 한 번의 분봉 스캔으로 s 그리드를 모두 커버한다(진입·TP는 s와 무관, 스톱만 s별로 다름).
    반환: e(분), x(분, 청산 다음 분), side(+1/-1), entry_px, exit_px, mae, etype(0 스톱·1 TP·2 구간끝), sidx(s 인덱스), widx(구간 인덱스)."""
    nw = len(w0); maxtr = nw * 2 * n_s
    e_ = np.zeros(maxtr, np.int64); x_ = np.zeros(maxtr, np.int64); side_ = np.zeros(maxtr, np.int64)
    entry_ = np.zeros(maxtr); exit_ = np.zeros(maxtr); mae_ = np.zeros(maxtr)
    etype_ = np.zeros(maxtr, np.int64); sidx_ = np.zeros(maxtr, np.int64); widx_ = np.zeros(maxtr, np.int64)
    nt = 0; nmin = len(o)
    for wi in range(nw):
        W0 = w0[wi]; b = b_[wi]
        if not (b > 0): continue
        i0 = W0 - t0
        if i0 < 0 or i0 >= nmin: continue
        A = o[i0]
        if not (A > 0): continue
        SL = A * (1 + b); BL = A * (1 - b)
        iend = i0 + H
        if iend > nmin: iend = nmin
        if iend <= i0: continue
        sell_used = False; buy_used = False
        state = np.zeros(n_s, np.int64)                      # 0 평면 1 롱 2 숏
        entry_px = np.zeros(n_s); entry_min = np.zeros(n_s, np.int64); mae_cur = np.zeros(n_s)
        used_sell = np.zeros(n_s, np.bool_); used_buy = np.zeros(n_s, np.bool_)
        lastm = iend - 1
        for m in range(i0, iend):
            hi = h[m]; lo = l[m]
            sell_fill = (not sell_used) and hi > SL * (1 + 1e-4)
            buy_fill = (not buy_used) and lo < BL * (1 - 1e-4)
            for si in range(n_s):
                if state[si] == 0:
                    if sell_fill and not used_sell[si]:
                        state[si] = 2; entry_px[si] = SL; entry_min[si] = m + t0; mae_cur[si] = 0.0; used_sell[si] = True
                    elif buy_fill and not used_buy[si]:
                        state[si] = 1; entry_px[si] = BL; entry_min[si] = m + t0; mae_cur[si] = 0.0; used_buy[si] = True
                elif state[si] == 1:                           # 롱
                    adv = 1.0 - lo / entry_px[si]
                    if adv > mae_cur[si]: mae_cur[si] = adv
                    sp = entry_px[si] * (1 - s_list[si] * b)
                    if lo <= sp:
                        e_[nt] = entry_min[si]; x_[nt] = m + t0 + 1; side_[nt] = 1; entry_[nt] = entry_px[si]
                        exit_[nt] = sp * (1 - 3e-4); mae_[nt] = mae_cur[si]; etype_[nt] = 0; sidx_[nt] = si; widx_[nt] = wi; nt += 1
                        state[si] = 0
                    elif hi >= A * (1 + 1e-4):
                        e_[nt] = entry_min[si]; x_[nt] = m + t0 + 1; side_[nt] = 1; entry_[nt] = entry_px[si]
                        exit_[nt] = A; mae_[nt] = mae_cur[si]; etype_[nt] = 1; sidx_[nt] = si; widx_[nt] = wi; nt += 1
                        state[si] = 0
                elif state[si] == 2:                           # 숏
                    adv = hi / entry_px[si] - 1.0
                    if adv > mae_cur[si]: mae_cur[si] = adv
                    sp = entry_px[si] * (1 + s_list[si] * b)
                    if hi >= sp:
                        e_[nt] = entry_min[si]; x_[nt] = m + t0 + 1; side_[nt] = -1; entry_[nt] = entry_px[si]
                        exit_[nt] = sp * (1 + 3e-4); mae_[nt] = mae_cur[si]; etype_[nt] = 0; sidx_[nt] = si; widx_[nt] = wi; nt += 1
                        state[si] = 0
                    elif lo <= A * (1 - 1e-4):
                        e_[nt] = entry_min[si]; x_[nt] = m + t0 + 1; side_[nt] = -1; entry_[nt] = entry_px[si]
                        exit_[nt] = A; mae_[nt] = mae_cur[si]; etype_[nt] = 1; sidx_[nt] = si; widx_[nt] = wi; nt += 1
                        state[si] = 0
        cl = c[lastm]
        for si in range(n_s):
            if state[si] != 0:
                sd = 1 if state[si] == 1 else -1
                e_[nt] = entry_min[si]; x_[nt] = lastm + t0 + 1; side_[nt] = sd; entry_[nt] = entry_px[si]
                exit_[nt] = cl; mae_[nt] = mae_cur[si]; etype_[nt] = 2; sidx_[nt] = si; widx_[nt] = wi; nt += 1
    return e_[:nt], x_[:nt], side_[:nt], entry_[:nt], exit_[:nt], mae_[:nt], etype_[:nt], sidx_[:nt], widx_[:nt]


_MIN_CACHE = {}


def minute_arrays(sym):
    if sym in _MIN_CACHE: return _MIN_CACHE[sym]
    k = store.load("klines_1m", sym)
    mi = J.mins(k.ts); t0 = int(mi.min()); T1 = int(mi.max()) + 1
    o = np.full(T1 - t0, np.nan, np.float32); h = o.copy(); l = o.copy(); c = o.copy()
    o[mi - t0] = k.open.values; h[mi - t0] = k.high.values; l[mi - t0] = k.low.values; c[mi - t0] = k.close.values
    o = pd.Series(o).ffill().bfill().values.astype(np.float64)
    h = pd.Series(h).ffill().bfill().values.astype(np.float64)
    l = pd.Series(l).ffill().bfill().values.astype(np.float64)
    c = pd.Series(c).ffill().bfill().values.astype(np.float64)
    lr = np.diff(np.log(c), prepend=np.log(c[0]))
    sig1m = pd.Series(lr).rolling(1440, min_periods=720).std().values           # 24시간 1분봉 실현변동성(시점까지)
    _MIN_CACHE[sym] = (t0, o, h, l, c, sig1m)
    return _MIN_CACHE[sym]


def events_for_config(D, sym_s, h, k, s_list, active_col):
    """한 심볼의 활성 구간에서 k(밴드 배율)별 사건(모든 s 변형 포함)을 만든다. s_list: numpy 배열."""
    t0, o, h_, l, c, sig1m = minute_arrays(sym_s)
    sub = D[(D.sym == sym_s) & (D[active_col])]
    w0 = J.mins(sub.t)
    sig_w = sig1m[np.clip(w0 - t0, 0, len(sig1m) - 1)] * np.sqrt(h * 60)
    b_ = k * sig_w
    e_, x_, side_, entry_, exit_, mae_, etype_, sidx_, widx_ = sim_fade(
        w0.astype(np.int64), b_.astype(np.float64), h * 60, s_list.astype(np.float64), o, h_, l, c, t0, len(s_list))
    out = pd.DataFrame(dict(e=e_, x=x_, side=side_, entry=entry_, exitpx=exit_, mae=mae_, etype=etype_, sidx=sidx_))
    out["sym"] = sym_s
    return out


def to_ret(ev):
    """체결가·청산가·수수료·펀딩으로 순수익률 계산(J.add_funding 재사용)."""
    ev = ev.copy()
    fee_in = 2e-4
    fee_out = np.where(ev.etype.values == 1, 2e-4, 5e-4)       # TP=메이커, 스톱·구간끝=테이커
    ev2 = J.add_funding(ev[["sym", "e", "x"]].copy())
    fund = ev2.fund.values
    long_ret = ev.exitpx.values / ev.entry.values - 1 - fee_in - fee_out - fund
    short_ret = 1 - ev.exitpx.values / ev.entry.values - fee_in - fee_out + fund
    ret = np.where(ev.side.values > 0, long_ret, short_ret)
    ev["ret"] = ret
    return ev


# ===================== 3) 평가 지표 =====================
def daily_sharpe(ev, a, b):
    """진입일별 수익 합산(동일 명목, 무레버리지). 거래 없는 날은 0으로 채워 기간 전체 변동성을 반영한다."""
    x = ev[(ev.e_ts >= pd.Timestamp(a, tz="UTC")) & (ev.e_ts < pd.Timestamp(b, tz="UTC"))]
    rng = pd.date_range(a, b, freq="D", tz="UTC", inclusive="left")
    if len(rng) < 5: return np.nan
    dly = x.groupby(x.e_ts.dt.floor("D")).ret.sum().reindex(rng).fillna(0.0)
    if dly.std() == 0: return np.nan
    return dly.mean() / dly.std() * np.sqrt(365)


def cfg_report(ev):
    ev = ev.copy(); ev["e_ts"] = D0 + pd.to_timedelta(ev.e, unit="min")
    tr = ev[(ev.e_ts >= TRAIN_A) & (ev.e_ts < TRAIN_B)]
    ea = ev[(ev.e_ts >= EVAL_A) & (ev.e_ts < EVAL_B)]
    days_tr = max((pd.Timestamp(TRAIN_B, tz="UTC") - pd.Timestamp(TRAIN_A, tz="UTC")).days, 1)
    days_ea = max((pd.Timestamp(EVAL_B, tz="UTC") - pd.Timestamp(EVAL_A, tz="UTC")).days, 1)
    out = dict(n_tr=len(tr), tpd_tr=len(tr) / days_tr, bp_tr=tr.ret.mean() * 1e4 if len(tr) else np.nan,
               wr_tr=(tr.ret > 0).mean() if len(tr) else np.nan, sharpe_tr=daily_sharpe(ev, TRAIN_A, TRAIN_B),
               n_ea=len(ea), tpd_ea=len(ea) / days_ea, bp_ea=ea.ret.mean() * 1e4 if len(ea) else np.nan,
               wr_ea=(ea.ret > 0).mean() if len(ea) else np.nan, sharpe_ea=daily_sharpe(ev, EVAL_A, EVAL_B),
               bp_long_ea=ea[ea.side > 0].ret.mean() * 1e4 if (ea.side > 0).any() else np.nan,
               bp_short_ea=ea[ea.side < 0].ret.mean() * 1e4 if (ea.side < 0).any() else np.nan,
               n_long_ea=(ea.side > 0).sum(), n_short_ea=(ea.side < 0).sum())
    yr = ev.groupby(ev.e_ts.dt.year).ret.mean() * 1e4
    out["yrbp"] = " ".join(f"{int(y)%100}:{v:+.0f}" for y, v in yr.items())
    return out


def event_daily_returns(ev):
    ev = ev.copy(); ev["e_ts"] = D0 + pd.to_timedelta(ev.e, unit="min")
    return ev.groupby(ev.e_ts.dt.floor("D")).ret.sum()


if __name__ == "__main__":
    t_ = time.time()
    D, FE = build_regime_frame()
    P(f"국면 프레임 {len(D):,}행, 특징 {len(FE)}개 ({time.time()-t_:.0f}s)")

    PREC = {}; DH = {}
    for h in (12, 24):
        Dh, prec = walkforward_sideways(D, FE, h)
        DH[h] = Dh; PREC[h] = prec
        P(f"\n국면 모델(다음 {h}h 횡보, M4) 연도별 OOS 정밀도 — top10%/20% (n) vs 기저")
        for Y, p10, n10, p20, n20, base, nte in prec:
            P(f"  {Y}: top10 {p10:.3f}(n={n10}) | top20 {p20:.3f}(n={n20}) | 기저 {base:.3f} | 평가표본 {nte}")

    # ===== 그리드: H x top_pct x k x s =====
    S_LIST = np.array([1.0, 2.0, 1e9])
    S_LABEL = ["s1.0", "s2.0", "s없음"]
    RESULTS = []        # (cfg_label, h, pct, k, s_idx, events(DataFrame))
    for h in (12, 24):
        Dh = DH[h]
        for pct, active_col in ((10, "active10"), (20, "active20")):
            for k in (0.5, 0.75, 1.0):
                parts = []
                for sym_s in SYMS:
                    raw = events_for_config(Dh, sym_s, h, k, S_LIST, active_col)
                    parts.append(raw)
                RAW = pd.concat(parts, ignore_index=True)
                for si, slab in enumerate(S_LABEL):
                    ev = RAW[RAW.sidx == si].copy()
                    if len(ev) == 0: continue
                    ev = to_ret(ev)
                    label = f"H{h} top{pct} k{k} {slab}"
                    RESULTS.append((label, h, pct, k, si, ev))
        P(f"H{h} 그리드 완료 ({time.time()-t_:.0f}s)")

    REP = []
    for label, h, pct, k, si, ev in RESULTS:
        r = cfg_report(ev)
        REP.append(dict(cfg=label, H=h, pct=pct, k=k, s=S_LABEL[si], **r))
    REPd = pd.DataFrame(REP).sort_values("sharpe_tr", ascending=False).reset_index(drop=True)
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 30)
    P("\n===== 전체 구성 — 학습(2021~23) 샤프 상위 10")
    P(REPd.head(10)[["cfg", "n_tr", "tpd_tr", "bp_tr", "wr_tr", "sharpe_tr", "n_ea", "tpd_ea", "bp_ea", "wr_ea", "sharpe_ea"]].round(3).to_string(index=False))

    best = REPd.iloc[0]
    sel_label = best.cfg
    sel_h, sel_pct, sel_k, sel_si = best.H, best.pct, best.k, S_LABEL.index(best.s)
    sel_ev = next(ev for (lab, h, pct, k, si, ev) in RESULTS if lab == sel_label)
    P(f"\n선택 구성(학습 샤프 최대): {sel_label} | 학습샤프 {best.sharpe_tr:.3f} 평가샤프 {best.sharpe_ea:.3f} | 연도별bp {cfg_report(sel_ev)['yrbp']}")
    P(f"  평가 롱 {best.bp_long_ea:+.1f}bp (n={best.n_long_ea}) · 숏 {best.bp_short_ea:+.1f}bp (n={best.n_short_ea})")

    # ===== 무작위 구간 위약 =====
    placebo_sharpes = []
    Dh_sel = DH[sel_h]
    for seed in range(20):
        rows = []
        for sym_s in SYMS:
            sub = Dh_sel[(Dh_sel.sym == sym_s) & Dh_sel.p_side.notna()]   # 워크포워드로 예측이 존재하는 연도(2021~)만
            n_active = int((Dh_sel[(Dh_sel.sym == sym_s)][f"active{sel_pct}"]).sum())
            rng = np.random.default_rng(seed * 100 + (0 if sym_s == "BTCUSDT" else 1))
            if len(sub) == 0 or n_active == 0: continue
            pick = rng.choice(len(sub), size=min(n_active, len(sub)), replace=False)
            fake = sub.iloc[pick].copy()
            t0_, o_, h_, l_, c_, sig1m_ = minute_arrays(sym_s)
            w0 = J.mins(fake.t)
            sig_w = sig1m_[np.clip(w0 - t0_, 0, len(sig1m_) - 1)] * np.sqrt(sel_h * 60)
            b_ = sel_k * sig_w
            e_, x_, side_, entry_, exit_, mae_, etype_, sidx_, widx_ = sim_fade(
                w0.astype(np.int64), b_.astype(np.float64), sel_h * 60, S_LIST[sel_si:sel_si + 1].astype(np.float64), o_, h_, l_, c_, t0_, 1)
            rows.append(pd.DataFrame(dict(e=e_, x=x_, side=side_, entry=entry_, exitpx=exit_, mae=mae_, etype=etype_, sidx=sidx_, sym=sym_s)))
        if not rows: continue
        fev = to_ret(pd.concat(rows, ignore_index=True))
        placebo_sharpes.append(daily_sharpe(fev.assign(e_ts=D0 + pd.to_timedelta(fev.e, unit="min")), EVAL_A, EVAL_B))
    placebo_sharpes = [x for x in placebo_sharpes if not np.isnan(x)]
    pb_med = np.median(placebo_sharpes) if placebo_sharpes else np.nan
    P(f"\n무작위 구간 위약(동일 빈도, 20 시드) 평가 샤프 중앙값 {pb_med:.3f} (n_seed={len(placebo_sharpes)}) vs 선택 구성 {best.sharpe_ea:.3f} "
      f"→ {'통과' if (not np.isnan(pb_med) and best.sharpe_ea > pb_med) else '미통과'}")

    # ===== 코어(z0.31+z0.33)와의 일간 상관 =====
    CORE_E = pd.concat([G.std_dip(), G.std_breakout()], ignore_index=True)
    d_core, pk_core = G.run_multi(CORE_E, 3, 12, {1: 12 / 48})
    if d_core is not None:
        core_daily = d_core[(d_core.index >= "2022-01-01")]
        sel_daily = event_daily_returns(sel_ev)
        sel_daily = sel_daily.reindex(pd.date_range(sel_daily.index.min(), sel_daily.index.max(), freq="D", tz="UTC")).fillna(0.0)
        joined = pd.concat([core_daily.rename("core"), sel_daily.rename("sel")], axis=1, join="inner").dropna()
        joined = joined[joined.index >= "2022-01-01"]
        corr = joined.core.corr(joined.sel) if len(joined) > 5 else np.nan
        P(f"\n선택 구성 vs 코어(z0.31+z0.33) 일간수익 상관(2022+, n={len(joined)}): {corr:.3f}")
    else:
        corr = np.nan
        P("\n코어 계좌가 파산 처리되어 상관을 계산할 수 없음")

    # ===== 표준 사건 저장(선택 + 학습 샤프 상위 3) =====
    top3 = REPd.head(3)
    OUTROWS = []
    for _, row in top3.iterrows():
        ev = next(ev for (lab, h, pct, k, si, ev) in RESULTS if lab == row.cfg)
        o = pd.DataFrame(dict(e=ev.e.values.astype(np.int64), x=ev.x.values.astype(np.int64), ret=ev.ret.values,
                               mae=ev.mae.values, k=2, mult=1.0, s=(ev.sym.values == "ETHUSDT").astype(int),
                               px=ev.entry.values, side=ev.side.values.astype(np.int64), cfg=row.cfg))
        OUTROWS.append(o)
    EVOUT = pd.concat(OUTROWS, ignore_index=True)
    EVOUT.to_parquet(R / "data" / "r118a_events.parquet", index=False)
    P(f"\n표준 사건 저장: data/r118a_events.parquet ({len(EVOUT):,}행, cfg={sorted(EVOUT.cfg.unique())})")

    # ===== 공유 계좌: 코어 + 선택 구성 =====
    EV_SEL = EVOUT[EVOUT.cfg == sel_label][G.COLS]
    E_JOINT = pd.concat([CORE_E, EV_SEL], ignore_index=True)
    P(f"\n===== 공유 계좌(w3N12): 코어 + 선택({sel_label}) — Wc(선택 비중) 그리드")
    for L in (12, 16):
        for Wc in (0.5, 1, 2, 4):
            d, pk = G.run_multi(E_JOINT, 3, 12, {1: L / 48, 2: Wc})
            if d is None:
                P(f"  L{L} Wc{Wc}: 파산 ({D0 + pd.Timedelta(minutes=int(pk[0])):%Y-%m-%d})")
            else:
                P(f"  L{L} Wc{Wc}: {({k_: (round(v_, 2) if isinstance(v_, float) else v_) for k_, v_ in G.summary(d).items()})}")
    P(f"\n참고 코어 단독: L12 평가 월 +32.9% 샤프 2.66 | L16 평가 월 +35.0% 샤프 2.45")

    (R / "logs" / "r118a.summary").write_text(
        f"R118a 완료. 선택 {sel_label} 학습샤프 {best.sharpe_tr:.3f} 평가샤프 {best.sharpe_ea:.3f} 코어상관 {corr:.3f} 위약중앙 {pb_med:.3f}\n")
    P(f"\n완료 ({time.time()-t_:.0f}s)")
