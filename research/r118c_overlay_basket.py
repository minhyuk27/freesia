"""R118c — R87~R89 "오버레이" 바구니(V·S·O01·H05)를 r118 공유 계좌 엔진 사건 형식으로 정확히 재현해 검증.

배경: R87~R89는 z0.31 딥 핵심 전략에 느리고 저상관인 "오버레이" 신호를 일간 수익 근사(명목 1, 종목 0.5)로 얹어
  샤프 2.56→2.72~2.77을 봤다(R88 로그). 정확한 공유 자본 엔진(r118_engine, 분 단위 마크가격 청산 판정)으로는
  한 번도 검증되지 않았다. 여기서는 선택된 변형을 "그 코드가 정의한 그대로" 사건으로 재구성한다.

선택된 변형(전부 2020~23만 보고 사전 선택된 것, R88/R87 로그 logs/r88.log·logs/r87.csv로 확인):
  V(R88 선택 3개, 계열 상위 3·t학습 순) : 깊은 변동성 압축(q<0.5) 양방향 스톱 진입 → 목표·손절·48h 시간 청산
      q0.5 x1.25 목표3(t학습 2.50) · q0.5 x1.0 목표3(t학습 2.36) · q0.5 x1.25 목표2(t학습 2.31)
  S(R88 선택 3개, 계열 상위 3) : 스테이블코인 7일 증가 상위 15~20% → 롱 5~7일(모두 look=7일)
      7일 상위20% 보유7일(t학습 2.71) · 7일 상위15% 보유5일(t학습 2.58) · 7일 상위15% 보유7일(t학습 2.42)
  O01(R87 원본, r87_new80.strategies) : 옵션 순매수(콜−풋) 24h z>2 → 롱 24h (t 2.59, 양 구간 흑자)
  H05(R87 원본, r87_new80.strategies) : 7·30·90일 추세 모두 상승 → 매일 00시 진입, 롱 24h (t 2.42, 양 구간 흑자)
  K·O(R88 평가)는 선택 기준(n≥30·초과>0·t≥2·흑자연도≥3) 통과가 0개라 바구니에서 제외.
  CMB(R89, 스테이블 7일+BTC 거래소 순유출 평균) 제외: R89 선택은 LO/LS 두 연속 비중 모드뿐이고(EV는 선택 안 됨),
    p=clip(z,±2,2)/2 가 날마다 바뀌는 "연속 재조정 노출"이라 e<x 단일 보유 사건(r118 표준 형식)으로 왜곡 없이 옮길 수 없다.
    → "trade-based가 아니므로" 제외하고 이 사실만 보고한다(과제 지시의 조건부 처리).

체결·비용(이 라운드의 통일 규칙, R87/R88 자체의 7bp/2bp 비용과는 다름 — 비교 시 유의):
  V는 r79 run_trades_x와 동일한 목표·손절·48h 시간청산 로직(원가, 비용 미반영)을 복제한 뒤,
  S·O01·H05(목표=손절=1000%라 사실상 시간청산만)도 같은 함수로 통일 체결.
  모든 체결가에 테이커 5bp + 슬리피지 2bp를 진입·청산 각각 반영(합 14bp, 매수는 불리하게 더 주고 매도는 불리하게 덜 받음),
  실제 보유기간 펀딩은 r102_joint_account.add_funding로 반영(숏은 양의 펀딩을 받음).
채널: k=3(바구니 전용), mult = 변형별 위험 균등 배수(2020~23 일간 손익 표준편차의 역수, 평균 1로 정규화).
저장: data/r118c_events.parquet (표준 9열 + var 열).
평가: 1) 변형별 거래 수·거래당 순bp(20~23 vs 24~26)·연도별·코어 일간 손익 상관
      2) 공유 계좌 run_multi(core+basket, 3, 12, {1:L/48, 3:Wb}), L∈{12,16}, Wb∈{.25,.5,1,2,3}, Wb는 학습월(22~23)로 선택.
사용: python research/r118c_overlay_basket.py"""
import sys, warnings, time
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store, costs
import r87_new80 as M87
import r102_joint_account as J
import r118_engine as E118
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
COLS = E118.COLS
SPLIT = pd.Timestamp("2024-01-01", tz="UTC")
TR_END = pd.Timestamp("2024-01-01", tz="UTC")


@numba.njit(cache=True)
def run_tts(ent, side, tg, st, hold_m, o, h, l, c):
    """목표·손절(%)·시간 청산 체결, r79_longshort_book.run_trades_x와 같은 논리의 비용 미반영(원가) 버전.
    반환: used(채택 여부), 청산 분봉 인덱스, 진입 원가(시가), 청산 원가, 보유 중 최대 역행가(정보용)."""
    n = len(c); m = len(ent)
    used = np.zeros(m, np.bool_); exi = np.zeros(m, np.int64)
    entpx = np.zeros(m); exitpx = np.zeros(m); advpx = np.zeros(m)
    busy = -1
    for q in range(m):
        i = ent[q]
        if i <= busy or i >= n - 1 or not (tg[q] > 0) or not (st[q] > 0):
            continue
        e = o[i]; s = side[q]
        up = e * (1 + s * tg[q] / 100); dn = e * (1 - s * st[q] / 100)
        j = i; r_px = np.nan; adv = e
        while j < min(n, i + hold_m[q]):
            if s > 0:
                if l[j] < adv: adv = l[j]
                if l[j] <= dn:
                    r_px = min(dn, o[j]) if j > i else dn; break
                if h[j] >= up:
                    r_px = up; break
            else:
                if h[j] > adv: adv = h[j]
                if h[j] >= dn:
                    r_px = max(dn, o[j]) if j > i else dn; break
                if l[j] <= up:
                    r_px = up; break
            j += 1
        if np.isnan(r_px):
            j = min(j, n - 1); r_px = c[j]
        used[q] = True; entpx[q] = e; exitpx[q] = r_px; advpx[q] = adv; exi[q] = j; busy = j
    return used, exi, entpx, exitpx, advpx


def fill(D, s, Ent):
    """Ent(i,side,tg,st,hold) → 표준 사건(e,x,ret,mae,s,px,side), 테이커 5bp+슬리피지 2bp(양측)·실제 펀딩 반영."""
    if Ent is None or len(Ent) == 0:
        return pd.DataFrame(columns=["e", "x", "ret", "mae", "s", "px", "side"])
    Ent = Ent.sort_values("i").reset_index(drop=True)
    k = D.K[s]; mi = J.mins(k.index)
    used, exi, entpx, exitpx, advpx = run_tts(
        Ent.i.values.astype(np.int64), Ent.side.values.astype(np.int64),
        Ent.tg.values.astype(np.float64), Ent.st.values.astype(np.float64),
        Ent.hold.values.astype(np.int64), k.open.values, k.high.values, k.low.values, k.close.values)
    if used.sum() == 0:
        return pd.DataFrame(columns=["e", "x", "ret", "mae", "s", "px", "side"])
    sd = Ent.side.values[used].astype(np.int64)
    e = mi[Ent.i.values[used]]; x = mi[exi[used]] + 1
    px = entpx[used] * (1 + sd * 2e-4)            # 불리한 슬리피지(롱은 더 비싸게, 숏은 더 싸게 체결)
    exf = exitpx[used] * (1 - sd * 2e-4)
    fund = J.add_funding(pd.DataFrame({"sym": s, "e": e, "x": x})).fund.values
    ret = sd * (exf / px - 1) - 2 * costs.TAKER - sd * fund   # 숏은 -sd*fund = +fund(양의 펀딩 수취)
    adv = advpx[used]
    mae = np.maximum(sd * (px - adv) / px, 0) + costs.TAKER
    return pd.DataFrame(dict(e=e, x=x, ret=ret, mae=mae, s=(0 if s == "BTCUSDT" else 1), px=px, side=sd))


def v_variant(D, q, x, tm):
    """R88 V계열: 깊은 변동성 압축(q 하향 교차) → 양방향 스톱 ±x·σ24, 목표 tm배, 48h 시간청산(복제)."""
    def f(s):
        H = D.H[s].copy(); rv2 = H.rv ** 2 * 60; RVd = rv2.rolling(24).mean(); RVm = rv2.rolling(720).mean()
        qq = RVd / RVm; sig = np.sqrt(RVm * 24)
        m = (qq < q) & (qq.shift(1) >= q); T = H[m]; k = D.K[s]
        st_ = k.index.searchsorted(pd.DatetimeIndex(T.t)); ok = st_ < len(k) - 2
        st_ = st_[ok]; xx = (x * sig[m].values)[ok]
        ent, sd = M87.straddle(st_.astype(np.int64), k.close.values[np.maximum(st_ - 1, 0)], xx,
                               np.full(len(st_), 2880, np.int64), k.high.values, k.low.values)
        g = ent >= 0
        return pd.DataFrame({"i": ent[g], "side": sd[g], "tg": xx[g] * 100 * tm, "st": xx[g] * 100,
                             "hold": np.full(g.sum(), 48 * 60)})
    return f


def s_variant(D, sc, look, top, hd):
    """R88 S계열: 스테이블코인 look일 증가 상위 top% → 롱 hd일(복제)."""
    def f(s):
        g = np.log(sc).diff(look); qq = g.rolling(365, min_periods=120).rank(pct=True)
        m = (qq >= 1 - top) & ~(qq.shift(1) >= 1 - top)
        return M87.entries_at(D, s, qq.index[m.values], 1, M87.BIG, M87.BIG, hd * 24)
    return f


def daily_approx(ev, idx):
    """근사 일간 손익(명목 1, 종목당 0.5 — R87/R88 표준 근사와 동일한 집계 방식), 진입일 기준."""
    if len(ev) == 0: return pd.Series(0.0, index=idx)
    t = (J.D0 + pd.to_timedelta(ev.e.values, unit="min"))
    d = pd.Series(ev.ret.values, index=pd.DatetimeIndex(t).floor("D")).groupby(level=0).sum() * 0.5
    return d.reindex(idx).fillna(0.0)


if __name__ == "__main__":
    t_ = time.time()
    D = M87.Data(); P(f"데이터 준비 ({time.time()-t_:.0f}s)")
    S = M87.strategies(D)
    onm = next(k for k in S if k.startswith("O01")); hnm = next(k for k in S if k.startswith("H05"))
    P(f"전략 사전(O01·H05 포함 {len(S)}개) 준비 ({time.time()-t_:.0f}s)")

    sc = pd.read_parquet(M87.EXT / "stablecoins.parquet").stable_total
    sc.index = pd.DatetimeIndex(sc.index); sc.index = (sc.index.tz_localize("UTC") if sc.index.tz is None else sc.index) + pd.Timedelta(days=1)

    variants = {
        "V_q.5x1.25t3": v_variant(D, 0.5, 1.25, 3),
        "V_q.5x1.0t3": v_variant(D, 0.5, 1.0, 3),
        "V_q.5x1.25t2": v_variant(D, 0.5, 1.25, 2),
        "S_7d20pct7d": s_variant(D, sc, 7, 0.20, 7),
        "S_7d15pct5d": s_variant(D, sc, 7, 0.15, 5),
        "S_7d15pct7d": s_variant(D, sc, 7, 0.15, 7),
        "O01": S[onm],
        "H05": S[hnm],
    }

    events = {}
    for name, fn in variants.items():
        parts = [fill(D, s, fn(s)) for s in M87.SYMS]
        ev = pd.concat(parts, ignore_index=True).sort_values("e").reset_index(drop=True)
        events[name] = ev
        P(f"  {name:14s} 거래 {len(ev):5d}건 준비 ({time.time()-t_:.0f}s)")

    # ---- 변형별 거래당 순수익(bp, 원가 반영) : 2020~23 vs 2024~26, 연도별 ----
    P("\n===== 변형별 원가 반영(테이커5bp+슬리피지2bp 양측 + 실제 펀딩) 거래당 순수익(bp)")
    for name, ev in events.items():
        t = J.D0 + pd.to_timedelta(ev.e, unit="min")
        tr = ev.ret.values[(t < SPLIT).values]; te = ev.ret.values[(t >= SPLIT).values]
        yr = t.dt.year
        yg = pd.Series(ev.ret.values, index=yr).groupby(level=0).agg(["mean", "size"])
        yrstr = " ".join(f"{y%100}:{row['mean']*1e4:+.0f}bp(n{int(row['size'])})" for y, row in yg.iterrows())
        P(f"  {name:14s} n={len(ev):4d} 20~23 {tr.mean()*1e4 if len(tr) else np.nan:+7.1f}bp(n{len(tr)}) "
          f"24~26 {te.mean()*1e4 if len(te) else np.nan:+7.1f}bp(n{len(te)}) | {yrstr}")

    # ---- 위험 균등 mult(2020~23 일간 손익 표준편차 역수, 평균 1) ----
    IDXtr = pd.date_range("2020-01-01", "2023-12-31", freq="D", tz="UTC")
    vol = {name: daily_approx(ev[J.D0 + pd.to_timedelta(ev.e, unit="min") < TR_END], IDXtr).std() for name, ev in events.items()}
    invvol = {k: (1.0 / v if v > 0 else 0.0) for k, v in vol.items()}
    meanw = np.mean(list(invvol.values()))
    mult = {k: v / meanw for k, v in invvol.items()}
    P("\n===== 위험 균등 배수(2020~23 일간 손익 표준편차 역수, 평균 1)")
    for name in events: P(f"  {name:14s} 20~23일 손익 표준편차 {vol[name]*1e4:.1f}bp/일 → mult {mult[name]:.2f}")

    # ---- 표준 사건 저장 ----
    parts = []
    for name, ev in events.items():
        e2 = ev.copy(); e2["k"] = 3; e2["mult"] = mult[name]; e2["var"] = name
        parts.append(e2[COLS + ["var"]])
    basket = pd.concat(parts, ignore_index=True)
    basket.to_parquet(R / "data" / "r118c_events.parquet", index=False)
    P(f"\n저장: data/r118c_events.parquet ({len(basket):,}행, {time.time()-t_:.0f}s)")

    # ---- 코어(z0.31+z0.33) 및 상관 ----
    from r118_engine import std_dip, std_breakout, run_multi, summary
    core = pd.concat([std_dip(), std_breakout()], ignore_index=True)
    P("\n===== 코어 단독(비교용, 과제 제시값: L12 평가 월 +32.9% 샤프 2.66 MDD -59% | L16 +35.0% 2.45 -67%)")
    d_core = {}
    for L in (12, 16):
        dd, pk = run_multi(core, 3, 12, {1: L / 48})
        d_core[L] = dd
        tail = ("파산 " + repr(pk)) if dd is None else (repr({k: (round(v, 2) if isinstance(v, float) else v) for k, v in summary(dd).items()}) + f" 최대명목 {pk:.1f}배")
        P(f"  코어 단독 L{L}: " + tail)
    d_corr = d_core[12]  # 과제 지정: run_multi(core, 3, 12, {1: 12/48})

    if d_corr is None:
        P("\n코어 L12 단독이 파산해 상관 계산 생략")
    else:
        P("\n===== 변형별 코어(z0.31+z0.33, w3N12 L12) 일간 손익 상관 (근사 일간 손익, 명목1·종목0.5)")
        for name, ev in events.items():
            dv = daily_approx(ev, d_corr.index)
            corr = np.corrcoef(dv.values, d_corr.reindex(dv.index).fillna(0).values)[0, 1]
            P(f"  {name:14s} 상관 {corr:+.2f}")

    # ---- 공유 계좌: 코어 + 바구니 ----
    P("\n===== 공유 계좌 run_multi(core+basket, 3, 12, {1:L/48, 3:Wb})")
    best = {}
    for L in (12, 16):
        rows = []
        for Wb in (0.25, 0.5, 1, 2, 3):
            dd, pk = run_multi(pd.concat([core, basket], ignore_index=True), 3, 12, {1: L / 48, 3: Wb})
            if dd is None:
                P(f"  L{L} Wb={Wb}: 파산 ({pk})"); continue
            sm = summary(dd); rows.append((Wb, sm))
            P(f"  L{L} Wb={Wb}: " + {k: (round(v, 2) if isinstance(v, float) else v) for k, v in sm.items()}.__repr__() + f" 최대명목 {pk:.1f}배")
        if rows:
            bw, bsm = max(rows, key=lambda r: r[1]["학습월"])
            best[L] = (bw, bsm)
            P(f"  → L{L} 학습월(22~23) 기준 최선 Wb={bw}: 평가 " + {k: (round(v, 2) if isinstance(v, float) else v) for k, v in bsm.items()}.__repr__())
    P(f"\n완료 ({time.time()-t_:.0f}s)")
