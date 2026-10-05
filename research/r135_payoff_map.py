"""R135 — 손익비(비대칭 장벽) 지도: 어떤 진입에서 '손절 d, 목표 R·d'의 실제 적중률이 무작위 기준선보다 높은가 (고배율 전제).

배경(사용자 2026-10-05): "적중 50%라도 손실 −0.5%·이익 +1%면 큰 엣지 아닌가 → 손익비 시스템 연구, 고배율 기반 동일".
원리: 방향 정보가 없으면 P(+R·d 먼저) = 1/(1+R) → 기대값 0(비용만큼 손해). 손익비 엣지 = 실제 적중 − 무작위 기준 적중.
무작위 기준선 F0: 매일 01:00 UTC(한국 10시) 롱·숏 둘 다 진입 → 같은 (d, R, 시간 청산)에서의 경험 적중(시간 청산까지 반영한 기준).
진입 가족:
  F0 무작위 기준(매일 01 UTC 롱+숏)      F1 매시 정각, 15분 EMA9>EMA21이면 롱·아니면 숏(추세 지속)
  F2 z0.33 돌파(롱, 스톱 진입가)         F3 z0.31 딥(롱, 지정가 체결가 — 진입 메이커)
  F4 R131 예보 창 첫 0.2% 이탈 추종       F5 R123 변동성 압축 양방향 스톱(VC, 롱·숏)
  F6 R123 다일 추세(TSMOM L28, 매일 00 UTC)
청산: 손절 d ∈ {0.2,0.3,0.4,0.5,0.75,1,1.5}%, 목표 R·d, R ∈ {1,1.5,2,3,4,6}, 최대 보유 24시간(시간 청산 종가).
  같은 분 양쪽 → 손실. 비용: 진입 테이커 5+2bp(F3은 메이커 2bp), 목표 지정가 2bp, 손절 테이커 5+2bp, 시간 청산 테이커 7bp.
  같은 가족·종목은 한 번에 한 포지션.
고배율 제약: 레버리지 L의 청산 거리 1/L − 0.4%(유지증거금)가 d + 0.1%(슬리피지 여유)보다 커야 한다 → L_max = 1/(d + 0.5%).
계좌: 증거금 = 자본 50%(사용자 구조), 명목 = 0.5·L·자본, 가족별 순차 복리(동시 거래는 시간순 근사). L ∈ {L_max, 켈리 최적 ≤ L_max}.
선택: 학습 2022~24에서 가족별 최선 (d, R) → 시험 2025~26 판정.
사용: python research/r135_payoff_map.py"""
import sys, warnings, time
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
import r102_joint_account as J
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
DS = [0.002, 0.003, 0.004, 0.005, 0.0075, 0.01, 0.015]; RS = [1, 1.5, 2, 3, 4, 6]; HOLD = 1440
CT, CM, SLIP = 5e-4, 2e-4, 2e-4


def grids():
    G = {}
    for s, sym in ((0, "BTCUSDT"), (1, "ETHUSDT")):
        k = store.load("klines_1m", sym); mi = J.mins(k.ts); T = mi.max() + 1
        H = np.full(T, np.nan); L = np.full(T, np.nan); C = np.full(T, np.nan); O = np.full(T, np.nan)
        H[mi] = k.high.values; L[mi] = k.low.values; C[mi] = k.close.values; O[mi] = k.open.values
        G[s] = (O, H, L, C, k, mi)
    return G


@numba.njit(cache=True)
def run(e, px, sd, H, L, C, d, Rm, hold):
    """반환 결과 코드(1 목표, 2 손절, 0 시간), 비용 전 수익, 청산 분."""
    n = len(e); res = np.zeros(n, np.int8); g = np.zeros(n); xe = np.zeros(n, np.int64)
    N = len(C)
    for i in range(n):
        st = px[i] * (1 - sd[i] * d); tp = px[i] * (1 + sd[i] * Rm * d); j1 = min(N, e[i] + hold); r = 0; last = px[i]; xj = j1 - 1
        for j in range(e[i] + 1, j1):
            if np.isnan(C[j]): continue
            last = C[j]
            hs = (L[j] <= st) if sd[i] > 0 else (H[j] >= st)
            ht = (H[j] >= tp) if sd[i] > 0 else (L[j] <= tp)
            if hs: r = 2; xj = j; break
            if ht: r = 1; xj = j; break
        res[i] = r; xe[i] = xj
        g[i] = Rm * d if r == 1 else (-d if r == 2 else sd[i] * (last / px[i] - 1))
    return res, g, xe


def one_at_a_time(e, xe, sym):
    keep = np.zeros(len(e), bool); busy = {0: -1, 1: -1}
    for i in np.argsort(e, kind="stable"):
        if e[i] > busy[sym[i]]: keep[i] = True; busy[sym[i]] = xe[i]
    return keep


def families(G):
    F = {}
    rows = []
    for s in (0, 1):
        O, H, L, C, k, mi = G[s]
        ts = pd.DatetimeIndex(k.ts); m01 = mi[(ts.hour == 1) & (ts.minute == 0)]
        for sd in (1, -1):
            rows.append(pd.DataFrame(dict(e=m01, px=O[m01], sd=sd, s=s)))
    F["F0 무작위(01UTC 롱+숏)"] = pd.concat(rows)
    rows = []
    for s in (0, 1):
        O, H, L, C, k, mi = G[s]
        c15 = pd.Series(k.close.values, index=pd.DatetimeIndex(k.ts)).resample("15min").last()
        st = np.sign(c15.ewm(span=9, adjust=False).mean() - c15.ewm(span=21, adjust=False).mean())
        hr = mi[pd.DatetimeIndex(k.ts).minute == 0]
        tsh = J.D0 + pd.to_timedelta(hr, unit="min")
        sd = st.reindex(tsh - pd.Timedelta("15min"), method="ffill").values          # 정시 직전 완료된 15분봉
        ok = np.isfinite(sd) & (sd != 0)
        rows.append(pd.DataFrame(dict(e=hr[ok], px=O[hr[ok]], sd=sd[ok].astype(int), s=s)))
    F["F1 매시 EMA 추세"] = pd.concat(rows)
    import r118_engine as E8
    B = E8.std_breakout(); F["F2 z0.33 돌파"] = pd.DataFrame(dict(e=B.e.values, px=B.px.values, sd=1, s=B.s.values))
    D = E8.std_dip(); F["F3 z0.31 딥(메이커)"] = pd.DataFrame(dict(e=D.e.values, px=D.px.values, sd=1, s=D.s.values))
    ev = pd.read_parquet(R / "data" / "r131_events.parquet", columns=["sym", "f", "s", "y", "X"])
    ev = ev[(ev.y.round(4) == 0.002) & (ev.X.round(4) == 0.005)]
    rows = []
    for s, sym in ((0, "BTCUSDT"), (1, "ETHUSDT")):
        O, H, L, C, k, mi = G[s]; g = ev[ev.sym == sym]; e_ = mi[g.f.values.astype(int)]
        rows.append(pd.DataFrame(dict(e=e_, px=C[e_], sd=g.s.values.astype(int), s=s)))
    F["F4 예보창 첫 이탈 추종"] = pd.concat(rows)
    T = pd.read_parquet(R / "data" / "r123_trades.parquet", columns=["mech", "fam", "side", "sym", "e", "px"])
    T["s"] = (T.sym == "ETHUSDT").astype(int)
    vc = T[T.fam == "vc"].drop_duplicates(["s", "e", "side"])
    F["F5 변동성 압축 스톱"] = pd.DataFrame(dict(e=vc.e.values, px=vc.px.values, sd=vc.side.values.astype(int), s=vc.s.values))
    ts_ = T[T.mech.str.startswith("TS L28 H1d")]
    F["F6 다일 추세 L28"] = pd.DataFrame(dict(e=ts_.e.values, px=ts_.px.values, sd=ts_.side.values.astype(int), s=ts_.s.values))
    return F


if __name__ == "__main__":
    t_ = time.time(); G = grids(); F = families(G); P(f"격자·가족 준비 ({time.time()-t_:.0f}s)")
    out = []
    for fn, Fe in F.items():
        Fe = Fe[(Fe.e >= J.mins(pd.DatetimeIndex(["2022-01-01"], tz="UTC"))[0])].reset_index(drop=True)
        Fe = Fe[np.isfinite(Fe.px.values)].reset_index(drop=True)
        yr = (J.D0 + pd.to_timedelta(Fe.e, unit="min")).dt.year.values
        entry_cost = CM if "딥" in fn else CT + SLIP
        for d in DS:
            for Rm in RS:
                res = np.zeros(len(Fe), np.int8); g = np.zeros(len(Fe)); xe = np.zeros(len(Fe), np.int64)
                for s in (0, 1):
                    m = (Fe.s == s).values
                    if not m.any(): continue
                    O, H, L, C, k, mi = G[s]
                    a, b, c = run(Fe.e.values[m].astype(np.int64), Fe.px.values[m].astype(float), Fe.sd.values[m].astype(np.int64), H, L, C, d, float(Rm), HOLD)
                    res[m], g[m], xe[m] = a, b, c
                keep = np.ones(len(Fe), bool) if fn.startswith("F0") else one_at_a_time(Fe.e.values, xe, Fe.s.values)
                cost = entry_cost + np.where(res == 1, CM, CT + SLIP)
                net = g - cost
                for per, pm in (("학습", (yr <= 2024)), ("시험", (yr >= 2025))):
                    mm = keep & pm
                    if mm.sum() < 20: continue
                    days = (pd.Timestamp("2025-01-01") - pd.Timestamp("2022-01-01")).days if per == "학습" else (pd.Timestamp("2026-09-22") - pd.Timestamp("2025-01-01")).days
                    out.append(dict(가족=fn, d=d * 100, R=Rm, 기간=per, n=int(mm.sum()), 하루=mm.sum() / days,
                                    적중=(res[mm] == 1).mean(), 손절=(res[mm] == 2).mean(), 시간=(res[mm] == 0).mean(),
                                    순bp=net[mm].mean() * 1e4, R당=net[mm].mean() / d, net=net[mm], e=Fe.e.values[mm]))
        P(f"  {fn} 완료 ({time.time()-t_:.0f}s)")
    O = pd.DataFrame(out)
    base = O[O.가족.str.startswith("F0")].set_index(["d", "R", "기간"]).적중
    O["기준적중"] = O.set_index(["d", "R", "기간"]).index.map(base)
    O["초과"] = O.적중 - O.기준적중
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    cols = ["가족", "d", "R", "기간", "n", "하루", "적중", "기준적중", "초과", "시간", "순bp", "R당"]
    P("\n===== 0) 무작위 기준선 적중 vs 이론 1/(1+R) (학습)")
    b0 = O[O.가족.str.startswith("F0") & (O.기간 == "학습")][["d", "R", "적중", "시간"]].assign(이론=lambda x: 1 / (1 + x.R))
    P(b0.round(3).to_string(index=False))
    P("\n===== 1) 가족별 학습 최선 (R당 기대값 기준, 하루 0.2건 이상) → 시험")
    rows = []
    for fn, g in O[~O.가족.str.startswith("F0")].groupby("가족"):
        tr = g[(g.기간 == "학습") & (g.하루 >= 0.2)]
        if not len(tr): tr = g[g.기간 == "학습"]
        b = tr.sort_values("R당", ascending=False).iloc[0]
        te = g[(g.기간 == "시험") & (g.d == b.d) & (g.R == b.R)]
        rows.append(b[cols]); rows += [te.iloc[0][cols]] if len(te) else []
    P(pd.DataFrame(rows).round(3).to_string(index=False))
    P("\n===== 2) 고배율 가능 구간(d ≤ 1.1% → 50배 이상 가능)에서 시험 기간 초과 적중 상위 15 (참고 — 사후)")
    P(O[(O.기간 == "시험") & (O.d <= 1.1) & ~O.가족.str.startswith("F0")].sort_values("R당", ascending=False).head(15)[cols].round(3).to_string(index=False))
    # 계좌: 학습 최선 구성, 시험 기간, 증거금 50%·레버리지
    P("\n===== 3) 계좌(증거금 = 자본 50%, 명목 = 0.5·L·자본) — 가족별 학습 최선 구성, 시험 2025~26")
    for fn, g in O[~O.가족.str.startswith("F0")].groupby("가족"):
        tr = g[(g.기간 == "학습") & (g.하루 >= 0.2)]
        if not len(tr): tr = g[g.기간 == "학습"]
        b = tr.sort_values("R당", ascending=False).iloc[0]
        te = g[(g.기간 == "시험") & (g.d == b.d) & (g.R == b.R)]
        if not len(te): continue
        te = te.iloc[0]; r = te.net[np.argsort(te.e)]; Lmax = int(1 / (b.d / 100 + 0.005))
        line = []
        for Lv in sorted({Lmax, 50, 20, 10, 5}):
            if Lv > Lmax: continue
            lam = 0.5 * Lv; eq = np.cumprod(np.maximum(1 + lam * r, 0)); mdd = (eq / np.maximum.accumulate(eq) - 1).min()
            mo = eq[-1] ** (1 / 20.7) - 1 if eq[-1] > 0 else -1
            line.append(f"L{Lv}: 최종 {eq[-1]:.3g}배 월 {mo*100:+.1f}% MDD {mdd*100:.0f}%")
        P(f"  {fn} (d {b.d}%, R {b.R}, 시험 n {te.n}, 적중 {te.적중:.3f} vs 기준 {te.기준적중:.3f}) L_max {Lmax}: " + " | ".join(line))
    O.drop(columns=["net", "e"]).to_parquet(R / "data" / "r135.parquet", index=False)
    P(f"완료 ({time.time()-t_:.0f}s)")
