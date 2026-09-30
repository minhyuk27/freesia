"""R109 — 딥 포지션 정확한 시가평가: 진입가(지정가 T)를 복원해, R102~R108의 보수적 가정(열린 딥 전부 최대 역행 동시)을 없앤다.

딥 진입가 T = 봉 시가 × (1 − k × rv1 × √B) (r26 run과 동일, rv1 = 1분 로그수익 300분 표준편차, 전 분까지). 복원 검증: 저장된 ret·mae와 비교.
엔진 sim4: 딥·돌파 모두 종목별 수량(Σ명목/진입가)으로 합산 → 분 저가(마크)로 평가, 자본 + 미실현 ≤ 0.65% × 총명목이면 청산.
격자: 딥 w ∈ {3, 4, 5, 6}, N ∈ {12, 16, 20, 24, 32}, 돌파 L ∈ {3, 4, 6, 8}. 선택 2022~23 월 복리 → 평가 2024~26.
사용: python research/r109_dip_exact_mtm.py"""
import sys, warnings, time, itertools
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
import r102_joint_account as J, r103_breakout_cost as K, r80_max_return as R80, r107_max_growth as G7
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)


@numba.njit(cache=True)
def sim4(e_, x_, r_, m_, k_, u_, s_, px_, LO, CL, t0, w, N, L33, X, Hh, Lk, mmr):
    """딥도 진입가(T)로 종목별 수량에 합산해 분 저가(마크)로 정확히 시가평가한다(딥 최대 역행 동시 가정 제거)."""
    T = LO.shape[1]; nd = (t0 + T) // 1440 + 1; day = np.full(nd, np.nan); info = np.zeros(3)
    ne = len(e_); order_x = np.argsort(x_)
    cap = 1.0; Q = np.zeros(2); NS = np.zeros(2); dipN = 0.0; dipL = 0.0
    isopen = np.zeros(ne, np.bool_); nn = np.zeros(ne)
    ce_t = np.zeros(ne, np.int64); ce_r = np.zeros(ne); nc = 0; lo_p = 0; rsum = 0.0; halt = -1
    pe = 0; px_i = 0; peak = 0.0
    for t in range(T):
        m = t0 + t
        while px_i < ne and x_[order_x[px_i]] <= m:
            q = order_x[px_i]; px_i += 1
            if not isopen[q]: continue
            isopen[q] = False; n = nn[q]; cap += n * r_[q]
            Q[s_[q]] -= n / px_[q]; NS[s_[q]] -= n
            if k_[q] == 0:
                dipN -= n
                ce_t[nc] = m; ce_r[nc] = r_[q]; nc += 1; rsum += r_[q]
        while lo_p < nc and ce_t[lo_p] <= m - Lk:
            rsum -= ce_r[lo_p]; lo_p += 1
        if rsum < -X and m + Hh > halt: halt = m + Hh
        while pe < ne and e_[pe] <= m:
            q = pe; pe += 1
            if e_[q] < m or x_[q] <= m: continue
            if k_[q] == 0:
                if m < halt: continue
                n = min(cap * w * u_[q], max(cap * N - dipN, 0.0))
            else:
                n = cap * L33 / 48.0 * u_[q]
            if n <= 1e-12: continue
            isopen[q] = True; nn[q] = n
            Q[s_[q]] += n / px_[q]; NS[s_[q]] += n
            if k_[q] == 0: dipN += n
        tot = NS[0] + NS[1]
        if tot > 0:
            eqlo = cap + Q[0] * LO[0, t] - NS[0] + Q[1] * LO[1, t] - NS[1]
            if eqlo <= mmr * tot:
                info[0] = m; info[1] = dipN / cap; info[2] = (NS[0] + NS[1] - dipN) / cap
                return day, -1.0, info
            if tot / cap > peak: peak = tot / cap
        if (m + 1) % 1440 == 0:
            day[m // 1440] = cap + Q[0] * CL[0, t] - NS[0] + Q[1] * CL[1, t] - NS[1]
    return day, peak, info



def dip_with_px():
    W = R80.dip_frame(0.0025)
    mult = np.minimum(np.clip(W.vreg.values ** 1.5, 0.3, 4) * np.clip(W.pred.values / 0.001, 0.5, 3), 6)
    px = np.full(len(W), np.nan); chk_r = np.full(len(W), np.nan); chk_m = np.full(len(W), np.nan)
    for sym in ("BTCUSDT", "ETHUSDT"):
        k = store.load("klines_1m", sym); o, l, c = k.open.values, k.low.values, k.close.values
        rv1 = pd.Series(np.log(c)).diff().rolling(300).std().shift(1).values
        m = (W.sym == sym).values; f = W.f.values[m].astype(np.int64); x = W.x.values[m].astype(np.int64); B = W.B.values[m].astype(np.int64)
        s0 = (f // B) * B; T = o[s0] * (1 - W.k.values[m] * rv1[s0] * np.sqrt(B)); px[m] = T
        lo = np.array([l[a:b + 1].min() for a, b in zip(f, x)]); chk_m[m] = 1 - lo / T
        chk_r[m] = c[x] / T - 1
    e = J.mins(W.ts)
    D = pd.DataFrame(dict(e=e, x=e + np.maximum(W.x.values - W.f.values, 1), ret=W.ret.values, mae=W.mae.values, k=0, mult=mult,
                          s=(W.sym.values == "ETHUSDT").astype(int), px=px))
    return D, chk_r, chk_m, W


def run(E, G, w, N, L):
    LO, CL, t0 = G
    day, pk, info = sim4(E.e.values.astype(np.int64), E.x.values.astype(np.int64), E.ret.values, E.mae.values, E.k.values.astype(np.int64),
                         E.mult.values, E.s.values.astype(np.int64), E.px.values, LO, CL, t0, float(w), float(N), float(L), 0.03, 720, 120, J.MMR)
    if pk < 0: return None, info
    eq = pd.Series(day, index=J.D0 + pd.to_timedelta(np.arange(len(day)), unit="D")).dropna()
    return R80.daily_from_eq(eq), pk


if __name__ == "__main__":
    t_ = time.time()
    D, cr, cm, W = dip_with_px()
    P(f"딥 {len(D):,}: 진입가 복원 결측 {np.isnan(D.px).mean()*100:.1f}% · mae 복원 오차 중앙 {np.nanmedian(np.abs(cm - W.mae.values))*1e4:.2f}bp 99% {np.nanquantile(np.abs(cm - W.mae.values), .99)*1e4:.1f}bp"
      f" · 수익(수수료 전) − 저장 ret 중앙 {np.nanmedian(cr - W.ret.values)*1e4:+.2f}bp")
    T8 = J.add_funding(J.breakout_trades(J.TOP8)); T8["lf"] = K.last_funding(T8)
    BK = K.events(T8, False, fmax=1e-4).drop(columns="lf")
    start = int(J.mins([pd.Timestamp("2022-01-01", tz="UTC")])[0])
    E = pd.concat([D.dropna(subset=["px"]), BK[BK.e >= start]], ignore_index=True).sort_values(["e", "k"]).reset_index(drop=True)
    Gd = G7.grid(start, "mark_1m")
    rows = []
    for w, N, L in itertools.product((3, 4, 5, 6), (12, 16, 20, 24, 32), (3, 4, 6, 8)):
        if N < 3 * w: continue
        d, pk = run(E, Gd, w, N, L)
        if d is None:
            rows.append(dict(w=w, N=N, L=L, ok=False, 파산=f"{J.D0 + pd.Timedelta(minutes=int(pk[0])):%Y-%m-%d %H:%M} 딥{pk[1]:.0f}·돌파{pk[2]:.0f}")); continue
        a = R80.metrics(d, "2022-01-01", "2024-01-01"); b = R80.metrics(d, "2024-01-01", "2026-09-23"); s = d["2024":"2026-09-22"]
        rows.append(dict(w=w, N=N, L=L, ok=True, 학습월=a["mo"], 학습MDD=a["mdd"], 평가월=b["mo"], 평가중앙=b["med"], 평가100달=b["big"], 평가MDD=b["mdd"],
                         평가샤프=s.mean() / s.std() * np.sqrt(365), 명목=pk, 연도=b["yr"]))
    O = pd.DataFrame(rows); pd.set_option("display.width", 250); pd.set_option("display.max_rows", 100)
    S = O[O.ok]; P(f"\n생존 {len(S)}/{len(O)}")
    P("파산 사례: " + " | ".join(f"w{r.w}N{r.N}L{r.L} {r.파산}" for r in O[~O.ok].head(8).itertuples()))
    P("\n===== 2022~23 월 복리 상위 12 → 평가 (사전 선택)"); P(S.sort_values("학습월", ascending=False).head(12).drop(columns=["ok", "파산"], errors="ignore").round(1).to_string(index=False))
    P("\n(비교) R107 보수적 엔진 최선 w3N12L4: 평가 월 +26.3%, w3N12L6: +28.3%")
    O.to_parquet(R / "data" / "r109_grid.parquet", index=False)
    P(f"완료 ({time.time()-t_:.0f}s)")
