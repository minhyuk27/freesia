"""R108 — 급락 조절기: 딥 누적이 급락 한가운데서 계좌를 청산시키는 것(R107: λ4+ 파산이 전부 2022-05-05)을 막으면 배율을 더 올릴 수 있는가.

조절기: 두 종목 중 하나라도 직전 gw분 수익 < −gd → 이후 gH분 동안 딥 총명목 한도 N → gN (새 진입만 막음, 보유분은 그대로).
  gw ∈ {30, 60, 240}분, gd ∈ {2, 3, 5}%, gN ∈ {0, 2}, gH ∈ {120, 720}분.
명목: 딥 w ∈ {3, 4, 5}, N ∈ {12, 16, 20, 24}, 돌파 L ∈ {3, 4, 6} (TOP8 펀딩≤1bp, 기본 크기).
선택: 2022~23 월 복리 최대(파산 제외) → 평가 2024~26 (2024-08-05·2025-10-10 급락은 평가 구간 = 표본 밖 급락 검정).
비교: 조절기 없음 최선(R107: w3N12L4 평가 월 +26.3%).
사용: python research/r108_crash_governor.py"""
import sys, warnings, time, itertools
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
import r102_joint_account as J, r103_breakout_cost as K, r80_max_return as R80, r107_max_growth as G7
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)


@numba.njit(cache=True)
def sim3(e_, x_, r_, m_, k_, u_, s_, px_, LO, CL, t0, w, N, L33, X, Hh, Lk, mmr, gd, gw, gN, gH):
    """sim2 + 급락 조절기: 두 종목 중 하나라도 직전 gw분 수익 < −gd 이면 이후 gH분 동안 딥 총명목 한도를 자본 × gN으로 낮춘다(전 분 종가까지만 사용)."""
    T = LO.shape[1]; nd = (t0 + T) // 1440 + 1; day = np.full(nd, np.nan); info = np.zeros(3)
    ne = len(e_); order_x = np.argsort(x_)
    cap = 1.0; Q = np.zeros(2); NS = np.zeros(2); dipN = 0.0; dipL = 0.0
    isopen = np.zeros(ne, np.bool_); nn = np.zeros(ne)
    ce_t = np.zeros(ne, np.int64); ce_r = np.zeros(ne); nc = 0; lo_p = 0; rsum = 0.0; halt = -1
    pe = 0; px_i = 0; peak = 0.0; gov = -1
    for t in range(T):
        m = t0 + t
        while px_i < ne and x_[order_x[px_i]] <= m:
            q = order_x[px_i]; px_i += 1
            if not isopen[q]: continue
            isopen[q] = False; n = nn[q]; cap += n * r_[q]
            if k_[q] == 1:
                Q[s_[q]] -= n / px_[q]; NS[s_[q]] -= n
            else:
                dipN -= n; dipL -= n * m_[q]
                ce_t[nc] = m; ce_r[nc] = r_[q]; nc += 1; rsum += r_[q]
        while lo_p < nc and ce_t[lo_p] <= m - Lk:
            rsum -= ce_r[lo_p]; lo_p += 1
        if rsum < -X and m + Hh > halt: halt = m + Hh
        if gd > 0 and t > gw:
            for sy in range(2):
                if CL[sy, t - 1] / CL[sy, t - 1 - gw] - 1 < -gd: gov = m + gH
        while pe < ne and e_[pe] <= m:
            q = pe; pe += 1
            if e_[q] < m or x_[q] <= m: continue
            if k_[q] == 0:
                if m < halt: continue
                n = min(cap * w * u_[q], max(cap * (gN if m < gov else N) - dipN, 0.0))
            else:
                n = cap * L33 / 48.0 * u_[q]
            if n <= 1e-12: continue
            isopen[q] = True; nn[q] = n
            if k_[q] == 1:
                Q[s_[q]] += n / px_[q]; NS[s_[q]] += n
            else:
                dipN += n; dipL += n * m_[q]
        tot = NS[0] + NS[1] + dipN
        if tot > 0:
            eqlo = cap + Q[0] * LO[0, t] - NS[0] + Q[1] * LO[1, t] - NS[1] - dipL
            if eqlo <= mmr * tot:
                info[0] = m; info[1] = dipN / cap; info[2] = (NS[0] + NS[1]) / cap
                return day, -1.0, info
            if tot / cap > peak: peak = tot / cap
        if (m + 1) % 1440 == 0:
            day[m // 1440] = cap + Q[0] * CL[0, t] - NS[0] + Q[1] * CL[1, t] - NS[1]
    return day, peak, info



def run(E, G, w, N, L, gov):
    LO, CL = G; t0 = int(E.e.min()); gd, gw, gN, gH = gov
    day, pk, info = sim3(E.e.values.astype(np.int64), E.x.values.astype(np.int64), E.ret.values, E.mae.values, E.k.values.astype(np.int64),
                         E.mult.values, E.s.values.astype(np.int64), E.px.values, LO, CL, t0, float(w), float(N), float(L), 0.03, 720, 120, J.MMR,
                         float(gd), int(gw), float(gN), int(gH))
    if pk < 0: return None, info
    eq = pd.Series(day, index=J.D0 + pd.to_timedelta(np.arange(len(day)), unit="D")).dropna()
    return R80.daily_from_eq(eq), pk


if __name__ == "__main__":
    t_ = time.time()
    DIP = J.dip_events()
    T = J.add_funding(J.breakout_trades(J.TOP8)); T["lf"] = K.last_funding(T)
    BK = K.events(T, False, fmax=1e-4).drop(columns="lf")
    start = int(J.mins([pd.Timestamp("2022-01-01", tz="UTC")])[0])
    E = pd.concat([DIP, BK[BK.e >= start]], ignore_index=True).sort_values(["e", "k"]).reset_index(drop=True)
    Gd = G7.grid(start, "mark_1m")
    govs = [(0, 1, 0, 1)] + [(gd, gw, gN, gH) for gw in (30, 60, 240) for gd in (0.02, 0.03, 0.05) for gN in (0, 2) for gH in (120, 720)]
    rows = []
    for gv in govs:
        for w, N, L in itertools.product((3, 4, 5), (12, 16, 20, 24), (3, 4, 6)):
            d, pk = run(E, Gd, w, N, L, gv)
            if d is None:
                rows.append(dict(gov=str(gv), w=w, N=N, L=L, ok=False, 파산=f"{J.D0 + pd.Timedelta(minutes=int(pk[0])):%Y-%m-%d}")); continue
            a = R80.metrics(d, "2022-01-01", "2024-01-01"); b = R80.metrics(d, "2024-01-01", "2026-09-23")
            s = d["2024":"2026-09-22"]
            rows.append(dict(gov=str(gv), w=w, N=N, L=L, ok=True, 학습월=a["mo"], 학습MDD=a["mdd"], 평가월=b["mo"], 평가중앙=b["med"], 평가100달=b["big"],
                             평가MDD=b["mdd"], 평가샤프=s.mean() / s.std() * np.sqrt(365), 명목=pk, 연도=b["yr"]))
        P(f"  조절기 {gv} 완료 ({time.time()-t_:.0f}s)")
    O = pd.DataFrame(rows); pd.set_option("display.width", 250); pd.set_option("display.max_rows", 100)
    S = O[O.ok]
    P(f"\n생존 {len(S)}/{len(O)} · 조절기 없음 생존 {S[S.gov == str(govs[0])].shape[0]}/36")
    P("파산 날짜 분포(상위): " + ", ".join(f"{k} {v}" for k, v in O[~O.ok].파산.value_counts().head(6).items()))
    P("\n===== 2022~23 월 복리 상위 12 → 평가 (사전 선택)"); P(S.sort_values("학습월", ascending=False).head(12).drop(columns="ok").round(1).to_string(index=False))
    best = S.sort_values("학습월", ascending=False).iloc[0]
    P(f"\n선택: {best.gov} w{best.w} N{best.N} L{best.L} → 평가 월 {best.평가월:+.1f}% · 중앙 {best.평가중앙:+.1f}% · ≥100% {best.평가100달}달 · MDD {best.평가MDD:.0f}%")
    P("\n조절기 없음 최선(학습 기준):"); P(S[S.gov == str(govs[0])].sort_values("학습월", ascending=False).head(3).drop(columns="ok").round(1).to_string(index=False))
    P("\n(참고) 평가 월 상위 8 — 사후"); P(S.sort_values("평가월", ascending=False).head(8).drop(columns="ok").round(1).to_string(index=False))
    O.to_parquet(R / "data" / "r108_grid.parquet", index=False)
    P(f"완료 ({time.time()-t_:.0f}s)")
