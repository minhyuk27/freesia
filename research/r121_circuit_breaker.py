"""R121 — 계좌 회로차단기: 평소엔 딥 총명목 한도를 크게(N 16~32), 계좌 미실현 손실이 자본의 BD를 넘으면 열린 딥 전부를 정리하고 BH분 쉰다.

배경: 딥 12배 벽(R109)은 급락 며칠(2022-05-05·2024-01-03·2024-08-05 등) 때문. 거래별 손절(R112)은 평소의 꼬리 되돌림까지 잘라 엣지를 죽였다.
계좌 수준 차단기는 급락 며칠에만 작동하므로 평소 엣지는 그대로 두고 파산만 막는다 — 사용자 조건(낙폭 무관, 완전 파산만 회피)과 일치.
차단 처리: 그 분 마크 저가 × (1 − 슬리피지 0.5%)에 테이커 청산(보수적), 딥만 정리(돌파 유지), 이후 BH분 새 딥 금지.
격자: 딥 w ∈ {3, 4, 5}, N ∈ {12, 16, 20, 24, 32}, BD ∈ {0(없음), 0.2, 0.3, 0.4, 0.5}, BH ∈ {720, 2880}분, 돌파 pnet15 L ∈ {12, 16}.
선택 2022~23 월 복리 → 평가 2024~26 · 2025~26(정직 구간). 차단 발동 횟수·날짜 보고.
사용: python research/r121_circuit_breaker.py"""
import sys, warnings, time, itertools
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
import r118_engine as G, r102_joint_account as J, r80_max_return as R80
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)


@numba.njit(cache=True)
def sim_brk(e_, x_, r_, k_, u_, s_, px_, sd_, LO, HI, CL, t0, w, N, Wk, X, Hh, Lk, mmr, BD, BH, bslip):
    T = LO.shape[1]; nd = (t0 + T) // 1440 + 1; day = np.full(nd, np.nan); info = np.zeros(3)
    ne = len(e_); order_x = np.argsort(x_)
    cap = 1.0; QL = np.zeros(2); NL = np.zeros(2); QS = np.zeros(2); NS = np.zeros(2); dipN = 0.0
    isopen = np.zeros(ne, np.bool_); nn = np.zeros(ne)
    ce_t = np.zeros(ne, np.int64); ce_r = np.zeros(ne); nc = 0; lo_p = 0; rsum = 0.0; halt = -1
    pe = 0; px_i = 0; peak = 0.0; QD = np.zeros(2); ND = np.zeros(2); nbrk = 0; bhalt = -1
    for t in range(T):
        m = t0 + t
        while px_i < ne and x_[order_x[px_i]] <= m:
            q = order_x[px_i]; px_i += 1
            if not isopen[q]: continue
            isopen[q] = False; n = nn[q]; cap += n * r_[q]; s = s_[q]
            if sd_[q] > 0:
                QL[s] -= n / px_[q]; NL[s] -= n
            else:
                QS[s] -= n / px_[q]; NS[s] -= n
            if k_[q] == 0:
                QD[s] -= n / px_[q]; ND[s] -= n
                dipN -= n; ce_t[nc] = m; ce_r[nc] = r_[q]; nc += 1; rsum += r_[q]
        while lo_p < nc and ce_t[lo_p] <= m - Lk:
            rsum -= ce_r[lo_p]; lo_p += 1
        if rsum < -X and m + Hh > halt: halt = m + Hh
        while pe < ne and e_[pe] <= m:
            q = pe; pe += 1
            if e_[q] < m or x_[q] <= m: continue
            if k_[q] == 0:
                if m < halt or m < bhalt: continue
                n = min(cap * w * u_[q], max(cap * N - dipN, 0.0))
            else:
                n = cap * Wk[k_[q]] * u_[q]
            if n <= 1e-12: continue
            isopen[q] = True; nn[q] = n; s = s_[q]
            if sd_[q] > 0:
                QL[s] += n / px_[q]; NL[s] += n
            else:
                QS[s] += n / px_[q]; NS[s] += n
            if k_[q] == 0:
                dipN += n; QD[s] += n / px_[q]; ND[s] += n
        tot = NL[0] + NL[1] + NS[0] + NS[1]
        if tot > 0:
            eqw = cap
            for s in range(2):
                eqw += QL[s] * LO[s, t] - NL[s] + NS[s] - QS[s] * HI[s, t]
            if BD > 0 and dipN > 0 and eqw - cap < -BD * cap:
                # 회로차단기: 열린 딥 전부를 이 분 마크 저가 × (1 − bslip)에 청산(테이커), 이후 BH분 딥 중지
                for s in range(2):
                    if ND[s] > 0:
                        cap += QD[s] * LO[s, t] * (1 - bslip) - ND[s] * (1 + 5e-4)
                        QL[s] -= QD[s]; NL[s] -= ND[s]; QD[s] = 0.0; ND[s] = 0.0
                for q2 in range(ne):
                    if isopen[q2] and k_[q2] == 0: isopen[q2] = False
                dipN = 0.0; bhalt = m + BH; nbrk += 1
                if cap <= 0: return day, -1.0, info
                eqw = cap
                for s in range(2):
                    eqw += QL[s] * LO[s, t] - NL[s] + NS[s] - QS[s] * HI[s, t]
                tot = NL[0] + NL[1] + NS[0] + NS[1]
            if eqw <= mmr * tot:
                info[0] = m; info[1] = dipN / cap; info[2] = (tot - dipN) / cap
                return day, -1.0, info
            if tot / cap > peak: peak = tot / cap
        if (m + 1) % 1440 == 0:
            v = cap
            for s in range(2):
                v += QL[s] * CL[s, t] - NL[s] + NS[s] - QS[s] * CL[s, t]
            day[m // 1440] = v
    info[0] = nbrk
    return day, peak, info



def run(E, w, N, L, BD, BH, start="2022-01-01"):
    t0 = int(J.mins([pd.Timestamp(start, tz="UTC")])[0])
    E = E[E.e >= t0].sort_values(["e", "k"]).reset_index(drop=True)
    LO, HI, CL, t0 = G.grid_hl(t0); wk = np.zeros(4); wk[1] = L / 48
    day, pk, info = sim_brk(E.e.values.astype(np.int64), E.x.values.astype(np.int64), E.ret.values.astype(float), E.k.values.astype(np.int64),
                            E.mult.values.astype(float), E.s.values.astype(np.int64), E.px.values.astype(float), E.side.values.astype(np.int64),
                            LO, HI, CL, t0, float(w), float(N), wk, 0.03, 720, 120, G.MMR, float(BD), int(BH), 0.005)
    if pk < 0: return None, info
    eq = pd.Series(day, index=G.D0 + pd.to_timedelta(np.arange(len(day)), unit="D")).dropna()
    return R80.daily_from_eq(eq), info


if __name__ == "__main__":
    t_ = time.time()
    E = pd.concat([G.std_dip(), G.std_breakout()], ignore_index=True)
    d0, _ = run(E, 3, 12, 12, 0, 720); s0 = G.summary(d0)
    P(f"재현(차단기 없음 w3N12L12): 평가 월 {s0['평가월']:+.2f}% (기대 +32.92%)")
    rows = []
    for BD, BH in [(0, 720)] + [(b, h) for b in (0.2, 0.3, 0.4, 0.5) for h in (720, 2880)]:
        for w, N, L in itertools.product((3, 4, 5), (12, 16, 20, 24, 32), (12, 16)):
            if N < 3 * w: continue
            d, info = run(E, w, N, L, BD, BH)
            if d is None: rows.append(dict(BD=BD, BH=BH, w=w, N=N, L=L, ok=False)); continue
            b2 = R80.metrics(d, "2025-01-01", "2026-09-23"); s2 = d["2025":"2026-09-22"]
            rows.append(dict(BD=BD, BH=BH, w=w, N=N, L=L, ok=True, 차단=int(info[0]), **G.summary(d), 월_25_26=b2["mo"], MDD_25_26=b2["mdd"], 샤프_25_26=s2.mean() / s2.std() * np.sqrt(365)))
        P(f"  BD{BD} BH{BH} 완료 ({time.time()-t_:.0f}s)")
    O = pd.DataFrame(rows); pd.set_option("display.width", 270); pd.set_option("display.max_rows", 80)
    S = O[O.ok]
    P("\n생존: " + " | ".join(f"BD{k[0]}/BH{k[1]}: {int(g.ok.sum())}/{len(g)} N≤{g[g.ok].N.max() if g.ok.any() else '-'}" for k, g in O.groupby(["BD", "BH"])))
    P("\n===== (BD, BH)별 학습(2022~23) 최선 → 평가")
    for k, g in S.groupby(["BD", "BH"]):
        b = g.sort_values("학습월", ascending=False).iloc[0]
        P(f"  BD{k[0]} BH{k[1]}: w{b.w} N{b.N} L{b.L} 차단 {b.차단}회 | 학습 {b.학습월:+.1f}% → 24~26 월 {b.평가월:+.1f}% 중앙 {b.평가중앙:+.1f}% ≥100% {b.평가100달}달 MDD {b.평가MDD:.0f}% | 25~26 월 {b.월_25_26:+.1f}% MDD {b.MDD_25_26:.0f}% 샤프 {b.샤프_25_26:.2f} | {b.연도}")
    P("\n===== 전체 학습 상위 12"); P(S.sort_values("학습월", ascending=False).head(12).drop(columns=["ok", "연도"]).round(1).to_string(index=False))
    O.to_parquet(R / "data" / "r121.parquet", index=False)
    P(f"완료 ({time.time()-t_:.0f}s)")
