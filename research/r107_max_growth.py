"""R107 — 결합 계좌 최대 성장: 파산 원인 진단, 마크가격 청산, 돌파 국면 크기, 딥·돌파 명목 분리 격자.

배경: 샤프 2.93이면 이론 켈리 성장은 월 +43%인데, 실제 최대는 λ3의 월 +25%다. λ4부터는 파산한다. 무엇이 파산을 만드는가?
1) 청산 판정 가격: 마지막 체결가 분 저가(R102) vs **마크가격 분 저가**(바이낸스 실제 청산 기준, mark_1m).
2) 돌파 크기: 기본(TOP8, 대부분 고정 크기) vs **국면 상향**(× clip((30일 변동성/365일 중앙)^1.5, 0.3, 4) — 딥 V3와 같은 방향, 전일까지).
3) 명목 격자(딥·돌파 분리): 딥 w ∈ {2, 3, 4, 5}, 딥 총명목 N = w × {3, 4, 6}, 돌파 L ∈ {0.5, 1, 2, 3, 4}.
   선택 = 2022~23 월 복리 최대(파산 제외) → 평가 2024~26. 전 기간 생존도 함께 표시.
진단: 파산 시각, 그때 딥·돌파 명목, 직전 1일 BTC 수익.
사용: python research/r107_max_growth.py"""
import sys, warnings, time, itertools
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
import r102_joint_account as J, r103_breakout_cost as K, r80_max_return as R80
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)


@numba.njit(cache=True)
def sim2(e_, x_, r_, m_, k_, u_, s_, px_, LO, CL, t0, w, N, L33, X, Hh, Lk, mmr):
    """J.sim_mtm과 같고, 파산 시 (분, 딥 명목/자본, 돌파 명목/자본)을 함께 돌려준다."""
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
            if k_[q] == 1:
                Q[s_[q]] -= n / px_[q]; NS[s_[q]] -= n
            else:
                dipN -= n; dipL -= n * m_[q]
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


def grid(t0, name):
    LOs, CLs = [], []
    for sym in ("BTCUSDT", "ETHUSDT"):
        k = store.load(name, sym); mi = J.mins(k.ts)
        T1 = mi.max() + 1; lo = np.full(T1 - t0, np.nan); cl = np.full(T1 - t0, np.nan); ok = mi >= t0
        lo[mi[ok] - t0] = k.low.values[ok]; cl[mi[ok] - t0] = k.close.values[ok]
        LOs.append(pd.Series(lo).ffill().bfill().values); CLs.append(pd.Series(cl).ffill().bfill().values)
    T = min(len(a) for a in LOs)
    return np.vstack([a[:T] for a in LOs]), np.vstack([a[:T] for a in CLs]), t0


def run(E, G, w, N, L):
    LO, CL, t0 = G                                   # 격자 시작 분 = t0 (사건 첫 분이 아님 — 2026-09-30 정렬 버그 수정)
    day, pk, info = sim2(E.e.values.astype(np.int64), E.x.values.astype(np.int64), E.ret.values, E.mae.values, E.k.values.astype(np.int64),
                         E.mult.values, E.s.values.astype(np.int64), E.px.values, LO, CL, t0, float(w), float(N), float(L), 0.03, 720, 120, J.MMR)
    if pk < 0:
        return None, info
    eq = pd.Series(day, index=J.D0 + pd.to_timedelta(np.arange(len(day)), unit="D")).dropna()
    return R80.daily_from_eq(eq), pk


if __name__ == "__main__":
    t_ = time.time()
    DIP = J.dip_events()
    T = J.add_funding(J.breakout_trades(J.TOP8)); T["lf"] = K.last_funding(T)
    BK = K.events(T, False, fmax=1e-4).drop(columns="lf")
    # 국면 배율(전일까지 30일 변동성 / 365일 중앙)
    for s_, sym in ((0, "BTCUSDT"), (1, "ETHUSDT")):
        d = store.load("klines_1m", sym).set_index("ts").close.resample("D").last(); r = np.log(d).diff()
        v30 = r.rolling(30).std(); reg = (v30 / v30.rolling(365, min_periods=120).median()).shift(1)
        m = BK.s == s_; day = (J.D0 + pd.to_timedelta(BK.loc[m, "e"], unit="min")).dt.normalize()
        ri = pd.DatetimeIndex(reg.index); reg.index = ri.tz_convert(None) if ri.tz is not None else ri
        BK.loc[m, "vreg"] = reg.reindex(day.dt.tz_convert(None).values).values
    P(f"돌파 국면 배율 결측 {BK.vreg.isna().mean()*100:.1f}%"); BK["vreg"] = BK.vreg.fillna(1.0)
    start = int(J.mins([pd.Timestamp("2022-01-01", tz="UTC")])[0])
    BKs = {"기본": BK, "국면↑": BK.assign(mult=BK.mult * np.clip(BK.vreg ** 1.5, 0.3, 4))}
    GR = {"체결가": grid(start, "klines_1m"), "마크": grid(start, "mark_1m")}
    P(f"준비 완료 ({time.time()-t_:.0f}s). 마크 vs 체결가 분 저가 평균 차이 BTC {np.mean(GR['마크'][0][0]/GR['체결가'][0][0]-1)*1e4:+.2f}bp, "
      f"분 저가가 마크 쪽이 더 높은 비율 {np.mean(GR['마크'][0][0] > GR['체결가'][0][0])*100:.0f}%")
    # 1) 파산 진단: 기존 λ 스케일에서
    P("\n===== 파산 진단 (w=λ, N=4λ, L=λ)")
    for gn, G in GR.items():
        for bn, B_ in BKs.items():
            E = pd.concat([DIP, B_[B_.e >= start].drop(columns="vreg")], ignore_index=True).sort_values(["e", "k"]).reset_index(drop=True)
            out = []
            for lam in (3, 3.5, 4, 5, 6):
                d, info = run(E, G, lam, 4 * lam, lam)
                if d is None:
                    tt = J.D0 + pd.Timedelta(minutes=int(info[0]))
                    out.append(f"λ{lam} 파산 {tt:%Y-%m-%d %H:%M} (딥 {info[1]:.1f}배·돌파 {info[2]:.1f}배)")
                else:
                    b = R80.metrics(d, "2024-01-01", "2026-09-23"); out.append(f"λ{lam} 평가 월 {b['mo']:+.1f}% MDD {b['mdd']:.0f}%")
            P(f"  {gn:3s} {bn:3s} | " + " | ".join(out))
    # 2) 격자: 2022~23 선택 → 평가
    rows = []
    for gn, G in GR.items():
        for bn, B_ in BKs.items():
            E = pd.concat([DIP, B_[B_.e >= start].drop(columns="vreg")], ignore_index=True).sort_values(["e", "k"]).reset_index(drop=True)
            for w, nm, L in itertools.product((2, 3, 4, 5), (3, 4, 6), (0.5, 1, 2, 3, 4)):
                d, pk = run(E, G, w, w * nm, L)
                if d is None:
                    rows.append(dict(가격=gn, 돌파=bn, w=w, N=w * nm, L=L, 생존=False)); continue
                a = R80.metrics(d, "2022-01-01", "2024-01-01"); b = R80.metrics(d, "2024-01-01", "2026-09-23")
                sh = d["2024":"2026-09-22"]; sh = sh.mean() / sh.std() * np.sqrt(365)
                rows.append(dict(가격=gn, 돌파=bn, w=w, N=w * nm, L=L, 생존=True, 학습월=a["mo"], 학습MDD=a["mdd"], 평가월=b["mo"], 평가중앙=b["med"],
                                 평가100달=b["big"], 평가MDD=b["mdd"], 평가샤프=sh, 최대명목=pk, 연도=b["yr"]))
            P(f"  격자 {gn}·{bn} 완료 ({time.time()-t_:.0f}s)")
    O = pd.DataFrame(rows); pd.set_option("display.width", 250); pd.set_option("display.max_rows", 100)
    S = O[O.생존]
    P(f"\n생존 {len(S)}/{len(O)}")
    for (gn, bn), g in S.groupby(["가격", "돌파"]):
        top = g.sort_values("학습월", ascending=False).head(5)
        P(f"\n===== {gn} · 돌파 {bn}: 2022~23 월 복리 상위 5 → 평가"); P(top.drop(columns=["가격", "돌파", "생존"]).round(1).to_string(index=False))
    P("\n===== (참고) 평가 월 복리 상위 8 — 사후 선택이라 판단에 쓰지 않음"); P(S.sort_values("평가월", ascending=False).head(8).drop(columns=["생존"]).round(1).to_string(index=False))
    O.to_parquet(R / "data" / "r107_grid.parquet", index=False)
    P(f"완료 ({time.time()-t_:.0f}s)")
