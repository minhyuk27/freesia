"""R116b — 새 데이터 필터 결합 계좌: z0.31 딥 × 업비트 김치 프리미엄 5분 변화(dkp5, R116 C) + z0.33 돌파 × 선물 순매수(pnet15 F3, R113b).

딥 필터(사전 지정 dkp5, 경계 = 학습 2022~23 딥 분위): D3 = 분위 ≥ 3만, D2 = 분위 ≥ 2만, DS = 크기 × (1 + a(q − 3)/2), a ∈ {0.5, 1.0}.
돌파: pnet15 F3(하위 40% 제외) 고정.
계좌: R109 정확 엔진, 딥 w ∈ {3, 4}, N ∈ {12, 16, 20}, 돌파 L ∈ {12, 16}. 선택 2022~23 → 평가 2024~26.
질문: ① 딥 필터가 성장·MDD를 개선하는가 ② 걸러낸 딥이 파산 원인(급락 지속)을 줄여 N 12배 벽을 풀어 주는가.
사용: python research/r116b_combined_filters_account.py"""
import sys, warnings, time, itertools
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
import r102_joint_account as J, r103_breakout_cost as K, r80_max_return as R80, r107_max_growth as G7, r109_dip_exact_mtm as M
import r113_spot_perp_flow as F13, r116_kimchi_premium as F16
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)


def attach(df, FE, col):
    out = np.full(len(df), np.nan)
    for s_, sym in ((0, "BTCUSDT"), (1, "ETHUSDT")):
        mk = (df.s == s_).values; ts = pd.DatetimeIndex(J.D0 + pd.to_timedelta(df.e.values[mk] - 1, unit="min"))
        out[mk] = FE[sym][col].reindex(ts).values
    return out


def quint(v, train):
    edges = pd.Series(v[train]).quantile([.2, .4, .6, .8]).values; q = np.searchsorted(edges, v) + 1
    return np.where(np.isnan(v), 3, q)


if __name__ == "__main__":
    t_ = time.time()
    T8 = J.add_funding(J.breakout_trades(J.TOP8)); T8["lf"] = K.last_funding(T8); BK = K.events(T8, False, fmax=1e-4).drop(columns="lf")
    FS = {s: F13.feats(s)[0][["pnet15"]] for s in ("BTCUSDT", "ETHUSDT")}
    BK["pnet15"] = attach(BK, FS, "pnet15"); del FS
    qb = quint(BK.pnet15.values, ((J.D0 + pd.to_timedelta(BK.e, unit="min")) < F13.SPLIT).values); BKf = BK[qb >= 3]
    D, _, _, W = M.dip_with_px()
    FK = {s: F16.feats(s)[0][["dkp5"]] for s in ("BTCUSDT", "ETHUSDT")}
    D["dkp5"] = attach(D, FK, "dkp5"); del FK
    trd = ((J.D0 + pd.to_timedelta(D.e, unit="min")) < F13.SPLIT).values
    qd = quint(D.dkp5.values, trd)
    P("딥 dkp5 분위별 거래당 bp 학습: " + " ".join(f"{D.ret[(qd == i) & trd].mean()*1e4:+.0f}" for i in range(1, 6)) +
      " | 평가: " + " ".join(f"{D.ret[(qd == i) & ~trd].mean()*1e4:+.0f}" for i in range(1, 6)))
    # 파산 원인 급락 구간의 딥이 어느 분위였나
    for day in ("2022-05-05", "2023-12-11", "2024-01-03", "2024-08-05", "2025-02-03"):
        m = (J.D0 + pd.to_timedelta(D.e, unit="min")).dt.strftime("%Y-%m-%d").values == day
        if m.any(): P(f"  {day} 딥 {m.sum()}건 분위 {np.bincount(qd[m], minlength=6)[1:].tolist()} 거래당 {D.ret[m].mean()*1e4:+.0f}bp")
    start = int(J.mins([pd.Timestamp("2022-01-01", tz="UTC")])[0]); Gd = G7.grid(start, "mark_1m")
    cols = ["e", "x", "ret", "mae", "k", "mult", "s", "px"]
    variants = {"딥 필터 없음": D, "D2(q≥2)": D[qd >= 2], "D3(q≥3)": D[qd >= 3],
                "DS0.5": D.assign(mult=D.mult * (1 + 0.5 * (qd - 3) / 2)), "DS1.0": D.assign(mult=D.mult * (1 + 1.0 * (qd - 3) / 2))}
    rows = []
    for nm, Dv in variants.items():
        E = pd.concat([Dv[cols], BKf[BKf.e >= start][cols]], ignore_index=True).sort_values(["e", "k"]).reset_index(drop=True)
        for w, N, L in itertools.product((3, 4), (12, 16, 20), (12, 16)):
            d, pk = M.run(E, Gd, w, N, L)
            if d is None:
                rows.append(dict(딥=nm, w=w, N=N, L=L, ok=False, 파산=f"{J.D0 + pd.Timedelta(minutes=int(pk[0])):%Y-%m-%d}")); continue
            a = R80.metrics(d, "2022-01-01", "2024-01-01"); b = R80.metrics(d, "2024-01-01", "2026-09-23"); sr = d["2024":"2026-09-22"]
            rows.append(dict(딥=nm, w=w, N=N, L=L, ok=True, 학습월=a["mo"], 학습MDD=a["mdd"], 평가월=b["mo"], 평가중앙=b["med"], 평가100달=b["big"], 평가MDD=b["mdd"],
                             평가샤프=sr.mean() / sr.std() * np.sqrt(365), 연도=b["yr"]))
        P(f"  {nm} 완료 ({time.time()-t_:.0f}s)")
    O = pd.DataFrame(rows); pd.set_option("display.width", 250); pd.set_option("display.max_rows", 100)
    S = O[O.ok]
    P("\n생존: " + " | ".join(f"{k}: {int(g.ok.sum())}/{len(g)} (N≤{g[g.ok].N.max() if g.ok.any() else '-'})" for k, g in O.groupby("딥", sort=False)))
    P("파산: " + ", ".join(f"{k} {v}" for k, v in O[~O.ok].파산.value_counts().items()))
    P("\n===== 딥 변형별 학습(2022~23) 최선 → 평가")
    for k, g in S.groupby("딥", sort=False):
        b = g.sort_values("학습월", ascending=False).iloc[0]
        P(f"  {k:10s} w{b.w} N{b.N} L{b.L} 학습 {b.학습월:+.1f}% → 평가 월 {b.평가월:+.1f}% 중앙 {b.평가중앙:+.1f}% ≥100% {b.평가100달}달 MDD {b.평가MDD:.0f}% 샤프 {b.평가샤프:.2f} | {b.연도}")
    P("\n===== 전체 학습 상위 10"); P(S.sort_values("학습월", ascending=False).head(10).drop(columns=["ok", "파산"], errors="ignore").round(1).to_string(index=False))
    O.to_parquet(R / "data" / "r116b.parquet", index=False)
    P(f"완료 ({time.time()-t_:.0f}s)")
