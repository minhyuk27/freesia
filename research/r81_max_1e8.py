"""R81 — 자본 1억 이하 전제의 최대 수익 정밀 탐색 (R80 확장). 봉우리가 아니라 고원을 고른다.

자본 1억(≈ 7.1만 달러) → w3·N12면 총명목 ≈ 86만 달러 = 바이낸스 75배 구간(유지증거금 0.65%) → 엔진 MMR = 0.0065.
격자: 딥 문턱 pc {10,15,20,25,30bp} × 국면 지수 α {1.5,2,2.5} × 배율 상한 cap {4,6,8} × w {2,2.5,3,3.5} × N {10,12,14}
      × 킬스위치 (X, H) {(2%,720분), (3%,720분), (2%,1440분)} (L = 120분 고정). 배율 = min(clip(국면^α,0.3,4) × clip(예측/10bp,0.5,3), cap).
분산: R79 롱·숏 조각 일간 수익을 비중 b {0, 0.15, 0.25} × 5배로 덧씌움(일간 가중합).
선택(사전 고정): 2022~23 성과로, MDD ≥ −90% 인 구성 중 '이웃 중앙값'(pc·α·cap·w·N을 한 칸씩 바꾼 이웃들의 월 복리 중앙값)이 최대인 구성.
확인: 선택 구성의 2024~26 성과와 전 기간 성과, 월 분포.
사용: python research/r81_max_1e8.py"""
import sys, warnings, itertools, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
PCS = (0.0010, 0.0015, 0.0020, 0.0025, 0.0030); ALS = (1.5, 2.0, 2.5); CAPS = (4.0, 6.0, 8.0)
WS = (2.0, 2.5, 3.0, 3.5); NS = (10, 12, 14); KS = ((0.02, 720), (0.03, 720), (0.02, 1440)); BS = (0.0, 0.15, 0.25)


def work(pc):
    import r40_regime_leverage as RL
    import r80_max_return as R80
    RL.MMR = 0.0065
    W = R80.dip_frame(pc); out = {}
    V2 = np.clip(W.pred.values / 0.001, 0.5, 3.0)
    for al, cap, w, N, (X, H) in itertools.product(ALS, CAPS, WS, NS, KS):
        mult = np.minimum(np.clip(W.vreg.values ** al, 0.3, 4.0) * V2, cap)
        res = RL.sim(W, w, N, mult, X=X, H=H)
        out[(pc, al, cap, w, N, X, H)] = R80.daily_from_eq(res[0]) if res is not None else None
    return out


if __name__ == "__main__":
    import r80_max_return as R80
    import r79_longshort_book as B7
    from freesia import store
    t_ = time.time()
    A = pd.read_parquet(R / "data" / "r79_trades.parquet")
    d = store.load("klines_1m", "BTCUSDT").set_index("ts").close.resample("1D").last(); reg = (np.log(d).diff(30) < 0).shift(1)
    Sx = A[A.sl.str[-1].eq("S")].copy(); Sx = Sx[reg.reindex(Sx.te.dt.floor("D")).fillna(False).values.astype(bool)]
    A2 = pd.concat([A, Sx.assign(sl=lambda x: x.sl.str.replace("|S", "|BEAR|S", regex=False))], ignore_index=True)
    picks = {Y: B7.select(A2, Y, 10, ("L", "S"), lookback=2) for Y in range(2022, 2027)}
    rows = []
    for Y, pk in picks.items():
        x = A2[A2.sl.isin(pk) & (A2.te.dt.year == Y)].copy(); x["pnl"] = x.r / len(pk) * 0.5; rows.append(x)
    X_ = pd.concat(rows); DIV = X_.groupby(X_.tx.dt.floor("D")).pnl.sum().reindex(R80.IDX).fillna(0)
    DD = {}
    with Pool(2) as pool:
        for o in pool.imap_unordered(work, PCS):
            DD.update(o); P(f"  문턱 완료 {len(DD)}/{len(PCS)*len(ALS)*len(CAPS)*len(WS)*len(NS)*len(KS)} ({time.time()-t_:.0f}s)")
    res = []
    for key, dd in DD.items():
        for b in BS:
            if dd is None: res.append((*key, b, -100, -100, -100, -100, 0)); continue
            dr = (1 - b) * dd + b * 5 * DIV
            m1 = R80.metrics(dr, "2022-01-01", "2024-01-01"); m2 = R80.metrics(dr, "2024-01-01", "2027-01-01")
            res.append((*key, b, m1["mo"], m1["mdd"], m2["mo"], m2["mdd"], 1))
    O = pd.DataFrame(res, columns=["pc", "al", "cap", "w", "N", "X", "H", "b", "tr_mo", "tr_mdd", "te_mo", "te_mdd", "ok"])
    O.to_csv(R / "logs" / "r81.csv", index=False)
    grids = {"pc": PCS, "al": ALS, "cap": CAPS, "w": WS, "N": NS}
    key = lambda r: tuple(r[c] for c in ("pc", "al", "cap", "w", "N", "X", "H", "b"))
    lut = {key(r): r.tr_mo for _, r in O.iterrows()}

    def neigh(r):
        vals = []
        for c, g in grids.items():
            i = g.index(r[c])
            for j in (i - 1, i + 1):
                if 0 <= j < len(g):
                    rr = r.copy(); rr[c] = g[j]; v = lut.get(key(rr))
                    if v is not None: vals.append(v)
        return np.median(vals) if vals else np.nan
    C = O[(O.ok == 1) & (O.tr_mdd >= -90)].copy(); C["nb"] = C.apply(neigh, axis=1); C["score"] = np.minimum(C.tr_mo, C.nb)
    P(f"\n===== 구성 {len(O):,}개 (파산 {(O.ok == 0).sum():,}), 학습 MDD≥−90% 후보 {len(C):,}")
    for lim in (-90, -80, -70, -50):
        Z = C[C.tr_mdd >= lim].sort_values("score", ascending=False).head(3)
        P(f"\n  [학습 MDD ≥ {lim}% · 고원 점수 상위 3]")
        for _, r in Z.iterrows():
            dd = DD[(r.pc, r.al, r.cap, r.w, r.N, r.X, r.H)]; dr = (1 - r.b) * dd + r.b * 5 * DIV; m = R80.metrics(dr)
            P(f"    문턱 {r.pc*1e4:.0f}bp α{r.al} 상한{r.cap:.0f} w{r.w} N{r.N:.0f} 킬({r.X*100:.0f}%,{r.H/60:.0f}h) 분산{r.b:.2f}×5 | "
              f"학습 월 {r.tr_mo:+.1f}% (이웃 {r.nb:+.1f}%) MDD {r.tr_mdd:.0f}% → 평가 24~26 월 {r.te_mo:+.1f}% MDD {r.te_mdd:.0f}% | "
              f"전체 월 {m['mo']:+.1f}% 중앙 {m['med']:+.1f}% 하위10% {m['p10']:+.1f}% 월≥100% {m['big']}회 MDD {m['mdd']:.0f}% | {m['yr']}")
    P("\n  참고: 평가 구간(24~26) 월 복리 상위 5 (사후, 선택에 쓰지 않음)")
    P(O[O.ok == 1].sort_values("te_mo", ascending=False).head(5).round(3).to_string(index=False))
    P("\n  파산 경계: N별 학습 파산 비율 / w별")
    P(O.groupby("N").ok.mean().round(3).to_string()); P(O.groupby("w").ok.mean().round(3).to_string())
    (R / "logs" / "r81.summary").write_text("R81 1억 이하 최대 수익 정밀 탐색 완료. 세부 logs/r81.log\n")
