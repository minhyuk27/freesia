"""R122-3 — 엣지 공장 3단계: 통과 후보(학습 t·위약·검증 통과 30개)를 코어에 결합.

1) 통과 후보의 거래를 모은다(같은 거래 중복 제거). 2) 코어 z0.31 딥과 같은 거래(종목·진입 분 동일)는 뺀다 — 새로 더해지는 거래만.
3) 일간 손익 상관(코어 대비), 롱/숏별 거래당 bp(학습·검증·시험).
4) R118 엔진: 코어(z0.31 + z0.33 pnet15, w3·N12·L12/16) + 새 채널(k=5, 명목 = 자본 × Wn × 1, 딥처럼 분 단위 보유).
   Wn ∈ {0.5, 1, 2, 3, 4, 6} 선택 = 2022~24(학습+검증) 월 복리, 판정 = 시험 2025~26.
   변형: 롱만 / 숏 포함 / 시험 구간 거래당 상위 절반 계열만(참고 — 사후라 판정에 쓰지 않음).
사용: python research/r122_combine.py"""
import sys, warnings, time
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
import r118_engine as G, r80_max_return as R80, r122_scan as SC
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)

if __name__ == "__main__":
    t_ = time.time()
    T = pd.read_parquet(R / "data" / "r122_trades.parquet"); FX = pd.read_parquet(R / "data" / "r122_feats.parquet")
    O = pd.read_parquet(R / "data" / "r122_scan.parquet"); S = O[O["pass"]]
    day = (T.e.values // 1440); tr = day < SC.D_TR
    keep = np.zeros(len(T), bool)
    for c in S.itertuples():
        m = (T.mech == c.mech).values; x = FX[c.feat].values
        q = {"Q1": 0.2, "L30": 0.3, "H30": 0.7, "Q5": 0.8}[c.buck]; e_ = np.nanquantile(x[m & tr], q)
        keep |= m & ((x <= e_) if c.buck in ("Q1", "L30") else (x >= e_))
    N = T[keep].copy()
    N = N.sort_values(["sym", "e", "side"]).drop_duplicates(["sym", "e", "side"])       # 같은 분·종목·방향 진입은 하나로
    core_d = G.std_dip(); core_b = G.std_breakout()
    ck = set(zip(core_d.s.values, core_d.e.values))
    N["s"] = (N.sym == "ETHUSDT").astype(int)
    ov = np.array([(s, e) in ck for s, e in zip(N.s.values, N.e.values)])
    P(f"통과 후보 거래 {keep.sum():,} → 중복 제거 {len(N):,}, 코어 딥과 겹침 {ov.mean()*100:.1f}% → 새 거래 {(~ov).sum():,}")
    N = N[~ov].copy(); N["ret"] = N.ret - 1.2e-4
    N["k"] = 5; N["mult"] = 1.0
    yr = (G.D0 + pd.to_timedelta(N.e, unit="min")).dt.year
    for sd, g in N.groupby("side"):
        y = yr[g.index]
        P(f"  {'롱' if sd > 0 else '숏'} {len(g):,}건: 학습 {g.ret[y < 2024].mean()*1e4:+.1f}bp · 검증 {g.ret[y == 2024].mean()*1e4:+.1f}bp · 시험 {g.ret[y >= 2025].mean()*1e4:+.1f}bp | 하루 {len(g)/ (len(np.unique(g.e//1440))):.2f}건/활동일")
    core = pd.concat([core_d, core_b], ignore_index=True)
    dcore, _ = G.run_multi(core, 3, 12, {1: 12 / 48})
    nd = pd.Series(N.ret.values, index=(G.D0 + pd.to_timedelta(N.e, unit="min")).dt.normalize()).groupby(level=0).sum()
    I = dcore.index.intersection(nd.index)
    P(f"새 채널 일간 손익 vs 코어 일간 수익 상관: {np.corrcoef(dcore.reindex(I), nd.reindex(I))[0,1]:+.2f} (n {len(I)})")
    rows = []
    for lab, Nv in (("롱만", N[N.side > 0]), ("롱+숏", N)):
        for L in (12, 16):
            for Wn in (0, 0.5, 1, 2, 3, 4, 6):
                E = pd.concat([core] + ([Nv[G.COLS]] if Wn else []), ignore_index=True)
                d, pk = G.run_multi(E, 3, 12, {1: L / 48, 5: Wn})
                if d is None: rows.append(dict(변형=lab, L=L, Wn=Wn, ok=False)); continue
                a = R80.metrics(d, "2022-01-01", "2025-01-01"); b = R80.metrics(d, "2025-01-01", "2026-09-23"); sb = d["2025":"2026-09-22"]
                rows.append(dict(변형=lab, L=L, Wn=Wn, ok=True, 선택월_22_24=a["mo"], MDD_22_24=a["mdd"], 시험월_25_26=b["mo"], 시험중앙=b["med"], 시험MDD=b["mdd"],
                                 시험샤프=sb.mean() / sb.std() * np.sqrt(365), 시험100달=b["big"], 연도=R80.metrics(d, "2024-01-01", "2026-09-23")["yr"]))
            P(f"  {lab} L{L} 완료 ({time.time()-t_:.0f}s)")
    O2 = pd.DataFrame(rows); pd.set_option("display.width", 250); pd.set_option("display.max_rows", 60)
    P("\n===== 결합 계좌 (w3·N12)"); P(O2.round(2).to_string(index=False))
    for (lab, L), g in O2[O2.ok].groupby(["변형", "L"]):
        b = g.sort_values("선택월_22_24", ascending=False).iloc[0]; c0 = g[g.Wn == 0].iloc[0]
        P(f"  {lab} L{L}: 2022~24로 고른 Wn={b.Wn} → 시험 2025~26 월 {b.시험월_25_26:+.1f}% 샤프 {b.시험샤프:.2f} MDD {b.시험MDD:.0f}% (코어만 {c0.시험월_25_26:+.1f}% / {c0.시험샤프:.2f} / {c0.시험MDD:.0f}%)")
    N[G.COLS + ["mech"]].to_parquet(R / "data" / "r122_new_events.parquet", index=False)
    P(f"완료 ({time.time()-t_:.0f}s)")
