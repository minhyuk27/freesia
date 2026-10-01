"""R123-3 — 엣지 공장 공용 증분 검정: 2단계 통과 후보마다 '코어와 겹치지 않는 거래'만 새 채널(k=5)로 코어에 더한다.

R122 교훈: 개별 후보 통과(일간 평균)와 코어 대비 증분은 다르다. 그래서 후보마다
  ① 코어 z0.31 딥과 같은 거래(종목·진입 분) 제외, ② 거래당 bp 학습·검증·시험, ③ R118 엔진 결합(w3·N12·L12, Wn ∈ {0.5, 1, 2, 4}),
  Wn 선택 = 2022~24 월 복리, 판정 = 시험 2025~26(코어만: 월 +28.60%, 샤프 2.33, MDD −58%).
사용: python research/r123_incremental.py <접두어>   (data/<접두어>_trades·feats·scan.parquet)"""
import sys, warnings, time
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
import r118_engine as G, r80_max_return as R80, r122_scan as SC
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)

if __name__ == "__main__":
    t_ = time.time(); PFX = sys.argv[1]
    T = pd.read_parquet(R / "data" / f"{PFX}_trades.parquet"); FX = pd.read_parquet(R / "data" / f"{PFX}_feats.parquet")
    S = pd.read_parquet(R / "data" / f"{PFX}_scan.parquet"); S = S[S["pass"]].reset_index(drop=True)
    P(f"통과 후보 {len(S)}")
    if len(S) == 0: sys.exit(0)
    day = T.e.values // 1440; tr = day < SC.D_TR
    core_d = G.std_dip(); core = pd.concat([core_d, G.std_breakout()], ignore_index=True)
    ck = set(zip(core_d.s.values, core_d.e.values))
    dc, _ = G.run_multi(core, 3, 12, {1: 12 / 48})
    sh = lambda x: x.mean() / x.std() * np.sqrt(365)
    rows = []
    for c in S.itertuples():
        m = (T.mech == c.mech).values; x = FX[c.feat].values
        q = {"Q1": .2, "L30": .3, "H30": .7, "Q5": .8}[c.buck]; e_ = np.nanquantile(x[m & tr], q)
        sel = m & ((x <= e_) if c.buck in ("Q1", "L30") else (x >= e_))
        N = T[sel].copy(); N["s"] = (N.sym == "ETHUSDT").astype(int)
        N = N[[(s, e) not in ck for s, e in zip(N.s.values, N.e.values)]].copy(); N["k"] = 5; N["mult"] = 1.0
        y = (G.D0 + pd.to_timedelta(N.e, unit="min")).dt.year.values
        nd = pd.Series(N.ret.values, index=(G.D0 + pd.to_timedelta(N.e, unit="min")).dt.normalize()).groupby(level=0).sum()
        I = dc.index.intersection(nd.index); corr = np.corrcoef(dc.reindex(I), nd.reindex(I))[0, 1] if len(I) > 30 else np.nan
        best = None
        for Wn in (0.5, 1, 2, 4):
            d, pk = G.run_multi(pd.concat([core, N[G.COLS]], ignore_index=True), 3, 12, {1: 12 / 48, 5: Wn})
            if d is None: continue
            a = R80.metrics(d, "2022-01-01", "2025-01-01")
            if best is None or a["mo"] > best[1]: best = (Wn, a["mo"], d)
        r = dict(cand=f"{c.mech}|{c.feat}|{c.buck}", 새거래=len(N), 상관=corr, 학습bp=N.ret[y < 2024].mean() * 1e4, 검증bp=N.ret[y == 2024].mean() * 1e4, 시험bp=N.ret[y >= 2025].mean() * 1e4)
        if best is None: r.update(파산=True)
        else:
            Wn, mo, d = best; b = R80.metrics(d, "2025-01-01", "2026-09-23")
            r.update(Wn=Wn, 선택월=mo, 시험월=b["mo"], 시험샤프=sh(d["2025":"2026-09-22"]), 시험MDD=b["mdd"])
        rows.append(r)
    O = pd.DataFrame(rows); pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    P("코어만(L12): 2022~24 월 +40.57%, 시험 2025~26 월 +28.60% 샤프 2.33 MDD -58%")
    P(O.round(2).sort_values("시험월", ascending=False).to_string(index=False))
    O.to_parquet(R / "data" / f"{PFX}_incremental.parquet", index=False)
    P(f"완료 ({time.time()-t_:.0f}s)")
