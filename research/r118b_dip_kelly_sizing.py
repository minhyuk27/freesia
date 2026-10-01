"""R118b — z0.31 딥 거래별 켈리 크기: R111 급락 지속 위험 p(표본 밖)로 위험한 딥은 작게, 안전한 딥은 크게.

배경: R111은 위험 딥을 '건너뛰기'만 했고(성장 감소·N 벽 유지), 켈리는 크기를 기대수익/분산에 비례시킨다.
기존 크기 V3 = 변동성 국면 × 예측 비례(pred/10bp) — 기대수익(μ) 쪽만 반영. 여기에 꼬리 위험 쪽을 더한다.
배율 g = clip((p̄_Y / p)^γ, 1/c, c), p = R111 표본 밖 위험(240분 −3%) 확률, p̄_Y = 그해 이전 표본 밖 p 중앙값(인과적), γ ∈ {0.5, 1, 1.5}, c ∈ {2, 3}.
계좌(R118 엔진): 딥 w ∈ {3, 4, 5}, N ∈ {12, 16}, 돌파(pnet15 필터) L ∈ {12, 16}. 선택 2022~23 → 평가 2024~26. 기준 = g 없음.
사용: python research/r118b_dip_kelly_sizing.py"""
import sys, warnings, time, itertools
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
import r118_engine as G, r109_dip_exact_mtm as M
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)

if __name__ == "__main__":
    t_ = time.time()
    Rk = pd.read_parquet(R / "data" / "r111_risk.parquet"); Rk["ts"] = pd.DatetimeIndex(Rk.ts)
    D, _, _, W = M.dip_with_px(); D["side"] = 1
    key = lambda df: df.sym.astype(str) + "|" + df.B.astype(int).astype(str) + "|" + df.k.astype(float).round(2).astype(str) + "|" + df.f.astype(np.int64).astype(str)
    pm = pd.Series(Rk.p_y.values, index=key(Rk)); pm = pm[~pm.index.duplicated()]
    D["p"] = pm.reindex(key(W)).values; yr = pd.DatetimeIndex(W.ts).year.values
    oos = Rk[Rk.p_y.notna()]
    pbar = {Y: oos[oos.ts.dt.year < Y].p_y.median() for Y in np.unique(yr)}
    D["pbar"] = pd.Series(yr).map(pbar).values
    P(f"딥 {len(D):,}, p 결측 {D.p.isna().mean()*100:.1f}%, 연도별 p̄ " + " ".join(f"{k}:{v:.3f}" for k, v in pbar.items()))
    # 위험 분위별 딥 거래당 수익(참고)
    q = pd.qcut(D.p, 5, labels=False)
    P("위험 5분위별 거래당 bp(전체 2022~): " + " ".join(f"{D.ret[q == i].mean()*1e4:+.0f}" for i in range(5)) + " · 최대 역행 중앙 " + " ".join(f"{D.mae[q == i].median()*100:.2f}%" for i in range(5)))
    BK = G.std_breakout()
    rows = []
    for gam, c in [(0, 1)] + [(g, c) for g in (0.5, 1.0, 1.5) for c in (2, 3)]:
        Dg = D.copy()
        if gam > 0: Dg["mult"] = Dg.mult * np.clip((Dg.pbar / Dg.p) ** gam, 1 / c, c).fillna(1.0)
        E = pd.concat([Dg[G.COLS], BK], ignore_index=True)
        for w, N, L in itertools.product((3, 4, 5), (12, 16), (12, 16)):
            d, pk = G.run_multi(E, w, N, {1: L / 48})
            if d is None:
                rows.append(dict(γ=gam, c=c, w=w, N=N, L=L, ok=False, 파산=f"{G.D0 + pd.Timedelta(minutes=int(pk[0])):%Y-%m-%d}")); continue
            rows.append(dict(γ=gam, c=c, w=w, N=N, L=L, ok=True, **G.summary(d)))
        P(f"  γ{gam} c{c} 완료 ({time.time()-t_:.0f}s)")
    O = pd.DataFrame(rows); pd.set_option("display.width", 250); pd.set_option("display.max_rows", 100)
    S = O[O.ok]
    P("\n생존: " + " | ".join(f"γ{k[0]}c{k[1]}: {int(g.ok.sum())}/{len(g)} N≤{g[g.ok].N.max() if g.ok.any() else '-'}" for k, g in O.groupby(["γ", "c"])))
    P("파산: " + ", ".join(f"{k} {v}" for k, v in O[~O.ok].파산.value_counts().items()))
    P("\n===== (γ, c)별 학습 최선 → 평가")
    for k, g in S.groupby(["γ", "c"]):
        b = g.sort_values("학습월", ascending=False).iloc[0]
        P(f"  γ{k[0]} c{k[1]}: w{b.w} N{b.N} L{b.L} 학습 {b.학습월:+.1f}% 샤프 {b.학습샤프:.2f} → 평가 월 {b.평가월:+.1f}% 중앙 {b.평가중앙:+.1f}% ≥100% {b.평가100달}달 MDD {b.평가MDD:.0f}% 샤프 {b.평가샤프:.2f} | {b.연도}")
    P("\n===== 전체 학습 상위 8"); P(S.sort_values("학습월", ascending=False).head(8).drop(columns=["ok", "파산"], errors="ignore").round(2).to_string(index=False))
    O.to_parquet(R / "data" / "r118b.parquet", index=False)
    P(f"완료 ({time.time()-t_:.0f}s)")
