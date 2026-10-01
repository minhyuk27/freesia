"""R125 — 꾸준함 목표 딥 크기 재설계: 기대수익 / 기대분산(켈리)으로 크기를 정하고, 계좌 수준 주간 위험 조절을 더한다.

배경(R124): 꾸준함 기준 기준선 = z0.31 문턱 10bp(w3·N12) + 돌파 L2 — 2025~26 양수 주 67%, 주 중앙 +2.3%, 최악 주 −40%.
현행 크기 V3 = 변동성 국면^1.5 × (예측/10bp), 상한 6 — 기대수익만 반영, 분산은 반영 안 함 → 급락 딥이 크게 들어가 최악 주를 만든다.
(a) 분산 모델: R32와 같은 특징·워크포워드로 목표 ret²(±2% 절단 후 제곱)를 예측 → σ̂² (표본 밖).
    크기 K1 = clip((μ̂/σ̂²) / 학습 연도 이전 중앙값, 0.25, 4), K2 = K1 × 변동성 국면^0.5(완만), K3 = V3 × clip(σ̄²/σ̂², 0.33, 3)(현행에 분산 보정만).
(b) 계좌 주간 위험 조절(줄이기만): 직전 4주 주간 수익 표준편차가 목표 τ를 넘으면 다음 주 노출 × τ/표준편차(하한 0.25). τ = 2022~24 주간 표준편차의 {0.5, 0.75}배.
    — 일간 수익에 배수를 곱하는 근사(노출 축소 쪽이라 청산 위험은 줄어든다). 양수 주 비율은 바뀌지 않고 크기 분포만 바뀐다.
평가: 2022~24 / 2025~26 양수 주·주 중앙·하위 25%·최악 주·주간 샤프·양수 월·월 중앙. 선택은 2022~24 주간 샤프.
사용: python research/r125_dip_variance_sizing.py"""
import sys, warnings, time, itertools
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
import r118_engine as G, r119_dip_model_v2 as V
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
META = {"sym", "f", "x", "ret", "mae", "ts", "yr"}


def wstats(d):
    wk = (1 + d).resample("W-SUN").prod() - 1; mo = (1 + d).resample("ME").prod() - 1; out = {}
    for tag, a, b in (("22_24", "2022", "2024"), ("25_26", "2025", "2026")):
        x = wk[a:b]; m = mo[a:b]
        out.update({f"양수주_{tag}": (x > 0).mean() * 100, f"중앙_{tag}": x.median() * 100, f"하위25_{tag}": x.quantile(.25) * 100,
                    f"최악_{tag}": x.min() * 100, f"주샤프_{tag}": x.mean() / x.std(), f"양수월_{tag}": (m > 0).mean() * 100, f"월중앙_{tag}": m.median() * 100})
    return out, wk


if __name__ == "__main__":
    t_ = time.time()
    E = pd.read_parquet(R / "data" / "r32_events.parquet"); E["ts"] = pd.DatetimeIndex(E.ts); E["yr"] = E.ts.dt.year
    E["isbtc"] = (E.sym == "BTCUSDT").astype(float); E = E.replace([np.inf, -np.inf], np.nan)
    ALL = [c for c in E.columns if c not in META]
    sig = []
    for Y in range(2022, 2027):
        tr = E[E.yr < Y]; te = E[E.yr == Y]; fu = [c for c in ALL if tr[c].notna().sum() > 1000]
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=400, l2_regularization=1.0, random_state=0)
        m.fit(tr[fu].values, np.clip(tr.ret.values, -0.02, 0.02) ** 2)
        sig.append(pd.Series(np.maximum(m.predict(te[fu].values), 1e-8), index=te.index))
        P(f"  분산 모델 {Y} ({time.time()-t_:.0f}s)")
    E["s2"] = pd.concat(sig)
    W = pd.read_parquet(R / "data" / "r32_wf.parquet"); W["ts"] = pd.DatetimeIndex(W.ts)
    W = W.merge(E[["sym", "B", "k", "f", "s2"]], on=["sym", "B", "k", "f"], how="left")
    P(f"분산 예측 결합 결측 {W.s2.isna().mean()*100:.2f}% · 분산 예측과 실현 ret² 순위 상관(2022~): {W[['s2']].assign(r2=W.ret**2).corr(method='spearman').iloc[0,1]:.3f}")
    B = G.std_breakout(); rows = []
    for pc in (0.0005, 0.001):
        D = V.dip_events(W, pc)
        Wp = W[W.pred > pc].sort_values(["f", "sym"]).reset_index(drop=True)          # dip_events와 같은 순서
        mu, s2, yr = Wp.pred.values, Wp.s2.values, Wp.ts.dt.year.values
        kel = mu / s2
        med = {Y: np.nanmedian(kel[yr < Y]) if (yr < Y).any() else np.nanmedian(kel[yr == Y][:200]) for Y in np.unique(yr)}
        kn = kel / pd.Series(yr).map(med).values
        vreg = np.clip(D.mult.values / np.clip(mu / 0.001, 0.5, 3), 0.3, 4) ** (1 / 1.5)        # V3에서 국면 배수 역산
        s2bar = {Y: np.nanmedian(s2[yr < Y]) if (yr < Y).any() else np.nanmedian(s2[yr == Y][:200]) for Y in np.unique(yr)}
        sb = pd.Series(yr).map(s2bar).values
        sizes = {"V3(현행)": D.mult.values, "K1 켈리": np.clip(kn, 0.25, 4), "K2 켈리×국면^0.5": np.clip(kn, 0.25, 4) * np.clip(vreg, 0.3, 4) ** 0.5,
                 "K3 V3×분산보정": D.mult.values * np.clip(sb / s2, 0.33, 3)}
        for sn, mlt in sizes.items():
            Dv = D.assign(mult=np.nan_to_num(mlt, nan=1.0))[G.COLS]
            for w_, N_ in ((2, 8), (3, 12)):
                d, pk = G.run_multi(pd.concat([Dv, B], ignore_index=True), w_, N_, {1: 2 / 48})
                if d is None: rows.append(dict(pc=pc * 1e4, 크기=sn, w=w_, N=N_, 조절="없음", 파산=True)); continue
                st, wk = wstats(d); rows.append(dict(pc=pc * 1e4, 크기=sn, w=w_, N=N_, 조절="없음", **st))
                # (b) 계좌 주간 위험 조절(축소만)
                sd0 = wk["2022":"2024"].std()
                for tf in (0.5, 0.75):
                    tau = tf * sd0; trail = wk.rolling(4).std().shift(1)
                    fct = np.clip(tau / trail, 0.25, 1.0).fillna(1.0)
                    dd = d * fct.reindex(d.index, method="bfill").fillna(1.0)        # 그 주 일간 수익에 그 주 배수(전주까지 정보)
                    st2, _ = wstats(dd); rows.append(dict(pc=pc * 1e4, 크기=sn, w=w_, N=N_, 조절=f"τ{tf}", **st2))
            P(f"  pc{pc*1e4:.0f} {sn} 완료 ({time.time()-t_:.0f}s)")
    O = pd.DataFrame(rows); pd.set_option("display.width", 270); pd.set_option("display.max_rows", 120)
    P(O.round(2).to_string(index=False))
    S = O.dropna(subset=["주샤프_22_24"])
    b = S.sort_values(["주샤프_22_24", "양수주_22_24"], ascending=False).iloc[0]
    P(f"\n2022~24 주간 샤프 최선: pc{b.pc:.0f} {b.크기} w{b.w}N{b.N} 조절 {b.조절} → 2025~26 양수주 {b.양수주_25_26:.0f}% 중앙 {b.중앙_25_26:+.2f}% 하위25 {b.하위25_25_26:+.2f}% "
      f"최악 {b.최악_25_26:+.1f}% 주샤프 {b.주샤프_25_26:.2f} 양수월 {b.양수월_25_26:.0f}% 월중앙 {b.월중앙_25_26:+.1f}%")
    P("기준선(pc10 V3 w3N12 L2, 조절 없음) 2025~26: 양수주 67% 중앙 +2.3% 하위25 −1.5% 최악 −40% 주샤프 0.34 양수월 67% 월중앙 +14%")
    O.to_parquet(R / "data" / "r125.parquet", index=False)
    P(f"완료 ({time.time()-t_:.0f}s)")
