"""R84 — z0.31 확장으로 샤프 올리기(월 100% 목표: 샤프 2.85 → 4+ 필요). ① 봉·깊이 격자 확장.

근거: 샤프는 독립 거래 수의 제곱근에 비례. 현재 봉 5종 × 깊이 5종(하루 약 1.3 사건). 자본은 대부분 놀고 있다.
격자: 봉 B ∈ {1,2,3,5,10,15,30,60,120,240}분 × 깊이 k ∈ {2,2.5,3,4,5,6,8,10,12} (기존 = B 5~60 × k 3~8).
이벤트·특징: R27·R32와 같은 규칙(지정가 2bp 관통, 봉 종가 청산, 체결 직전 분 특징), 실시간 동등 50개(bookDepth 제외).
모델: 같은 HistGB, 연도별 확장창 워크포워드(평가 2022~2026), 전 격자로 한 모델(B·k가 특징).
평가: 예측 문턱 15·25·35bp × 부분집합(기존 격자 / 전체 / 신규만) → 거래당, 연간 거래, 일간 연 샤프(V3·w1N4),
      R40 엔진으로 명목 w ∈ {1,2,2.5,3,3.5,4}(N = 4w) 최대 성장(파산 전) 월 복리·MDD, 2024~26.
사용: python research/r84_expand.py [build|walk|eval]"""
import sys, warnings, time, gc
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
BARS = (1, 2, 3, 5, 10, 15, 30, 60, 120, 240); KS = (2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0)
OLD_B = (5, 10, 15, 30, 60); OLD_K = (3.0, 4.0, 5.0, 6.0, 8.0)
EV = R / "data" / "r84_events.parquet"; WF = R / "data" / "r84_wf.parquet"


def build():
    from r26_hf_twosided import run, EPS, FM, FT
    import r27_dip_events as R27, r32_dip_events as R32
    rows = []; t_ = time.time()
    for s in ("BTCUSDT", "ETHUSDT"):
        d, F = R27.feats(s); X, piv = R32.extra(s, d); F.update(X)
        for c in ("bslope", "bimb"): F.pop(c, None)
        for B in BARS:
            for kx in KS:
                a, b, r, m = run(d["o"], d["h"], d["l"], d["c"], d["rv1"], B, 1, kx, 0, 1.0, 0, EPS, FM, FT)
                if len(a) == 0: continue
                fm1 = a - 1; s0 = (a // B) * B; x = kx * d["rv1"][s0] * np.sqrt(B); T = d["o"][s0] * (1 - x)
                df = pd.DataFrame({"sym": s, "B": np.float32(B), "k": np.float32(kx), "f": a, "x": b, "ret": r.astype(np.float32), "mae": m.astype(np.float32),
                                   "ts": d["idx"][a], "el": ((a % B) / B).astype(np.float32), "pre": ((d["c"][fm1] / d["o"][s0] - 1) / x).astype(np.float32),
                                   "tpiv": (np.log(T / piv[fm1]) / x).astype(np.float32), "qh": (a % 15).astype(np.float32), "hr0": (a % 60).astype(np.float32)})
                for kk, v in F.items(): df[kk] = v[fm1].astype(np.float32)
                rows.append(df)
            P(f"  {s} B{B} 누적 이벤트 {sum(len(x) for x in rows):,} ({time.time()-t_:.0f}s)")
        del d, F, X; gc.collect()
    E = pd.concat(rows, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    E.to_parquet(EV, index=False); P(f"저장 {EV} {E.shape}")


def walk():
    from sklearn.ensemble import HistGradientBoostingRegressor
    E = pd.read_parquet(EV); E["ts"] = pd.DatetimeIndex(E.ts); E["yr"] = E.ts.dt.year; E["isbtc"] = (E.sym == "BTCUSDT").astype(np.float32)
    feats = [c for c in E.columns if c not in {"sym", "f", "x", "ret", "mae", "ts", "yr"}]
    out = []
    for Y in range(2022, 2027):
        tr = E[E.yr < Y]; te = E[E.yr == Y]
        if len(tr) > 3_000_000: tr = tr.sample(3_000_000, random_state=0)
        fu = [c for c in feats if tr[c].notna().sum() > 1000]
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=400, l2_regularization=1.0,
                                          random_state=0).fit(tr[fu].values, np.clip(tr.ret.values, -0.02, 0.02))
        out.append(te[["sym", "B", "k", "f", "x", "ret", "mae", "ts", "yr"]].assign(pred=m.predict(te[fu].values).astype(np.float32)))
        P(f"  {Y}: 학습 {len(tr):,} → 평가 {len(te):,} (특징 {len(fu)})")
    pd.concat(out).to_parquet(WF, index=False); P(f"저장 {WF}")


def evaluate():
    import r40_regime_leverage as RL, r80_max_return as R80
    from freesia import store
    RL.MMR = 0.0065
    W = pd.read_parquet(WF); W["ts"] = pd.DatetimeIndex(W.ts); W["ret"] = W.ret - 1.2e-4
    for s in ("BTCUSDT", "ETHUSDT"):
        k = store.load("klines_1m", s).set_index("ts").close.resample("D").last(); r = np.log(k).diff()
        v30 = r.rolling(30).std(); reg = (v30 / v30.rolling(365, min_periods=120).median()).shift(1)
        m = W.sym == s; W.loc[m, "vreg"] = reg.reindex(W.loc[m, "ts"].dt.normalize()).values
    W["vreg"] = W.vreg.fillna(1.0)
    old = W.B.isin(OLD_B) & W.k.isin(OLD_K)
    P(f"워크포워드 이벤트 {len(W):,} (기존 격자 {old.sum():,})")
    P("\n  격자별 거래당 순익(예측>25bp, bp) — 행 B, 열 k")
    S = W[W.pred > 0.0025]
    P((S.groupby(["B", "k"]).ret.mean().unstack() * 1e4).round(1).to_string())
    P((S.groupby(["B", "k"]).size().unstack() / 4.75).round(0).to_string(header=False))
    for th in (0.0015, 0.0025, 0.0035):
        for sub, mk in (("기존 격자", old), ("전체 격자", pd.Series(True, index=W.index)), ("신규만", ~old)):
            S = W[mk & (W.pred > th)].sort_values(["f", "sym"]).reset_index(drop=True)
            if len(S) < 100: continue
            mult = np.minimum(np.clip(S.vreg.values ** 1.5, 0.3, 4.0) * np.clip(S.pred.values / 0.001, 0.5, 3.0), 6.0)
            base = RL.sim(S, 1, 4, mult, X=0.03, H=720)
            sh = np.nan
            if base is not None:
                d = R80.daily_from_eq(base[0]); sh = d.mean() / d.std() * np.sqrt(365)
            best = None
            for w in (1, 2, 2.5, 3, 3.5, 4):
                res = RL.sim(S, w, 4 * w, mult, X=0.03, H=720)
                if res is None: break
                d = R80.daily_from_eq(res[0]); m_ = R80.metrics(d); mt = R80.metrics(d, "2024-01-01", "2027-01-01")
                if best is None or m_["mo"] > best[1]["mo"]: best = (w, m_, mt)
            line = f"  문턱 {th*1e4:.0f}bp {sub:6s}: 거래 {len(S)/4.75:7.0f}/년 거래당 {S.ret.mean()*1e4:+6.2f}bp 승률 {(S.ret>0).mean():.3f} | 샤프(w1N4) {sh:.2f}"
            if best: line += f" | 최대 w{best[0]}N{4*best[0]:.0f}: 월 {best[1]['mo']:+.1f}% (중앙 {best[1]['med']:+.1f}%) MDD {best[1]['mdd']:.0f}% 월≥100% {best[1]['big']} | 24~26 월 {best[2]['mo']:+.1f}% MDD {best[2]['mdd']:.0f}%"
            P(line)
    (R / "logs" / "r84.summary").write_text("R84 z0.31 격자 확장 완료. 세부 logs/r84.log\n")


if __name__ == "__main__":
    {"build": build, "walk": walk, "eval": evaluate}[sys.argv[1] if len(sys.argv) > 1 else "eval"]()
