"""R55 — 5분 격자 방향 모델: 후보를 3배로 늘려 확신 상위 선택의 빈도를 올린다.

특징: R27 차트·거래량·파생 + R32 오더플로우·호가·청산 개념 (t−1 분까지) + 다중 척도 수익·7일 위치 (R50과 동일 정의)
라벨: 12시간 안에 ±1.5%·±1% 먼저 도달 (R48과 동일 규칙)
모델: 결판 표본 학습 분류기, 시드 3개 평균, 연도별 확장창 워크포워드
선택: R50 방식(양 꼬리 각각 직전 30일 상·하위 q, 인과적), 12시간 겹침 제거 → R47 30배 계좌
사용: python research/r55_grid5.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
from r27_dip_events import feats
from r32_dip_events import extra
from r47_x30_frame import outcome, simulate_entries, stat, CT, CM
from r50_direction_model import select
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
STEP = 5; H = 720


def build():
    rows = []
    for s in ("BTCUSDT", "ETHUSDT"):
        d, F = feats(s); X, _ = extra(s, d); F.update(X)
        k = store.load("klines_1m", s)
        o, h, l, c = (k[x].values for x in ("open", "high", "low", "close"))
        idx = np.arange(10080 + 5, len(c) - H - 1, STEP)
        df = pd.DataFrame({"sym": s, "t": idx, "ts": k.ts.values[idx]})
        for kk, v in F.items():
            df[kk] = v[idx - 1].astype(np.float32)
        lc = np.log(c)
        rv = pd.Series(lc).diff().rolling(1440).std().values
        lo7 = pd.Series(l).rolling(10080).min().values; hi7 = pd.Series(h).rolling(10080).max().values
        t1 = idx - 1
        for w in (60, 240, 720, 1440):
            df[f"r{w}"] = (lc[t1] - lc[t1 - w]).astype(np.float32)
        df["pos7d"] = ((c[t1] - lo7[t1]) / (hi7[t1] - lo7[t1])).astype(np.float32)
        df["r1440n"] = (df.r1440.values / (rv[t1] * np.sqrt(1440))).astype(np.float32)
        df["r240n"] = (df.r240.values / (rv[t1] * np.sqrt(240))).astype(np.float32)
        for dd in (0.01, 0.015):
            _, kind, _ = outcome(o, h, l, c, h, l, idx.astype(np.int64), 1, dd, 0.99, H, CT, CM)
            df[f"y{int(dd*1000)}"] = np.where(kind == 1, 1, np.where(kind == -1, -1, 0)).astype(np.int8)
        rows.append(df)
        P(f"{s} 5분 격자 {len(df):,}")
    D = pd.concat(rows, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    D["ts"] = pd.to_datetime(D.ts, utc=True); D["yr"] = D.ts.dt.year
    D["isbtc"] = (D.sym == "BTCUSDT").astype(np.float32)
    return D.sort_values(["t", "sym"]).reset_index(drop=True)


if __name__ == "__main__":
    D = build()
    D.to_parquet(R / "data" / "r55_grid5.parquet", index=False)
    FEAT = [c for c in D.columns if c not in {"sym", "t", "ts", "yr", "y10", "y15"}]
    P(f"시점 {len(D):,}, 특징 {len(FEAT)}")
    for y, dd in (("y15", 0.015), ("y10", 0.01)):
        out = []
        for Y in range(2022, 2027):
            tr = D[(D.yr < Y) & (D[y] != 0)]; te = D[D.yr == Y].copy()
            fu = [c for c in FEAT if tr[c].notna().sum() > 15000]
            ps = []
            for sd in (0, 1, 2):
                m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.03, max_leaf_nodes=31,
                                                   min_samples_leaf=3000, l2_regularization=1.0, random_state=sd)
                m.fit(tr[fu].values, (tr[y].values == 1).astype(int))
                ps.append(m.predict_proba(te[fu].values)[:, 1])
            te["p"] = np.mean(ps, axis=0); out.append(te)
            P(f"  {y} {Y} 완료")
        W = pd.concat(out).sort_values(["t", "sym"]).reset_index(drop=True)
        W[["sym", "t", "ts", "yr", y, "p"]].to_parquet(R / "data" / f"r55_wf_{y}.parquet", index=False)
        wk = (W.ts.max() - W.ts.min()).days / 7
        P(f"\n===== 5분 격자 ±{dd*100:.1f}% / 12시간 (시드 3개 평균)")
        for q in (0.003, 0.001, 0.0005, 0.0003):
            S = select(W, q); dec = S[S[y] != 0]
            acc = (dec[y] == dec.side).mean() if len(dec) else np.nan
            by = dec.groupby("yr").apply(lambda g: (g[y] == g.side).mean())
            eq, T = simulate_entries(S[["ts", "sym", "side"]], dd, 720)
            cg, mo, mmed, mdd, fin = stat(eq)
            P(f"  q={q*100:.2f}%: 주 {len(S)/wk:4.1f}회 결판 적중 {acc:.3f} (n={len(dec)}) 무결판 {(S[y]==0).mean():.2f} | "
              f"계좌 월 {mo*100:+6.1f}% MDD {mdd*100:4.0f}% | " + " ".join(f"{yy%100}:{v:.2f}" for yy, v in by.items()))
