"""R98 — 강건한 단일 방향 신호의 단순 투표 + 변동성 예보 결합 (ML 과적합 회피).

신호 선택(선택 편향 차단): R97 특징 119개 각각을 연도별 확장창으로 방향·문턱을 정해, **2022~23 평가 결과만으로** 고른다
  (고변동 예보 pr ≥ 0.8 표본의 부호 4시간 수익 t ≥ 1.5, 두 해 모두 양수, 가족당 최대 2개). 2024~26은 건드리지 않는다.
점수: 각 신호의 학습 분포 백분위 u(0~1) → 방향 부호 × (u − 0.5) × 2, 선택 신호 평균 = S ∈ [−1, 1]. (극단만 쓰는 변형 V: 상·하위 10%면 ±1 투표 합)
매매: pr ≥ th 이고 |S| ≥ s(학습 구간 |S| 분위) 이면 sign(S) 방향 다음 분 시가 진입, 청산 E1 ±2% / E3 4시간 / E5 ±1.5%·4h.
      종목별 한 포지션, 비용 테이커 7bp×2(목표는 지정가 2bp). 2022~23 최선 → 2024~26 검증, 3단 레버리지. 비교: 변동성 예보 없이(th=0).
사용: python research/r98_vote.py"""
import sys, warnings, time
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
import r97_direction as Q
from freesia import store
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)

if __name__ == "__main__":
    t_ = time.time(); D, OLD, NEW = Q.frame(); FE = OLD + NEW
    U = {}; SG = {}; rows = []
    for c in FE:
        u = np.full(len(D), np.nan); sgs = {}
        for Y in range(2022, 2027):
            tr = D.loc[D.yr < Y, [c, "fr"]].dropna(); m = (D.yr == Y).values
            if len(tr) < 5000: continue
            srt = np.sort(tr[c].values); x = D.loc[m, c].values
            u[m] = np.where(np.isnan(x), np.nan, np.searchsorted(srt, x) / len(srt))
            lo, hi = np.quantile(srt, [0.1, 0.9]); sgs[Y] = np.sign(tr.fr[tr[c] >= hi].mean() - tr.fr[tr[c] <= lo].mean())
        if len(sgs) < 5: continue
        sg = np.zeros(len(D))
        for Y, v in sgs.items(): sg[(D.yr == Y).values] = v
        U[c] = u; SG[c] = sg
        ext = np.where(u >= 0.9, 1, np.where(u <= 0.1, -1, 0)) * sg
        x = D.assign(pred=ext); x = x[(x.pred != 0) & (x.pr >= 0.8) & (x.yr <= 2023)].dropna(subset=["fr"])
        if len(x) < 200: continue
        s = x.pred * x.fr; by = s.groupby(x.yr).mean()
        rows.append(dict(f=c, fam=Q.fam(c), n=len(x), bp=s.mean() * 1e4, t=s.mean() / s.std() * np.sqrt(len(x) / 16), y22=by.get(2022, np.nan) * 1e4, y23=by.get(2023, np.nan) * 1e4))
    O = pd.DataFrame(rows).sort_values("t", ascending=False)
    ok = O[(O.t >= 1.5) & (O.y22 > 0) & (O.y23 > 0)]
    sel = ok.groupby("fam").head(2)
    P(f"신호 후보 {len(O)}개 → 2022~23 기준 통과 {len(ok)} → 가족당 2개 {len(sel)} ({time.time()-t_:.0f}s)"); P(sel.round(2).to_string())
    SEL = sel.f.tolist()
    Sc = np.nanmean(np.vstack([SG[c] * (U[c] - 0.5) * 2 for c in SEL]), axis=0)
    Vt = np.nansum(np.vstack([SG[c] * np.where(U[c] >= 0.9, 1, np.where(U[c] <= 0.1, -1, 0)) for c in SEL]), axis=0)
    D["S"] = Sc; D["V"] = Vt
    for nm in ("S", "V"):
        x = D[(D.side != 0) & D[nm].notna() & (D[nm] != 0)]
        P(f"  점수 {nm}: 큰 움직임 결판 적중 전체 {(np.sign(x[nm]) == x.side).mean():.3f} | pr≥0.8 {(np.sign(x[x.pr>=.8][nm]) == x[x.pr>=.8].side).mean():.3f} | "
          f"학습(22~23) {(np.sign(x[x.yr<=2023][nm]) == x[x.yr<=2023].side).mean():.3f} 평가 {(np.sign(x[x.yr>=2024][nm]) == x[x.yr>=2024].side).mean():.3f}")
    K = {s: store.load("klines_1m", s) for s in Q.SYMS}; rows = []
    for nm in ("S", "V"):
        A = D[D[nm].notna()]; aq = A[A.yr <= 2023][nm].abs()
        levels = [(q, aq.quantile(q)) for q in (0.5, 0.75, 0.9, 0.97)] if nm == "S" else [(v, v - 0.5) for v in (1, 2, 3)]
        for th in (0.0, 0.5, 0.8, 0.95):
            for lab, lv in levels:
                for ex, (tp, sl) in (("E1 ±2%", (0.02, 0.02)), ("E3 4h", (0, 0)), ("E5 ±1.5%", (0.015, 0.015))):
                    parts = []
                    for s in Q.SYMS:
                        g = A[(A.sym == s) & (A.pr >= th) & (A[nm].abs() >= lv)].sort_values("t"); k = K[s]
                        r = Q.trade(g.t.values.astype(np.int64), np.sign(g[nm].values).astype(np.int64), k.open.values, k.high.values, k.low.values, k.close.values, tp, sl, 240)
                        parts.append(pd.DataFrame({"ts": g.ts.values, "yr": g.yr.values, "r": r}))
                    T = pd.concat(parts).dropna(); T["ts"] = Q.utc(T.ts); a = T[T.yr <= 2023]; b = T[T.yr >= 2024]
                    dd = lambda x, s0, s1: x.groupby(x.ts.dt.floor("D")).r.sum().reindex(pd.date_range(s0, s1, freq="D", tz="UTC")).fillna(0)
                    da, db = dd(a, "2022-01-01", "2023-12-31"), dd(b, "2024-01-01", "2026-09-19")
                    rows.append(dict(sc=nm, th=th, lv=lab, ex=ex, na=len(a) / 730, bpa=a.r.mean() * 1e4, sha=da.mean() / da.std() * np.sqrt(365),
                                     nb=len(b) / 993, bpb=b.r.mean() * 1e4, shb=db.mean() / db.std() * np.sqrt(365), wb=(b.r > 0).mean(),
                                     yb=" ".join(f"{y%100}:{v*1e4:+.0f}" for y, v in T.groupby("yr").r.mean().items()), T=T))
    O = pd.DataFrame([{k: v for k, v in r.items() if k != "T"} for r in rows]); pd.set_option("display.width", 260); pd.set_option("display.max_rows", 200)
    P("\n===== 변동성 예보 문턱별 평균 (모든 점수·강도·청산): 거래당 bp 학습/평가")
    P(O.groupby("th")[["bpa", "bpb", "sha", "shb", "na"]].mean().round(2).to_string())
    P("\n===== 학습 샤프 상위 15 → 평가"); P(O.sort_values("sha", ascending=False).head(15).round(2).to_string())
    best = rows[int(O.sha.idxmax())]; T = best["T"]
    P(f"\n학습 최선 {best['sc'], best['th'], best['lv'], best['ex']} 평가 3단:")
    d = T[T.yr >= 2024]; d = d.groupby(d.ts.dt.floor("D")).r.sum().reindex(pd.date_range("2024-01-01", "2026-09-19", freq="D", tz="UTC")).fillna(0)
    for L in (1, 3, 10):
        eq = (1 + (d * L).clip(lower=-0.99)).cumprod(); yrs = len(d) / 365.25
        P(f"  ×{L:2d}: 월 {(eq.iloc[-1]**(1/12/yrs)-1)*100:+.2f}% MDD {(eq/eq.cummax()-1).min()*100:.0f}%")
    D[["sym", "t", "ts", "yr", "S", "V", "pr", "side", "fr"]].to_parquet(R / "data" / "r98_scores.parquet", index=False)
    (R / "logs" / "r98.summary").write_text("done\n")
