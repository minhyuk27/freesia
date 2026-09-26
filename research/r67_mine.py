"""R67 — 분기 단위 무작위 학습/테스트 분할로 차트 선 패턴을 찾고 검증한다 (R66 거래 기록 사용).

분할: 2020Q1~2026Q3 27개 분기를 무작위로 학습 16 / 테스트 11, 20회 반복(시드 0~19). 분기 마지막 2일의 거래는 제외(경계 누수 방지).
패턴 ① 규칙 탐색: 기본 칸 = (격자, 선, 규칙, 역할). 조건 = 없음 또는 특징 하나의 학습 3분위 구간
         (volp trend mom20 vrel tbr lslope dist, 시간대 3구간, 주말 여부). 약 2,300칸 × 26조건 ≈ 6만 개 후보.
         학습에서 n≥150, 평균 순수익>0 중 t값 상위 20개 선택 → 테스트에서 개별·합산(중복 거래 1회) 성과.
패턴 ② 모델: HistGB 회귀(특징 + 격자·선·규칙·역할 범주)로 순수익 예측, 학습 예측값 상위 5% 문턱 이상만 테스트에서 거래.
기준선: 같은 절차를 위약 선 거래에 그대로 적용 → '고르는 과정'만으로 나오는 성과.
판정: 20개 분할에서 테스트 순수익 평균·흑자 분할 수, 실제 vs 위약.
사용: python research/r67_mine.py"""
import sys, warnings, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
CONT = ["volp", "trend", "mom20", "vrel", "tbr", "lslope", "dist"]
NSPLIT, NTR, TOPK, MINN = 20, 16, 20, 150


def load():
    A = pd.concat([pd.read_parquet(R / "data" / f"r67_trades_{tf}.parquet") for tf in ("15min", "1h", "4h")], ignore_index=True)
    A["tf"] = A.tf.astype("category")
    t = A.t.dt.tz_convert("UTC")
    A["q"] = (t.dt.year * 4 + (t.dt.month - 1) // 3).astype(np.int32)
    qend = (t.dt.to_period("Q").dt.end_time.dt.tz_localize("UTC"))
    A = A[(qend - t) > pd.Timedelta(days=2)].reset_index(drop=True)
    A["cell"] = (A.tf.astype(str) + "|" + A.s.astype(str) + "|" + A.rule.astype(str) + "|" + A.ori.astype(str)).astype("category").cat.codes.astype(np.int32)
    A["sess"] = np.select([A.hour < 8, A.hour < 16], [0, 1], 2).astype(np.int8)
    A["wknd"] = (A.wday >= 5).astype(np.int8)
    A["day"] = t.dt.floor("D")
    return A


def mine(D, tr, te, test_days):
    """D: 한 종류(실제 또는 위약) 거래. tr/te: 불리언 마스크. 반환: 선택 패턴 표, 합산 테스트 성과."""
    n = D.n.values.astype(np.float64); cell = D.cell.values.astype(np.int64)
    conds = [("없음", np.zeros(len(D), np.int64))]
    for c in CONT:
        x = D[c].values; qs = np.nanquantile(x[tr], [1 / 3, 2 / 3])
        conds.append((c, np.where(np.isnan(x), -1, np.digitize(x, qs)).astype(np.int64)))
    conds.append(("sess", D.sess.values.astype(np.int64))); conds.append(("wknd", D.wknd.values.astype(np.int64)))
    cand = []
    for ci, (cname, b) in enumerate(conds):
        key = cell * 100 + ci * 5 + (b + 1)
        s = pd.DataFrame({"k": key[tr], "n": n[tr]}).groupby("k").n.agg(["count", "mean", "std"])
        s = s[s.index % 5 != 0]                                       # 특징 결측 구간은 패턴으로 쓰지 않음
        s = s[(s["count"] >= MINN) & (s["mean"] > 0)]
        s["t"] = s["mean"] / (s["std"] / np.sqrt(s["count"]))
        s["cname"] = cname; cand.append(s)
    C = pd.concat(cand).sort_values("t", ascending=False).head(TOPK)
    sel = np.zeros(len(D), bool); rows = []
    keys_all = {ci: cell * 100 + ci * 5 + (b + 1) for ci, (_, b) in enumerate(conds)}
    cmap = {c: i for i, (c, _) in enumerate(conds)}
    for k, r in C.iterrows():
        m = (keys_all[cmap[r.cname]] == k) & te
        sel |= m
        rows.append((k, r.cname, r["count"], r["mean"] * 1e4, r.t, m.sum(), n[m].mean() * 1e4 if m.any() else np.nan))
    S = pd.DataFrame(rows, columns=["key", "cond", "n_tr", "tr_bp", "t", "n_te", "te_bp"])
    S["cell"] = S.key // 100; S["bin"] = S.key % 5 - 1
    pooled = (n[sel].mean() * 1e4 if sel.any() else np.nan, sel.sum() / test_days, (S.te_bp > 0).mean())
    return S, pooled


def model(D, tr, te, test_days, seed):
    X = D[CONT + ["sess", "wknd"]].copy()
    for c in ("tf", "s", "rule"): X[c] = D[c].cat.codes
    X["ori"] = (D.ori.values > 0).astype(int)
    y = np.clip(D.n.values.astype(np.float64), -0.03, 0.03)
    rng = np.random.default_rng(seed); itr = np.where(tr)[0]
    if len(itr) > 1_500_000: itr = rng.choice(itr, 1_500_000, replace=False)
    catm = np.array([c in ("tf", "s", "rule", "ori", "sess", "wknd") for c in X.columns])
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=500,
                                      l2_regularization=1.0, categorical_features=catm, random_state=seed)
    m.fit(X.values[itr], y[itr])
    ptr = m.predict(X.values[itr]); thr = np.quantile(ptr, 0.95)
    pte = m.predict(X.values[te]); s = pte >= thr
    nte = D.n.values[te]
    return nte[s].mean() * 1e4 if s.any() else np.nan, s.sum() / test_days


if __name__ == "__main__":
    t_ = time.time(); A = load()
    Q = np.sort(A.q.unique()); P(f"거래 {len(A):,}, 분기 {len(Q)}개, 기본 칸 {A.cell.nunique():,} ({time.time()-t_:.0f}s)")
    res = []; picks = []
    for sp in range(NSPLIT):
        rng = np.random.default_rng(sp); trq = set(rng.choice(Q, NTR, replace=False))
        line = [f"분할{sp:2d} 테스트분기 " + ",".join(f"{q//4%100}Q{q%4+1}" for q in sorted(set(Q) - trq))]
        for plc in (False, True):
            D = A[A.plc == plc].reset_index(drop=True)
            tr = D.q.isin(trq).values; te = ~tr
            test_days = D.day[te].nunique()
            S, (pb, pday, frac) = mine(D, tr, te, test_days)
            mb, mday = model(D, tr, te, test_days, sp) if sp < 10 else (np.nan, np.nan)
            res.append((sp, plc, S.tr_bp.mean(), pb, pday, frac, mb, mday))
            if not plc: picks.append(S.assign(split=sp))
            line.append(f"{'위약' if plc else '실제'}: 규칙 학습 {S.tr_bp.mean():+6.1f} → 테스트 {pb:+6.1f}bp (하루 {pday:5.1f}, 흑자 패턴 {frac:.0%}) | 모델 테스트 {mb:+6.1f}bp (하루 {mday:5.1f})")
        P(" | ".join(line) + f" ({time.time()-t_:.0f}s)")
    Rz = pd.DataFrame(res, columns=["split", "plc", "tr_bp", "te_bp", "te_day", "frac_pos", "ml_bp", "ml_day"])
    P("\n===== 요약 (20개 분할, 모델은 10개)")
    for plc in (False, True):
        x = Rz[Rz.plc == plc]
        P(f"  {'위약' if plc else '실제'}: 규칙 학습 평균 {x.tr_bp.mean():+.1f}bp → 테스트 평균 {x.te_bp.mean():+.1f}bp (분할별 {x.te_bp.min():+.1f}~{x.te_bp.max():+.1f}, "
          f"흑자 분할 {(x.te_bp > 0).sum()}/{len(x)}), 하루 {x.te_day.mean():.1f}건, 선택 패턴 중 테스트 흑자 {x.frac_pos.mean():.0%} | "
          f"모델 테스트 평균 {x.ml_bp.mean():+.1f}bp (흑자 분할 {(x.ml_bp > 0).sum()}/{x.ml_bp.notna().sum()}), 하루 {x.ml_day.mean():.1f}건")
    Pk = pd.concat(picks)
    cells = A.drop_duplicates("cell").set_index("cell")[["tf", "s", "rule", "ori"]]
    Pk = Pk.join(cells, on="cell")
    top = Pk.groupby(["tf", "s", "rule", "ori", "cond", "bin"]).agg(선택횟수=("split", "size"), 학습bp=("tr_bp", "mean"), 테스트bp=("te_bp", "mean"),
                                                                  테스트n=("n_te", "mean")).sort_values("선택횟수", ascending=False).head(25)
    P("\n  자주 선택된 패턴 상위 25 (실제 선, 20개 분할 중 선택 횟수)")
    pd.set_option("display.width", 250); P(top.round(1).to_string())
    Rz.to_csv(R / "logs" / "r67_splits.csv", index=False); Pk.to_csv(R / "logs" / "r67_picks.csv", index=False)
    x, y = Rz[~Rz.plc], Rz[Rz.plc]
    (R / "logs" / "r67.summary").write_text(f"R67 분기 무작위 분할 20회: 규칙 테스트 실제 {x.te_bp.mean():+.1f}bp / 위약 {y.te_bp.mean():+.1f}bp, "
                                            f"모델 실제 {x.ml_bp.mean():+.1f} / 위약 {y.ml_bp.mean():+.1f}\n")
