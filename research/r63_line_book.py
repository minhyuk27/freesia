"""R63 — 선 접근 순간의 호가·체결 정보로 반등 vs 돌파를 가를 수 있는가 (R62 사건, 2023~).

접근 순간 τ: 접근봉 안에서 1분봉이 처음 선 띠에 닿은 분(지지: 저가 ≤ 선+δ, 저항: 고가 ≥ 선−δ).
특징은 τ 이전 분까지만 사용(선에 지정가를 미리 걸어두는 매매를 가정). 방향은 '선 쪽으로 미는 쪽'이 +가 되게 맞춘다.
  체결(agg1s → 1분): flow5/flow30 = 선 쪽 테이커 순체결 비율, big30 = 10만$ 이상 대량 체결의 선 쪽 순비율,
        vrel5 = 5분 거래대금 / 직전 24시간 5분 중앙값, absorb5 = log(5분 거래대금 / |5분 수익|) − 24시간 중앙값,
        speed15 = 15분 동안 선 쪽 이동 / 선 시간대 ATR
  호가(bookDepth 30초 → 1분 마지막 값): wall1 = log(선 쪽 1% 잔량 / 반대쪽 1%), slope_s/slope_o = log(5%/1%) 선 쪽/반대쪽,
        dwall5/dwall30 = 선 쪽 1% 잔량의 5·30분 변화(벽 추가 +, 철수 −)
  포지션(OI 5분, 5분 지연 적용): oi60 = 1시간 OI 변화율
  선 맥락: nprev(이전 반등 수), age(생성 후 봉 수), slope, tf, kind
판정 ① 단변량: 특징 5분위별 반등률, 학습 2023~24 / 평가 2025~26 (같은 방향이면 신뢰)
     ② 모델: HistGB(시드 5 평균) 학습 → 평가 AUC, 상·하위 10% 반등률, 라벨 셔플 위약
     ③ 위약 선 사건에도 같은 특징을 계산해 '선 특유'인지 확인
사용: python research/r63_line_book.py"""
import sys, warnings, glob
from pathlib import Path
import numpy as np, pandas as pd, numba
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R))
from freesia import store
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
T0 = pd.Timestamp("2023-01-01", tz="UTC"); SPLIT = pd.Timestamp("2025-01-01", tz="UTC")


def minute_table(sym):
    k = store.load("klines_1m", sym).set_index("ts")
    k = k[k.index >= T0 - pd.Timedelta(days=2)]
    parts = []
    for f in sorted(glob.glob(str(store.PQ / "agg1s" / sym / "*.parquet"))):
        a = pd.read_parquet(f, columns=["ts", "bq", "sq", "lbq", "lsq"]).set_index("ts")
        parts.append(a.resample("1min").sum())
    A = pd.concat(parts); A = A[~A.index.duplicated()]
    M = k[["open", "high", "low", "close"]].join(A, how="left").fillna({"bq": 0, "sq": 0, "lbq": 0, "lsq": 0})
    b = store.load("bookdepth", sym).set_index("ts")[["n-1", "n+1", "n-5", "n+5"]]
    M = M.join(b.resample("1min").last().ffill(limit=10), how="left")
    o = store.load("metrics_5m", sym).set_index("ts")[["sum_open_interest_value"]]
    o.index = o.index + pd.Timedelta(minutes=5)                 # 5분 지연(보수)
    M = M.join(o.resample("1min").last().ffill(limit=15), how="left")
    v = M.bq + M.sq; lc = np.log(M.close)
    M["v5"] = v.rolling(5).sum(); M["v30"] = v.rolling(30).sum()
    M["f5"] = (M.sq - M.bq).rolling(5).sum() / M.v5.clip(lower=1)          # 매도 우위 + (지지 쪽으로 미는 흐름)
    M["f30"] = (M.sq - M.bq).rolling(30).sum() / M.v30.clip(lower=1)
    M["g30"] = (M.lsq - M.lbq).rolling(30).sum() / M.v30.clip(lower=1)
    M["vrel5"] = np.log(M.v5.clip(lower=1) / M.v5.rolling(1440).median().clip(lower=1))
    ab = np.log(M.v5.clip(lower=1) / (np.abs(lc - lc.shift(5)) + 1e-4))
    M["absorb5"] = ab - ab.rolling(1440).median()
    M["r15"] = lc - lc.shift(15)
    M["wall1"] = np.log(M["n-1"] / M["n+1"])                     # 매수벽 우위 (지지 기준)
    M["sl_b"] = np.log(M["n-5"] / M["n-1"]); M["sl_a"] = np.log(M["n+5"] / M["n+1"])
    M["dwb5"] = np.log(M["n-1"] / M["n-1"].shift(5)); M["dwb30"] = np.log(M["n-1"] / M["n-1"].shift(30))
    M["dwa5"] = np.log(M["n+1"] / M["n+1"].shift(5)); M["dwa30"] = np.log(M["n+1"] / M["n+1"].shift(30))
    M["oi60"] = np.log(M.sum_open_interest_value / M.sum_open_interest_value.shift(60))
    return M


@numba.njit(cache=True)
def touch_idx(st, en, lo, hi, thr, sup):
    out = np.full(len(st), -1)
    for e in range(len(st)):
        for i in range(st[e], en[e]):
            if (sup[e] and lo[i] <= thr[e]) or ((not sup[e]) and hi[i] >= thr[e]):
                out[e] = i; break
    return out


FEAT = ["flow5", "flow30", "big30", "vrel5", "absorb5", "speed15", "wall1", "slope_s", "slope_o", "dwall5", "dwall30", "oi60"]
CTX = ["nprev", "age", "slope", "tfc", "kindc"]

if __name__ == "__main__":
    E = pd.read_parquet(R / "data" / "r62_events.parquet")
    E = E[(E.t >= T0) & (E.res != 0)].reset_index(drop=True)
    TFL = {"15min": 15, "1h": 60, "4h": 240}
    out = []
    for sym in ("BTCUSDT", "ETHUSDT"):
        M = minute_table(sym); ix = M.index
        P(f"{sym} 1분 표 {M.shape}, 호가 커버 {M.wall1.notna().mean():.2f}")
        X = E[E.sym == sym].copy()
        st = ix.searchsorted(pd.DatetimeIndex(X.t)); en = st + X.tf.map(TFL).values
        sup = (X.side == "지지").values
        thr = np.where(sup, np.exp(X.lvl + X.d), np.exp(X.lvl - X.d))
        ti = touch_idx(st, np.minimum(en, len(ix)), M.low.values, M.high.values, thr, sup)
        X = X[ti > 0].copy(); ti = ti[ti > 0]; sup = (X.side == "지지").values; sg = np.where(sup, 1, -1)
        r = M.iloc[ti - 1]                                        # τ 직전 분까지
        X["flow5"] = sg * r.f5.values; X["flow30"] = sg * r.f30.values; X["big30"] = sg * r.g30.values
        X["vrel5"] = r.vrel5.values; X["absorb5"] = r.absorb5.values
        X["speed15"] = -sg * r.r15.values / X.atr.values
        X["wall1"] = sg * r.wall1.values
        X["slope_s"] = np.where(sup, r.sl_b.values, r.sl_a.values); X["slope_o"] = np.where(sup, r.sl_a.values, r.sl_b.values)
        X["dwall5"] = np.where(sup, r.dwb5.values, r.dwa5.values); X["dwall30"] = np.where(sup, r.dwb30.values, r.dwa30.values)
        X["oi60"] = r.oi60.values
        X["tau"] = ix[ti]
        out.append(X); del M
    D = pd.concat(out, ignore_index=True)
    D["y"] = (D.res == 1).astype(int); D["age"] = D.j - D.i0
    D["tfc"] = D.tf.map({"15min": 0, "1h": 1, "4h": 2}); D["kindc"] = (D.kind == "T").astype(int)
    D.to_parquet(R / "data" / "r63_events_feat.parquet", index=False)
    P(f"사건 {len(D):,} (실제 {(~D.plc).sum():,} / 위약 {D.plc.sum():,}), 특징 결측률: " +
      " ".join(f"{c}:{D[c].isna().mean():.2f}" for c in FEAT))

    P("\n===== ① 단변량: 5분위 반등률 (Q1 → Q5), 학습 23~24 | 평가 25~26 | 위약선 평가")
    for c in FEAT + ["nprev", "age"]:
        seg = []
        for nm, S in (("학습", D[~D.plc & (D.t < SPLIT)]), ("평가", D[~D.plc & (D.t >= SPLIT)]), ("위약", D[D.plc & (D.t >= SPLIT)])):
            s = S[[c, "y"]].dropna()
            if len(s) < 500 or s[c].nunique() < 5: seg.append(f"{nm} -"); continue
            q = pd.qcut(s[c].rank(method="first"), 5, labels=False)
            m = s.groupby(q).y.mean()
            seg.append(f"{nm} " + " ".join(f"{v:.3f}" for v in m) + f" (Δ{(m.iloc[-1]-m.iloc[0])*100:+.1f})")
        P(f"  {c:8s} " + " | ".join(seg))

    P("\n===== ② 모델 (학습 2023~24 → 평가 2025~26, 시드 5 평균)")
    def fit_eval(S, feats, shuffle=False, tag=""):
        tr = S[S.t < SPLIT]; te = S[S.t >= SPLIT]
        yl = tr.y.values.copy()
        if shuffle: yl = np.random.default_rng(1).permutation(yl)
        p = np.mean([HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=300,
                                                     l2_regularization=1.0, random_state=sd).fit(tr[feats].values, yl)
                     .predict_proba(te[feats].values)[:, 1] for sd in range(5)], axis=0)
        auc = roc_auc_score(te.y, p)
        hi = te.y[p >= np.quantile(p, 0.9)].mean(); lo = te.y[p <= np.quantile(p, 0.1)].mean()
        P(f"  {tag:28s} AUC {auc:.3f} | 상위10% 반등 {hi:.3f} 하위10% {lo:.3f} (기저 {te.y.mean():.3f}, n={len(te):,})")
        return p
    real = D[~D.plc]
    fit_eval(real, CTX, tag="실제선: 맥락만")
    fit_eval(real, CTX + FEAT, tag="실제선: 맥락+호가·체결")
    fit_eval(real, CTX + FEAT, shuffle=True, tag="실제선: 라벨 셔플 위약")
    fit_eval(D[D.plc], CTX + FEAT, tag="위약선: 맥락+호가·체결")
    for tf in ("15min", "1h", "4h"):
        fit_eval(real[real.tf == tf], CTX + FEAT, tag=f"실제선 {tf}")
    (R / "logs" / "r63.summary").write_text("R63 선 × 호가·체결 완료. 세부 logs/r63.log\n")
