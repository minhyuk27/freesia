"""실전 엔진 대조 검증: 실시간 코드(freesia.live)가 연구 코드와 같은 값을 내는가.

A 특징: 저장된 과거 데이터를 실전 코드(minute_features + level_features)에 넣어 R32 이벤트(2026년 표본)의 특징과 비교.
         실전처럼 '최근 N분 창'만 넣어 계산(롤링 창이 잘리는 영향까지 확인).
B 예측: 실전 모델(data/models/z031_live_2026.joblib) 예측 vs R83 워크포워드 예측(data/r83_wf_live.parquet).
C 지표 매핑: REST 5분 지표(최근 30일) vs 저장된 일별 파일 값(같은 시각 비교, 시각 기준 차이 탐지).
D 1분봉: REST vs 저장 파일(최근 하루).
사용: python scripts/live_parity.py"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, joblib
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R))
from freesia import store
from freesia.live import features as LF, feed

WINDOW = 10080 + 300 + 1500                                         # rvreg(10080) + rv(300) + 여유


def main():
    E = pd.read_parquet(R / "data" / "r32_events.parquet"); E["ts"] = pd.DatetimeIndex(E.ts)
    M = joblib.load(R / "data" / "models" / "z031_live_2026.joblib")
    W = pd.read_parquet(R / "data" / "r83_wf_live.parquet"); W["ts"] = pd.DatetimeIndex(W.ts)
    print(f"모델: 학습 ~{M['trained_until']}, {M['n']:,}건, 특징 {len(M['feats'])}")
    for s in ("BTCUSDT", "ETHUSDT"):
        K = store.load("klines_1m", s).set_index("ts"); PR = store.load("premium_1m", s).set_index("ts").close
        MT = store.load("metrics_5m", s).set_index("ts"); FU = store.load("funding", s).set_index("ts").rate
        ev = E[(E.sym == s) & (E.ts >= "2026-06-01") & (E.ts < "2026-09-15")]
        ev = ev.sample(min(300, len(ev)), random_state=0)
        diffs = {}; preds = []
        for _, e in ev.iterrows():
            f = int(e.f); lo = f - WINDOW
            k = K.iloc[lo:f]                                           # 분 f−1 까지(실전과 동일하게 창만)
            t0, t1 = k.index[0] - pd.Timedelta("1D"), k.index[-1]
            F, rv1, piv = LF.minute_features(k, PR[(PR.index >= t0) & (PR.index <= t1)], MT[(MT.index >= t0) & (MT.index <= t1)],
                                             FU[(FU.index >= t0 - pd.Timedelta("3D")) & (FU.index <= t1)])
            B = int(e.B); a = f; s0 = (a // B) * B; j_s0 = s0 - lo; j = len(k) - 1
            T, lv = LF.level_features(a, B, float(e.k), k.open.values[j_s0], rv1[j_s0], k.close.values[j], piv[j], s == "BTCUSDT")
            row = {**{kk: v[j] for kk, v in F.items()}, **lv}
            for c in M["feats"]:
                ref = float(e[c]); val = float(row[c])
                if np.isnan(ref) and np.isnan(val): d = 0.0
                elif np.isnan(ref) or np.isnan(val): d = np.inf
                else: d = abs(val - ref) / (abs(ref) + 1e-6)
                diffs.setdefault(c, []).append(d)
            x = np.array([[row[c] for c in M["feats"]]], dtype=np.float64)
            preds.append((M["model"].predict(x)[0], e.ts, int(e.f), float(e.B), float(e.k)))
        D = pd.DataFrame({c: pd.Series(v) for c, v in diffs.items()})
        bad = D.quantile(0.99).sort_values(ascending=False)
        print(f"\n[A {s}] 이벤트 {len(ev)}개, 특징별 상대오차 99% 분위 상위(float32 저장 오차 ~1e-6 수준이면 정상):")
        print("  " + " ".join(f"{c}:{v:.1e}" for c, v in bad.head(10).items()))
        print(f"  상대오차 > 1e-3 인 값 비율: {(D > 1e-3).mean().mean():.4f}")
        Wt = W[W.sym == s][["f", "B", "k", "pred"]].astype({"B": float, "k": float})
        Pd = pd.DataFrame(preds, columns=["p", "ts", "f", "B", "k"]).merge(Wt, on=["f", "B", "k"], how="left")
        print(f"[B {s}] 예측 대조(같은 체결 분 기준 근사): 상관 {np.corrcoef(Pd.p, Pd.pred.fillna(Pd.p))[0,1]:.4f}, "
              f"|차| 중앙 {np.nanmedian(np.abs(Pd.p - Pd.pred))*1e4:.3f}bp")
        # C 지표
        api = feed.metrics_5m(s, 500); st = MT[MT.index >= api.index.min()]
        for c in api.columns:
            j0 = api[c].reindex(st.index).dropna(); j1 = st[c].reindex(j0.index)
            shift = {}
            for sh in (-2, -1, 0, 1, 2):
                a_ = api[c].shift(sh).reindex(st.index).dropna(); b_ = st[c].reindex(a_.index)
                shift[sh] = (np.abs(a_ - b_) / b_.abs()).median() if len(a_) else np.nan
            print(f"[C {s}] {c}: 겹침 {len(j0)}개, 시각 이동별 상대오차 중앙 " + " ".join(f"{k_:+d}:{v:.1e}" for k_, v in shift.items()))
        # D 1분봉
        kk = feed.klines_1m(s, 600); ks = K.reindex(kk.index).dropna()
        dd = (kk.loc[ks.index, ["close", "quote_volume", "taker_buy_quote_volume", "trades"]] - ks[["close", "quote_volume", "taker_buy_quote_volume", "trades"]]).abs().max()
        print(f"[D {s}] REST 1분봉 vs 저장 최대 차: " + " ".join(f"{c}:{v:.3g}" for c, v in dd.items()) + f" (겹침 {len(ks)}분)")


if __name__ == "__main__":
    main()
