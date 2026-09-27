"""z0.31 실전 모델 학습·저장. 연구 R83(실시간 동등 50특징)과 같은 모델·설정.

운용 규칙(z0.31 정의): 매년 1월 1일, 그 전 해까지의 전체 이벤트로 재학습. 지금(2026)은 2020~2025로 학습 = 워크포워드 2026 모델.
저장: data/models/z031_live_<평가연도>.joblib  {model, feats, trained_until, n}
사용: python scripts/z031_train_live.py [평가연도=2026]"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import HistGradientBoostingRegressor
R = Path(__file__).resolve().parents[1]

Y = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
E = pd.read_parquet(R / "data" / "r32_events.parquet")
E["ts"] = pd.DatetimeIndex(E.ts); E["yr"] = E.ts.dt.year; E["isbtc"] = (E.sym == "BTCUSDT").astype(float)
E = E.replace([np.inf, -np.inf], np.nan)
ALL = [c for c in E.columns if c not in {"sym", "f", "x", "ret", "mae", "ts", "yr"}]
LIVE = [c for c in ALL if c not in ("bslope", "bimb")]
tr = E[E.yr < Y]
fu = [c for c in LIVE if tr[c].notna().sum() > 1000]
m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=400,
                                  l2_regularization=1.0, random_state=0).fit(tr[fu].values, np.clip(tr.ret.values, -0.02, 0.02))
out = R / "data" / "models"; out.mkdir(parents=True, exist_ok=True)
joblib.dump({"model": m, "feats": fu, "trained_until": f"{Y-1}-12-31", "n": len(tr)}, out / f"z031_live_{Y}.joblib")
print(f"저장 {out / f'z031_live_{Y}.joblib'} — 학습 {len(tr):,}건, 특징 {len(fu)}개")
