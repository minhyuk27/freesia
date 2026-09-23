"""z0.20 신호·가격 캐시 — 이후 스크립트가 무거운 임포트 체인을 안 타게 한다(결과 불변)."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from z0_20_sim import SIG, D, SYM, STEP, MAXH, MMR
out = {}
for s in SYM:
    d = D[s]
    lr = pd.Series(np.log(d["c"])).diff()
    out[f"{s}_o"] = d["o"].astype(np.float64); out[f"{s}_h"] = d["h"].astype(np.float64)
    out[f"{s}_l"] = d["l"].astype(np.float64); out[f"{s}_c"] = d["c"].astype(np.float64)
    out[f"{s}_sig"] = SIG[s].astype(np.float64)
    out[f"{s}_rv"] = lr.rolling(1440).std().shift(1).values.astype(np.float64)
    out[f"{s}_idx"] = d["idx"].asi8
np.savez_compressed(R/"data"/"z2_cache.npz", STEP=STEP, MAXH=MAXH, MMR=MMR,
                    SYM=np.array(SYM), **out)
print("saved", (R/"data"/"z2_cache.npz").stat().st_size/1e6, "MB", flush=True)
