"""4차 스크리닝: 고점 근접, 낙폭 레짐, OBV 추세, 볼린저 돌파, 종가 위치 모멘텀."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import screen as S  # noqa: E402


def f_nearhigh(x):
    P, N = [], []
    for h in [720, 1440, 2160, 4320, 8760]:
        n = x.nb(h)
        hh = x.C.rolling(n, min_periods=max(2, n // 2)).max()
        prox = (x.C / hh).fillna(0).to_numpy()
        for th in [0.8, 0.85, 0.9, 0.95]:
            P.append(np.where(prox > th, 1.0, 0.0)); N.append(f"nearhigh {h}h th={th}")
            P.append(S.hold(prox > th + 0.03, prox < th)); N.append(f"nearhigh-hyst {h}h th={th}")
    return N, P


def f_ddregime(x):
    P, N = [], []
    for h in [336, 720, 1440]:
        n = x.nb(h)
        hh = x.C.rolling(n, min_periods=2).max()
        dd = (x.C / hh - 1).fillna(0).to_numpy()
        for th in [-0.1, -0.15, -0.2, -0.3]:
            P.append(np.where(dd > th, 1.0, 0.0)); N.append(f"dd-regime {h}h > {th}")
    return N, P


def f_obv(x):
    P, N = [], []
    sgn = np.sign(x.C.diff().fillna(0))
    obv = (sgn * x.b.quote_volume).cumsum()
    for h in [168, 336, 720, 1440]:
        n = x.nb(h)
        slope = (obv - obv.shift(n)).fillna(0).to_numpy()
        pr = (x.C / x.C.shift(n) - 1).fillna(0).to_numpy()
        P.append(np.where(slope > 0, 1.0, 0.0)); N.append(f"obv slope>0 {h}h")
        P.append(np.where((slope > 0) & (pr > 0), 1.0, 0.0)); N.append(f"obv&price {h}h")
    return N, P


def f_bollinger(x):
    P, N = [], []
    c = x.C.to_numpy()
    for h in [168, 336, 720]:
        n = x.nb(h)
        ma, sd = x.C.rolling(n).mean(), x.C.rolling(n).std()
        for k in [1.0, 1.5, 2.0]:
            up = (x.C > ma + k * sd).to_numpy(); lo = (x.C < ma).to_numpy()
            P.append(S.hold(up, lo)); N.append(f"bollinger {h}h k={k}")
    return N, P


def f_clv(x):
    P, N = [], []
    rng = (x.b.high - x.b.low).replace(0, np.nan)
    clv = (((x.b.close - x.b.low) - (x.b.high - x.b.close)) / rng).fillna(0)
    for h in [24, 72, 168, 336]:
        n = x.nb(h)
        m = clv.rolling(n).mean().fillna(0).to_numpy()
        for th in [0.0, 0.1, 0.2]:
            P.append(np.where(m > th, 1.0, 0.0)); N.append(f"clv {h}h>{th}")
            P.append(np.where(m > th, 1.0, np.where(m < -th, -1.0, 0.0))); N.append(f"clv-LS {h}h ±{th}")
    return N, P


def main():
    tfs, tag = sys.argv[1].split(","), sys.argv[2]
    out, t0 = [], time.time()
    for fam, fn in (("nearhigh", f_nearhigh), ("ddregime", f_ddregime), ("obv", f_obv), ("bollinger", f_bollinger), ("clv", f_clv)):
        for tf in tfs:
            for sym in S.SYMS:
                try:
                    x = S.Ctx(sym, tf)
                    N, P = fn(x)
                    r = S.evaluate(fam, x, N, P, 2022, 1.5)
                except Exception as e:  # noqa: BLE001
                    import traceback; traceback.print_exc()
                    r = {"family": fam, "sym": sym, "tf": tf, "error": repr(e)}
                if r:
                    out.append(r); print(fam, tf, sym, "OOS Sh", r.get("oos_sharpe"), flush=True)
    Path(f"logs/screen_{tag}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
