"""캐리(저변동 엣지) + 추세(배율 조합) 포트폴리오. 일 단위, 2021~.
자본 배분: a = 캐리 비중(현물+무기한 숏, 무레버리지), 1-a = 추세 레그(BTC/ETH 50:50, 배율 L, 격리·청산 반영)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from freesia.bt import core  # noqa: E402
import screen as S  # noqa: E402
import screen2 as S2  # noqa: E402
import combo as CB  # noqa: E402
import lev as LV  # noqa: E402


def main():
    frames = {s: LV.daily_frame(s) for s in S.SYMS}
    exps, carry = {}, {}
    for s in S.SYMS:
        x = S2.CarryCtx(s, "1d")
        idx = frames[s].index
        exps[s] = pd.Series(CB.trend_exposure(x, "ALL", None), index=x.idx).reindex(idx).fillna(0).to_numpy()
        c = core.run_ret(np.ones(len(x.idx)), x.ret, np.zeros(len(x.idx)), cost=0.0017 / 1.2)
        c[0] = 0
        carry[s] = pd.Series(c, index=x.idx).reindex(idx).fillna(0).to_numpy()
    idx = frames["BTCUSDT"].index
    carry_p = np.mean([carry[s] for s in S.SYMS], axis=0)          # BTC/ETH 캐리 50:50
    yrs = idx.year.to_numpy()
    res = {}
    for L in [1, 2, 3, 4]:
        legs = [LV.simulate(frames[s], exps[s], L, 0.7)[0] for s in S.SYMS]
        trend = np.mean(legs, axis=0)
        for a in [0.0, 0.25, 0.5, 0.75, 1.0]:
            pr = a * carry_p + (1 - a) * trend
            for nm, m in (("2021+", yrs >= 2021), ("2022+", yrs >= 2022)):
                res[f"L={L} a={a} {nm}"] = LV.stat(pr[m])
    # 추세 내부 헤지 결합 H (현물+조건부 헤지) 병기
    out = {"grid": res}
    Path("logs/portfolio.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("캐리 비중 a / 추세 배율 L → CAGR, MDD, Sharpe  (2021~ | 2022~)")
    for L in [1, 2, 3, 4]:
        for a in [0.0, 0.25, 0.5, 0.75, 1.0]:
            r1, r2 = res[f"L={L} a={a} 2021+"], res[f"L={L} a={a} 2022+"]
            print(f"L={L} a={a:4}: {r1['cagr']:6.3f} {r1['mdd']:7.3f} Sh{r1['sharpe']:5.2f} | {r2['cagr']:6.3f} {r2['mdd']:7.3f} Sh{r2['sharpe']:5.2f}")


if __name__ == "__main__":
    main()
