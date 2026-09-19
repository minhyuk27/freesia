"""결합 H의 BTC+ETH 혼합, 그리고 '추세 다리 배율 + 꺼진 구간 캐리' 스위칭 구조(일 단위, 청산 반영)."""
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
    res = {}
    # (1) H 혼합 (4시간봉)
    Hs, yrs = [], None
    for s in S.SYMS:
        x = S2.CarryCtx(s, "4h"); x.r_price = x.C.pct_change().fillna(0).to_numpy()
        e = CB.trend_exposure(x, "ALL", None)
        H, T, C = CB.series(x, e)
        Hs.append(pd.Series(H, index=x.idx))
        yrs = x.idx.year.to_numpy() if yrs is None else yrs
    h = pd.concat(Hs, axis=1).dropna().mean(axis=1).to_numpy()
    y = pd.concat(Hs, axis=1).dropna().index.year.to_numpy()
    ppy = 6 * 365
    res["H_BTC+ETH_4h"] = {"전체": CB.stats(h, ppy, "4h"), "2022+": CB.stats(h[y >= 2022], ppy, "4h"),
                          "연도별Sharpe": {int(Y): round(CB.sh(h[y == Y], ppy), 2) for Y in sorted(set(y))},
                          "연도별수익%": {int(Y): round(float(h[y == Y].sum() * 100), 1) for Y in sorted(set(y))}}
    # (2) 스위칭: 추세 켜짐 → 무기한 롱 L배(격리, 청산 반영) / 꺼짐 → 캐리(무레버리지, 자본 1.2)
    frames = {s: LV.daily_frame(s) for s in S.SYMS}
    out2 = {}
    for L in (1, 2, 3):
        legs = []
        for s in S.SYMS:
            x = S2.CarryCtx(s, "1d")
            d = frames[s]
            e = pd.Series(CB.trend_exposure(x, "ALL", None), index=x.idx).reindex(d.index).fillna(0).to_numpy()
            tr, liq = LV.simulate(d, e, L, 0.7)
            c = core.run_ret(np.ones(len(x.idx)), x.ret, np.zeros(len(x.idx)), cost=0.0017 / 1.2); c[0] = 0
            c = pd.Series(c, index=x.idx).reindex(d.index).fillna(0).to_numpy()
            e_prev = np.concatenate([[0.0], e[:-1]])
            legs.append(tr + (1 - np.minimum(e_prev * 1.0, 1)) * c)     # 노출 안 쓴 몫은 캐리
        a = np.mean(legs, axis=0)
        yy = frames["BTCUSDT"].index.year.to_numpy()
        out2[f"L={L}"] = {"2021+": LV.stat(a[yy >= 2021]), "2022+": LV.stat(a[yy >= 2022])}
    res["스위칭(추세 L배 + 꺼짐 캐리)"] = out2
    Path("logs/hlev.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))
    print(json.dumps(res, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
