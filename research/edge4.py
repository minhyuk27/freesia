"""테이커 매수 우위 필터 정밀 검증 (BTC·ETH). 창·임계값 격자의 고원, 연도별, 플라시보, 4시간 버전."""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from freesia.bt import core  # noqa: E402
import screen as S  # noqa: E402
import edge2 as E2  # noqa: E402


def sh(a, m):
    a = a[m]
    return round(float(a.mean() / a.std() * np.sqrt(365)), 2) if a.std() > 0 else 0.0


for sym in S.SYMS:
    x, F = E2.build(sym)
    b = x.b; C, fund, idx = x.C.to_numpy(), x.fund, x.idx
    yrs = idx.year.to_numpy(); e = F["ens"].to_numpy()
    raw = (b.taker_buy_quote_volume / b.quote_volume.replace(0, np.nan))
    Rb = core.run(e, C, fund)
    print("=====", sym, "기준 추세 2021+ %.2f 2022+ %.2f | 매수후보유 2021+ %.2f" % (sh(Rb, yrs >= 2021), sh(Rb, yrs >= 2022), sh(x.r.to_numpy(), yrs >= 2021)))
    print("창(일) × 임계값 → 필터 적용 추세 Sharpe(2021+ / 2022+), 통과율")
    grid = {}
    for w in (1, 3, 5, 10, 20):
        row = []
        for th in (0.48, 0.49, 0.50, 0.51, 0.52):
            v = (raw.rolling(w).mean().shift(0) > th).fillna(False).to_numpy().astype(float)   # 봉 마감 후 값: 다음 봉에 적용(core.run이 처리)
            R = core.run(e * v, C, fund)
            grid[(w, th)] = (sh(R, yrs >= 2021), sh(R, yrs >= 2022), float(v[yrs >= 2021].mean()))
            row.append(f"{grid[(w,th)][0]:5}/{grid[(w,th)][1]:5}({grid[(w,th)][2]:.2f})")
        print(f"  {w:2}일:", " | ".join(row))
    better = np.mean([g[0] > sh(Rb, yrs >= 2021) and g[1] > sh(Rb, yrs >= 2022) for g in grid.values()])
    print("  격자 25개 중 기준 추세보다 두 구간 모두 나은 비율: %.2f" % better)
    # 연도별 (창3, 0.50)
    v = (raw.rolling(3).mean() > 0.50).fillna(False).to_numpy().astype(float)
    R = core.run(e * v, C, fund)
    print("  연도별 Sharpe 필터 / 기준:", {int(Y): (sh(R, yrs == Y), sh(Rb, yrs == Y)) for Y in sorted(set(yrs)) if Y >= 2020})
    # 플라시보: 필터를 무작위 회전
    rng = np.random.default_rng(5); real = sh(R, yrs >= 2021); null = []
    for _ in range(400):
        k = int(rng.integers(60, len(v) - 60)); null.append(sh(core.run(e * np.roll(v, k), C, fund), yrs >= 2021))
    print("  플라시보(2021+): 실제 %.2f, 무작위 회전 중앙 %.2f, 95%% %.2f, p=%.3f" % (real, np.median(null), np.percentile(null, 95), (np.array(null) >= real).mean()))
    # 필터 단독(롱플랫)과 상관: 추세 신호와 겹치는 부분?
    print("  필터 통과일 평균 다음날 수익률(bp): %.1f, 불통과: %.1f, 전체: %.1f" % (
        1e4 * pd.Series(x.r.to_numpy()[1:])[v[:-1] > 0].mean(), 1e4 * pd.Series(x.r.to_numpy()[1:])[v[:-1] == 0].mean(), 1e4 * x.r.mean()))
