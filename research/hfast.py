"""결합 H를 빠른 신호로: 거래(=헤지 비율 25% 이상 변경) 횟수 vs 성과. 룩백 세트를 바꿔 빈도를 조절. BTC·ETH만."""
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

SETS = {"초고속(6~48h)": [6, 12, 24, 48], "고속(12~120h)": [12, 24, 48, 72, 120], "중속(24~336h)": [24, 48, 72, 120, 168, 336],
        "기준(24~2160h)": CB.HS, "저속(168~2160h)": [168, 336, 720, 1080, 1440, 2160]}
tf = sys.argv[1]
rows = {}
for sym in S.SYMS:
    x = S2.CarryCtx(sym, tf); x.r_price = x.C.pct_change().fillna(0).to_numpy()
    yrs = x.idx.year.to_numpy(); ppy = x.ppy
    for nm, hs in SETS.items():
        sg = np.mean([np.sign((x.C / x.C.shift(x.nb(h)) - 1).fillna(0).to_numpy()) for h in hs], axis=0)
        for kind in ("연속", "이진"):
            e = np.clip(sg, 0, 1) if kind == "연속" else (sg > 0.0).astype(float)
            ev = float((np.abs(np.diff(e, prepend=0)) >= 0.25).sum() / (len(e) / ppy))
            H = CB.series(x, e)[0]
            rows.setdefault((nm, kind), []).append((sym, ev, pd.Series(H, index=x.idx)))
print(f"{tf}: 룩백 세트 / 신호 형태 → 거래(연/종목), BTC+ETH 혼합 H: 전체 Sharpe·연·MDD | 2022+ Sharpe·연 | 연도별 수익%")
for (nm, kind), l in rows.items():
    h = pd.concat([s for _, _, s in l], axis=1).dropna().mean(axis=1); y = h.index.year.to_numpy()
    a, b = CB.stats(h.to_numpy(), ppy, tf), CB.stats(h.to_numpy()[y >= 2022], ppy, tf)
    ev = np.mean([e for _, e, _ in l]); wk = ev / 52
    yr = {int(Y): round(float(h[y == Y].sum() * 100), 0) for Y in sorted(set(y))}
    print(f"{nm:16}{kind:3} 거래 {ev:6.0f}/년 (주 {wk:4.1f}회) | {a['sharpe']:5} {a['ann']:6.3f} {a['mdd']:7.3f} | {b['sharpe']:5} {b['ann']:6.3f} | {yr}")
