"""R101b — R101 돌파 앙상블의 베타 대조: 같은 24시각 앙상블·같은 추세 필터·같은 청산으로 '세션 시가 즉시 진입'(돌파 조건 없음).
돌파가 대조보다 샤프·거래당 수익이 높아야 진짜 엣지. 사용: python research/r101b_control.py"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
import r100_lw_deep as B, r101_lw_ensemble as E
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
CFG = [("DT", 4, 0.5, 5, 0, "변동성조절"), ("DT", 5, 0.5, 20, 4, "고정"), ("LW", 3, 1.2, 10, 4, "고정"), ("DT", 4, 0.7, 20, 4, "고정")]
acc = {}
for sym in ("BTCUSDT", "ETHUSDT"):
    k = store.load("klines_1m", sym); k["ts"] = pd.DatetimeIndex(k.ts); o, h, l, c = k.open.values, k.high.values, k.low.values, k.close.values
    for a in range(24):
        s0, s1, O, H, L, C, full, sts = B.sessions(k, 24, a); sts = pd.DatetimeIndex(sts); sts = sts.tz_localize("UTC") if sts.tz is None else sts
        dayi = ((sts - E.D0).days).values.astype(np.int64); ok = (dayi >= 0) & (dayi < E.NDAY)
        Ro = pd.Series(H - L).shift(1).values; vt = np.clip(0.02 / (Ro / O), 0, 3); LV = E.lv_sets(O, H, L, C)
        for cfg in CFG:
            fam, N, K, M, xm, sz = cfg; ma = pd.Series(C).rolling(M).mean().shift(1).values; g = O > ma
            for tag, lv in (("돌파", np.where(full, O + LV[(fam, N, K)], np.nan)), ("대조(시가 진입)", np.where(full & ~np.isnan(LV[(fam, N, K)]), O * 1.0, np.nan))):
                r = E.run(s0.astype(np.int64), s1.astype(np.int64), o, h, l, c, lv.astype(float), O - 0.5 * Ro, 1, xm)
                y = np.where(g, r, np.nan) * (vt if sz == "변동성조절" else 1); m = ~np.isnan(y) & ok
                key = (cfg, tag); acc.setdefault(key, [np.zeros(E.NDAY), 0, 0.0])
                acc[key][0] += np.bincount(dayi[m], weights=y[m], minlength=E.NDAY) / 48; acc[key][1] += int(m.sum()); acc[key][2] += float(y[m].sum())
idx = pd.date_range(E.D0, periods=E.NDAY, freq="D")
for cfg in CFG:
    out = []
    for tag in ("돌파", "대조(시가 진입)"):
        d, n, s = acc[(cfg, tag)]
        out.append(f"{tag}: 샤프 학습 {E.sh(d[:E.TRD]):.2f} 평가 {E.sh(d[E.TRD:]):.2f} 거래당 {s/n*1e4:+.1f}bp 하루 {n/E.NDAY:.1f}건 | 연도 " +
                   " ".join(f"{y%100}:{E.sh(d[idx.year==y]):+.1f}" for y in range(2020, 2027)))
    P(f"{cfg}\n   " + "\n   ".join(out))
