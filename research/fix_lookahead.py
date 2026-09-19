"""룩어헤드 수정 전후 비교 — 저변동 조건에 현재 봉 종가가 들어갔던 문제.
사용: python research/fix_lookahead.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R))
sys.path.insert(0, str(R / "research"))
from lib_dip import bars, run, summarize
TE = pd.Timestamp("2024-01-01", tz="UTC")
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)

for sym in ("BTCUSDT", "ETHUSDT"):
    D = bars(sym, 240)
    s = pd.Series
    rp = s((D["bh"] - D["bl"]) / D["bo"]).shift(1)          # 직전 봉 범위 (이미 안전)
    lr = s(np.log(D["bc"])).diff()
    rv_bad = lr.rolling(24).std()                            # ← 현재 봉 종가 포함 (룩어헤드)
    rv_ok = lr.rolling(24).std().shift(1)                    # ← 직전 봉까지만
    P("=" * 92 + f"\n### {sym}")
    for nm, rv in (("① 기존(룩어헤드 있음)", rv_bad), ("② 수정(1봉 지연)", rv_ok)):
        q = rv.rolling(720).quantile(0.3)
        if nm.startswith("②"):
            q = rv.rolling(720).quantile(0.3)                # rv 자체가 이미 지연됨
        cd = (rv < q).fillna(False).values
        ret, mae, xs, tin, hold = run(D, (rp * 1.0).values, cd)
        o = summarize(ret, D["bi"], TE, nm)
        if o and o["tr"] and o["te"]:
            P(f"  {nm:22s} 학습 {o['tr']['mean']:7.2f}bp(승률 {o['tr']['win']:.3f}, n{o['tr']['n']}) | "
              f"테스트 {o['te']['mean']:7.2f}bp(승률 {o['te']['win']:.3f}, n{o['te']['n']}, t{o['te']['t']:.2f}, 주{o['te']['wk']:.2f})")
    # 추가 안전: 2봉 지연
    rv2 = lr.rolling(24).std().shift(2)
    cd2 = (rv2 < rv2.rolling(720).quantile(0.3)).fillna(False).values
    ret, _, _, _, _ = run(D, (rp * 1.0).values, cd2)
    o = summarize(ret, D["bi"], TE, "③ 2봉 지연")
    if o and o["tr"] and o["te"]:
        P(f"  {'③ 2봉 지연(더 보수적)':22s} 학습 {o['tr']['mean']:7.2f}bp | 테스트 {o['te']['mean']:7.2f}bp(t{o['te']['t']:.2f}, n{o['te']['n']})")
    # 연도별 (수정판)
    cd = (rv_ok < rv_ok.rolling(720).quantile(0.3)).fillna(False).values
    ret, mae, xs, tin, hold = run(D, (rp * 1.0).values, cd)
    ok = ~np.isnan(ret); ser = pd.Series(ret[ok], index=D["bi"][ok])
    P("  수정판 연도별: " + " | ".join(f"{y} {v:+.1f}bp({int(c)})" for (y, v), (_, c) in
                                 zip(ser.groupby(ser.index.year).mean().items(), ser.groupby(ser.index.year).count().items())))
