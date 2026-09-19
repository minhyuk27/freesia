"""직전 봉 위꼬리 조건 심화 — 룩어헤드 없음. 정의·강도·봉크기·깊이 격자 + 개별 위약.
사용: python research/wick_study.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from lib_dip import bars
from grid_clean import run2
TE = pd.Timestamp("2024-01-01", tz="UTC")
rng = np.random.default_rng(223)
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)

for sym in ("ETHUSDT", "BTCUSDT"):
    P("=" * 104 + f"\n### {sym}")
    rows = []
    for B, bn in ((240, "4h"), (480, "8h"), (720, "12h"), (1440, "1d")):
        D = bars(sym, B)
        s = pd.Series
        bo, bc, bh, bl = D["bo"], D["bc"], D["bh"], D["bl"]
        rng_ = s((bh - bl) / bo)
        uw = s((bh - np.maximum(bo, bc)) / bo)              # 위꼬리 절대폭
        uwr = s((bh - np.maximum(bo, bc)) / (bh - bl))       # 위꼬리 비율
        body = s(np.abs(bc - bo) / bo)
        uwb = uw / body.replace(0, np.nan)                   # 위꼬리/몸통
        L = lambda x: x.shift(1)
        W = {
            "위꼬리비율>0.4": (L(uwr) > 0.4).fillna(False).values,
            "위꼬리비율>0.5": (L(uwr) > 0.5).fillna(False).values,
            "위꼬리비율>0.6": (L(uwr) > 0.6).fillna(False).values,
            "위꼬리 상위30%": (L(uw) > L(uw).rolling(168).quantile(.7)).fillna(False).values,
            "위꼬리 상위15%": (L(uw) > L(uw).rolling(168).quantile(.85)).fillna(False).values,
            "위꼬리/몸통>2": (L(uwb) > 2).fillna(False).values,
            "위꼬리>범위30%+음봉": ((L(uwr) > 0.3) & (L(s((bc - bo))) < 0)).fillna(False).values,
            "위꼬리>범위30%+양봉": ((L(uwr) > 0.3) & (L(s((bc - bo))) > 0)).fillna(False).values,
            "(대조)전체": np.ones(D["n"], bool),
            "(대조)아래꼬리 상위30%": (L(s((np.minimum(bo, bc) - bl) / bo)) >
                                 L(s((np.minimum(bo, bc) - bl) / bo)).rolling(168).quantile(.7)).fillna(False).values,
        }
        for xn, xa in (("직전범위0.75", (L(rng_) * 0.75).values), ("직전범위1.0", (L(rng_) * 1.0).values),
                       ("직전범위1.5", (L(rng_) * 1.5).values), ("고정0.5%", np.full(D["n"], 0.005))):
            for wn, cd in W.items():
                ret = run2(D["o"], D["h"], D["l"], D["c"], B, np.asarray(xa, float), np.asarray(cd, np.bool_), 1, 7.0)
                ok = ~np.isnan(ret); ts = D["bi"][ok]
                if ok.sum() < 100:
                    continue
                tr = ts < TE; te = ~tr
                if tr.sum() < 50 or te.sum() < 40:
                    continue
                a = ret[ok][tr]; b_ = ret[ok][te]
                # 개별 위약: 같은 조건 개수를 무작위 봉에 배치
                pl = []
                k_ = int(cd.sum())
                for _ in range(60):
                    cr = np.zeros(D["n"], bool)
                    cr[rng.choice(D["n"], k_, replace=False)] = True
                    r2 = run2(D["o"], D["h"], D["l"], D["c"], B, np.asarray(xa, float), cr, 1, 7.0)
                    m2 = ~np.isnan(r2) & (D["bi"] >= TE)
                    if m2.sum() > 30:
                        pl.append(r2[m2].mean())
                pct = float((np.array(pl) < b_.mean()).mean() * 100) if pl else np.nan
                rows.append((bn, xn, wn, a.mean(), int(tr.sum()), b_.mean(), float((b_ > 0).mean()),
                             int(te.sum()), b_.mean() / b_.std() * np.sqrt(len(b_)) if b_.std() > 0 else 0,
                             te.sum() / max((ts[te][-1] - ts[te][0]).days / 7, 1), pct,
                             np.percentile(pl, 95) if pl else np.nan))
    d = pd.DataFrame(rows, columns=["봉", "깊이", "위꼬리조건", "학습bp", "학습n", "테스트bp", "테스트승률",
                                    "테스트n", "t", "주당", "위약백분위", "위약95%"])
    P(f"총 {len(d)}종")
    P("\n[테스트 상위 18 — 위약 백분위 포함]")
    P(d.sort_values("테스트bp", ascending=False).head(18).round(2).to_string(index=False))
    ok = d[(d.학습bp > 0) & (d.테스트bp > 0) & (d.위약백분위 >= 95)]
    P(f"\n학습·테스트 양수 + 위약 95퍼센타일 이상: {len(ok)} / {len(d)}")
    if len(ok):
        P(ok.sort_values("테스트bp", ascending=False).round(2).to_string(index=False))
    P("\n[조건별 평균 — 대조군과 비교]")
    P(d.groupby("위꼬리조건")[["학습bp", "테스트bp", "위약백분위"]].mean().round(2).sort_values("테스트bp", ascending=False).to_string())
