"""1분봉 기본 통계: 시가 대비 종가 방향별로 봉 안에서 반대쪽으로 얼마나 갔는가(MAE), 그리고 이동 크기 분포.
사용: python research/barstats.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R))
from freesia import store
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
pd.set_option("display.width", 200)

for sym in ("BTCUSDT", "ETHUSDT"):
    k = store.load("klines_1m", sym).set_index("ts")
    o, h, l, c = k.open.values, k.high.values, k.low.values, k.close.values
    n = len(k)
    up = c > o; dn = c < o; fl = c == o
    P("=" * 96 + f"\n### {sym}  1분봉 {n:,}개  ({k.index[0].date()} ~ {k.index[-1].date()})")
    P(f"\n[1] 봉 유형  상승 {up.mean():.4%} ({up.sum():,}) | 하락 {dn.mean():.4%} ({dn.sum():,}) | 보합 {fl.mean():.4%} ({fl.sum():,})")

    # [2] 방향별 반대쪽 이탈 비율
    dip = (l < o)          # 시가 아래로 내려간 적 있음
    pop = (h > o)          # 시가 위로 올라간 적 있음
    P("\n[2] 봉 안에서 시가 반대편을 건드린 비율")
    P(f"  상승봉({up.sum():,}개) 중 시가 아래로 내려간 적 있음 : {dip[up].mean():.4%}")
    P(f"  하락봉({dn.sum():,}개) 중 시가 위로 올라간 적 있음   : {pop[dn].mean():.4%}")
    P(f"  보합봉({fl.sum():,}개) 중 아래로 {dip[fl].mean():.4%} / 위로 {pop[fl].mean():.4%} / 양쪽 {(dip & pop)[fl].mean():.4%} / 완전정지 {(~dip & ~pop)[fl].mean():.4%}")
    P(f"  전체 중 위아래 모두 건드림 : {(dip & pop).mean():.4%}")

    # [3] 역행 폭(MAE) 분포
    mae_up = (o - l) / o * 1e4      # 상승봉에서 시가 대비 최대 하락(bp)
    mae_dn = (h - o) / o * 1e4      # 하락봉에서 시가 대비 최대 상승(bp)
    P("\n[3] 역행 폭(MAE) 분포 — bp 단위, 시가 기준")
    rows = []
    for nm, x in (("상승봉의 하락 역행", mae_up[up]), ("하락봉의 상승 역행", mae_dn[dn])):
        rows.append((nm, x.mean(), np.median(x), *[np.quantile(x, q) for q in (.25, .75, .9, .99, .999)], x.max()))
    P(pd.DataFrame(rows, columns=["구분", "평균", "중앙", "25%", "75%", "90%", "99%", "99.9%", "최대"]).round(2).to_string(index=False))
    P("  역행이 특정 폭을 넘은 비율(진입 후 손절·청산 판단용)")
    rows = []
    for x in (1, 2, 5, 10, 20, 60, 100):
        rows.append((x, float((mae_up[up] >= x).mean()), float((mae_dn[dn] >= x).mean())))
    P(pd.DataFrame(rows, columns=["역행 bp", "상승봉 중 비율", "하락봉 중 비율"]).round(4).to_string(index=False))

    # [4] 이동 크기
    body = (c - o) / o * 1e4
    rng_ = (h - l) / o * 1e4
    uw = (h - np.maximum(o, c)) / o * 1e4
    lw = (np.minimum(o, c) - l) / o * 1e4
    P("\n[4] 1분봉 이동 크기 (bp)")
    rows = []
    for nm, x in (("몸통 |종가-시가|", np.abs(body)), ("전체 범위 고-저", rng_), ("위꼬리", uw), ("아래꼬리", lw)):
        rows.append((nm, x.mean(), np.median(x), *[np.quantile(x, q) for q in (.75, .9, .99, .999)], x.max()))
    P(pd.DataFrame(rows, columns=["구분", "평균", "중앙", "75%", "90%", "99%", "99.9%", "최대"]).round(2).to_string(index=False))
    P(f"  몸통/범위 비율 평균 {np.nanmean(np.abs(body) / np.where(rng_ == 0, np.nan, rng_)):.4f} (1에 가까울수록 일방향 봉)")
    P(f"  종가가 고가와 같음 {float((c == h).mean()):.4%} | 종가가 저가와 같음 {float((c == l).mean()):.4%}")

    # [5] 연도별
    P("\n[5] 연도별 추이")
    df = pd.DataFrame({"up": up, "dn": dn, "fl": fl, "dip_up": np.where(up, dip, np.nan),
                       "pop_dn": np.where(dn, pop, np.nan), "body": np.abs(body), "rng": rng_,
                       "mae_up": np.where(up, mae_up, np.nan), "mae_dn": np.where(dn, mae_dn, np.nan)}, index=k.index)
    g = df.groupby(df.index.year).agg(상승=("up", "mean"), 하락=("dn", "mean"), 보합=("fl", "mean"),
                                      상승봉_하락터치=("dip_up", "mean"), 하락봉_상승터치=("pop_dn", "mean"),
                                      몸통bp=("body", "mean"), 범위bp=("rng", "mean"),
                                      상승봉역행bp=("mae_up", "mean"), 하락봉역행bp=("mae_dn", "mean"))
    P(g.round(4).to_string())

    # [6] 0.1% 관점: 시가 진입 후 봉 안에서 ±10bp 도달
    P("\n[6] 시가 진입 후 그 1분 안에 ±10bp(0.1%) 도달 비율")
    up10 = (h - o) / o >= 0.001; dn10 = (o - l) / o >= 0.001
    P(f"  위로 10bp 도달 {up10.mean():.4%} | 아래로 10bp 도달 {dn10.mean():.4%} | 양쪽 모두 {(up10 & dn10).mean():.4%} | 한쪽만 {(up10 ^ dn10).mean():.4%} | 둘 다 아님 {(~up10 & ~dn10).mean():.4%}")
    onlyup = up10 & ~dn10; onlydn = dn10 & ~up10
    P(f"  한쪽만 도달한 경우 중 위쪽 {onlyup.sum() / (onlyup.sum() + onlydn.sum()):.4%}")
