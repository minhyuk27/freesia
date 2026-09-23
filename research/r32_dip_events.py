"""R32-1 — R27 딥 이벤트에 전문가·실전 개념 특징을 추가한다 ([[freesia 고빈도 실전 조사 (R31)]]).

추가 특징(전부 f-1 종료까지의 정보):
  CVD·흡수  : cvd5/cvd30 (순공격 비율), cvddiv (가격은 저점권인데 최근 순매도가 약해짐), absorb (거래량 대비 가격 무반응)
  스윕      : piv (직전 30~240분 스윙 저점 대비 현재가), swept (최근 30분에 그 저점을 깼는가), tpiv (지정가가 스윙 저점 아래 얼마나)
  VPIN 근사 : vpin15/vpin60 = Σ|매수−매도| / Σ거래량 (1분 단위 대량 분류)
  캐스케이드: accel (최근 5분 매도량 / 직전 30분 평균), oi15 (15분 OI 변화)
  OI 사분면 : quad = sign(ret60)×2 + sign(oi60) 부호 조합
  프리미엄  : dprem5/dprem30 (선물 프리미엄 변화 — 선물 주도 매도인지)
  분기점    : qh (체결 분의 15분 내 위상), hr0 (정시까지 거리)
  호가(2023~): bslope (−5% 매수벽 / −1% 매수벽), bimb (±1% 불균형) — 이전은 결측
출력: data/r32_events.parquet
사용: python research/r32_dip_events.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
from r26_hf_twosided import run, EPS, FM, FT
from r27_dip_events import feats, BARS, KS
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
SYM = ("BTCUSDT", "ETHUSDT")


def extra(s, d):
    k = store.load("klines_1m", s).set_index("ts")
    idx = k.index
    qv = pd.Series(k.quote_volume.values); tb = pd.Series(k.taker_buy_quote_volume.values)
    c = pd.Series(k.close.values); l = pd.Series(k.low.values)
    net = 2 * tb - qv                                     # 매수공격 − 매도공격
    X = {}
    for w in (5, 30):
        X[f"cvd{w}"] = (net.rolling(w).sum() / qv.rolling(w).sum()).values
    early = (net.rolling(30).sum() - net.rolling(5).sum()) / (qv.rolling(30).sum() - qv.rolling(5).sum())
    X["cvddiv"] = (net.rolling(5).sum() / qv.rolling(5).sum() - early).values   # +면 최근 매도 약화
    lr = np.log(c).diff()
    rv = lr.rolling(300).std()
    X["absorb"] = ((qv.rolling(5).mean() / qv.rolling(1440).mean()) /
                   (np.abs(np.log(c / c.shift(5))) / (rv * np.sqrt(5)) + 0.25)).values
    piv = l.shift(30).rolling(210).min()                  # 30~240분 전 스윙 저점
    X["piv"] = (np.log(c / piv) / (rv * np.sqrt(60))).values
    X["swept"] = (l.rolling(30).min() < piv).astype(float).values
    ab = (net.abs()).rolling(15).sum() / qv.rolling(15).sum()
    X["vpin15"] = ab.values
    X["vpin60"] = ((net.abs()).rolling(60).sum() / qv.rolling(60).sum()).values
    sell = qv - tb
    X["accel"] = (sell.rolling(5).mean() / sell.rolling(30).mean()).values
    m = store.load("metrics_5m", s).set_index("ts")
    m.index = m.index + pd.Timedelta(minutes=5)
    oi = m.sum_open_interest.reindex(idx, method="ffill")
    X["oi15"] = np.log(oi / oi.shift(15)).values
    oi60 = np.log(oi / oi.shift(60)).values
    X["quad"] = (np.sign(np.log(c / c.shift(60)).values) * 2 + np.sign(oi60))
    pr = store.load("premium_1m", s).set_index("ts").close.reindex(idx).ffill()
    X["dprem5"] = (pr - pr.shift(5)).values
    X["dprem30"] = (pr - pr.shift(30)).values
    bd = store.load("bookdepth", s)
    bd["m"] = bd.ts.dt.floor("min") + pd.Timedelta(minutes=1)   # 스냅샷 분이 끝난 뒤부터 사용
    bd = bd.groupby("m").last()
    sl = (bd["n-5"] / bd["n-1"]).reindex(idx, method="ffill", limit=10)
    im = ((bd["n-1"] - bd["n+1"]) / (bd["n-1"] + bd["n+1"])).reindex(idx, method="ffill", limit=10)
    X["bslope"] = sl.values; X["bimb"] = im.values
    return X, piv.values


if __name__ == "__main__":
    rows = []
    for s in SYM:
        d, F = feats(s)
        X, piv = extra(s, d)
        F.update(X)
        P(f"{s} 특징 {len(F)}개")
        for B in BARS:
            for kx in KS:
                a, b, r, m = run(d["o"], d["h"], d["l"], d["c"], d["rv1"], B, 1, kx, 0, 1.0, 0, EPS, FM, FT)
                if len(a) == 0: continue
                fm1 = a - 1; s0 = (a // B) * B
                x = kx * d["rv1"][s0] * np.sqrt(B)
                T = d["o"][s0] * (1 - x)
                df = pd.DataFrame({"sym": s, "B": B, "k": kx, "f": a, "x": b, "ret": r, "mae": m,
                                   "ts": d["idx"][a], "el": (a % B) / B,
                                   "pre": (d["c"][fm1] / d["o"][s0] - 1) / x,
                                   "tpiv": np.log(T / piv[fm1]) / x,
                                   "qh": (a % 15).astype(float), "hr0": (a % 60).astype(float)})
                for kk, v in F.items():
                    df[kk] = v[fm1]
                rows.append(df)
        P(f"  {s} 완료")
    E = pd.concat(rows, ignore_index=True)
    for c_ in E.columns:
        if E[c_].dtype == np.float64: E[c_] = E[c_].astype(np.float32)
    E = E.replace([np.inf, -np.inf], np.nan)
    E.to_parquet(R / "data" / "r32_events.parquet", index=False)
    P(f"저장 data/r32_events.parquet {E.shape}")
