"""엣지 전략에 배율을 얹었을 때의 성장·MDD·청산 (일 단위 리밸런싱, 시간봉 저가/고가로 청산 판정).
격리마진 가정: 하루 동안 포지션 고정. 청산가 = 진입가 × (1 - 1/(L·e) + mmr). 청산 시 증거금 전액 손실 + 청산수수료.
mmr=0.4%(미검증). 손절 없음(최악 기준)과 '청산거리 70%에서 손절' 두 방식을 병기."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from freesia import store  # noqa: E402
from freesia.bt import core  # noqa: E402
import screen as S  # noqa: E402
import combo as CB  # noqa: E402

MMR, LIQ_FEE = 0.004, 0.0125


def daily_frame(sym):
    h = core.bars(sym, "1h")
    d = h.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    d["prev_close"] = d.close.shift(1)
    d["low_dev"] = d.low / d.prev_close - 1      # 전일 종가 기준 하루 중 최대 하락
    d["high_dev"] = d.high / d.prev_close - 1
    d["ret"] = d.close / d.prev_close - 1
    d["fund"] = core.funding_bars(sym, d.index, "1d")
    return d


def simulate(d, e_signal, L, stop_frac=None, cost=core.COST):
    """e_signal[i]: i일 종료 시점에 정한 다음날 노출(0~1). 반환: 일별 수익률, 청산 횟수."""
    e = np.concatenate([[0.0], e_signal[:-1]])
    e_prev = np.concatenate([[0.0], e[:-1]])
    lev = L * e
    r = d.ret.to_numpy(); lo = d.low_dev.to_numpy(); fu = d.fund.to_numpy()
    out = np.zeros(len(d)); liqs = 0
    for i in range(1, len(d)):
        if lev[i] <= 0:
            out[i] = -abs(lev[i] - L * e_prev[i]) * cost * 0
            continue
        dist = 1.0 / lev[i] - MMR                         # 청산까지 허용 하락폭
        ret = lev[i] * r[i] - lev[i] * fu[i] - abs(lev[i] - L * e_prev[i]) * cost
        if lo[i] <= -dist:                                # 청산
            liqs += 1
            ret = -1.0
        elif stop_frac is not None and lo[i] <= -dist * stop_frac:   # 손절 (청산 전 이탈)
            ret = -dist * stop_frac * lev[i] - abs(lev[i] - L * e_prev[i]) * cost
        out[i] = max(ret, -1.0)
    return out, liqs


def stat(a):
    eq = np.cumprod(1 + a)
    yrs = len(a) / 365.0
    cagr = eq[-1] ** (1 / yrs) - 1 if eq[-1] > 0 else -1.0
    mdd = (eq / np.maximum.accumulate(eq) - 1).min()
    return {"cagr": round(float(cagr), 3), "mdd": round(float(mdd), 3), "final_x": round(float(eq[-1]), 2),
            "sharpe": round(float(a.mean() / a.std() * np.sqrt(365)) if a.std() > 0 else 0, 2)}


def main():
    res = {}
    frames = {s: daily_frame(s) for s in S.SYMS}
    exps = {}
    for s in S.SYMS:
        x = S.Ctx(s, "1d")
        exps[s] = pd.Series(CB.trend_exposure(x, "ALL", None), index=x.idx).reindex(frames[s].index).fillna(0).to_numpy()
    for label, syms in (("BTC", ["BTCUSDT"]), ("ETH", ["ETHUSDT"]), ("BTC+ETH 50/50", S.SYMS)):
        res[label] = {}
        for L in [1, 1.5, 2, 3, 4, 5, 7, 10]:
            row = {}
            for nm, stop in (("무손절", None), ("손절70%", 0.7)):
                legs, liq = [], 0
                for s in syms:
                    a, q = simulate(frames[s], exps[s], L, stop)
                    legs.append(a); liq += q
                a = np.mean(legs, axis=0)
                m = np.arange(len(a)) >= 0
                row[nm] = {**stat(a[365:]), "liq": liq, "2022+": stat(a[frames[syms[0]].index.year >= 2022]) ["cagr"]}
            res[label][str(L)] = row
        # 참고: 매수후보유 1배
    Path("logs/lev.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    for label, rr in res.items():
        print("=====", label, "(추세 앙상블 ALL 롱플랫, 일 리밸런싱, 2021~)")
        print(f"{'배율':>4} | {'무손절 CAGR':>10} {'MDD':>7} {'청산':>4} {'x배':>7} | {'손절70% CAGR':>11} {'MDD':>7} {'x배':>7} {'Sh':>5}")
        for L, row in rr.items():
            a, b = row["무손절"], row["손절70%"]
            print(f"{L:>4} | {a['cagr']:>10} {a['mdd']:>7} {a['liq']:>4} {a['final_x']:>7} | {b['cagr']:>11} {b['mdd']:>7} {b['final_x']:>7} {b['sharpe']:>5}")


if __name__ == "__main__":
    main()
