"""옛 Atdroid V1 규칙을 문서 그대로 재현 (Binance 데이터, BTC·ETH, 1배, 비용 편도 6bp).
진입: 일봉 MACD 히스토그램>0(직전 마감 일봉) AND 4시간 RSI<55(ETH)/<60(BTC) AND BTC 일봉 MA20 상승·횡보(종가>=MA20 또는 이격 1% 이내)
청산: 목표 +5%(ETH)/+4%(BTC), 손절 -1.5%, 최대 보유 72시간, 4시간 RSI>70(ETH)/75(BTC). 같은 봉에서 손절·목표 동시 도달 시 손절 우선(보수적)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freesia.bt import core, ind  # noqa: E402

COST = 0.0006
PARAM = {"ETHUSDT": dict(rsi_in=55, tp=0.05, rsi_out=70), "BTCUSDT": dict(rsi_in=60, tp=0.04, rsi_out=75)}
btc_d = core.bars("BTCUSDT", "1d").close
ma20 = btc_d.rolling(20).mean()
mkt_ok_d = ((btc_d >= ma20) | ((btc_d / ma20 - 1).abs() < 0.01)).shift(1)     # 직전 마감 일봉 기준


def run(sym, start=None):
    p = PARAM[sym]
    b4 = core.bars(sym, "4h"); d1 = core.bars(sym, "1d").close
    _, _, mh = ind.macd(d1, 12, 26, 9)
    macd_ok = (mh > 0).shift(1)                                                # 직전 마감 일봉
    rsi = ind.rsi(b4.close, 14)
    day = b4.index.floor("1D")
    mk = pd.Series(mkt_ok_d.reindex(day).to_numpy(), index=b4.index).fillna(False)
    md = pd.Series(macd_ok.reindex(day).to_numpy(), index=b4.index).fillna(False)
    o, h, l, c = b4.open.to_numpy(), b4.high.to_numpy(), b4.low.to_numpy(), b4.close.to_numpy()
    r = rsi.to_numpy(); ent_ok = (mk & md & (rsi < p["rsi_in"])).to_numpy()
    n = len(c); eq = np.ones(n); ret = np.zeros(n); trades = []
    pos = False; ep = 0.0; ei = 0
    for i in range(1, n):
        if pos:
            stop, tgt = ep * (1 - 0.015), ep * (1 + p["tp"])
            reason = None; px = c[i]
            if l[i] <= stop: reason, px = "손절", stop
            elif h[i] >= tgt: reason, px = "목표", tgt
            elif i - ei >= 18: reason = "72h"
            elif r[i] > p["rsi_out"]: reason = "RSI"
            if reason:
                g = px / ep - 1 - 2 * COST
                ret[i] += g; trades.append((b4.index[ei], b4.index[i], g, reason)); pos = False
        elif ent_ok[i - 1]:                                                   # 직전 봉 마감 조건 → 이번 봉 시가 진입
            pos, ep, ei = True, o[i], i
    net = ret
    return b4.index, net, trades


def stats(idx, net, trades, sym):
    yrs = idx.year.to_numpy(); ppy = 6 * 365
    T = pd.DataFrame(trades, columns=["in", "out", "g", "why"])
    def sh(a): return round(float(a.mean() / a.std() * np.sqrt(ppy)), 2) if a.std() > 0 else 0
    tot = np.cumprod(1 + net)
    mdd = (tot / np.maximum.accumulate(tot) - 1).min()
    print(f"== {sym}: 거래 {len(T)}회 (연 {len(T)/(len(idx)/ppy):.0f}), 승률 {100*(T.g>0).mean():.0f}%, 평균 {100*T.g.mean():.2f}%, 청산 {T.why.value_counts().to_dict()}")
    print(f"   전체 Sharpe {sh(net)}, 연수익 {net.mean()*ppy*100:.1f}%, MDD {mdd*100:.0f}%, 시장 노출 {100*np.mean(np.abs(net)>0):.1f}%(수익 발생 봉)")
    print("   연도별 수익%:", {int(Y): round(float(net[yrs == Y].sum() * 100), 1) for Y in sorted(set(yrs))})
    print("   연도별 Sharpe:", {int(Y): sh(net[yrs == Y]) for Y in sorted(set(yrs))})
    m22 = yrs >= 2022
    print(f"   2022~ Sharpe {sh(net[m22])}, 연수익 {net[m22].mean()*ppy*100:.1f}%")


for sym in ("ETHUSDT", "BTCUSDT"):
    idx, net, tr = run(sym)
    stats(idx, net, tr, sym)
    bh = core.bars(sym, "4h").close.pct_change().fillna(0).to_numpy()
    print(f"   (참고) 매수후보유 2022~ Sharpe {round(float(bh[idx.year>=2022].mean()/bh[idx.year>=2022].std()*np.sqrt(6*365)),2)}")
