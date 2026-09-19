"""단기 방향 예측 실험: 1분봉 기반 triple-barrier(목표·손절·시간) 라벨 + 워크포워드 ML. BTC·ETH만.
질문: "몇 시간 안에 +tp%가 -sl%보다 먼저 오는가"를 65~75% 이상 맞힐 수 있는가?
- 결정 시점: 5분마다. 진입가 = 그 분 종가 ± 슬리피지 1bp. 같은 분봉에서 양쪽 터치 시 손절 우선(보수적).
- 학습: 해당 연도 이전 전체(15분 간격 부표본), 테스트: 그 해. 신호 임계값 τ는 손익분기 승률+마진으로 사전 지정(최적화 안 함).
- 포지션은 한 번에 하나. 레버리지 시뮬: 100배, 자본 30% 증거금(명목 30배), 수수료 taker 5bp×2 + 슬리피지 1bp×2, 청산 거리 0.6%.
사용법: python research/hf_barrier.py <SYMBOL> <configs> [shuffle]"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freesia import store  # noqa: E402

STEP = 5                       # 결정 간격(분)
SLIP = 0.0001
FEE = 0.0005
LIQ = 0.006                    # 100배 격리 청산 거리 (mmr 0.4% 가정)
CONFIGS = {"A": (0.005, 0.005, 120), "B": (0.005, 0.010, 240), "C": (0.003, 0.003, 60), "D": (0.010, 0.010, 480)}


def load(sym, other):
    k = store.load("klines_1m", sym).set_index("ts")
    k = k[k.index >= "2020-01-01"]
    pr = store.load("premium_1m", sym).set_index("ts").close.reindex(k.index)
    f = store.load("funding", sym).set_index("ts").rate
    fr = f.reindex(k.index, method="ffill")
    met = store.load("metrics_5m", sym)
    oi = None
    if met is not None:
        oi = met.set_index("ts")["sum_open_interest_value"].reindex(k.index, method="ffill", limit=10)
    ko = store.load("klines_1m", other).set_index("ts").reindex(k.index)
    return k, pr, fr, oi, ko


def features(k, pr, fr, oi, ko):
    c = k.close.astype("float32")
    F = pd.DataFrame(index=k.index)
    lr = np.log(c).diff()
    for m in (1, 5, 15, 30, 60, 240, 1440):
        F[f"ret{m}"] = np.log(c).diff(m)
    v30 = lr.rolling(30).std(); v240 = lr.rolling(240).std(); v1440 = lr.rolling(1440).std()
    F["vol30"], F["vol240"], F["vol1440"] = v30, v240, v1440
    F["volratio"] = v30 / v240
    for m in (60, 240):
        hi, lo = k.high.rolling(m).max(), k.low.rolling(m).min()
        F[f"pos{m}"] = (c - lo) / (hi - lo).replace(0, np.nan)
        F[f"dhi{m}"] = (hi / c - 1) / v240.replace(0, np.nan)
        F[f"dlo{m}"] = (c / lo - 1) / v240.replace(0, np.nan)
    qv, tr, tb = k.quote_volume, k.trades, k.taker_buy_quote_volume
    F["qv_z"] = (qv.rolling(5).mean() - qv.rolling(1440).mean()) / qv.rolling(1440).std()
    F["tr_z"] = (tr.rolling(5).mean() - tr.rolling(1440).mean()) / tr.rolling(1440).std()
    for m in (5, 15, 60):
        F[f"tbr{m}"] = tb.rolling(m).sum() / qv.rolling(m).sum() - 0.5
    F["tbr_chg"] = F["tbr15"] - F["tbr60"]
    F["avg_trade"] = (qv.rolling(15).sum() / tr.rolling(15).sum())
    F["avg_trade"] = F["avg_trade"] / F["avg_trade"].rolling(1440).mean()
    F["prem"] = pr; F["prem_chg60"] = pr - pr.shift(60); F["fund"] = fr
    if oi is not None:
        F["oi_chg60"] = oi.pct_change(60); F["oi_chg1440"] = oi.pct_change(1440)
    co = ko.close.astype("float32")
    for m in (5, 15, 60):
        F[f"o_ret{m}"] = np.log(co).diff(m)
    F["o_tbr15"] = ko.taker_buy_quote_volume.rolling(15).sum() / ko.quote_volume.rolling(15).sum() - 0.5
    F["rel60"] = F["ret60"] - F["o_ret60"]
    idx = k.index
    F["min_to_fund"] = ((8 * 60 - (idx.hour * 60 + idx.minute) % (8 * 60))).astype("float32")
    F["hour"], F["dow"] = idx.hour, idx.dayofweek
    return F.astype("float32")


def barrier(high, low, close, ii, tp, sl, H, side):
    """반환: outcome(+1 목표, -1 손절, 0 시간종료), exit_k(분), 순 가격수익(비용 전, 진입가 대비)"""
    n = len(close)
    ent = close[ii] * (1 + SLIP * (1 if side > 0 else -1))
    tp_px, sl_px = ent * (1 + side * tp), ent * (1 - side * sl)
    out = np.zeros(len(ii), np.int8); ex = np.full(len(ii), H, np.int32); ret = np.zeros(len(ii), np.float32)
    open_ = np.ones(len(ii), bool)
    for kk in range(1, H + 1):
        j = np.minimum(ii + kk, n - 1)
        hi, lo = high[j], low[j]
        hit_sl = (lo <= sl_px) if side > 0 else (hi >= sl_px)
        hit_tp = (hi >= tp_px) if side > 0 else (lo <= tp_px)
        lose = open_ & hit_sl
        win = open_ & ~hit_sl & hit_tp
        out[lose], ex[lose], ret[lose] = -1, kk, -sl
        out[win], ex[win], ret[win] = 1, kk, tp
        open_ &= ~(lose | win)
        if not open_.any():
            break
    jt = np.minimum(ii + H, n - 1)
    r_t = side * (close[jt] / ent - 1)
    ret[open_] = r_t[open_]
    return out, ex, ret


def sim(times, side_out, out, ex, ret, tp, sl, ii, lev=100, frac=0.30):
    """한 번에 하나만 보유. 자본 대비 레버리지 시뮬. 반환: 거래 목록, 최종 자본배수, 최대낙폭, 청산 횟수"""
    cap = 1.0; peak = 1.0; mdd = 0.0; trades = []; liq = 0; next_free = -1
    for t in np.argsort(ii):
        if not side_out[t] or ii[t] <= next_free:
            continue
        r = ret[t] - 2 * FEE - 2 * SLIP                           # 순 가격수익
        notional = cap * frac * lev
        if -ret[t] >= LIQ and out[t] == -1 or (r < -LIQ):         # 손절이 청산 거리보다 깊거나 실제 손실이 청산 거리 초과
            pnl = -frac                                            # 증거금 전액 손실
            liq += 1
        else:
            pnl = frac * lev * r
        cap *= (1 + pnl); peak = max(peak, cap); mdd = min(mdd, cap / peak - 1)
        trades.append((times[t], out[t], r, pnl))
        next_free = ii[t] + ex[t]
        if cap <= 0.02: break
    return trades, cap, mdd, liq


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def main():
    sym, cfgs = sys.argv[1], sys.argv[2].split(",")
    shuffle = len(sys.argv) > 3 and sys.argv[3] == "shuffle"
    other = "ETHUSDT" if sym == "BTCUSDT" else "BTCUSDT"
    t0 = time.time()
    k, pr, fr, oi, ko = load(sym, other)
    F = features(k, pr, fr, oi, ko)
    high, low, close = k.high.to_numpy(np.float64), k.low.to_numpy(np.float64), k.close.to_numpy(np.float64)
    idx_all = np.arange(1441, len(k) - 500, STEP)
    Xa = F.iloc[idx_all]; ts = k.index[idx_all]
    print(sym, "특징", F.shape[1], "결정 시점", len(idx_all), f"{time.time()-t0:.0f}s", flush=True)
    from sklearn.ensemble import HistGradientBoostingClassifier
    res = {}
    for cname in cfgs:
        tp, sl, H = CONFIGS[cname]
        lab = {}
        for side in (1, -1):
            lab[side] = barrier(high, low, close, idx_all, tp, sl, H, side)
        print(cname, "라벨 완료", {s: (round(float((lab[s][0] == 1).mean()), 3), round(float((lab[s][0] == -1).mean()), 3), round(float((lab[s][0] == 0).mean()), 3)) for s in lab}, f"{time.time()-t0:.0f}s", flush=True)
        yrs = ts.year.to_numpy()
        cost = 2 * FEE + 2 * SLIP
        # 손익분기 승률(레버리지 무관, 가격 기준 기대값 0): p*(tp-cost) = (1-p)*(sl+cost)
        be = (sl + cost) / (tp + sl)
        res[cname] = {"tp": tp, "sl": sl, "H": H, "breakeven_win": round(be, 3), "years": {}}
        for Y in range(2022, 2027):
            tr = np.where((yrs < Y) & (np.arange(len(yrs)) % 3 == 0))[0]
            te = np.where(yrs == Y)[0]
            if len(tr) < 50000 or len(te) < 1000: continue
            P = {}
            for side in (1, -1):
                y = (lab[side][0][tr] == 1).astype(int)
                if shuffle: y = np.random.default_rng(1).permutation(y)
                m = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.06, max_depth=4, min_samples_leaf=300, l2_regularization=2.0, random_state=0)
                m.fit(Xa.iloc[tr].to_numpy(), y)
                P[side] = m.predict_proba(Xa.iloc[te].to_numpy())[:, 1]
            base = {s: float((lab[s][0][te] == 1).mean()) for s in (1, -1)}
            row = {"base_win_long": round(base[1], 3), "base_win_short": round(base[-1], 3), "taus": {}}
            for tau in (be + 0.03, be + 0.06, be + 0.10):
                sig = np.where((P[1] >= P[-1]) & (P[1] > tau), 1, np.where((P[-1] > P[1]) & (P[-1] > tau), -1, 0))
                pick = np.where(sig != 0)[0]
                if len(pick) == 0:
                    row["taus"][round(tau, 3)] = {"n": 0}; continue
                # 선택된 신호의 라벨/수익 결합
                outs = np.where(sig[pick] == 1, lab[1][0][te][pick], lab[-1][0][te][pick])
                exs = np.where(sig[pick] == 1, lab[1][1][te][pick], lab[-1][1][te][pick])
                rets = np.where(sig[pick] == 1, lab[1][2][te][pick], lab[-1][2][te][pick])
                iis = idx_all[te][pick]
                tr_list, cap, mdd, liq = sim(ts[te][pick], np.ones(len(pick), bool), outs, exs, rets, tp, sl, iis)
                n = len(tr_list); wins = sum(1 for t in tr_list if t[1] == 1)
                gross_ret = np.mean([t[2] for t in tr_list]) if n else 0
                row["taus"][round(tau, 3)] = {"신호수": int(len(pick)), "거래수": n, "주당거래": round(n / (len(te) * STEP / 60 / 24 / 7), 1),
                                              "승률(목표)": round(wins / n, 3) if n else None, "평균 순가격수익%": round(100 * gross_ret, 3),
                                              "100배·30% 최종배수": round(cap, 3), "MDD": round(mdd, 3), "청산": liq}
            # 보정: 예측확률 구간별 실제 승률
            pp = np.where(P[1] >= P[-1], P[1], P[-1]); ss = np.where(P[1] >= P[-1], 1, -1)
            act = np.where(ss == 1, lab[1][0][te] == 1, lab[-1][0][te] == 1)
            qs = np.quantile(pp, [0, .5, .9, .99, 1.0]); cal = []
            for a, b in zip(qs[:-1], qs[1:]):
                mk = (pp >= a) & (pp <= b); cal.append((round(float(a), 3), round(float(b), 3), round(float(act[mk].mean()), 3), int(mk.sum())))
            row["보정(예측확률 구간→실제 승률, n)"] = cal
            res[cname]["years"][Y] = row
            print(sym, cname, Y, "base", row["base_win_long"], row["base_win_short"], "be", round(be, 3), json.dumps(clean(row["taus"]), ensure_ascii=False)[:400], f"{time.time()-t0:.0f}s", flush=True)
    Path(f"logs/hf_{sym[:3]}_{'_'.join(cfgs)}{'_shuf' if shuffle else ''}.json").write_text(json.dumps(clean(res), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
