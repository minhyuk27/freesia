"""초단기 규칙 탐색: 1분봉 EMA 교차·괴리, 거래량 급증, 체결(테이커) 불균형, 돌파, VWAP, 그리고 이들의 조합.
5분마다 결정(1분 종가), 결과는 triple-barrier(목표/손절/시간)로 판정. BTC·ETH 각각.
평가: 학습 2020~2023 / 테스트 2024~2026, 비용 taker 왕복 12bp(수수료 10bp + 슬리피지 2bp). 한 규칙은 포지션 하나(쿨다운).
사용: python research/hf_rules.py <SYMBOL>"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from freesia import store  # noqa: E402
from hf_barrier import barrier, FEE, SLIP  # noqa: E402

GEOMS = {"g30": (0.003, 0.003, 60), "g55": (0.005, 0.005, 120), "g510": (0.005, 0.010, 240), "g11": (0.010, 0.010, 480), "g105": (0.010, 0.005, 480)}
COST = 2 * FEE + 2 * SLIP
STEP = 5


def build(sym):
    k = store.load("klines_1m", sym).set_index("ts")
    k = k[k.index >= "2020-01-01"]
    n = (len(k) // STEP) * STEP
    k = k.iloc[:n]
    c, h, l = k.close.to_numpy(np.float64), k.high.to_numpy(np.float64), k.low.to_numpy(np.float64)
    v, tb = k.quote_volume.to_numpy(np.float64), k.taker_buy_quote_volume.to_numpy(np.float64)
    cs, vs, tbs = pd.Series(c), pd.Series(v), pd.Series(tb)
    dec = np.arange(1500 + 4 - (1500 % STEP), n - 1450, STEP)      # 5분 봉 마감 시점(1분 인덱스)
    dec = dec[(dec + 1) % STEP == 0]
    rules = {}

    def S(a):  # 1분 배열 → 결정 시점 값
        return np.asarray(a)[dec]

    def prev(a):
        return np.asarray(a)[dec - STEP]

    # 1) EMA 교차 (초단기 이동평균)
    for f, s in [(3, 10), (5, 20), (10, 40), (20, 80), (40, 160)]:
        st = (cs.ewm(span=f, adjust=False).mean() > cs.ewm(span=s, adjust=False).mean()).to_numpy()
        up, dn = S(st) & ~prev(st), ~S(st) & prev(st)
        rules[f"EMA교차{f}/{s} 추종"] = (up, dn)
        rules[f"EMA교차{f}/{s} 역추종"] = (dn, up)
    # 2) EMA 괴리(z): 늘어난 고무줄
    for nn in (10, 30, 60, 240):
        dev = cs - cs.ewm(span=nn, adjust=False).mean()
        z = (dev / dev.rolling(60).std()).to_numpy()
        for kk in (1.5, 2.0, 2.5, 3.0):
            hi, lo = S(z) > kk, S(z) < -kk
            hi, lo = hi & ~(prev(z) > kk), lo & ~(prev(z) < -kk)
            rules[f"괴리z EMA{nn} k{kk} 되돌림"] = (lo, hi)
            rules[f"괴리z EMA{nn} k{kk} 추종"] = (hi, lo)
    # 3) 거래량 급증 + 봉 방향
    q5 = vs.rolling(5).sum()
    z5 = ((q5 - q5.rolling(1440).mean()) / q5.rolling(1440).std()).to_numpy()
    d5 = np.sign(S(c) - np.asarray(c)[dec - 5])
    for kk in (2, 3, 4):
        sp = S(z5) > kk
        rules[f"거래량급증z{kk} 방향추종"] = (sp & (d5 > 0), sp & (d5 < 0))
        rules[f"거래량급증z{kk} 방향역추종"] = (sp & (d5 < 0), sp & (d5 > 0))
    # 4) 테이커 매수 불균형
    for w in (5, 15, 60):
        tr = (tbs.rolling(w).sum() / vs.rolling(w).sum() - 0.5).to_numpy()
        for th in (0.03, 0.06, 0.10):
            rules[f"테이커{w}m>{th} 추종"] = (S(tr) > th, S(tr) < -th)
            rules[f"테이커{w}m>{th} 역추종"] = (S(tr) < -th, S(tr) > th)
    # 5) 돌파(+거래량) / 돌파 실패
    for N in (30, 60, 240):
        hmax = pd.Series(h).rolling(N).max().shift(STEP).to_numpy(); lmin = pd.Series(l).rolling(N).min().shift(STEP).to_numpy()
        bu, bd = S(c) > S(hmax), S(c) < S(lmin)
        bu, bd = bu & ~(np.asarray(c)[dec - STEP] > prev(hmax)), bd & ~(np.asarray(c)[dec - STEP] < prev(lmin))
        for vz in (0, 2):
            ok = S(z5) > vz
            rules[f"{N}분 돌파 거래량z>{vz} 추종"] = (bu & ok, bd & ok)
            rules[f"{N}분 돌파 거래량z>{vz} 실패매매"] = (bd & ok, bu & ok)
    # 6) UTC 일 VWAP
    day = k.index.floor("1D")
    vw = ((cs * vs).groupby(day).cumsum() / vs.groupby(day).cumsum()).to_numpy()
    dv = cs.to_numpy() / vw - 1
    zv = dv / pd.Series(dv).rolling(1440).std().to_numpy()
    for kk in (1.0, 2.0):
        hi, lo = (S(zv) > kk) & ~(prev(zv) > kk), (S(zv) < -kk) & ~(prev(zv) < -kk)
        rules[f"VWAP이격z{kk} 추종"] = (hi, lo)
        rules[f"VWAP이격z{kk} 되돌림"] = (lo, hi)
    # 7) 조합: 초단기 추세 + 트리거
    tb15 = (tbs.rolling(15).sum() / vs.rolling(15).sum() - 0.5).to_numpy()
    z30 = ((cs - cs.ewm(span=30, adjust=False).mean()) / (cs - cs.ewm(span=30, adjust=False).mean()).rolling(60).std()).to_numpy()
    for f, s in [(5, 20), (10, 40), (20, 80)]:
        up = S((cs.ewm(span=f, adjust=False).mean() > cs.ewm(span=s, adjust=False).mean()).to_numpy())
        for th in (0.03, 0.06):
            rules[f"조합 EMA{f}/{s}추세+테이커15m>{th}"] = (up & (S(tb15) > th) & ~(prev(tb15) > th), ~up & (S(tb15) < -th) & ~(prev(tb15) < -th))
        for kk in (1.0, 2.0):
            rules[f"조합 EMA{f}/{s}추세+눌림z<{kk}"] = (up & (S(z30) < -kk) & ~(prev(z30) < -kk), ~up & (S(z30) > kk) & ~(prev(z30) > kk))
        for vz in (2, 3):
            spk = S(z5) > vz
            rules[f"조합 EMA{f}/{s}추세+거래량z>{vz}방향"] = (up & spk & (d5 > 0), ~up & spk & (d5 < 0))
    ts = k.index[dec]
    return k, c, h, l, dec, ts, rules


def main():
    sym = sys.argv[1]
    t0 = time.time()
    k, c, h, l, dec, ts, rules = build(sym)
    yrs = ts.year.to_numpy()
    print(sym, "결정 시점", len(dec), "규칙", len(rules), f"{time.time()-t0:.0f}s", flush=True)
    out = []
    for gname, (tp, sl, H) in GEOMS.items():
        lab = {s: barrier(h, l, c, dec, tp, sl, H, s) for s in (1, -1)}
        steps = {s: np.ceil(lab[s][1] / STEP).astype(int) for s in (1, -1)}
        print(gname, "라벨", f"{time.time()-t0:.0f}s", flush=True)
        for rname, (L, S_) in rules.items():
            both = L & S_
            ev = np.where((L | S_) & ~both)[0]
            if len(ev) < 50: continue
            sel_side, sel_i = [], []
            nxt = -1
            for e in ev:
                if e < nxt: continue
                sd = 1 if L[e] else -1
                sel_i.append(e); sel_side.append(sd)
                nxt = e + steps[sd][e]
            si = np.array(sel_i); sd = np.array(sel_side)
            ret = np.where(sd == 1, lab[1][2][si], lab[-1][2][si]).astype(np.float64) - COST
            win = np.where(sd == 1, lab[1][0][si], lab[-1][0][si]) == 1
            y = yrs[si]
            row = {"rule": rname, "geom": gname, "sym": sym[:3], "n": int(len(si)), "gross_bps": round(float(1e4 * (ret + COST).mean()), 1)}
            for nm, m in (("train", y <= 2023), ("test", y >= 2024)):
                a = ret[m]
                row[nm] = {"n": int(m.sum()), "bps": round(float(1e4 * a.mean()), 1) if m.sum() else None,
                           "win": round(float(win[m].mean()), 3) if m.sum() else None,
                           "t": round(float(a.mean() / a.std() * np.sqrt(len(a))), 2) if m.sum() > 20 and a.std() > 0 else None}
            row["years"] = {int(Y): round(float(1e4 * ret[y == Y].mean()), 1) for Y in sorted(set(y)) if (y == Y).sum() >= 20}
            out.append(row)
        print(gname, "완료", len(out), f"{time.time()-t0:.0f}s", flush=True)
    Path(f"logs/hf_rules_{sym[:3]}.json").write_text(json.dumps(out, ensure_ascii=False))
    print("저장", len(out))


if __name__ == "__main__":
    main()
