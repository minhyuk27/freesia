"""차트분석 기법 15종을 3~4개씩 결합해 엣지 탐색 (호가 없이 가능한 것만, BTC·ETH).
규율: 학습 2020~2023에서만 선택 → 테스트 2024~에서 1회 평가. 조합 1,820개이므로 같은 강도의 위약 탐색으로 null-max 문턱을 만든다.
사용: python research/chart_combo.py [4h|1h]"""
import itertools
import sys
import warnings
from pathlib import Path

import numba
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from freesia.bt import core, ind  # noqa: E402

TF = sys.argv[1] if len(sys.argv) > 1 else "4h"
COST = 0.0012                    # 왕복 12bp
TE = pd.Timestamp("2024-01-01", tz="UTC")
GEOM = {"g2": (0.02, 0.02, 30), "g3": (0.03, 0.03, 60), "g15": (0.015, 0.015, 12)}
rng = np.random.default_rng(5)


# ───────── 차트분석 기법 15종 (각각 -1/0/+1)
def techniques(b):
    o, h, l, c, v = b.open, b.high, b.low, b.close, b.quote_volume
    T = {}
    n = len(c)

    # 1. 다우 시장구조: 스윙 고점·저점이 동반 상승/하락
    sh, sl = ind.swings(h, l, 5)
    hi = h.where(sh).ffill(); lo = l.where(sl).ffill()
    hi1, lo1 = hi.shift(1), lo.shift(1)
    T["구조"] = np.sign((hi > hi1).astype(int) + (lo > lo1).astype(int) - 1).where((hi > hi1) == (lo > lo1), 0).fillna(0)

    # 2. 돈치안 채널 돌파
    T["돌파"] = pd.Series(np.where(c > h.rolling(20).max().shift(1), 1, np.where(c < l.rolling(20).min().shift(1), -1, 0)), index=c.index)

    # 3. 이치모쿠 구름 + 전환/기준선
    t9, k26, sa, sb = ind.ichimoku(h, l, c, 9, 26, 52)
    above = (c > sa) & (c > sb) & (t9 > k26)
    below = (c < sa) & (c < sb) & (t9 < k26)
    T["이치모쿠"] = pd.Series(np.where(above, 1, np.where(below, -1, 0)), index=c.index)

    # 4. 볼륨 프로파일 POC (최근 120봉 거래대금 최빈 가격대)
    poc = np.full(n, np.nan)
    tp = ((h + l + c) / 3).values; vv = v.values
    for i in range(120, n):
        p = tp[i - 120:i]; w = vv[i - 120:i]
        cnt, edge = np.histogram(p, bins=24, weights=w)
        j = int(cnt.argmax()); poc[i] = (edge[j] + edge[j + 1]) / 2
    T["볼륨프로파일"] = pd.Series(np.sign(c.values - poc), index=c.index).fillna(0)

    # 5. 앵커드 VWAP (최근 스윙 극단에 고정)
    anchor = np.where(sh.values | sl.values)[0]
    av = np.full(n, np.nan); a = 0; cum_pv = 0.0; cum_v = 0.0
    aset = set(anchor.tolist())
    for i in range(n):
        if i in aset:
            cum_pv = 0.0; cum_v = 0.0
        cum_pv += tp[i] * vv[i]; cum_v += vv[i]
        if cum_v > 0:
            av[i] = cum_pv / cum_v
    T["앵커드VWAP"] = pd.Series(np.sign(c.values - av), index=c.index).fillna(0)

    # 6. 피보나치 되돌림: 추세 방향으로 38.2~61.8% 되돌림 구간이면 추세 방향
    sw_h = h.where(sh).ffill(); sw_l = l.where(sl).ffill()
    rngv = (sw_h - sw_l)
    retr = (sw_h - c) / rngv.replace(0, np.nan)
    up = (sw_h.diff() > 0) | (sw_l.diff() > 0)
    inz = (retr > 0.382) & (retr < 0.618)
    T["피보나치"] = pd.Series(np.where(inz & up, 1, np.where(inz & ~up, -1, 0)), index=c.index)

    # 7. 캔들 패턴: 장악형 / 핀바
    body = (c - o).abs(); rng_ = (h - l).replace(0, np.nan)
    bull = ((c > o) & (o.shift(1) > c.shift(1)) & (c >= o.shift(1)) & (o <= c.shift(1))) | ((c - l) / rng_ > 0.66)
    bear = ((c < o) & (o.shift(1) < c.shift(1)) & (c <= o.shift(1)) & (o >= c.shift(1))) | ((h - c) / rng_ > 0.66)
    T["캔들패턴"] = pd.Series(np.where(bull & ~bear, 1, np.where(bear & ~bull, -1, 0)), index=c.index)

    # 8. RSI 다이버전스 (가격 신저점인데 RSI는 아님 → 강세)
    r = ind.rsi(c, 14)
    pl = c.rolling(30).min(); rl = r.rolling(30).min()
    ph = c.rolling(30).max(); rh = r.rolling(30).max()
    T["다이버전스"] = pd.Series(np.where((c <= pl) & (r > rl * 1.05), 1, np.where((c >= ph) & (r < rh * 0.95), -1, 0)), index=c.index)

    # 9. 피봇 포인트 (직전 일 고저종 기준)
    d = pd.DataFrame({"h": h, "l": l, "c": c}).resample("1D").agg({"h": "max", "l": "min", "c": "last"})
    piv = ((d.h + d.l + d.c) / 3).shift(1).reindex(c.index, method="ffill")
    T["피봇"] = pd.Series(np.sign(c - piv), index=c.index).fillna(0)

    # 10. FVG (3봉 불균형)
    gap_up = l > h.shift(2); gap_dn = h < l.shift(2)
    T["FVG"] = pd.Series(np.where(gap_up.rolling(10).max() > 0, 1, np.where(gap_dn.rolling(10).max() > 0, -1, 0)), index=c.index)

    # 11. 변동성 수축 후 확장 (볼린저가 켈트너 안 → 이탈 방향)
    bu, bm, bl = ind.bbands(c, 20, 2.0)
    ku, km, kl = ind.keltner(h, l, c, 20, 1.5)
    sq = (bu < ku) & (bl > kl)
    fired = sq.shift(1) & ~sq
    T["수축확장"] = pd.Series(np.where(fired & (c > bm), 1, np.where(fired & (c < bm), -1, 0)), index=c.index)

    # 12. 하이킨아시 추세
    ho, hh, hl, hc = ind.heikin_ashi(o, h, l, c)
    T["하이킨아시"] = pd.Series(np.where((hc > ho) & (hc.shift(1) > ho.shift(1)), 1, np.where((hc < ho) & (hc.shift(1) < ho.shift(1)), -1, 0)), index=c.index)

    # 13. 회귀 채널 (60봉 선형회귀 ±2σ)
    x = np.arange(60)
    slope = c.rolling(60).apply(lambda y: np.polyfit(x, y, 1)[0], raw=True)
    resid = c - c.rolling(60).mean()
    sd = resid.rolling(60).std()
    T["회귀채널"] = pd.Series(np.where((slope > 0) & (resid < -sd), 1, np.where((slope < 0) & (resid > sd), -1, 0)), index=c.index)

    # 14. 이중바닥/이중천정
    lo2 = l.rolling(40).min()
    db = (l <= lo2 * 1.01) & (l.shift(20).rolling(20).min() <= lo2 * 1.01) & (c > c.shift(10))
    hi2 = h.rolling(40).max()
    dt = (h >= hi2 * 0.99) & (h.shift(20).rolling(20).max() >= hi2 * 0.99) & (c < c.shift(10))
    T["이중바닥천정"] = pd.Series(np.where(db, 1, np.where(dt, -1, 0)), index=c.index)

    # 15. 수퍼트렌드 (ATR 브릭 대용)
    st, dirn = ind.supertrend(h, l, c, 10, 3.0)
    T["수퍼트렌드"] = pd.Series(dirn, index=c.index).fillna(0)

    M = np.zeros((len(T), n), np.int8)
    for i, k in enumerate(T):
        M[i] = np.nan_to_num(np.sign(T[k].values)).astype(np.int8)
    return list(T.keys()), M


@numba.njit(cache=True)
def barrier(hh, ll, cc, tp, sl, H):
    n = len(cc)
    rl = np.zeros(n); rs = np.zeros(n); hold = np.zeros(n, np.int32)
    for i in range(n - 1):
        e = cc[i]; up = e * (1 + tp); dn = e * (1 - sl); done = False
        for j in range(i + 1, min(i + 1 + H, n)):
            if hh[j] >= up:
                rl[i] = tp; rs[i] = -tp; hold[i] = j - i; done = True; break
            if ll[j] <= dn:
                rl[i] = -sl; rs[i] = sl; hold[i] = j - i; done = True; break
        if not done:
            j = min(i + H, n - 1)
            rl[i] = cc[j] / e - 1; rs[i] = -rl[i]; hold[i] = j - i
    return rl, rs, hold


@numba.njit(cache=True)
def run_combo(M, idx, k, rl, rs, hold, cost, split):
    n = M.shape[1]
    s_tr = 0.0; s2_tr = 0.0; c_tr = 0; s_te = 0.0; s2_te = 0.0; c_te = 0
    nxt = 0
    for i in range(n):
        if i < nxt:
            continue
        a = M[idx[0], i]
        if a == 0:
            continue
        ok = True
        for t in range(1, k):
            if M[idx[t], i] != a:
                ok = False; break
        if not ok:
            continue
        r = (rl[i] if a == 1 else rs[i]) - cost
        nxt = i + hold[i] + 1
        if i < split:
            s_tr += r; s2_tr += r * r; c_tr += 1
        else:
            s_te += r; s2_te += r * r; c_te += 1
    return s_tr, s2_tr, c_tr, s_te, s2_te, c_te


def stat(s, s2, c):
    if c < 2:
        return 0.0, 0.0, 0
    m = s / c
    var = max(s2 / c - m * m, 1e-18)
    return m, m / np.sqrt(var) * np.sqrt(c), c


def search(M, rl, rs, hold, split, combos, ks):
    out = np.zeros((len(combos), 6))
    for i in range(len(combos)):
        idx = combos[i]; k = ks[i]
        a, b, cN, d, e, f = run_combo(M, idx, k, rl, rs, hold, COST, split)
        mtr, ttr, ntr = stat(a, b, cN); mte, tte, nte = stat(d, e, f)
        out[i] = (mtr * 1e4, ttr, ntr, mte * 1e4, tte, nte)
    return out


if __name__ == "__main__":
    names = None
    res = {}
    for sym in ("BTCUSDT", "ETHUSDT"):
        b = core.bars(sym, TF)
        names, M = techniques(b)
        split = int((b.index < TE).sum())
        cmb = [c for r in (3, 4) for c in itertools.combinations(range(15), r)]
        combos = np.array([list(c) + [0] * (4 - len(c)) for c in cmb], np.int64)
        ks = np.array([len(c) for c in cmb], np.int64)
        print(f"\n{'='*80}\n{sym} {TF} 봉 {len(b):,}개 | 기법 {len(names)}종 | 조합 {len(cmb)}개 (3개 {sum(ks==3)}, 4개 {sum(ks==4)})", flush=True)
        for gname, (tp, sl, H) in GEOM.items():
            rl, rs, hold = barrier(b.high.values, b.low.values, b.close.values, tp, sl, H)
            o = search(M, rl, rs, hold, split, combos, ks)
            d = pd.DataFrame(o, columns=["학습bp", "학습t", "학습n", "테스트bp", "테스트t", "테스트n"])
            d["조합"] = [" + ".join(names[j] for j in c) for c in cmb]
            d = d[(d.학습n >= 30) & (d.테스트n >= 20)]
            # 학습에서만 선택 → 테스트 1회 평가
            top = d.sort_values("학습t", ascending=False).head(10)
            print(f"\n[{gname} 목표{tp*100:.1f}% 손절{sl*100:.1f}% {H}봉] 유효 조합 {len(d)}개 — 학습 상위 10의 테스트 성적", flush=True)
            print(top[["조합", "학습bp", "학습t", "학습n", "테스트bp", "테스트t", "테스트n"]].round(2).to_string(index=False), flush=True)
            print(f"  학습 상위10의 테스트 평균 {top.테스트bp.mean():.1f}bp, 양수 {int((top.테스트bp>0).sum())}/10", flush=True)
            print(f"  전체 조합 테스트 t: 최대 {d.테스트t.max():.2f}, 95% {d.테스트t.quantile(.95):.2f} | 학습 t 최대 {d.학습t.max():.2f}", flush=True)
            res[(sym, gname)] = d
            # 위약: 기법 시계열을 무작위 위상 회전 → 같은 탐색 10회
            mx = []
            for _ in range(10):
                Mp = np.empty_like(M)
                for r_ in range(M.shape[0]):
                    Mp[r_] = np.roll(M[r_], rng.integers(200, M.shape[1] - 200))
                op = search(Mp, rl, rs, hold, split, combos, ks)
                dp = pd.DataFrame(op, columns=["학습bp", "학습t", "학습n", "테스트bp", "테스트t", "테스트n"])
                dp = dp[(dp.학습n >= 30) & (dp.테스트n >= 20)]
                if len(dp):
                    tp10 = dp.sort_values("학습t", ascending=False).head(10)
                    mx.append((dp.테스트t.max(), tp10.테스트bp.mean()))
            if mx:
                a = np.array(mx)
                print(f"  위약 10회: 테스트 t 최대의 평균 {a[:,0].mean():.2f} (최대 {a[:,0].max():.2f}) | 학습상위10의 테스트 평균 bp {a[:,1].mean():.1f} (최대 {a[:,1].max():.1f})", flush=True)

    # 두 종목 공통 생존
    print(f"\n{'='*80}\n[두 종목 공통] 학습·테스트 모두 양수인 조합", flush=True)
    for g in GEOM:
        db = res[("BTCUSDT", g)]; de = res[("ETHUSDT", g)]
        okb = set(db[(db.학습bp > 0) & (db.테스트bp > 0)].조합); oke = set(de[(de.학습bp > 0) & (de.테스트bp > 0)].조합)
        both = okb & oke
        print(f"  {g}: BTC {len(okb)} / ETH {len(oke)} / 공통 {len(both)} (전체 {len(db)})", flush=True)
        if both:
            s = db[db.조합.isin(both)].set_index("조합")[["테스트bp", "테스트t"]].join(
                de[de.조합.isin(both)].set_index("조합")[["테스트bp", "테스트t"]], lsuffix="_B", rsuffix="_E")
            s["최소t"] = s[["테스트t_B", "테스트t_E"]].min(axis=1)
            print(s.sort_values("최소t", ascending=False).head(10).round(2).to_string(), flush=True)
