"""차트 패턴 연구 B·C·D — 손으로 정의하지 않고 데이터가 패턴을 찾게 한다.
B 형상 군집(KMeans, 정규화 창) / C 최근접이웃 형상 매칭(상관) / D 캔들 시퀀스 기호화(3~4봉)
규율: 군집·선택은 학습 2020~2023에서만. 테스트 2024~ 1회. 위약(시간 회전) 비교. 시도 수 기록.
사용: python research/patterns2.py [tf]"""
import sys
import warnings
from pathlib import Path

import numba
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from freesia.bt import core  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402

TF = sys.argv[1] if len(sys.argv) > 1 else "1h"
TE = pd.Timestamp("2024-01-01", tz="UTC")
rng = np.random.default_rng(41)
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
TRIES = {}


def tstat(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return float(x.mean() / x.std() * np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else 0.0


@numba.njit(cache=True)
def barrier(hh, ll, cc, tp, sl, H):
    n = len(cc); rl = np.zeros(n); rs = np.zeros(n); hold = np.zeros(n, np.int32)
    for i in range(n - 1):
        e = cc[i]; up = e * (1 + tp); dn = e * (1 - sl); done = False
        for j in range(i + 1, min(i + 1 + H, n)):
            if hh[j] >= up:
                rl[i] = tp; rs[i] = -tp; hold[i] = j - i; done = True; break
            if ll[j] <= dn:
                rl[i] = -sl; rs[i] = sl; hold[i] = j - i; done = True; break
        if not done:
            j = min(i + H, n - 1); rl[i] = cc[j] / e - 1; rs[i] = -rl[i]; hold[i] = j - i
    return rl, rs, hold


@numba.njit(cache=True)
def pick(ev, hold):
    out = np.zeros(len(ev), np.bool_); nxt = 0
    for i in range(len(ev)):
        if ev[i] and i >= nxt:
            out[i] = True; nxt = i + hold[i] + 1
    return out


def windows(lp, W):
    """정규화된 W봉 형상 (현재 봉 포함, 룩어헤드 없음)"""
    n = len(lp)
    X = np.full((n, W), np.nan)
    for j in range(W):
        X[W - 1:, j] = lp[j:n - W + 1 + j]
    m = X.mean(1, keepdims=True); s = X.std(1, keepdims=True)
    return (X - m) / np.where(s == 0, np.nan, s)


def report(name, sig, rl, rs, hold, idx, extra=""):
    rows = []
    for nm, m in (("학습", idx < TE), ("테스트", idx >= TE)):
        ev = (sig != 0) & m
        sel = pick(ev, hold)
        if sel.sum() < 15:
            rows.append(None); continue
        r = np.where(sig[sel] == 1, rl[sel], rs[sel])
        wk = sel.sum() / max((idx[m][-1] - idx[m][0]).days / 7, 1)
        rows.append((r.mean() * 1e4, tstat(r), int(sel.sum()), wk))
    if rows[0] and rows[1]:
        P(f"  {name:28s} 학습 {rows[0][0]:7.1f}bp t{rows[0][1]:5.2f} n{rows[0][2]:6d} | "
          f"테스트 {rows[1][0]:7.1f}bp t{rows[1][1]:5.2f} n{rows[1][2]:6d} 주당{rows[1][3]:5.1f} | "
          f"테이커순 {rows[1][0]-12:6.1f} 메이커순 {rows[1][0]-4:6.1f} {extra}")
    return rows


for sym in ("BTCUSDT", "ETHUSDT"):
    b = core.bars(sym, TF)
    idx = b.index; lp = np.log(b.close.values)
    tr = idx < TE
    P("=" * 100 + f"\n### {sym} {TF} 봉 {len(b):,}개  (학습 {tr.sum():,} / 테스트 {(~tr).sum():,})")
    rl, rs, hold = barrier(b.high.values, b.low.values, b.close.values, 0.03, 0.03, 48)
    P("기하: ±3%, 최대 48봉 보유. 비용 테이커 12bp / 메이커 4bp")

    # ── B. 형상 군집
    P("\n[B] 형상 군집 (KMeans, 학습 구간에서만 적합)")
    nb = 0
    for W in (12, 24, 48):
        X = windows(lp, W)
        ok = ~np.isnan(X).any(1)
        for K in (20, 50):
            nb += K
            fitm = ok & tr
            km = KMeans(K, n_init=4, random_state=0).fit(X[fitm])
            lab = np.full(len(b), -1)
            lab[ok] = km.predict(X[ok])
            stats = []
            for c in range(K):
                m = (lab == c) & tr
                if m.sum() < 60:
                    continue
                r = np.where(np.ones(m.sum()) == 1, rl[m], rl[m])
                stats.append((c, rl[m].mean() * 1e4, m.sum()))
            if not stats:
                continue
            s = pd.DataFrame(stats, columns=["c", "학습롱bp", "n"]).sort_values("학습롱bp")
            longs = set(s[s.학습롱bp > 20].c); shorts = set(s[s.학습롱bp < -20].c)
            sig = np.where(np.isin(lab, list(longs)), 1, np.where(np.isin(lab, list(shorts)), -1, 0)).astype(np.int8)
            report(f"W{W} K{K} (롱{len(longs)}/숏{len(shorts)}군집)", sig, rl, rs, hold, idx)
            # 위약: 군집 라벨을 시간 회전
            pls = []
            for _ in range(60):
                sh = rng.integers(500, len(b) - 500)
                sg = np.roll(sig, sh)
                ev = (sg != 0) & (~tr); sel = pick(ev, hold)
                if sel.sum() > 15:
                    pls.append(np.where(sg[sel] == 1, rl[sel], rs[sel]).mean() * 1e4)
            if pls:
                ev = (sig != 0) & (~tr); sel = pick(ev, hold)
                real = np.where(sig[sel] == 1, rl[sel], rs[sel]).mean() * 1e4 if sel.sum() > 15 else np.nan
                P(f"       위약60회: 평균 {np.mean(pls):5.1f} 95% {np.percentile(pls,95):5.1f} 최대 {np.max(pls):5.1f} | 실제 {real:5.1f}")
    TRIES["B"] = nb

    # ── C. 최근접이웃 형상 매칭
    P("\n[C] 최근접이웃 형상 매칭 (학습 구간 형상과 상관 상위 M개의 이후 수익 평균)")
    nc = 0
    for W in (24, 48):
        X = windows(lp, W)
        ok = ~np.isnan(X).any(1)
        Xn = X / np.sqrt(W)
        train_i = np.where(ok & tr)[0]
        train_i = train_i[::3]                       # 부분 표본
        A = Xn[train_i]
        for M in (50, 200):
            nc += 1
            test_i = np.where(ok & ~tr)[0]
            pred = np.zeros(len(b))
            B_ = Xn[test_i]
            step = 4000
            for s0 in range(0, len(test_i), step):
                sub = B_[s0:s0 + step]
                C = sub @ A.T                        # 상관
                topk = np.argpartition(-C, M, axis=1)[:, :M]
                pred[test_i[s0:s0 + step]] = rl[train_i][topk].mean(1)
            thr = np.percentile(np.abs(pred[test_i]), 80)
            sig = np.where(pred > thr, 1, np.where(pred < -thr, -1, 0)).astype(np.int8)
            report(f"W{W} M{M} 상위20%확신", sig, rl, rs, hold, idx)
    TRIES["C"] = nc

    # ── D. 캔들 시퀀스 기호화
    P("\n[D] 캔들 시퀀스 (방향×몸통크기 기호, 3~4봉 조합)")
    o, h, l, c = b.open.values, b.high.values, b.low.values, b.close.values
    body = (c - o) / np.where(h - l == 0, np.nan, h - l)
    bq = pd.Series(np.abs(body)).rolling(500).rank(pct=True).values
    sym_ = np.where(np.isnan(bq), -1,
                    np.where(np.abs(body) < 0.1, 0,
                             np.where(body > 0, 1 + (bq > .5).astype(int) + (bq > .85).astype(int),
                                      4 + (bq > .5).astype(int) + (bq > .85).astype(int))))
    nd = 0
    for L in (3, 4):
        key = np.zeros(len(b), np.int64) - 1
        valid = np.ones(len(b), bool)
        for j in range(L):
            s = np.roll(sym_, j)
            valid &= (s >= 0)
            key = key * 7 + s
        key[~valid] = -1
        df = pd.DataFrame({"k": key, "r": rl, "tr": tr})
        g = df[df.tr & (df.k >= 0)].groupby("k").r.agg(["mean", "count"])
        g = g[g["count"] >= 80]
        nd += len(g)
        longs = set(g[g["mean"] > 0.002].index); shorts = set(g[g["mean"] < -0.002].index)
        sig = np.where(np.isin(key, list(longs)), 1, np.where(np.isin(key, list(shorts)), -1, 0)).astype(np.int8)
        report(f"{L}봉 시퀀스 (후보 {len(g)}개 → 롱{len(longs)}/숏{len(shorts)})", sig, rl, rs, hold, idx)
        if len(g):
            best = g.sort_values("mean", ascending=False)
            P(f"       학습 최고 시퀀스 {best['mean'].iloc[0]*1e4:.0f}bp(n={int(best['count'].iloc[0])}) / 최저 {best['mean'].iloc[-1]*1e4:.0f}bp(n={int(best['count'].iloc[-1])})")
    TRIES["D"] = nd
    P(f"\n  시도 장부: B {TRIES.get('B')}군집, C {TRIES.get('C')}설정, D {TRIES.get('D')}시퀀스")
