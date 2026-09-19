"""차트 패턴 연구 4갈래 (Binance BTC·ETH 무기한만).
A 고전 기하 패턴 18종(피벗 기반)  B 데이터 주도 형상 군집(KMeans)  C 최근접이웃 형상 매칭  D 캔들 시퀀스(3~4봉 기호화)
규율: 학습 2020~2023 / 테스트 2024~. 수수료 전 총엣지(bp)를 먼저 보고, 테이커 왕복 12bp·메이커 4bp와 비교. 위약 비교. 시도 수 기록.
사용: python research/patterns.py [tf]"""
import sys
import warnings
from collections import Counter
from pathlib import Path

import numba
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from freesia.bt import core  # noqa: E402

TF = sys.argv[1] if len(sys.argv) > 1 else "1h"
TE = pd.Timestamp("2024-01-01", tz="UTC")
rng = np.random.default_rng(31)
TRIES = Counter()
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)


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


def eval_sig(sig, rl, rs, hold, idx, cost=0.0012):
    """sig: -1/0/1. 반환 학습·테스트 (총bp, 순bp, t, n, 주당)"""
    out = {}
    for nm, m in (("학습", idx < TE), ("테스트", idx >= TE)):
        ev = (sig != 0) & m
        sel = pick(ev, hold)
        if sel.sum() < 15:
            out[nm] = None; continue
        r = np.where(sig[sel] == 1, rl[sel], rs[sel])
        wk = sel.sum() / max((idx[m][-1] - idx[m][0]).days / 7, 1)
        out[nm] = (r.mean() * 1e4, (r.mean() - cost) * 1e4, tstat(r), int(sel.sum()), wk)
    return out


def piv(h, l, k):
    """프랙탈 피벗. 반환: (확정시점 인덱스, 피벗 인덱스, 가격, 고점여부)"""
    n = len(h); H = h.values; L = l.values
    ish = (h.rolling(2 * k + 1).max().shift(-k).values == H)
    isl = (l.rolling(2 * k + 1).min().shift(-k).values == L)
    out = []
    for i in range(k, n - k):
        if ish[i]:
            out.append((i + k, i, H[i], True))
        elif isl[i]:
            out.append((i + k, i, L[i], False))
    # 교대 정리: 같은 타입 연속이면 더 극단만 유지
    cl = []
    for c, i, p, ih in out:
        if cl and cl[-1][3] == ih:
            if (ih and p > cl[-1][2]) or (not ih and p < cl[-1][2]):
                cl[-1] = (c, i, p, ih)
        else:
            cl.append((c, i, p, ih))
    return cl


def classic(b, k=3):
    """고전 패턴 18종 → dict[name] = 신호배열(-1/0/1)"""
    n = len(b); c = b.close.values
    pv = piv(b.high, b.low, k)
    S = {nm: np.zeros(n, np.int8) for nm in
         ["헤드앤숄더", "역헤드앤숄더", "이중천정", "이중바닥", "삼중천정", "삼중바닥",
          "상승삼각형", "하락삼각형", "대칭삼각형", "상승쐐기", "하락쐐기",
          "직사각형", "확산형", "강세깃발", "약세깃발", "컵앤핸들", "둥근바닥", "둥근천정"]}
    near = lambda a, b_, t=0.03: abs(a - b_) / max(a, 1e-9) < t
    for j in range(5, len(pv)):
        cf = pv[j][0]                      # 확정 시점
        if cf >= n:
            break
        p = [pv[j - i][2] for i in range(5, 0, -1)]      # p0..p4 (오래된 순)
        ih = [pv[j - i][3] for i in range(5, 0, -1)]
        idx5 = [pv[j - i][1] for i in range(5, 0, -1)]
        px = c[cf]
        if ih[0]:      # H-L-H-L-H
            neck = min(p[1], p[3])
            if p[2] > p[0] and p[2] > p[4] and near(p[0], p[4]) and px < neck:
                S["헤드앤숄더"][cf] = -1
            if near(p[0], p[2], .02) and near(p[2], p[4], .02) and px < neck:
                S["삼중천정"][cf] = -1
            if p[0] > p[2] > p[4] and near(p[1], p[3], .02) and px < min(p[1], p[3]):
                S["하락삼각형"][cf] = -1
            if p[0] < p[2] < p[4] and p[1] < p[3] and px < p[3]:
                S["상승쐐기"][cf] = -1
            if near(p[0], p[2], .015) and near(p[2], p[4], .015) and near(p[1], p[3], .015):
                if px > max(p[0], p[2], p[4]):
                    S["직사각형"][cf] = 1
                elif px < min(p[1], p[3]):
                    S["직사각형"][cf] = -1
            if p[4] > p[2] > p[0] and p[3] < p[1] and px > p[2]:
                S["확산형"][cf] = 1
            if p[0] > p[2] > p[4] and p[1] < p[3] and px > p[2]:
                S["대칭삼각형"][cf] = 1
            # 컵앤핸들: 왼쪽 고점 ≈ 오른쪽 고점, 가운데 깊은 저점, 마지막 얕은 눌림
            if near(p[0], p[2], .05) and p[1] < min(p[0], p[2]) * 0.85 and p[3] > p[1] * 1.1 and px > p[2]:
                S["컵앤핸들"][cf] = 1
            if p[1] < p[3] and p[0] < p[2] and px > p[4] and (idx5[4] - idx5[0]) > 20:
                S["둥근바닥"][cf] = 1
        else:          # L-H-L-H-L
            neck = max(p[1], p[3])
            if p[2] < p[0] and p[2] < p[4] and near(p[0], p[4]) and px > neck:
                S["역헤드앤숄더"][cf] = 1
            if near(p[0], p[2], .02) and near(p[2], p[4], .02) and px > neck:
                S["삼중바닥"][cf] = 1
            if p[0] < p[2] < p[4] and near(p[1], p[3], .02) and px > max(p[1], p[3]):
                S["상승삼각형"][cf] = 1
            if p[0] > p[2] > p[4] and p[1] > p[3] and px > p[3]:
                S["하락쐐기"][cf] = 1
            if near(p[0], p[2], .015) and near(p[2], p[4], .015) and near(p[1], p[3], .015):
                if px < min(p[0], p[2], p[4]):
                    S["직사각형"][cf] = -1
                elif px > max(p[1], p[3]):
                    S["직사각형"][cf] = 1
            if p[4] < p[2] < p[0] and p[3] > p[1] and px < p[2]:
                S["확산형"][cf] = -1
            if p[0] < p[2] < p[4] and p[1] > p[3] and px < p[2]:
                S["대칭삼각형"][cf] = -1
            if p[1] > p[3] and p[0] > p[2] and px < p[4] and (idx5[4] - idx5[0]) > 20:
                S["둥근천정"][cf] = -1
        # 이중 천정·바닥 (마지막 3피벗)
        q = [pv[j - i][2] for i in range(3, 0, -1)]; qh = [pv[j - i][3] for i in range(3, 0, -1)]
        if qh[0] and near(q[0], q[2], .02) and px < q[1]:
            S["이중천정"][cf] = -1
        if (not qh[0]) and near(q[0], q[2], .02) and px > q[1]:
            S["이중바닥"][cf] = 1
    # 깃발: 급등/급락 후 반대방향 얕은 조정 뒤 재돌파
    ret = pd.Series(c).pct_change(10)
    rng20 = (b.high.rolling(10).max() - b.low.rolling(10).min()) / b.close
    flagpole_up = (ret.shift(10) > 0.08).values & (rng20 < 0.05).values
    flagpole_dn = (ret.shift(10) < -0.08).values & (rng20 < 0.05).values
    brk_up = (b.close > b.high.rolling(10).max().shift(1)).values
    brk_dn = (b.close < b.low.rolling(10).min().shift(1)).values
    S["강세깃발"] = np.where(flagpole_up & brk_up, 1, 0).astype(np.int8)
    S["약세깃발"] = np.where(flagpole_dn & brk_dn, -1, 0).astype(np.int8)
    return S


if __name__ == "__main__":
    GEOM = {"g2": (0.02, 0.02, 30), "g4": (0.04, 0.04, 60)}
    for sym in ("BTCUSDT", "ETHUSDT"):
        b = core.bars(sym, TF)
        idx = b.index
        P("=" * 94 + f"\n### {sym} {TF} 봉 {len(b):,}개")
        for gname, (tp, sl, H) in GEOM.items():
            rl, rs, hold = barrier(b.high.values, b.low.values, b.close.values, tp, sl, H)
            P(f"\n[A] 고전 기하 패턴 18종 — 기하 {gname}(±{tp*100:.0f}%, {H}봉), 수수료 전 총bp / 테이커순 / 메이커순")
            rows = []
            for pk in (3, 5):
                S = classic(b, pk)
                for nm, sig in S.items():
                    TRIES["A"] += 1
                    r = eval_sig(sig, rl, rs, hold, idx)
                    if r["학습"] and r["테스트"]:
                        rows.append((nm, pk, *r["학습"][:1], r["학습"][2], r["학습"][3],
                                     r["테스트"][0], r["테스트"][0] - 12, r["테스트"][0] - 4, r["테스트"][2], r["테스트"][3], r["테스트"][4]))
            d = pd.DataFrame(rows, columns=["패턴", "피벗k", "학습총bp", "학습t", "학습n", "테스트총bp", "테이커순", "메이커순", "테스트t", "테스트n", "주당"])
            P(d.sort_values("테스트총bp", ascending=False).round(2).to_string(index=False))
            ok = d[(d.학습총bp > 0) & (d.테스트총bp > 0)]
            P(f"  → 학습·테스트 모두 양수 {len(ok)}/{len(d)} | 테스트 총bp가 메이커 4bp 초과 {int((d.테스트총bp>4).sum())}개, 테이커 12bp 초과 {int((d.테스트총bp>12).sum())}개")
        P(f"\n  시도 누적 A={TRIES['A']}")
