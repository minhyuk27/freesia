"""양방향 고배율(더블 녹아웃 스트래들) 구조 — 변동성 타이밍으로 개선 가능한가.
구조: 자본 대비 증거금 m씩 롱·숏 동시 L배. 한쪽은 역행 d=1/L-0.4%에서 청산(-m), 살아남은 쪽은 계속 보유.
손실은 '톱질'에서만 발생. 따라서 핵심 질문 = 고변동 구간이 일방향인가 톱질인가.
사용: python research/straddle.py
"""
import sys
import warnings
from pathlib import Path

import numba
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from freesia import store  # noqa: E402
from freesia.bt import ind  # noqa: E402

TE = pd.Timestamp("2024-01-01", tz="UTC")
FEE = 0.0005          # 편도 taker
rng = np.random.default_rng(71)
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)


@numba.njit(cache=True)
def run(hi, lo, cl, ent, L, H, tgt, vdec, vwin):
    """ent: 진입 여부. 반환 배열(진입 인덱스별 총스테이크 대비 수익률, 보유 분)"""
    n = len(cl)
    d = 1.0 / L - 0.004
    out = np.full(n, np.nan)
    hold = np.zeros(n, np.int32)
    nxt = 0
    for i in range(n - 2):
        if (not ent[i]) or i < nxt:
            continue
        p0 = cl[i]
        longdead = False; shortdead = False
        pnl_dead = 0.0
        j = i
        ex = -1
        for j in range(i + 1, min(i + 1 + H, n)):
            up = hi[j] / p0 - 1.0
            dn = 1.0 - lo[j] / p0
            if (not longdead) and dn >= d:
                longdead = True; pnl_dead -= 1.0
            if (not shortdead) and up >= d:
                shortdead = True; pnl_dead -= 1.0
            if longdead and shortdead:
                ex = j; break
            mv = cl[j] / p0 - 1.0
            # 살아남은 다리 손익(증거금 배수)
            alive = 0.0
            if not longdead:
                alive += L * mv
            if not shortdead:
                alive += -L * mv
            if tgt > 0 and alive >= tgt:
                ex = j; break
            if vdec > 0 and j - i > vwin:
                s1 = 0.0; s2 = 0.0
                for t in range(j - vwin + 1, j + 1):
                    s1 += abs(cl[t] / cl[t - 1] - 1.0)
                for t in range(i + 1, i + vwin + 1):
                    s2 += abs(cl[t] / cl[t - 1] - 1.0)
                if s2 > 0 and s1 / s2 < vdec:
                    ex = j; break
        if ex < 0:
            ex = min(i + H, n - 1)
        mv = cl[ex] / p0 - 1.0
        alive = 0.0
        nalive = 0
        if not longdead:
            alive += L * mv; nalive += 1
        if not shortdead:
            alive += -L * mv; nalive += 1
        fee = 2.0 * L * FEE + nalive * L * FEE      # 진입 2다리 + 생존 다리 청산
        tot = (pnl_dead + alive - fee) / 2.0         # 총 스테이크 2m 대비
        out[i] = tot
        hold[i] = ex - i
        nxt = ex + 1
    return out, hold


def er(cl, W):
    """효율비 = |순이동| / 경로길이 (1에 가까울수록 일방향)"""
    net = np.abs(pd.Series(cl).shift(-W) / pd.Series(cl) - 1)
    path = pd.Series(np.abs(np.diff(cl, prepend=cl[0]) / cl)).rolling(W).sum().shift(-W)
    return (net / path.replace(0, np.nan)).values


for sym in ("BTCUSDT", "ETHUSDT"):
    k = store.load("klines_1m", sym).set_index("ts")
    idx = k.index
    hi, lo, cl, v, n_ = k.high.values, k.low.values, k.close.values, k.quote_volume.values, k.trades.values
    P("=" * 96 + f"\n### {sym} 1분봉 {len(k):,}개")

    # ── 트리거 정의 (룩어헤드 없음)
    lp = np.log(cl)
    r1 = pd.Series(lp).diff()
    rv5 = r1.rolling(5).std(); rv60 = r1.rolling(60).std(); rv1440 = r1.rolling(1440).std()
    trig = {}
    trig["무조건"] = np.ones(len(k), bool)
    q = rv60.rolling(10080).rank(pct=True)
    trig["저변동(60분 하위20%)"] = (q < 0.2).values
    trig["고변동(60분 상위20%)"] = (q > 0.8).values
    # 스퀴즈: 5분봉 볼린저 ⊂ 켈트너 해제
    c5 = pd.Series(cl, index=idx).resample("5min").last()
    h5 = pd.Series(hi, index=idx).resample("5min").max(); l5 = pd.Series(lo, index=idx).resample("5min").min()
    bm, bu, bl = ind.bbands(c5, 20, 2.0); km, ku, kl = ind.keltner(h5, l5, c5, 20, 1.5)
    sq = ((bu < ku) & (bl > kl))
    fired = (sq.shift(1) & ~sq).reindex(idx, method="ffill").fillna(False).values
    trig["스퀴즈 해제"] = fired
    # 체결 급증
    tz = pd.Series(n_).rolling(60).mean() / pd.Series(n_).rolling(1440).mean()
    trig["체결급증(1시간/1일>2)"] = (tz > 2).values
    # ATR 수축
    atr = ind.atr(pd.Series(hi), pd.Series(lo), pd.Series(cl), 60) / pd.Series(cl)
    trig["ATR 수축(하위10%)"] = (atr.rolling(10080).rank(pct=True) < 0.1).values
    # 거래량 고갈 후 첫 확장
    vz = pd.Series(v).rolling(30).sum() / pd.Series(v).rolling(1440).sum() * 48
    trig["거래량 고갈→확장"] = ((vz.shift(1) < 0.5) & (vz > 1.0)).values
    # 펀딩 정산 ±5분
    mins = idx.hour % 8 * 60 + idx.minute
    trig["펀딩정산 ±5분"] = ((mins <= 5) | (mins >= 475))
    # 미국 개장·장중 (13~16 UTC)
    trig["미국장(13~16 UTC)"] = ((idx.hour >= 13) & (idx.hour < 16))
    # 점프 직후
    sig = r1.abs().rolling(1440).median() / 0.6745
    jump = (r1.abs() > 5 * sig)
    trig["점프 직후 1시간"] = jump.rolling(60).max().fillna(0).astype(bool).values
    # OI 급증
    try:
        oi = store.load("metrics_5m", sym).set_index("ts")["sum_open_interest"].resample("1min").ffill().reindex(idx)
        trig["OI 1시간 +1%"] = (oi.pct_change(60) > 0.01).values
    except Exception:
        pass
    # 복합
    trig["스퀴즈+체결급증"] = trig["스퀴즈 해제"] & trig["체결급증(1시간/1일>2)"]
    trig["저변동+체결급증"] = trig["저변동(60분 하위20%)"] & trig["체결급증(1시간/1일>2)"]

    # ── 효율비: 고변동 구간이 일방향인가?
    P("\n[효율비] |순이동|/경로길이 — 1에 가까울수록 일방향(스트래들에 유리)")
    rows = []
    for W in (60, 240):
        e = er(cl, W)
        for nm, m in trig.items():
            mm = m & ~np.isnan(e)
            if mm.sum() < 500:
                continue
            rows.append((W, nm, np.nanmean(e[mm]), int(mm.sum())))
    d = pd.DataFrame(rows, columns=["창(분)", "트리거", "효율비", "n"])
    P(d.pivot_table(index="트리거", columns="창(분)", values="효율비").round(4).sort_values(60, ascending=False).to_string())

    # ── 구조 시뮬레이션
    P("\n[구조 손익] 100배 양방향, 총 스테이크 대비 % (진입 수수료 2다리 + 생존 다리 청산 반영)")
    rows = []
    for name, m in trig.items():
        mm = np.nan_to_num(m, nan=0).astype(bool).copy()
        mm[:11000] = False
        for H, tgt, vdec in ((60, 0, 0), (240, 0, 0), (240, 1.0, 0), (240, 0, 0.5), (1440, 0, 0.5)):
            out, hold = run(hi, lo, cl, mm, 100.0, H, tgt, vdec, 30)
            ok = ~np.isnan(out)
            if ok.sum() < 30:
                continue
            ts = idx[ok]
            for nm2, sel in (("학습", ts < TE), ("테스트", ts >= TE)):
                x = out[ok][sel]
                if len(x) < 20:
                    continue
                rows.append((name, H, tgt, vdec, nm2, x.mean() * 100, np.median(x) * 100,
                             float((x > 0).mean()), len(x), len(x) / max((ts[sel][-1] - ts[sel][0]).days / 7, 1)))
    dd = pd.DataFrame(rows, columns=["트리거", "H", "목표", "변동감쇠", "구간", "평균%", "중앙%", "승률", "n", "주당"])
    pv = dd[dd.구간 == "테스트"].sort_values("평균%", ascending=False)
    P(pv.head(18).round(2).to_string(index=False))
    P("\n  학습·테스트 모두 양수인 설정:")
    g = dd.pivot_table(index=["트리거", "H", "목표", "변동감쇠"], columns="구간", values="평균%")
    both = g[(g.get("학습", 0) > 0) & (g.get("테스트", 0) > 0)]
    P(f"  {len(both)} / {len(g)}개")
    if len(both):
        P(both.round(2).sort_values("테스트", ascending=False).head(12).to_string())
