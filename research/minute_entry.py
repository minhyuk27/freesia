"""분 단위 조건부 진입 — 매 분 t에서 t까지의 정보만으로 판단. 룩어헤드 구조적으로 불가능.
진입: 조건 충족 시 시장가(테이커). 청산: H분 후 또는 목표/손절. 쿨다운으로 중복 방지.
비용: 왕복 12bp(테이커) 기준, 지정가 변형은 별도 표기.
사용: python research/minute_entry.py [SYMBOL]"""
import sys, warnings, itertools
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R))
from freesia import store
TE = pd.Timestamp("2024-01-01", tz="UTC")
rng = np.random.default_rng(229)
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)


@numba.njit(cache=True)
def sim(hi, lo, cl, ent, H, tp, sl, fee, cool):
    n = len(cl)
    out = np.full(n, np.nan); hold = np.zeros(n, np.int32)
    nxt = 0
    for i in range(n - 1):
        if (not ent[i]) or i < nxt:
            continue
        p0 = cl[i]
        up = p0 * (1 + tp) if tp > 0 else 1e18
        dn = p0 * (1 - sl) if sl > 0 else -1.0
        j = min(i + H, n - 1); px = cl[j]
        for t in range(i + 1, min(i + 1 + H, n)):
            if hi[t] >= up:
                px = up; j = t; break
            if lo[t] <= dn:
                px = dn; j = t; break
        out[i] = (px / p0 - 1.0) * 1e4 - fee
        hold[i] = j - i
        nxt = j + cool
    return out, hold


if __name__ == "__main__":
    sym = sys.argv[1] if len(sys.argv) > 1 else "ETHUSDT"
    k = store.load("klines_1m", sym).set_index("ts")
    idx = k.index
    o, hi, lo, cl, v, tb, nt = (k.open.values, k.high.values, k.low.values, k.close.values,
                                k.quote_volume.values, k.taker_buy_quote_volume.values, k.trades.values)
    s = pd.Series
    lp = np.log(cl)
    # 모든 특징은 시점 t까지의 정보만 사용(rolling은 현재 포함, 이는 t 시점에 알 수 있음)
    r5 = s(lp).diff(5).values; r15 = s(lp).diff(15).values; r30 = s(lp).diff(30).values; r60 = s(lp).diff(60).values
    rv60 = s(lp).diff().rolling(60).std()
    z5 = r5 / (rv60.values * np.sqrt(5)); z15 = r15 / (rv60.values * np.sqrt(15)); z30 = r30 / (rv60.values * np.sqrt(30))
    volr = (s(v).rolling(15).sum() / s(v).rolling(1440).sum() * 96).values
    imb = (2 * s(tb).rolling(15).sum() / s(v).rolling(15).sum() - 1).values
    ntr = (s(nt).rolling(15).mean() / s(nt).rolling(1440).mean()).values
    low60 = (cl <= s(lo).rolling(60).min().values * 1.0005)
    low240 = (cl <= s(lo).rolling(240).min().values * 1.0005)
    pr = store.load("premium_1m", sym).set_index("ts").close.reindex(idx).values * 1e4
    dprem = s(pr).diff(15).values

    COND = {
        "급락 z5<-2": (z5 < -2), "급락 z5<-3": (z5 < -3), "급락 z15<-2": (z15 < -2), "급락 z15<-3": (z15 < -3),
        "급락 z30<-2": (z30 < -2), "급락 z30<-3": (z30 < -3),
        "60분 저점": low60, "240분 저점": low240,
        "급락z15<-2 + 거래량2배": (z15 < -2) & (volr > 2),
        "급락z15<-2 + 거래량0.5배": (z15 < -2) & (volr < 0.5),
        "급락z15<-2 + 매도우위": (z15 < -2) & (imb < -0.2),
        "급락z15<-2 + 매수우위": (z15 < -2) & (imb > 0.2),
        "급락z15<-2 + 60분저점": (z15 < -2) & low60,
        "급락z15<-2 + 프리미엄급락": (z15 < -2) & (dprem < -5),
        "급락z15<-2 + 프리미엄유지": (z15 < -2) & (dprem > -2),
        "급락z30<-2 + 240분저점": (z30 < -2) & low240,
        "급락z15<-2 + 체결급증": (z15 < -2) & (ntr > 2),
        "(대조)무조건 매시": (np.arange(len(cl)) % 60 == 0),
    }
    rows = []; pool = []
    for cn, cd in COND.items():
        cdv = np.nan_to_num(cd, nan=0).astype(np.bool_).copy(); cdv[:2000] = False
        if cdv.sum() < 200:
            continue
        for H, tp, sl, gn in ((60, 0, 0, "60분보유"), (240, 0, 0, "240분보유"), (1440, 0, 0, "1일보유"),
                              (240, 0.01, 0.01, "±1%/240분"), (240, 0.02, 0.02, "±2%/240분"),
                              (1440, 0.02, 0.02, "±2%/1일"), (1440, 0.03, 0.015, "+3/-1.5%/1일")):
            out, hold = sim(hi, lo, cl, cdv, H, tp, sl, 12.0, 60)
            ok = ~np.isnan(out); ts = idx[ok]
            if ok.sum() < 150:
                continue
            tr = ts < TE; te = ~tr
            if tr.sum() < 60 or te.sum() < 50:
                continue
            a = out[ok][tr]; b_ = out[ok][te]
            wk = te.sum() / max((ts[te][-1] - ts[te][0]).days / 7, 1)
            rows.append((cn, gn, a.mean(), int(tr.sum()), b_.mean(), float((b_ > 0).mean()), int(te.sum()),
                         b_.mean() / b_.std() * np.sqrt(len(b_)) if b_.std() > 0 else 0, wk))
            for _ in range(3):
                cr = np.roll(cdv, rng.integers(5000, len(cdv) - 5000))
                o2, _ = sim(hi, lo, cl, cr, H, tp, sl, 12.0, 60)
                m2 = ~np.isnan(o2) & (idx >= TE)
                if m2.sum() > 50:
                    pool.append(o2[m2].mean())
    d = pd.DataFrame(rows, columns=["조건", "청산", "학습bp", "학습n", "테스트bp", "테스트승률", "테스트n", "t", "주당"])
    pool = np.array(pool)
    P(f"### {sym}  분 단위 조건부 진입 {len(d)}종 (테이커 12bp)")
    P(f"위약 {len(pool)}회: 평균 {pool.mean():.2f} 95% {np.percentile(pool,95):.2f} **최대 {pool.max():.2f}**")
    P("\n[테스트 상위 18]")
    P(d.sort_values("테스트bp", ascending=False).head(18).round(2).to_string(index=False))
    both = d[(d.학습bp > 0) & (d.테스트bp > 0)]
    P(f"\n학습·테스트 모두 양수 {len(both)}/{len(d)} | 테스트 > 위약최대: {int((d.테스트bp > pool.max()).sum())}개")
    if len(both):
        P(both.sort_values("테스트bp", ascending=False).head(12).round(2).to_string(index=False))
