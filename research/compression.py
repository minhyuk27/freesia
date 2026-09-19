"""'거래대금은 쌓이는데 가격은 안 움직인다'(흡수) → 이후 큰 일방향 이동이 오는가.
핵심 대조: 거래 많음+무변동(흡수) vs 거래 적음+무변동(고갈). 주식에서 온 가설을 크립토에서 검정.
측정: 이후 이동 크기 / 일방향성(효율비) / 첫 X% 도달 시간 / 양방향 구조 손익.
사용: python research/compression.py"""
import sys, warnings
from pathlib import Path
import numba, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R))
from freesia import store
TE = pd.Timestamp("2024-01-01", tz="UTC")
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
BASE = 20160          # 2주 기준선


@numba.njit(cache=True)
def first_hit(hi, lo, cl, ent, x, H):
    """진입 후 ±x 중 먼저 닿는 쪽과 걸린 분. 반환 (방향, 분)"""
    n = len(cl)
    side = np.zeros(n, np.int8); mins = np.zeros(n, np.int32)
    for i in range(n - 1):
        if not ent[i]:
            continue
        p0 = cl[i]
        for j in range(i + 1, min(i + 1 + H, n)):
            if hi[j] / p0 - 1.0 >= x:
                side[i] = 1; mins[i] = j - i; break
            if 1.0 - lo[j] / p0 >= x:
                side[i] = -1; mins[i] = j - i; break
        else:
            mins[i] = H
    return side, mins


for sym in ("BTCUSDT", "ETHUSDT"):
    k = store.load("klines_1m", sym).set_index("ts")
    idx = k.index
    hi, lo, cl, v, nt, tb = k.high.values, k.low.values, k.close.values, k.quote_volume.values, k.trades.values, k.taker_buy_quote_volume.values
    lp = np.log(cl)
    try:
        oi = store.load("metrics_5m", sym).set_index("ts")["sum_open_interest"].resample("1min").ffill().reindex(idx).values
    except Exception:
        oi = np.full(len(k), np.nan)
    P("=" * 100 + f"\n### {sym} 1분봉 {len(k):,}개")

    for W in (240, 1440):
        s = pd.Series
        vol = s(v).rolling(W).sum()
        volz = vol / vol.rolling(BASE).median()                 # 거래대금 상대 강도
        trd = s(nt).rolling(W).sum(); trdz = trd / trd.rolling(BASE).median()
        net = (s(lp) - s(lp).shift(W)).abs()                    # 구간 순이동
        netz = net / net.rolling(BASE).median()
        rng_ = (s(hi).rolling(W).max() - s(lo).rolling(W).min()) / s(cl)
        vpr = vol / rng_.replace(0, np.nan)                     # 거래대금/범위
        vprz = vpr / vpr.rolling(BASE).median()
        imb = (s(2 * tb - v).rolling(W).sum() / vol.replace(0, np.nan)).abs()
        doi = s(oi).pct_change(W)

        hiv = (volz > volz.rolling(BASE).quantile(0.7))
        lov = (volz < volz.rolling(BASE).quantile(0.3))
        lonet = (netz < netz.rolling(BASE).quantile(0.3))
        T = {
            "① 흡수(거래많음+무변동)": (hiv & lonet).values,
            "② 고갈(거래적음+무변동)": (lov & lonet).values,
            "③ 거래대금/범위 상위10%": (vprz > vprz.rolling(BASE).quantile(0.9)).values,
            "④ 체결많음+무변동": ((trdz > trdz.rolling(BASE).quantile(0.7)) & lonet).values,
            "⑤ OI축적+무변동": ((s(doi) > 0.01) & lonet).values,
            "⑥ 불균형큼+무변동": ((imb > imb.rolling(BASE).quantile(0.7)) & lonet).values,
            "⑦ 무변동만": lonet.values,
            "⑧ 거래많음만": hiv.values,
            "⑨ 무조건": np.ones(len(k), bool),
        }
        P(f"\n── 형성 구간 W={W}분 ──")
        rows = []
        for nm, m in T.items():
            mm = np.nan_to_num(m, nan=0).astype(bool).copy(); mm[:BASE + W] = False
            if mm.sum() < 2000:
                continue
            for Hf in (240, 1440):
                fwd_net = (s(lp).shift(-Hf) - s(lp)).abs().values
                fwd_path = s(np.abs(np.diff(lp, prepend=lp[0]))).rolling(Hf).sum().shift(-Hf).values
                er = fwd_net / np.where(fwd_path == 0, np.nan, fwd_path)
                base_m = np.nan_to_num(T["⑨ 무조건"], nan=0).astype(bool).copy(); base_m[:BASE + W] = False
                rows.append((nm, Hf, np.nanmean(fwd_net[mm]) * 100, np.nanmean(fwd_net[base_m]) * 100,
                             np.nanmean(er[mm]), np.nanmean(er[base_m]),
                             float(np.nanmean(fwd_net[mm] > 0.02)), int(mm.sum())))
        d = pd.DataFrame(rows, columns=["조건", "Hf", "이후 |순이동|%", "무조건 대비 기준", "효율비", "무조건 효율비", "P(>2%)", "n"])
        d["이동배수"] = (d["이후 |순이동|%"] / d["무조건 대비 기준"]).round(3)
        d["효율비배수"] = (d["효율비"] / d["무조건 효율비"]).round(3)
        P(d[["조건", "Hf", "이후 |순이동|%", "이동배수", "효율비", "효율비배수", "P(>2%)", "n"]].round(4).to_string(index=False))

        # 첫 2% 도달 방향의 편향(방향 예측력) + 도달 시간
        P(f"  [±2% 먼저 닿는 쪽 · {W}분 형성, 최대 7일 추적]")
        for nm in ("① 흡수(거래많음+무변동)", "② 고갈(거래적음+무변동)", "⑦ 무변동만", "⑨ 무조건"):
            mm = np.nan_to_num(T[nm], nan=0).astype(bool).copy(); mm[:BASE + W] = False
            if mm.sum() < 1000:
                continue
            side, mins = first_hit(hi, lo, cl, mm, 0.02, 10080)
            sel = mm & (side != 0)
            if sel.sum() < 200:
                continue
            up = float((side[sel] == 1).mean())
            P(f"    {nm:24s} 위 {up:.3f} / 아래 {1-up:.3f} | 도달 중앙 {np.median(mins[sel]):.0f}분 | 7일내 도달 {sel.sum()/mm.sum():.3f} | n={int(sel.sum())}")
