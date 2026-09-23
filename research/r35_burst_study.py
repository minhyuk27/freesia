"""R35 — 1초 오더플로우 사건 연구: 공격적 체결 폭발 뒤 가격은 어디로 가나.

목표(사용자 2026-09-23): 월 100%+. 필요조건 추정 = 하루 30건 이상 × 거래당 순 +10bp 이상 × 꼬리 작게.
사건: 초 t에서 끝나는 W초(5·15·60) 순공격 대금 nf = Σ(매수−매도 테이커 대금)을
      직전 1시간 초당 평균 거래대금×W 로 나눈 z (인과적). z ≤ −Z = 매도 폭발, z ≥ +Z = 매수 폭발. 같은 방향 쿨다운 60초.
측정: 사건 초 종가 기준 1·5·15·30·60·120·300·600초 뒤 수익(bp, 사건 방향 반대로 서면 +가 되도록 부호 조정 전 원값),
      그리고 이후 60·300초 안의 최저·최고(지정가 체결·MAE 판단용).
출력: data/r35_events.parquet + 요약 (연도별, Z 구간별, 하루 사건 수)
사용: python research/r35_burst_study.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R))
from freesia import store
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
SYM = ("BTCUSDT", "ETHUSDT")
WS = (5, 15, 60)
ZS = (1.5, 2.5, 4.0)
HS = (1, 5, 15, 30, 60, 120, 300, 600)


def month_events(s, f):
    d = pd.read_parquet(f)
    t0 = d.ts.min().floor("D"); t1 = d.ts.max().ceil("min")
    idx = pd.date_range(t0, t1, freq="s", inclusive="left")
    d = d.set_index("ts").reindex(idx)
    c = d.c.ffill().values.astype(np.float64)
    h = d.h.fillna(pd.Series(c, index=idx)).values; l = d.l.fillna(pd.Series(c, index=idx)).values
    bq = d.bq.fillna(0).values.astype(np.float64); sq = d.sq.fillna(0).values.astype(np.float64)
    n = len(c)
    net = np.cumsum(bq - sq); vol = np.cumsum(bq + sq)
    vbar = pd.Series(bq + sq).rolling(3600, min_periods=1800).mean().shift(1).values   # 직전 1시간 초당 대금
    lr = np.diff(np.log(c), prepend=np.nan)
    sd1 = pd.Series(lr).rolling(3600, min_periods=1800).std().shift(1).values
    fl = pd.Series(l[::-1]).rolling(300, min_periods=1).min().values[::-1]   # t..t+299 최저
    fh = pd.Series(h[::-1]).rolling(300, min_periods=1).max().values[::-1]
    fl60 = pd.Series(l[::-1]).rolling(60, min_periods=1).min().values[::-1]
    fh60 = pd.Series(h[::-1]).rolling(60, min_periods=1).max().values[::-1]
    out = []
    for W in WS:
        nf = np.full(n, np.nan); nf[W:] = net[W:] - net[:-W]
        z = nf / (vbar * W)
        rW = np.full(n, np.nan); rW[W:] = np.log(c[W:] / c[:-W])
        for side in (-1, 1):
            cand = np.where((side * z >= ZS[0]) & np.isfinite(z))[0]
            keep = []; last = -10**9
            for i in cand:
                if i - last >= 60 and i + 600 < n and i + 1 < n:
                    keep.append(i); last = i
            if not keep: continue
            k = np.array(keep)
            row = {"sym": s, "W": W, "side": side, "ts": idx[k], "z": z[k], "rW": rW[k] * 1e4,
                   "rWn": rW[k] / (sd1[k] * np.sqrt(W)), "vbar": vbar[k], "sd1": sd1[k] * 1e4}
            for H in HS:
                row[f"m{H}"] = np.log(c[k + H] / c[k]) * 1e4
            # 이후 최저·최고: t+1 부터
            row["lo60"] = np.log(fl60[k + 1] / c[k]) * 1e4; row["hi60"] = np.log(fh60[k + 1] / c[k]) * 1e4
            row["lo300"] = np.log(fl[k + 1] / c[k]) * 1e4; row["hi300"] = np.log(fh[k + 1] / c[k]) * 1e4
            out.append(pd.DataFrame(row))
    return pd.concat(out) if out else None


if __name__ == "__main__":
    ev = []
    for s in SYM:
        for f in sorted((store.PQ / "agg1s" / s).glob("*.parquet")):
            e = month_events(s, f)
            if e is not None: ev.append(e)
            P(f"  {s} {f.stem}: {0 if e is None else len(e)}")
    E = pd.concat(ev, ignore_index=True)
    E.to_parquet(R / "data" / "r35_events.parquet", index=False)
    E["yr"] = E.ts.dt.year
    days = (E.ts.max() - E.ts.min()).days
    P(f"\n사건 {len(E):,} ({E.ts.min()} ~ {E.ts.max()}, {days}일)")
    P("\n[1] 폭발 뒤 수익 (bp, 사건 반대 방향으로 섰을 때 +) — 평균 / 승률(m) / 하루 사건 수(두 종목)")
    for W in WS:
        for side in (-1, 1):
            for lo, hi in zip(ZS, ZS[1:] + (99,)):
                g = E[(E.W == W) & (E.side == side) & (side * E.z >= lo) & (side * E.z < hi)]
                if len(g) < 100: continue
                rev = lambda H: (-side * g[f"m{H}"])
                P(f"  W{W:2d} {'매도' if side<0 else '매수'}폭발 z{lo:>3}~{hi:<3} 하루 {len(g)/days:6.1f} | " +
                  " ".join(f"{H}s{rev(H).mean():+5.1f}" for H in HS) +
                  f" | 60s승 {(rev(60)>0).mean():.2f} 300s승 {(rev(300)>0).mean():.2f} | 폭발중 이동 {g.rW.mean():+5.1f}bp")
    P("\n[2] 연도별 (W15, |z|≥2.5) 반대방향 60s·300s 평균 bp")
    for side in (-1, 1):
        g = E[(E.W == 15) & (E.side == side) & (side * E.z >= 2.5)]
        by = g.groupby("yr")
        P(f"  {'매도' if side<0 else '매수'}폭발 " + " | ".join(
            f"{y}: n/일 {len(x)/365:5.1f} 60s {(-side*x.m60).mean():+5.1f} 300s {(-side*x.m300).mean():+5.1f}" for y, x in by))
    P("\n[3] 폭발 중 가격이 많이 밀린 사건만 (rWn 상위/하위 20%, 방향 일치) — 과잉반응 되돌림?")
    for W in WS:
        for side in (-1, 1):
            g = E[(E.W == W) & (E.side == side) & (side * E.z >= 2.5)]
            q = g.rWn.quantile(0.2 if side < 0 else 0.8)
            gg = g[(g.rWn <= q) if side < 0 else (g.rWn >= q)]
            if len(gg) < 50: continue
            P(f"  W{W:2d} {'매도' if side<0 else '매수'} 큰 이동 하루 {len(gg)/days:5.1f} | " +
              " ".join(f"{H}s{(-side*gg[f'm{H}']).mean():+5.1f}" for H in HS))
