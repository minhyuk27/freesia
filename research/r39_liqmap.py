"""R39 — 추정 청산 지도(liquidation heatmap) 재구성과 '자석·스윕 반전' 검증.

재구성 (5분, 2021-12~, 인과적):
  OI 증가분 ΔOI>0 = 새 롱 ΔOI + 새 숏 ΔOI 가 그 5분 종가 P에서 생겼다고 본다.
  레버리지 분포(10·20·50·100배, 가중 .35·.30·.20·.15)로 청산가를 계산해 로그가격 격자(0.1%)에 쌓는다.
    롱 청산가 = P×(1−1/L+mmr), 숏 청산가 = P×(1+1/L−mmr)
  OI 감소분은 전체 질량을 같은 비율로 줄인다. 반감기 7일로 자연 감소(포지션 정리).
  5분 봉 저가 이하의 롱 청산 질량, 고가 이상의 숏 청산 질량은 '쓸렸다'고 보고 제거·기록.
특징 (시점 t 봉 종료 후):
  dnL_k = 현재가 아래 k% 안의 롱 청산 질량 / 현재 OI, upS_k = 위 k% 안의 숏 청산 질량 / OI (k=1,2,4)
  asym_k = (upS_k − dnL_k) / (upS_k + dnL_k)   ← +면 위쪽(숏 청산)이 두텁다 → 자석 가설상 상승
  swL_h, swS_h = 직전 h분(15·60) 동안 쓸린 롱/숏 청산 질량 / OI  ← 스윕 반전 가설
목표: 다음 15·60·240분 수익. 5분위 경계는 2022만, 평가 2023~2026.
사용: python research/r39_liqmap.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R))
from freesia import store
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
LEV = np.array([10.0, 20.0, 50.0, 100.0]); WGT = np.array([0.35, 0.30, 0.20, 0.15]); MMR = 0.004
BIN = 0.001; HL = 7 * 288


@numba.njit(cache=True)
def build(lp, lh, ll, oi, g0, nb, lev, wgt, mmr, binw, hl):
    n = len(lp)
    mL = np.zeros(nb); mS = np.zeros(nb)
    dec = 0.5 ** (1.0 / hl)
    K = (1, 2, 4)
    out = np.full((n, 8), np.nan)       # dnL1 dnL2 dnL4 upS1 upS2 upS4 swL swS
    for t in range(1, n):
        if np.isnan(oi[t]) or np.isnan(oi[t - 1]) or oi[t - 1] <= 0:
            continue
        mL *= dec; mS *= dec
        d = oi[t] - oi[t - 1]
        if d < 0:
            f = max(0.0, 1.0 + d / oi[t - 1]); mL *= f; mS *= f
        else:
            for j in range(len(lev)):
                bl = int((lp[t] + np.log(1 - 1 / lev[j] + mmr) - g0) / binw)
                bs = int((lp[t] + np.log(1 + 1 / lev[j] - mmr) - g0) / binw)
                if 0 <= bl < nb: mL[bl] += d * wgt[j]
                if 0 <= bs < nb: mS[bs] += d * wgt[j]
        lo = int((ll[t] - g0) / binw); hi = int((lh[t] - g0) / binw)
        cur = int((lp[t] - g0) / binw)
        sw = 0.0
        for b in range(max(lo, 0), min(cur + 1, nb)):
            sw += mL[b]; mL[b] = 0.0
        out[t, 6] = sw / oi[t]
        sw = 0.0
        for b in range(max(cur, 0), min(hi + 1, nb)):
            sw += mS[b]; mS[b] = 0.0
        out[t, 7] = sw / oi[t]
        for q in range(3):
            w = int(K[q] * 0.01 / binw)
            a = 0.0; c = 0.0
            for b in range(max(cur - w, 0), cur):
                a += mL[b]
            for b in range(cur + 1, min(cur + w + 1, nb)):
                c += mS[b]
            out[t, q] = a / oi[t]; out[t, 3 + q] = c / oi[t]
    return out


def quint(df, sig, tgt, lab):
    tr, te = df[df.ts.dt.year == 2022], df[df.ts.dt.year >= 2023]
    x = tr[sig].replace([np.inf, -np.inf], np.nan).dropna()
    if x.nunique() < 5: return
    qs = np.unique(np.nanquantile(x, [0.1, 0.3, 0.5, 0.7, 0.9]))
    f = lambda d: [d[tgt][np.digitize(d[sig], qs) == i].mean() * 1e4 for i in range(len(qs) + 1)]
    a, b = f(tr), f(te)
    by = te.groupby(te.ts.dt.year).apply(lambda d: d[tgt][np.digitize(d[sig], qs) == len(qs)].mean() * 1e4
                                          - d[tgt][np.digitize(d[sig], qs) == 0].mean() * 1e4)
    P(f"  {lab:34s} 2022 " + " ".join(f"{v:+6.1f}" for v in a) + " | 평가 " + " ".join(f"{v:+6.1f}" for v in b)
      + " | 연도별 극단차 " + " ".join(f"{y%100}{v:+.1f}" for y, v in by.items()))


if __name__ == "__main__":
    for s in ("BTCUSDT", "ETHUSDT"):
        k = store.load("klines_1m", s).set_index("ts")
        k5 = k.resample("5min").agg({"high": "max", "low": "min", "close": "last"}).dropna()
        m = store.load("metrics_5m", s).set_index("ts")
        m.index = m.index + pd.Timedelta(minutes=5)          # 5분 구간 종료 후 확정
        oi = m.sum_open_interest.reindex(k5.index)
        k5 = k5[oi.notna().cumsum() > 0]; oi = oi.reindex(k5.index)
        lp = np.log(k5.close.values); lh = np.log(k5.high.values); ll = np.log(k5.low.values)
        g0 = np.log(k5.low.min()) - 0.5; nb = int((np.log(k5.high.max()) + 0.5 - g0) / BIN) + 1
        F = build(lp, lh, ll, oi.values.astype(float), g0, nb, LEV, WGT, MMR, BIN, HL)
        D = pd.DataFrame(F, index=k5.index, columns=["dnL1", "dnL2", "dnL4", "upS1", "upS2", "upS4", "swL", "swS"])
        for kk in (1, 2, 4):
            D[f"asym{kk}"] = (D[f"upS{kk}"] - D[f"dnL{kk}"]) / (D[f"upS{kk}"] + D[f"dnL{kk}"])
        for h in (3, 12):
            D[f"swL{h*5}"] = D.swL.rolling(h).sum(); D[f"swS{h*5}"] = D.swS.rolling(h).sum()
        D["swNet60"] = D.swS60 - D.swL60
        c = k5.close
        for H in (3, 12, 48):
            D[f"f{H*5}"] = np.log(c.shift(-H) / c)
        D["ts"] = D.index
        D = D[D.ts >= "2022-01-01"].dropna(subset=["f15"])
        P(f"\n===== {s} ({len(D):,} 5분 시점) — 평균 청산질량/OI: 아래 2% {D.dnL2.mean():.3f}, 위 2% {D.upS2.mean():.3f}")
        P("[자석] 비대칭(+ = 위쪽 숏청산 두꺼움) → 이후 수익 (6구간, bp)")
        for kk in (1, 2, 4):
            for H in (15, 60, 240):
                quint(D, f"asym{kk}", f"f{H}", f"asym{kk} → {H}분")
        P("[스윕 반전] 직전 롱청산 쓸림(swL) 많을수록 → 이후 수익 (+면 반등)")
        for sg in ("swL15", "swL60", "swS15", "swS60", "swNet60"):
            for H in (15, 60, 240):
                quint(D, sg, f"f{H}", f"{sg} → {H}분")
        D.to_parquet(R / "data" / f"r39_liq_{s}.parquet")
