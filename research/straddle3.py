"""양방향 구조의 유망 영역 정밀 검증: 저변동 구간 + 저배율 + 장기 보유.
학습 2020~2023 / 테스트 2024~, 위약 300회, 실제 구조 손익.
사용: python research/straddle3.py"""
import sys, warnings
from pathlib import Path
import numba, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R))
from freesia import store
from freesia.bt import ind
from straddle import run
TE = pd.Timestamp("2024-01-01", tz="UTC")
rng = np.random.default_rng(83)
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)

for sym in ("BTCUSDT", "ETHUSDT"):
    k = store.load("klines_1m", sym).set_index("ts")
    idx = k.index; hi, lo, cl = k.high.values, k.low.values, k.close.values
    r1 = pd.Series(np.log(cl)).diff()
    rv60 = r1.rolling(60).std(); q60 = rv60.rolling(10080).rank(pct=True)
    rv1440 = r1.rolling(1440).std(); q1d = rv1440.rolling(43200).rank(pct=True)
    c5 = pd.Series(cl, index=idx).resample("5min").last(); h5 = pd.Series(hi, index=idx).resample("5min").max(); l5 = pd.Series(lo, index=idx).resample("5min").min()
    bm, bu, bl = ind.bbands(c5, 20, 2.0); km, ku, kl = ind.keltner(h5, l5, c5, 20, 1.5)
    sq = ((bu < ku) & (bl > kl)); fired = (sq.shift(1) & ~sq).reindex(idx, method="ffill").fillna(False).values
    T = {"저변동60 하위5%": (q60 < 0.05).values, "저변동60 하위2%": (q60 < 0.02).values,
         "저변동1일 하위10%": (q1d < 0.10).values, "저변동5%+스퀴즈": (q60 < 0.05).values & fired,
         "저변동1일10%+저변동60 5%": (q1d < 0.10).values & (q60 < 0.05).values}
    P("=" * 98 + f"\n### {sym}  (저변동+저배율 영역, 총 스테이크 대비 %)")
    rows = []
    for name, m in T.items():
        mm = np.nan_to_num(m, nan=0).astype(bool).copy(); mm[:50000] = False
        for L in (10, 15, 20, 25):
            for H, tgt, vd in ((4320, 0, 0), (10080, 0, 0), (20160, 0, 0), (10080, 2.0, 0), (10080, 0, 0.4)):
                out, hold = run(hi, lo, cl, mm, float(L), H, tgt, vd, 60)
                ok = ~np.isnan(out); ts = idx[ok]
                if ok.sum() < 40:
                    continue
                a = out[ok][ts < TE]; b_ = out[ok][ts >= TE]
                if len(a) < 20 or len(b_) < 15:
                    continue
                rows.append((name, L, H // 1440, tgt, vd, a.mean() * 100, float((a > 0).mean()), len(a),
                             b_.mean() * 100, float((b_ > 0).mean()), len(b_),
                             len(b_) / max((ts[ts >= TE][-1] - TE).days / 7, 1)))
    d = pd.DataFrame(rows, columns=["트리거", "배율", "H일", "목표", "감쇠", "학습%", "학습승률", "학습n", "테스트%", "테스트승률", "테스트n", "주당"])
    both = d[(d["학습%"] > 0) & (d["테스트%"] > 0)]
    P(f"설정 {len(d)}개 중 학습·테스트 모두 양수 {len(both)}개")
    P(d.sort_values("테스트%", ascending=False).head(14).round(2).to_string(index=False))
    if len(both):
        P("\n[학습·테스트 모두 양수]")
        P(both.sort_values("테스트%", ascending=False).round(2).to_string(index=False))
        # 위약: 같은 개수 무작위 진입
        r0 = both.iloc[0]
        L = float(r0.배율); H = int(r0.H일) * 1440
        nev = int(r0.테스트n)
        cand = np.where((idx >= TE))[0]; cand = cand[cand < len(cl) - H - 10]
        pls = []
        for _ in range(300):
            ev = np.zeros(len(cl), bool); ev[rng.choice(cand, min(nev * 4, len(cand)), replace=False)] = True
            o2, _ = run(hi, lo, cl, ev, L, H, float(r0.목표), float(r0.감쇠), 60)
            x = o2[~np.isnan(o2)]
            if len(x) > 10:
                pls.append(x.mean() * 100)
        P(f"  위약 {len(pls)}회 (배율 {L:.0f}, {int(r0.H일)}일): 평균 {np.mean(pls):.2f}% 95% {np.percentile(pls,95):.2f}% 최대 {np.max(pls):.2f}% | 실제 {r0['테스트%']:.2f}%")
        P(f"  → 백분위 {(np.array(pls) < r0['테스트%']).mean()*100:.1f}%")
