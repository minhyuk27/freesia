"""양방향 구조 최종 검증: 펀딩 반영 + 학습에서만 선택 → 테스트 1회 + 앵커드 워크포워드.
한쪽이 청산되면 살아남은 다리는 펀딩에 노출된다(숏 생존=수취, 롱 생존=지불).
사용: python research/straddle4.py"""
import sys, warnings
from pathlib import Path
import numba, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R))
from freesia import store
TE = pd.Timestamp("2024-01-01", tz="UTC")
FEE = 0.0005
rng = np.random.default_rng(97)
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)


@numba.njit(cache=True)
def runf(hi, lo, cl, fnd, ent, L, H):
    """펀딩 반영. fnd[j] = 해당 분의 펀딩률(정산 분에만 값). 숏 생존시 +, 롱 생존시 -."""
    n = len(cl); d = 1.0 / L - 0.004
    out = np.full(n, np.nan); hold = np.zeros(n, np.int32)
    nxt = 0
    for i in range(n - 2):
        if (not ent[i]) or i < nxt:
            continue
        p0 = cl[i]; ld = False; sd = False; dead = 0.0; fac = 0.0
        ex = min(i + H, n - 1)
        for j in range(i + 1, min(i + 1 + H, n)):
            up = hi[j] / p0 - 1.0; dn = 1.0 - lo[j] / p0
            if (not ld) and dn >= d:
                ld = True; dead -= 1.0
            if (not sd) and up >= d:
                sd = True; dead -= 1.0
            if ld and sd:
                ex = j; break
            if fnd[j] != 0.0:
                if (not sd) and ld:
                    fac += L * fnd[j]        # 숏만 생존 → 수취
                elif (not ld) and sd:
                    fac -= L * fnd[j]        # 롱만 생존 → 지불
            ex = j
        mv = cl[ex] / p0 - 1.0
        alive = 0.0; na = 0
        if not ld:
            alive += L * mv; na += 1
        if not sd:
            alive += -L * mv; na += 1
        fee = 2.0 * L * FEE + na * L * FEE
        out[i] = (dead + alive + fac - fee) / 2.0
        hold[i] = ex - i
        nxt = ex + 1
    return out, hold


for sym in ("BTCUSDT", "ETHUSDT"):
    k = store.load("klines_1m", sym).set_index("ts")
    idx = k.index; hi, lo, cl = k.high.values, k.low.values, k.close.values
    f = store.load("funding", sym).set_index("ts").rate
    fnd = pd.Series(0.0, index=idx); fnd.loc[fnd.index.intersection(f.index)] = f.reindex(fnd.index.intersection(f.index)).values
    fv = fnd.values
    r1 = pd.Series(np.log(cl)).diff()
    q60 = r1.rolling(60).std().rolling(10080).rank(pct=True).values
    q1d = r1.rolling(1440).std().rolling(43200).rank(pct=True).values
    T = {}
    for p_ in (0.02, 0.05, 0.10, 0.20):
        T[f"저변동60<{int(p_*100)}%"] = (q60 < p_)
        T[f"저변동1일<{int(p_*100)}%"] = (q1d < p_)
    P("=" * 98 + f"\n### {sym} — 펀딩 반영, 학습에서만 선택 → 테스트 1회")
    rows = []
    for name, m in T.items():
        mm = np.nan_to_num(m, nan=0).astype(bool).copy(); mm[:50000] = False
        for L in (10, 15, 20, 25, 40):
            for Hd in (7, 14, 30, 60):
                out, hold = runf(hi, lo, cl, fv, mm, float(L), Hd * 1440)
                ok = ~np.isnan(out); ts = idx[ok]
                a = out[ok][ts < TE]; b_ = out[ok][ts >= TE]
                if len(a) < 25 or len(b_) < 15:
                    continue
                rows.append((name, L, Hd, a.mean() * 100, float((a > 0).mean()), len(a),
                             b_.mean() * 100, float((b_ > 0).mean()), len(b_),
                             len(b_) / max((ts[ts >= TE][-1] - TE).days / 7, 1)))
    d = pd.DataFrame(rows, columns=["트리거", "배율", "H일", "학습%", "학습승률", "학습n", "테스트%", "테스트승률", "테스트n", "주당"])
    P(f"설정 {len(d)}개")
    top = d.sort_values("학습%", ascending=False).head(8)
    P("\n[학습 상위 8 → 그대로 테스트]")
    P(top.round(2).to_string(index=False))
    P(f"  학습 상위8의 테스트 평균 {top['테스트%'].mean():.2f}%, 양수 {int((top['테스트%']>0).sum())}/8")
    # 위약: 학습 상위 설정으로 무작위 진입
    r0 = top.iloc[0]; L = float(r0.배율); H = int(r0.H일) * 1440
    cand = np.where(idx >= TE)[0]; cand = cand[cand < len(cl) - H - 10]
    pls = []
    for _ in range(200):
        ev = np.zeros(len(cl), bool); ev[rng.choice(cand, min(int(r0.테스트n) * 4, len(cand)), replace=False)] = True
        o2, _ = runf(hi, lo, cl, fv, ev, L, H)
        x = o2[~np.isnan(o2)]
        if len(x) > 10:
            pls.append(x.mean() * 100)
    P(f"  위약 {len(pls)}회: 평균 {np.mean(pls):.2f}% 95% {np.percentile(pls,95):.2f}% 최대 {np.max(pls):.2f}% | 실제 {r0['테스트%']:.2f}% → 백분위 {(np.array(pls)<r0['테스트%']).mean()*100:.1f}%")
    # 앵커드 워크포워드
    P("\n[앵커드 워크포워드] 매년 직전까지 학습해 최고 설정 선택 → 그 해 평가")
    wf = []
    for y in range(2022, 2027):
        s = pd.Timestamp(f"{y}-01-01", tz="UTC"); e = pd.Timestamp(f"{y+1}-01-01", tz="UTC")
        best = None
        for name, m in T.items():
            mm = np.nan_to_num(m, nan=0).astype(bool).copy(); mm[:50000] = False
            for L in (10, 15, 20, 25, 40):
                for Hd in (7, 14, 30):
                    out, hold = runf(hi, lo, cl, fv, mm, float(L), Hd * 1440)
                    ok = ~np.isnan(out); ts = idx[ok]
                    tr = out[ok][ts < s]
                    if len(tr) < 25:
                        continue
                    if best is None or tr.mean() > best[0]:
                        best = (tr.mean(), name, L, Hd, out, ok, ts)
        if best is None:
            continue
        _, name, L, Hd, out, ok, ts = best
        yr = out[ok][(ts >= s) & (ts < e)]
        if len(yr) < 5:
            continue
        wf.append((y, name, L, Hd, yr.mean() * 100, float((yr > 0).mean()), len(yr)))
    P(pd.DataFrame(wf, columns=["연", "선택 트리거", "배율", "H일", "그해%", "승률", "n"]).round(2).to_string(index=False))
