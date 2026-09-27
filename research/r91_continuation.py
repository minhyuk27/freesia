"""R91 — '초기 추세 따라가기'의 근본 검증: 타이밍 신호 뒤 가격이 기준가(c[t−1])에서 먼저 ±y 닿으면,
그 방향으로 기준가 ±2%까지 가는 확률(기준가로 되돌아오기 전) vs 무작위보행 이론값 y/2%.
매매: 닿은 분에 스톱 진입(갭 반영), 목표 = 기준가 ±2%(지정가 2bp), 손절 = 기준가 ± s·y (되돌림), 창 240분 + 보유 최대 240분(시간 청산 테이커).
y ∈ {0.5, 0.75, 1.0, 1.25, 1.5}%, 손절 위치 s ∈ {0(기준가), 0.5(절반 되돌림)}, 타이밍 상위 5/10/20/30/100%.
사용: python research/r91_continuation.py"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, numba
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R))
from freesia import store
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
CT, CM = 7e-4, 2e-4


@numba.njit(cache=True)
def run(t, on, o, h, l, c, y, s, W, H):
    n = len(c); Rr = np.zeros(len(t)); E = np.zeros(len(t), np.int64); m = 0; busy = -1
    for q in range(len(t)):
        i = t[q]
        if not on[q] or i <= busy or i + W + H >= n: continue
        ref = c[i - 1]; f = -1; sd = 0; px = 0.0
        for j in range(i, i + W):
            if h[j] >= ref * (1 + y) and l[j] <= ref * (1 - y): break
            if h[j] >= ref * (1 + y): f = j; sd = 1; px = max(ref * (1 + y), o[j]); break
            if l[j] <= ref * (1 - y): f = j; sd = -1; px = min(ref * (1 - y), o[j]); break
        if f < 0: continue
        tgt = ref * (1 + sd * 0.02); stp = ref * (1 + sd * y * s); r = np.nan; j = f + 1
        if (sd > 0 and px >= tgt) or (sd < 0 and px <= tgt): continue
        while j < f + H:
            if sd > 0:
                if l[j] <= stp: r = min(stp, o[j]) / px - 1 - 2 * CT; break
                if h[j] >= tgt: r = tgt / px - 1 - CT - CM; break
            else:
                if h[j] >= stp: r = -(max(stp, o[j]) / px - 1) - 2 * CT; break
                if l[j] <= tgt: r = -(tgt / px - 1) - CT - CM; break
            j += 1
        if np.isnan(r): r = sd * (c[j - 1] / px - 1) - 2 * CT
        Rr[m] = r; E[m] = f; m += 1; busy = j
    return E[:m], Rr[:m]


if __name__ == "__main__":
    W_ = pd.read_parquet(R / "data" / "r90_wf.parquet").sort_values(["sym", "t"])
    K = {s: store.load("klines_1m", s) for s in ("BTCUSDT", "ETHUSDT")}
    days = (W_.ts.max() - W_.ts.min()).days
    P("y·손절 / 타이밍 상위 q: 거래 수(하루) | 목표 도달률 vs 무작위보행 이론 | 거래당 순bp | 연도별 bp")
    for y in (0.005, 0.0075, 0.01, 0.0125, 0.015):
        for s in (0.0, 0.5):
            th = (2 - y * 100) and (y - y * s) / (0.02 - y * s)      # 무작위보행: 손절까지 거리/(전체 폭)
            for q in (5, 10, 20, 30, 100):
                parts = []
                for sym in K:
                    g = W_[W_.sym == sym]; on = (g.p >= g[f"th{q}"]).values if q < 100 else np.ones(len(g), bool); k = K[sym]
                    E, Rr = run(g.t.values.astype(np.int64), on, k.open.values, k.high.values, k.low.values, k.close.values, y, s, 240, 240)
                    parts.append(pd.DataFrame({"t": pd.to_datetime(k.ts.values[E], utc=True), "r": Rr}))
                T = pd.concat(parts)
                if len(T) < 30: continue
                hit = (T.r > 0.02 - y - 0.003).mean()
                yr = T.groupby(T.t.dt.year).r.mean() * 1e4
                P(f"  y={y*100:.2f}% 손절{'기준가' if s == 0 else '절반'} 상위{q:3d}%: {len(T):5d}({len(T)/days:4.2f}/일) | 목표 {hit:.3f} vs 이론 {th:.3f} | {T.r.mean()*1e4:+6.1f}bp | " +
                  " ".join(f"{a%100}:{b:+.0f}" for a, b in yr.items()))
    (R / "logs" / "r91.summary").write_text("R91 초기 추세 지속 확률 완료\n")
