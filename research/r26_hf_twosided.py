"""R26 — 고빈도 양방향 유동성 공급 전면 스캔.

질문: '미리 지정가를 깔고 사건이 오게 한다'(z0.3·z0.14의 유일한 작동 원리)를
      하루 여러 번 거래하는 빈도로 끌어올릴 수 있는가.
새로 보는 축 (이전 연구에서 안 본 것):
  ① 양방향 — 스파이크 매도(숏) 지정가. z0.14는 롱 전용이었다
  ② 지정가 청산(TP) — 이전 출구는 전부 봉 종가 시장가(테이커 5bp)
  ③ 봉 길이 3~60분 전 구간을 양방향으로
규칙: 봉 b의 지정가는 봉 시가와 직전 분까지의 rv만 사용(룩어헤드 없음).
      체결은 지정가를 eps만큼 관통해야 인정, 체결 분에는 청산 불가(같은 분 왕복 불인정).
사용: python research/r26_hf_twosided.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
SYM = ("BTCUSDT", "ETHUSDT")
EPS = 0.0002; FM = 2.0; FT = 5.0
D = {}
for s in SYM:
    k = store.load("klines_1m", s).set_index("ts")
    c = k.close.values
    lr = pd.Series(np.log(c)).diff()
    D[s] = dict(o=k.open.values, h=k.high.values, l=k.low.values, c=c, idx=k.index,
                yr=k.index.year.values.astype(np.int64),
                rv1=lr.rolling(300).std().shift(1).values)
    P(f"{s}: {len(k):,}분 {k.index[0]} ~ {k.index[-1]}")
YEARS = (k.index[-1] - k.index[0]).days / 365.25


@numba.njit(cache=True)
def run(o, h, l, c, rv1, B, side, kx, mode, g, H, eps, fm, ft):
    """side +1 롱(시가 아래 매수) / -1 숏(시가 위 매도).
    mode 0 = 봉 종가 테이커 청산, 1 = TP 지정가(진입가→시가 방향 g 비율 회복) 미체결 시 종가.
    H = 봉 끝 이후 추가 보유 봉 수(TP 대기 연장)."""
    n = len(c) // B
    ei = np.full(n, -1, np.int64); ex = np.full(n, -1, np.int64)
    rr = np.zeros(n); mm = np.zeros(n); k = 0
    for b in range(n):
        s0 = b * B; e = s0 + B * (1 + H)
        if e > len(c): break
        v = rv1[s0]
        if np.isnan(v) or v <= 0: continue
        A = o[s0]; x = kx * v * np.sqrt(B)
        T = A * (1.0 - side * x)
        f = -1
        for t in range(s0, s0 + B):
            if side > 0:
                if l[t] <= T * (1 - eps): f = t; break
            else:
                if h[t] >= T * (1 + eps): f = t; break
        if f < 0: continue
        TP = T + g * (A - T)
        adv = l[f] if side > 0 else h[f]
        px = -1.0; mk = 0; hit = -1
        for t in range(f + 1, e):
            if mode == 1:
                if side > 0 and h[t] >= TP * (1 + eps):
                    px = TP; mk = 1; hit = t; break
                if side < 0 and l[t] <= TP * (1 - eps):
                    px = TP; mk = 1; hit = t; break
            if side > 0:
                if l[t] < adv: adv = l[t]
            else:
                if h[t] > adv: adv = h[t]
        if hit < 0:
            hit = e - 1; px = c[e - 1]
        fee = (fm + fm) if mk == 1 else (fm + ft)
        if side > 0:
            rr[k] = px / T - 1.0 - fee / 1e4; mm[k] = 1.0 - adv / T
        else:
            rr[k] = 1.0 - px / T - fee / 1e4; mm[k] = adv / T - 1.0
        ei[k] = f; ex[k] = hit; k += 1
    return ei[:k], ex[:k], rr[:k], mm[:k]


def events(s, B, side, kx, mode=0, g=1.0, H=0, eps=EPS, fm=FM, ft=FT):
    d = D[s]
    a, b, r, m = run(d["o"], d["h"], d["l"], d["c"], d["rv1"], B, side, kx, mode, g, H, eps, fm, ft)
    return a, b, r, m, d["yr"][a]


def line(lab, parts):
    """parts: [(a,b,r,m,yr), ...] 종목 합산 요약"""
    r = np.concatenate([p[2] for p in parts]); yr = np.concatenate([p[4] for p in parts])
    m = np.concatenate([p[3] for p in parts])
    if len(r) < 50:
        P(f"  {lab:44s} 표본부족({len(r)})"); return None
    by = {y: r[yr == y] for y in range(2020, 2027)}
    ys = " ".join(f"{str(y)[2:]}{by[y].mean()*1e4:+5.1f}" if len(by[y]) else f"{str(y)[2:]}  nan" for y in by)
    P(f"  {lab:44s} n={len(r):6d}({len(r)/YEARS:6.0f}/년) {r.mean()*1e4:+6.2f}bp 승률{(r>0).mean():.3f} "
      f"MAE95 {np.percentile(m,95)*100:4.2f}% | {ys}")
    return r.mean(), len(r) / YEARS


if __name__ == "__main__":
    P("\n[1] 봉 길이 × 방향 × 깊이 — 종가 테이커 청산 (z0.14의 양방향·전 봉 확장)")
    res = []
    for B in (3, 5, 10, 15, 30, 60):
        for side in (1, -1):
            for kx in (2.0, 3.0, 4.0, 5.0, 6.0, 8.0):
                out = line(f"{B:2d}분 {'롱' if side>0 else '숏'} rv×{kx}",
                           [events(s, B, side, kx) for s in SYM])
                if out: res.append((out[0], out[1], B, side, kx))

    P("\n[2] 지정가 청산(TP) — 진입가에서 시가 방향으로 g 비율 회복 시 메이커 청산, 아니면 종가")
    for B in (5, 10, 15, 30):
        for side in (1, -1):
            for kx in (3.0, 4.0, 6.0):
                for g in (0.25, 0.5, 1.0):
                    for H in (0, 1):
                        line(f"{B:2d}분 {'롱' if side>0 else '숏'} rv×{kx} TP{g} +{H}봉",
                             [events(s, B, side, kx, mode=1, g=g, H=H) for s in SYM])

    P("\n[3] 비용 0 기준 순수 엣지 (구조 확인용: 수수료·관통 제거)")
    for B in (5, 15):
        for side in (1, -1):
            for kx in (2.0, 4.0, 6.0):
                line(f"{B:2d}분 {'롱' if side>0 else '숏'} rv×{kx} 비용0",
                     [events(s, B, side, kx, eps=0.0, fm=0.0, ft=0.0) for s in SYM])
