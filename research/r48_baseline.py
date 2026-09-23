"""R48 — 단기 방향 예측의 기준선: 라벨 생성 + 무조건 비율 + 단순 예측자 + 통계적 해상도.

라벨(이후 모든 방향 공통, data/r48_labels.parquet):
  15분 간격 시점 t(2020~), 시가 진입 가정. 12시간 안에 +d 먼저 = +1, −d 먼저 = −1, 둘 다 없음 = 0.
  d = 1%, 1.5%. (같은 분에 둘 다 닿으면 −1로: R47 구조의 보수적 처리와 동일)
기준선:
  [1] 무조건 비율(위 먼저·아래 먼저·무승부) — 연도·종목별
  [2] 단순 예측자의 적중률: 항상 롱 / 직전 1·4·12·24시간 수익 부호(추세) / 그 반대(되돌림)
      적중률 = 결판난 표본(±1) 중 방향 일치 비율. 확신 상위 범위는 |직전 수익| 상위 x% (서술용, 인과 문턱 아님)
  [3] 해상도: 주 3~6회로 3.7년 평가 시 적중률의 표준오차와 '65%'의 의미
사용: python research/r48_baseline.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
from r47_x30_frame import outcome, LIQ, CT, CM
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
H = 720; STEP = 15; DS = (0.01, 0.015)

rows = []
for s in ("BTCUSDT", "ETHUSDT"):
    k = store.load("klines_1m", s).set_index("ts")
    o, h, l, c = (k[x].values for x in ("open", "high", "low", "close"))
    idx = np.arange(1440, len(k) - H - 1, STEP)
    df = pd.DataFrame({"sym": s, "t": idx, "ts": k.index[idx]})
    for d in DS:
        _, kind, tt = outcome(o, h, l, c, h, l, idx, 1, d, 0.99, H, CT, CM)   # 청산 없음(라벨용), 롱 기준
        df[f"y{int(d*1000)}"] = np.where(kind == 1, 1, np.where(kind == -1, -1, 0)).astype("int8")
        df[f"tt{int(d*1000)}"] = tt.astype("int16")
    lc = np.log(c)
    for w in (60, 240, 720, 1440):
        df[f"r{w}"] = (lc[idx - 1] - lc[idx - 1 - w]).astype("float32")      # t 시가 직전까지
    rows.append(df)
L = pd.concat(rows, ignore_index=True)
L["yr"] = L.ts.dt.year
L.to_parquet(R / "data" / "r48_labels.parquet", index=False)
P(f"라벨 {len(L):,} 시점 (15분 간격, 2종목) 저장 data/r48_labels.parquet")

for d in DS:
    y = f"y{int(d*1000)}"
    P(f"\n===== ±{d*100:.1f}% / 12시간")
    P("[1] 무조건 비율: 위 먼저 / 아래 먼저 / 무승부  (결판 중 위 먼저)")
    for (s, yr), g in L.groupby(["sym", "yr"]):
        up, dn, nt = (g[y] == 1).mean(), (g[y] == -1).mean(), (g[y] == 0).mean()
        P(f"  {s[:3]} {yr}: {up:.3f} / {dn:.3f} / {nt:.3f}  ({up/(up+dn):.3f})")
    D = L[L[y] != 0]
    P(f"  전체 결판 중 위 먼저 {(D[y]==1).mean():.3f} (n={len(D):,}), 평균 결판 시간 위 {D[D[y]==1][f'tt{int(d*1000)}'].mean():.0f}분 아래 {D[D[y]==-1][f'tt{int(d*1000)}'].mean():.0f}분")
    P("[2] 단순 예측자 적중률 (결판 표본) — 전체 / 확신 상위 10%·3%·1%·0.3% (|직전 수익| 기준)")
    for w in (60, 240, 720, 1440):
        x = D[f"r{w}"]
        for lab, sgn in (("추세", 1), ("되돌림", -1)):
            acc = lambda m: ((np.sign(x[m]) * sgn) == D[y][m]).mean()
            parts = [f"{acc(np.ones(len(D), bool)):.3f}"]
            for q in (0.9, 0.97, 0.99, 0.997):
                m = (x.abs() >= x.abs().quantile(q)).values
                parts.append(f"{acc(m):.3f}")
            P(f"  직전 {w//60:2d}시간 {lab:3s}: " + " / ".join(parts))
    P(f"  항상 롱: {(D[y]==1).mean():.3f}")

P("\n[3] 해상도 — 평가 3.7년, 주 n회면 거래 수와 적중률 표준오차(참 적중률 60% 가정)")
for n in (3, 6):
    N = int(n * 52 * 3.7)
    se = np.sqrt(0.6 * 0.4 / N)
    P(f"  주 {n}회: {N}거래, 표준오차 ±{se*100:.1f}%p → 65% 관측의 95% 구간 {65-1.96*se*100:.1f}~{65+1.96*se*100:.1f}%")
