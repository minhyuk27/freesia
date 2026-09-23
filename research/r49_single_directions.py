"""R49 — 방향 A·B·C·E 단독 검증: 각 정보가 '확신 쪽'에서 ±1%/±1.5% 먼저 도달을 얼마나 맞히나.

특징표: R46 후보표(15분 격자, R27·R32 특징을 t−1 분까지) ⋈ R48 라벨 (sym, t).
절차(특징 하나씩):
  학습(2020~2022)에서 분위 경계를 정하고, 평가(2023~2026)에서
  상단 q% → 롱 예측 / 하단 q% → 숏 예측 을 각각 채점 (q = 3%, 1%, 0.3%). 방향이 반대인 해석(역발상)도 함께 본다.
  적중률 = 결판 표본(±1) 중 일치 비율. 주당 신호 수도 보고.
  특징 묶음: A 사건(급락·청산·펀딩·OI·프리미엄) / B 호가 / C 포지셔닝 / E 추세·거래량 / 기타
사건 직접 채점:
  A0 z0.31 선택 딥 체결(R32 워크포워드 예측>10bp) → 롱, 체결 시각 기준 라벨
  E0 R41 돌파(3일 고점 + 거래대금 2배) → 롱 / 3일 저점 이탈 → 숏
사용: python research/r49_single_directions.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
from r47_x30_frame import outcome, CT, CM
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
SPLIT = 2023
L = pd.read_parquet(R / "data" / "r48_labels.parquet")
C = pd.read_parquet(R / "data" / "r46_cands.parquet")
C = C[C.side == 1].drop(columns=[c for c in C.columns if c.startswith("m") or c.startswith("liq") or c in ("side", "r12", "yr")])
D = L.merge(C.drop(columns=["ts"]), on=["sym", "t"], how="inner").replace([np.inf, -np.inf], np.nan)
P(f"결합 {len(D):,} 시점, 특징 {len(C.columns)-3}개")
GROUP = {
    "A": ["ret5", "ret15", "ret60", "pre", "dlo60", "dlo240", "dlo1440", "accel", "oi15", "oi60", "oi1440", "fund", "prem", "premz",
          "dprem5", "dprem30", "vol5", "trd5", "rvshort", "piv", "swept", "absorb"],
    "B": ["bslope", "bimb"],
    "C": ["tlsr", "lsr", "quad"],
    "E": ["ret240", "ret1440", "pos60", "pos240", "pos1440", "vol60", "sell5", "sell60", "cvd5", "cvd30", "cvddiv", "avgsz5", "rvreg",
          "vpin15", "vpin60", "wick5"],
}
WEEKS = (D.ts.max() - pd.Timestamp("2023-01-01", tz="UTC")).days / 7


def score(y, f):
    tr = D[D.yr < SPLIT]; te = D[(D.yr >= SPLIT) & (D[y] != 0)]
    if f in ("bslope", "bimb"):                               # 호가는 2023~ 만 존재 → 2023 경계, 2024~ 평가
        tr = D[D.yr == 2023]; te = D[(D.yr >= 2024) & (D[y] != 0)]
    x = tr[f].dropna()
    if x.nunique() < 20 or te[f].notna().sum() < 1000: return []
    out = []
    wk = (te.ts.max() - te.ts.min()).days / 7
    for q in (0.03, 0.01, 0.003):
        hi, lo = x.quantile(1 - q), x.quantile(q)
        up = te[te[f] >= hi]; dn = te[te[f] <= lo]
        for tail, g, pred in (("상단", up, 1), ("하단", dn, -1)):
            if len(g) < 30: continue
            acc = (g[y] == pred).mean()
            yrs = g.groupby("yr")[y].apply(lambda v: (v == pred).mean())
            # 원래 방향(acc)과 역방향(1-acc) 중 좋은 쪽을 기록하되 방향 표시
            best = max(acc, 1 - acc); dirn = "순" if acc >= 0.5 else "역"
            ymin = (yrs if acc >= 0.5 else 1 - yrs).min()
            out.append((best, f, q, tail, dirn, len(g), len(g) / wk, ymin))
    return out


for y in ("y10", "y15"):
    P(f"\n===== 라벨 {y} (±{int(y[1:])/10:.1f}%, 12시간) — 평가 2023~ (호가는 2024~), 결판 표본 적중률")
    for grp, fs in GROUP.items():
        res = []
        for f in fs:
            if f in D.columns: res += score(y, f)
        res.sort(reverse=True)
        P(f"[{grp}] 상위 8 (적중률, 특징, 꼬리 q, 쪽, 순/역, n, 주당, 연도 최저)")
        for r in res[:8]:
            P(f"   {r[0]:.3f}  {r[1]:9s} q{r[2]*100:4.1f}% {r[3]} {r[4]} n={r[5]:5d} 주{r[6]:5.1f} 최저해 {r[7]:.3f}")

P("\n[A0] z0.31 선택 딥 체결 → 롱 (체결 분 시가 진입 가정, 2022~ 워크포워드 선택)")
W = pd.read_parquet(R / "data" / "r32_wf.parquet")
W["ts"] = pd.DatetimeIndex(W.ts)
for th in (0.001, 0.002, 0.004):
    S = W[W.pred > th]
    accs = {}
    for d in (0.01, 0.015):
        ys = []
        for s, g in S.groupby("sym"):
            k = store.load("klines_1m", s)
            _, kind, _ = outcome(k.open.values, k.high.values, k.low.values, k.close.values, k.high.values, k.low.values,
                                 g.f.values.astype(np.int64) + 1, 1, d, 0.99, 720, CT, CM)
            ys.append(pd.DataFrame({"yr": g.ts.dt.year.values, "y": kind}))
        Y = pd.concat(ys); Yd = Y[Y.y != 0]
        accs[d] = (Yd.y == 1).mean(), len(Yd), Yd.groupby("yr").y.apply(lambda v: (v == 1).mean()).round(3).to_dict()
    epi = S.assign(h=S.ts.dt.floor("15min")).drop_duplicates(["sym", "h"])
    wk = (S.ts.max() - S.ts.min()).days / 7
    P(f"  예측>{th*1e4:.0f}bp: 사건 주 {len(epi)/wk:.1f}회 | ±1% 적중 {accs[0.01][0]:.3f} (n={accs[0.01][1]}) {accs[0.01][2]} | "
      f"±1.5% {accs[0.015][0]:.3f} {accs[0.015][2]}")

P("\n[E0] 3일 고점 돌파(거래대금 2배) → 롱 / 3일 저점 이탈 → 숏, 2023~")
for s in ("BTCUSDT", "ETHUSDT"):
    k = store.load("klines_1m", s).set_index("ts")
    c = k.close; qv = k.quote_volume
    hi = k.high.rolling(4320).max().shift(1); lo = k.low.rolling(4320).min().shift(1)
    vok = qv.rolling(60).mean() / qv.rolling(1440).mean() >= 2
    for lab, sig, side in (("돌파 롱", (c > hi) & vok, 1), ("이탈 숏", (c < lo) & vok, -1)):
        pos = np.where(sig.values)[0]
        keep = []; last = -10**9
        for p in pos:
            if p - last >= 720 and k.index[p] >= pd.Timestamp("2023-01-01", tz="UTC") and p + 722 < len(k):
                keep.append(p + 1); last = p
        keep = np.array(keep, np.int64)
        for d in (0.01, 0.015):
            _, kind, _ = outcome(k.open.values, k.high.values, k.low.values, k.close.values, k.high.values, k.low.values,
                                 keep, side, d, 0.99, 720, CT, CM)
            dec = kind != 0
            P(f"  {s[:3]} {lab} ±{d*100:.1f}%: 적중 {(kind[dec]==1).mean():.3f} n={dec.sum()} (주 {len(keep)/ (3.7*52):.2f})")
