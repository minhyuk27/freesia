"""R82 — 월 100%를 위한 '두 번째 독립 엣지' 탐색: z0.31과 다른 상황에서의 지정가 유동성 공급.

배경: z0.31(연 샤프 2.85)만으로는 켈리 최대 성장 ≈ 월 40%. 월 100%는 샤프 4.1+ 필요 → 상관 낮은 같은 품질 엣지가 필요.
엔진(1분봉): 조건이 켜진 분 t에 직전 종가 × (1 − k·σ) 지정가 매수(숏은 대칭)를 M분 유지. 체결 = 저가가 지정가보다 5bp 더 관통.
  청산: 진입가 × (1 + x·σ) 지정가 또는 T분 뒤 종가(테이커). σ = 직전 60분 1분 수익 표준편차 × √60 (1시간 변동성).
  비용: 진입·목표 메이커 2bp, 시간청산 테이커 5+2bp. 종목·조건마다 한 번에 한 포지션.
조건(상황)
  IDIO  ETH 15분 수익 − BTC 15분 수익 ≤ −1.5σ (ETH만 급락, 종목 고유 충격) → ETH 매수 / 반대면 BTC 매수
  FUND  펀딩 정산(00·08·16시 UTC) 전 30분 ~ 후 30분
  WKND  토·일
  HOUR  매시 정각 전 5분 ~ 후 10분
  OPEN  한국장(00:00 UTC) · 미국장(13:30 UTC) 개장 전후 30분
  ALL   조건 없음(기준선, 15분마다 한 번 걸기)
격자: k ∈ {1.5, 2.5, 3.5}, x ∈ {0.5, 1.0}, M = 15분, T ∈ {60, 240}분, 방향 롱(숏은 ALL만 확인).
판정: 거래당 순익, 일간 연 샤프(자본 = 명목 1 가정), z0.31 일간 손익(w1N4, 문턱 25bp·V3) 상관, 결합 샤프.
사용: python research/r82_independent_lp.py"""
import sys, warnings, itertools, time
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
CM, CT, PEN = 2e-4, 7e-4, 5e-4


@numba.njit(cache=True)
def lp(on, o, h, l, c, sig, side, k, x, M, T):
    """on: 조건 켜진 분(불리언). 반환 (진입 분, 청산 분, 순수익) 배열."""
    n = len(c); E = np.zeros(n // 10 + 10, np.int64); X = np.zeros_like(E); Rr = np.zeros(n // 10 + 10); m = 0; t = 61
    while t < n - T - M - 2:
        if not on[t] or np.isnan(sig[t]) or sig[t] <= 0:
            t += 1; continue
        ref = c[t - 1]; lim = ref * (1 - side * k * sig[t]); fill = -1
        for i in range(t, t + M):
            if (side > 0 and l[i] <= lim * (1 - PEN)) or (side < 0 and h[i] >= lim * (1 + PEN)):
                fill = i; break
        if fill < 0:
            t += M; continue
        e = lim; tp = e * (1 + side * x * sig[t]); r = np.nan; j = fill + 1
        while j < fill + T:
            if (side > 0 and h[j] >= tp) or (side < 0 and l[j] <= tp):
                r = side * (tp / e - 1) - 2 * CM_; break
            j += 1
        if np.isnan(r):
            j = fill + T; r = side * (c[j] / e - 1) - CM_ - CT_
        E[m] = fill; X[m] = j; Rr[m] = r; m += 1
        t = j + 1
    return E[:m], X[:m], Rr[:m]


CM_, CT_ = CM, CT


if __name__ == "__main__":
    t_ = time.time()
    K = {s: store.load("klines_1m", s).set_index("ts") for s in ("BTCUSDT", "ETHUSDT")}
    idx = K["BTCUSDT"].index.intersection(K["ETHUSDT"].index)
    for s in K: K[s] = K[s].reindex(idx).ffill()
    lr = {s: np.log(K[s].close).diff() for s in K}
    sig = {s: (lr[s].rolling(60).std() * np.sqrt(60)).shift(1).values for s in K}
    r15 = {s: (np.log(K[s].close) - np.log(K[s].close).shift(15)).values for s in K}
    hh, mm, wd = idx.hour.values, idx.minute.values, idx.weekday.values
    mins = hh * 60 + mm
    fund = np.zeros(len(idx), bool)
    for f in (0, 480, 960): fund |= (np.abs(((mins - f + 720) % 1440) - 720) <= 30)
    cond = {"FUND": fund, "WKND": wd >= 5, "HOUR": (mm >= 55) | (mm <= 10),
            "OPEN": (np.abs(((mins - 0 + 720) % 1440) - 720) <= 30) | (np.abs(((mins - 810 + 720) % 1440) - 720) <= 30),
            "ALL": (mm % 15 == 0)}
    spread = r15["ETHUSDT"] - r15["BTCUSDT"]; sref = pd.Series(sig["ETHUSDT"]) / 2
    # z0.31 일간 손익(비교용)
    import r40_regime_leverage as RL, r80_max_return as R80
    RL.MMR = 0.0065; W = R80.dip_frame(0.0025); V2 = np.clip(W.pred.values / 0.001, 0.5, 3)
    zd = R80.daily_from_eq(RL.sim(W, 1, 4, np.minimum(np.clip(W.vreg.values ** 1.5, 0.3, 4) * V2, 6), X=0.03, H=720)[0])
    zsh = zd.mean() / zd.std() * np.sqrt(365)
    P(f"z0.31(w1N4) 일간 연 샤프 {zsh:.2f} ({time.time()-t_:.0f}s)")
    rows = []
    for cname in ("IDIO", "FUND", "WKND", "HOUR", "OPEN", "ALL"):
        for k, x, T in itertools.product((1.5, 2.5, 3.5), (0.5, 1.0), (60, 240)):
            sides = (1, -1) if cname == "ALL" else (1,)
            for side in sides:
                trades = []
                for s in ("BTCUSDT", "ETHUSDT"):
                    if cname == "IDIO":
                        other = "BTCUSDT" if s == "ETHUSDT" else "ETHUSDT"
                        on = ((r15[s] - r15[other]) <= -1.5 * np.nan_to_num(sig[s])) & (mm % 5 == 0)
                    else:
                        on = cond[cname]
                    k_ = K[s]
                    E, X, Rr = lp(on, k_.open.values, k_.high.values, k_.low.values, k_.close.values, np.nan_to_num(sig[s], nan=-1), side, k, x, 15, T)
                    trades.append(pd.DataFrame({"t": idx[X], "r": Rr, "sym": s}))
                Tt = pd.concat(trades)
                if len(Tt) < 30: continue
                d = Tt.groupby(Tt.t.dt.floor("D")).r.sum() * 0.5
                d = d.reindex(pd.date_range(idx[0].floor("D"), idx[-1].floor("D"), freq="D")).fillna(0)
                sh = d.mean() / d.std() * np.sqrt(365) if d.std() > 0 else 0
                dz = d[d.index >= zd.index[0]]; zz = zd.reindex(dz.index).fillna(0)
                corr = np.corrcoef(dz.values, zz.values)[0, 1] if dz.std() > 0 else np.nan
                # 결합: 각 흐름을 샤프 비례로 단순 합(변동성 정규화 동일 위험) → 결합 샤프
                a = dz / dz.std() if dz.std() > 0 else dz; b = zz / zz.std()
                comb = (a + b); csh = comb.mean() / comb.std() * np.sqrt(365)
                yr = Tt.groupby(Tt.t.dt.year).r.mean() * 1e4
                rows.append((cname, k, x, T, "롱" if side > 0 else "숏", len(Tt), len(Tt) / ((idx[-1] - idx[0]).days), Tt.r.mean() * 1e4, (Tt.r > 0).mean(),
                             sh, corr, csh, " ".join(f"{y%100}:{v:+.0f}" for y, v in yr.items())))
        P(f"  {cname} 완료 ({time.time()-t_:.0f}s)")
    O = pd.DataFrame(rows, columns=["상황", "k", "x", "T", "방향", "n", "하루", "거래당bp", "승률", "샤프", "z031상관", "결합샤프", "연도별bp"])
    O.to_csv(R / "logs" / "r82.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    P(f"\n===== 구성 {len(O)}개 — 샤프 상위 20 (z0.31 단독 {zsh:.2f}, 목표 결합 샤프 ≥ 4)")
    P(O.sort_values("샤프", ascending=False).head(20).round(2).to_string(index=False))
    P("\n  상황별 최고 샤프"); P(O.sort_values("샤프", ascending=False).groupby("상황").head(1).round(2).to_string(index=False))
    (R / "logs" / "r82.summary").write_text(f"R82 독립 유동성 공급 엣지 탐색 완료. 최고 샤프 {O.샤프.max():.2f}, 최고 결합 샤프 {O.결합샤프.max():.2f}\n")
