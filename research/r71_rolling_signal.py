"""R71 — 매 4시간마다 최근 30일로 측정해 '유의미하게 큰' 롱·숏 신호가 나올 때만 거래 (R70의 주 단위 → 4시간 단위).

시점 t = 매 4시간 봉 종가 (2020-03 ~ 2026-09, BTC·ETH 각각 신호, 통계는 두 종목 합산).
창 = 봉 종가 t' ∈ [t−30일, t] 이고 라벨이 t까지 확정된 표본(t' + h봉 ≤ t). 최근 7일 = 창 중 t' ≥ t−7일.
신호 종류와 '유의미' 기준
  F1 캔들 시퀀스: 현재 최근 3봉 방향 코드(8)와 같은 창 표본의 미래수익 t값 ≥ τ (n ≥ 10), 최근 7일 같은 코드 평균이 같은 부호(있으면)
  F2 단일 지표: 91개 지표 중 창에서 |순위상관| 최대인 지표, |ρ|·√n ≥ τ+1 (91개 중 최대를 고르므로 문턱을 1 올림),
      현재 값이 창의 상·하위 10%이면 부호 방향, 최근 7일 부분 상관이 같은 부호
  F3 로지스틱(L2 C=0.1, 창 표준화): 현재 예측 확률 p, |p−0.5| ≥ τ·0.05 이고 최근 7일 창 내 적중률 > 0.5
  F4 유사 모양: 최근 6봉 수익/ATR 벡터의 창 kNN(k=20) 미래수익 t값 ≥ τ, 이웃 중 최근 7일 것들의 평균이 같은 부호(있으면)
τ ∈ {2, 2.5, 3}. 목표 h = 1봉(4시간), 6봉(24시간). 비용 왕복 14bp.
위약: 각 종목의 미래수익을 무작위 시간 이동(순환)해 관계를 끊은 뒤 같은 절차.
계좌(최선 구성): 레버리지 1·3·5, 포지션 = 자본 × 레버리지 / (h × 2종목).
사용: python research/r71_rolling_signal.py"""
import sys, warnings, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
from r68_bar_patterns import build, COST
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
BAR = pd.Timedelta("4h"); TAUS = (2.0, 2.5, 3.0)


def tstat(x):
    n = len(x)
    if n < 2: return 0.0
    s = x.std(ddof=1)
    return x.mean() / (s / np.sqrt(n)) if s > 0 else 0.0


def run(D, FE, ycol, H):
    t_all = D.t.values; order = np.argsort(t_all, kind="stable"); D = D.iloc[order].reset_index(drop=True)
    tv = D.t.values.astype("datetime64[ns]").astype(np.int64); y = D[ycol].values
    code = ((D.G1_r0 > 0).astype(int) * 4 + (D.G1_r1 > 0).astype(int) * 2 + (D.G1_r2 > 0).astype(int)).values
    shape = np.nan_to_num(D[[f"G1_r{k}" for k in range(6)]].values)
    X = D[FE].values
    ns = lambda d: int(pd.Timedelta(d).value)
    times = np.unique(tv); times = times[times >= tv[0] + ns("35D")]
    out = []
    for T in times:
        cur = np.where(tv == T)[0]
        lo = np.searchsorted(tv, T - ns("30D")); hi = np.searchsorted(tv, T - H * ns("4h"), side="right")
        if hi - lo < 150: continue
        w = np.arange(lo, hi); yw = y[w]; rec = tv[w] >= T - ns("7D")
        # F1
        for i in cur:
            m = code[w] == code[i]
            if m.sum() >= 10:
                ts = tstat(yw[m]); r7 = yw[m & rec]
                ok7 = len(r7) == 0 or np.sign(r7.mean()) == np.sign(ts)
                out.append(("F1", i, np.sign(ts), abs(ts), ok7))
        # F4
        for i in cur:
            dd = ((shape[w] - shape[i]) ** 2).sum(1); nn = np.argpartition(dd, 20)[:20]
            ts = tstat(yw[nn]); r7 = yw[nn][rec[nn]]
            ok7 = len(r7) == 0 or np.sign(r7.mean()) == np.sign(ts)
            out.append(("F4", i, np.sign(ts), abs(ts), ok7))
        # F2
        Xw = X[w]; good = ~np.isnan(Xw).any(0) & (np.nanstd(Xw, 0) > 0)
        if good.sum() > 0:
            Rx = np.apply_along_axis(rankdata, 0, Xw[:, good]); Ry = rankdata(yw)
            Rx = (Rx - Rx.mean(0)) / Rx.std(0); Ryz = (Ry - Ry.mean()) / Ry.std()
            rho = (Rx * Ryz[:, None]).mean(0); j = np.argmax(np.abs(rho)); f = np.where(good)[0][j]
            z = abs(rho[j]) * np.sqrt(len(w))
            q10, q90 = np.quantile(Xw[:, f], [0.1, 0.9])
            rr = rec.sum()
            rho7 = np.corrcoef(Xw[rec, f], yw[rec])[0, 1] if rr >= 10 else np.nan
            ok7 = np.isnan(rho7) or np.sign(rho7) == np.sign(rho[j])
            for i in cur:
                v = X[i, f]
                if np.isnan(v): continue
                d_ = 1 if v >= q90 else (-1 if v <= q10 else 0)
                if d_ != 0: out.append(("F2", i, d_ * np.sign(rho[j]), z - 1, ok7))
        # F3 (매 봉 재학습)
        med = np.nanmedian(Xw, 0); Xf = np.where(np.isnan(Xw), med, Xw); mu = Xf.mean(0); sd = Xf.std(0); sd[sd == 0] = 1
        Z = (Xf - mu) / sd
        try:
            m3 = LogisticRegression(C=0.1, max_iter=300).fit(Z, (yw > 0).astype(int))
        except Exception:
            continue
        pw = m3.predict_proba(Z)[:, 1]; hit7 = ((pw[rec] > 0.5) == (yw[rec] > 0)).mean() if rec.any() else 0.5
        for i in cur:
            xi = np.where(np.isnan(X[i]), med, X[i]); p = m3.predict_proba(((xi - mu) / sd)[None])[0, 1]
            out.append(("F3", i, np.sign(p - 0.5), abs(p - 0.5) / 0.05, hit7 > 0.5))
    O = pd.DataFrame(out, columns=["fam", "i", "side", "score", "ok7"])
    O["t"] = D.t.values[O.i]; O["sym"] = D.sym.values[O.i]; O["fwd"] = y[O.i]
    return O


def account(T, H, lev):
    per = (T.side * T.fwd - COST).groupby(T.t).sum() * lev / (H * 2)
    eq = (1 + per.sort_index().clip(lower=-0.999)).cumprod()
    mon = eq.resample("ME").last().pct_change().dropna(); yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 0.1)
    return (eq.iloc[-1] ** (1 / 12 / yrs) - 1) * 100, (eq / eq.cummax() - 1).min() * 100, (mon > 0).mean()


if __name__ == "__main__":
    t_ = time.time()
    closes = {s: store.load("klines_1m", s).set_index("ts").close.resample("4h").last() for s in ("BTCUSDT", "ETHUSDT")}
    D = pd.concat([build(s, "4h", 6, closes["ETHUSDT" if s == "BTCUSDT" else "BTCUSDT"]).assign(sym=s) for s in ("BTCUSDT", "ETHUSDT")], ignore_index=True)
    D = D.replace([np.inf, -np.inf], np.nan).dropna(subset=["f1", "fW", "G1_r5"]).reset_index(drop=True)
    D = D[D.t >= pd.Timestamp("2020-02-01", tz="UTC")].reset_index(drop=True)
    FE = [c for c in D.columns if c[:2] in ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9") and c[2] == "_"]
    rng = np.random.default_rng(0)
    for ycol, H, tname in (("f1", 1, "다음 4시간"), ("fW", 6, "다음 24시간")):
        res = {}
        for plc in (False, True):
            Dx = D.copy()
            if plc:
                for s in ("BTCUSDT", "ETHUSDT"):
                    m = (Dx.sym == s).values; v = Dx.loc[m, ycol].values; Dx.loc[m, ycol] = np.roll(v, int(rng.integers(500, len(v) - 500)))
            res[plc] = run(Dx, FE, ycol, H)
            P(f"  {tname} {'위약' if plc else '실제'} 측정 완료: 후보 {len(res[plc]):,} ({time.time()-t_:.0f}s)")
        days = (D.t.max() - D.t.min()).days
        P(f"\n===== 목표 {tname} (매 4시간 측정, 최근 30일 창) — 거래당 비용 후 bp")
        P(f"  {'신호':4s} {'τ':>4s} {'7일확인':>6s} | {'실제':>34s} | {'위약':>18s} | 연도(실제)")
        best = []
        for fam in ("F1", "F2", "F3", "F4"):
            for tau in TAUS:
                for need7 in (False, True):
                    line = []
                    for plc in (False, True):
                        O = res[plc]; S = O[(O.fam == fam) & (O.score >= tau) & (O.side != 0)]
                        if need7: S = S[S.ok7]
                        r = S.side * S.fwd - COST
                        line.append((r.mean() * 1e4 if len(S) else np.nan, len(S) / days, (S.side * S.fwd > 0).mean() if len(S) else np.nan, S))
                    (a, ad, ah, S), (b, bd, _, _) = line
                    yr = (S.side * S.fwd - COST).groupby(S.t.dt.year).mean() * 1e4 if len(S) else pd.Series(dtype=float)
                    P(f"  {fam:4s} {tau:4.1f} {'예' if need7 else '아니오':>6s} | {a:+7.1f}bp 적중 {ah:.3f} 하루 {ad:5.2f} | {b:+7.1f}bp 하루 {bd:5.2f} | " +
                      " ".join(f"{k%100}:{v:+.0f}" for k, v in yr.items()))
                    if len(S) >= 200: best.append((a - (b if np.isfinite(b) else 0), a, fam, tau, need7, S))
        best.sort(key=lambda z: z[1], reverse=True)
        for _, a, fam, tau, need7, S in best[:3]:
            line = " / ".join(f"{nm} 월 {cm:+.2f}% MDD {dd:.0f}% 흑자달 {wp:.0%}" for (cm, dd, wp), nm in
                              ((account(S, H, lev), nm) for lev, nm in ((1, "무레버"), (3, "3배"), (5, "5배"))))
            P(f"  계좌 [{fam} τ≥{tau} 7일확인 {'예' if need7 else '아니오'}, 거래당 {a:+.1f}bp] {line}")
    (R / "logs" / "r71.summary").write_text("R71 매 4시간 유의 신호 거래 완료. 세부 logs/r71.log\n")
