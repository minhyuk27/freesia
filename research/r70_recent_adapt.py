"""R70 — 4시간봉: 최근 1달 학습 → 직전 1주일에도 패턴이 반복되면 다음 1주일 진입 (적응형).

매주 월요일 00:00 UTC(T)마다
  학습 = 봉 종가 t ∈ [T−37일, T−7일) 이고 라벨(t+h봉)이 T−7일 전에 끝난 표본 (BTC·ETH 합산, 약 360개)
  검증 = t ∈ [T−7일, T) 이고 라벨이 T 전에 끝난 표본
  거래 = t ∈ [T, T+7일) 의 매 4시간 봉, 학습 모델(검증 데이터는 학습에 안 씀)로 신호
패턴 종류
  F1 캔들 시퀀스: 최근 3봉 방향(8코드). 학습에서 코드별 평균 수익 t ≥ 1, n ≥ 6인 코드만, 방향 = 평균 부호
  F2 지표 규칙: R68 특징 91개 중 학습 순위상관 |ρ| 상위 5개(|ρ| ≥ 0.1). 각 지표가 학습 30/70% 분위 밖이면 부호 투표, |합| ≥ 2면 진입
  F3 로지스틱(L2, C=0.1, 표준화): 학습 예측 15/85% 분위 밖이면 진입
  F4 유사 모양: 최근 6봉 수익/ATR 벡터의 학습 kNN(k=15) 평균 미래수익, 학습(자기 제외) |예측| 80% 분위 이상이면 진입
검증 통과('반복'): 검증 신호 n ≥ 3 이고 평균(방향×수익) > 0
비교: 항상 / 검증 통과 시 / 검증 실패 시 / 무작위 켜기(통과율과 같은 확률, 30회 평균)
목표: h = 1봉(4시간), 6봉(24시간). 비용 왕복 14bp. 계좌 레버리지 1·3·5 (24시간은 포지션 1개 = 자본 1/6).
사용: python research/r70_recent_adapt.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
from r68_bar_patterns import build, COST
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
BAR = pd.Timedelta("4h")


def f1(tr, X, y):
    code = lambda d: ((d.G1_r0 > 0).astype(int) * 4 + (d.G1_r1 > 0).astype(int) * 2 + (d.G1_r2 > 0).astype(int)).values
    ct = code(tr); s = pd.DataFrame({"c": ct, "y": y}).groupby("c").y.agg(["mean", "std", "count"])
    s = s[(s["count"] >= 6) & (np.abs(s["mean"]) / (s["std"] / np.sqrt(s["count"])) >= 1.0)]
    m = dict(np.sign(s["mean"]))
    return lambda d: np.array([m.get(c, 0.0) for c in code(d)])


def f2(tr, X, y, feats):
    rs = []
    for c in feats:
        x = tr[c].values; ok = ~np.isnan(x)
        if ok.sum() < 50 or np.nanstd(x) == 0: continue
        r = spearmanr(x[ok], y[ok]).correlation
        if np.isfinite(r): rs.append((abs(r), r, c))
    rs = sorted(rs, reverse=True)[:5]; rs = [z for z in rs if z[0] >= 0.1]
    th = {c: (np.nanquantile(tr[c], 0.3), np.nanquantile(tr[c], 0.7), np.sign(r)) for _, r, c in rs}
    def sig(d):
        v = np.zeros(len(d))
        for c, (lo, hi, sg) in th.items():
            x = d[c].values; v += np.where(x > hi, sg, np.where(x < lo, -sg, 0))
        return np.where(np.abs(v) >= 2, np.sign(v), 0)
    return sig


def f3(tr, X, y, feats):
    med = tr[feats].median(); mu = tr[feats].fillna(med).mean(); sd = tr[feats].fillna(med).std().replace(0, 1)
    Z = lambda d: ((d[feats].fillna(med) - mu) / sd).fillna(0).values
    m = LogisticRegression(C=0.1, max_iter=500).fit(Z(tr), (y > 0).astype(int))
    p = m.predict_proba(Z(tr))[:, 1]; lo, hi = np.quantile(p, 0.15), np.quantile(p, 0.85)
    return lambda d: (lambda q: np.where(q >= hi, 1, np.where(q <= lo, -1, 0)))(m.predict_proba(Z(d))[:, 1])


def f4(tr, X, y):
    cols = [f"G1_r{k}" for k in range(6)]
    A = np.nan_to_num(tr[cols].values); nn = NearestNeighbors(n_neighbors=16).fit(A)
    _, ix = nn.kneighbors(A); ptr = y[ix[:, 1:]].mean(1); thr = np.quantile(np.abs(ptr), 0.8)
    def sig(d):
        _, i2 = nn.kneighbors(np.nan_to_num(d[cols].values), n_neighbors=15); p = y[i2].mean(1)
        return np.where(np.abs(p) >= thr, np.sign(p), 0)
    return sig


def account(T, H, lev):
    per = (T.side * T.fwd - COST).groupby(T.t).sum() * lev / H
    eq = (1 + per.sort_index().clip(lower=-0.999)).cumprod()
    mon = eq.resample("ME").last().pct_change().dropna(); yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    return (eq.iloc[-1] ** (1 / 12 / yrs) - 1) * 100, (eq / eq.cummax() - 1).min() * 100, (mon > 0).mean()


if __name__ == "__main__":
    closes = {s: store.load("klines_1m", s).set_index("ts").close.resample("4h").last() for s in ("BTCUSDT", "ETHUSDT")}
    D = pd.concat([build(s, "4h", 6, closes["ETHUSDT" if s == "BTCUSDT" else "BTCUSDT"]).assign(sym=s) for s in ("BTCUSDT", "ETHUSDT")], ignore_index=True)
    D = D.replace([np.inf, -np.inf], np.nan).dropna(subset=["f1", "fW", "G1_r5"]).sort_values("t").reset_index(drop=True)
    FE = [c for c in D.columns if c[:2] in ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9") and c[2] == "_"]
    weeks = pd.date_range("2020-03-02", "2026-09-21", freq="W-MON", tz="UTC")
    for ycol, H, tname in (("f1", 1, "다음 4시간"), ("fW", 6, "다음 24시간")):
        recs = []
        for T in weeks:
            tr = D[(D.t >= T - pd.Timedelta(days=37)) & (D.t + H * BAR <= T - pd.Timedelta(days=7))]
            va = D[(D.t >= T - pd.Timedelta(days=7)) & (D.t + H * BAR <= T)]
            te = D[(D.t >= T) & (D.t < T + pd.Timedelta(days=7))]
            if len(tr) < 200 or len(va) < 20 or len(te) == 0: continue
            y = tr[ycol].values
            fams = {"F1 캔들시퀀스": f1(tr, None, y), "F2 지표규칙": f2(tr, None, y, FE), "F3 로지스틱": f3(tr, None, y, FE), "F4 유사모양": f4(tr, None, y)}
            for fn, sig in fams.items():
                sv = sig(va); kv = sv != 0
                ok = kv.sum() >= 3 and (sv[kv] * va[ycol].values[kv]).mean() > 0
                st = sig(te); kt = st != 0
                if kt.any():
                    recs.append(pd.DataFrame({"fam": fn, "wk": T, "ok": ok, "t": te.t.values[kt], "sym": te.sym.values[kt],
                                              "side": st[kt], "fwd": te[ycol].values[kt]}))
        Rr = pd.concat(recs, ignore_index=True); Rr["r"] = Rr.side * Rr.fwd - COST
        days = (Rr.t.max() - Rr.t.min()).days
        P(f"\n===== 목표 {tname}: 주 {Rr.wk.nunique()}개")
        P(f"  {'패턴':12s} | {'항상':>22s} | {'검증 통과 시(요청 방식)':>34s} | {'검증 실패 시':>16s} | 무작위 켜기 위약")
        rng = np.random.default_rng(0)
        for fn, G in list(Rr.groupby("fam")) + [("합산 F1~F4", Rr)]:
            wk_ok = G.groupby("wk").ok.first(); rate = wk_ok.mean()
            a = G.r.mean() * 1e4; b = G[G.ok]; c = G[~G.ok]
            pl = []
            for _ in range(30):
                on = pd.Series(rng.random(len(wk_ok)) < rate, index=wk_ok.index); pl.append(G[G.wk.map(on).values].r.mean() * 1e4)
            yr = b.groupby(b.t.dt.year).r.mean() * 1e4
            P(f"  {fn:12s} | {a:+6.1f}bp 하루 {len(G)/days:4.1f} | {b.r.mean()*1e4:+6.1f}bp 적중 {(b.r + COST > 0).mean():.3f} 하루 {len(b)/days:4.1f} 통과율 {rate:.0%} | "
              f"{c.r.mean()*1e4:+6.1f}bp | {np.mean(pl):+6.1f}bp (95% {np.percentile(pl, 97.5):+.1f}) | 연도 " + " ".join(f"{k%100}:{v:+.0f}" for k, v in yr.items()))
        for fn, G in list(Rr.groupby("fam")) + [("합산 F1~F4", Rr)]:
            b = G[G.ok]
            if len(b) < 50: continue
            line = " / ".join(f"{nm} 월 {cm:+.2f}% MDD {dd:.0f}% 흑자달 {wp:.0%}" for (cm, dd, wp), nm in
                              ((account(b, H * (4 if fn.startswith("합산") else 1), lev), nm) for lev, nm in ((1, "무레버"), (3, "3배"), (5, "5배"))))
            P(f"  계좌 [{fn} 검증 통과 시] {line}")
    (R / "logs" / "r70.summary").write_text("R70 최근 1달 학습·1주 검증 적응형 완료. 세부 logs/r70.log\n")
