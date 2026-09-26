"""R72 — 매 4시간마다 가장 최근 1달(4시간봉 180개)로 모델을 다시 학습해 그 신호로 거래.

종목별로 봉 i의 종가 시점에: 학습 = 같은 종목의 봉 [i−h−179, i−h] (라벨 i 시점까지 확정된 최근 180개), 예측 = 봉 i.
특징: R68 4시간 특징 91개(캔들·추세·오실레이터·변동성·거래량·파생·교차·시간·차트선 거리).
모델: M1 로지스틱(L2 C=0.1, 전체) · M2 얕은 부스팅(HistGB 50회, 깊이 3, 잎 최소 10) · M3 엑스트라트리(50그루, 잎 최소 5) · M4 로지스틱(캔들+교차만)
진입: 예측 확률 p ≥ θ 롱, p ≤ 1−θ 숏, θ ∈ {0.55, 0.60, 0.65, 0.70}. 목표 h = 1봉(4시간), 6봉(24시간). 비용 왕복 14bp.
위약: 종목별 미래수익을 무작위 시간 이동(순환)해 같은 절차.
계좌(상위 구성): 레버리지 1·3·5, 포지션 = 자본 × 레버리지 / (h × 2종목).
사용: python research/r72_retrain_180.py"""
import sys, warnings, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
N = 180; THETA = (0.55, 0.60, 0.65, 0.70)


def job(args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
    sym, ycol, H, plc, X, XC, y, t = args
    if plc:
        y = np.roll(y, int(np.random.default_rng(int(sym == "ETHUSDT") * 2 + int(ycol == "fW")).integers(500, len(y) - 500)))
    out = []
    for i in range(N + H + 200, len(y)):
        a, b = i - H - N + 1, i - H + 1
        Xtr, ytr = X[a:b], (y[a:b] > 0).astype(int)
        if ytr.min() == ytr.max(): continue
        med = np.nanmedian(Xtr, 0); med = np.where(np.isnan(med), 0, med)
        F = np.where(np.isnan(Xtr), med, Xtr); mu = F.mean(0); sd = F.std(0); sd[sd == 0] = 1
        xi = np.where(np.isnan(X[i]), med, X[i])
        z = (F - mu) / sd; zi = ((xi - mu) / sd)[None]
        p1 = LogisticRegression(C=0.1, max_iter=300).fit(z, ytr).predict_proba(zi)[0, 1]
        p2 = HistGradientBoostingClassifier(max_iter=50, max_depth=3, min_samples_leaf=10, learning_rate=0.1,
                                            random_state=0).fit(Xtr, ytr).predict_proba(X[i][None])[0, 1]
        p3 = ExtraTreesClassifier(n_estimators=50, min_samples_leaf=5, random_state=0, n_jobs=1).fit(F, ytr).predict_proba(xi[None])[0, 1]
        C = XC[a:b]; mc = np.nanmedian(C, 0); mc = np.where(np.isnan(mc), 0, mc); Cf = np.where(np.isnan(C), mc, C)
        cm, cs = Cf.mean(0), Cf.std(0); cs[cs == 0] = 1; ci = np.where(np.isnan(XC[i]), mc, XC[i])
        p4 = LogisticRegression(C=0.1, max_iter=300).fit((Cf - cm) / cs, ytr).predict_proba(((ci - cm) / cs)[None])[0, 1]
        out.append((t[i], sym, y[i], p1, p2, p3, p4))
    return ycol, plc, pd.DataFrame(out, columns=["t", "sym", "fwd", "M1", "M2", "M3", "M4"])


def account(T, H, lev):
    per = (T.side * T.fwd - COST).groupby(T.t).sum() * lev / (H * 2)
    eq = (1 + per.sort_index().clip(lower=-0.999)).cumprod()
    mon = eq.resample("ME").last().pct_change().dropna(); yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 0.1)
    return (eq.iloc[-1] ** (1 / 12 / yrs) - 1) * 100, (eq / eq.cummax() - 1).min() * 100, (mon > 0).mean()


if __name__ == "__main__":
    from freesia import store
    from r68_bar_patterns import build, COST
    t_ = time.time()
    closes = {s: store.load("klines_1m", s).set_index("ts").close.resample("4h").last() for s in ("BTCUSDT", "ETHUSDT")}
    jobs = []
    for s in ("BTCUSDT", "ETHUSDT"):
        D = build(s, "4h", 6, closes["ETHUSDT" if s == "BTCUSDT" else "BTCUSDT"])
        D = D.replace([np.inf, -np.inf], np.nan).dropna(subset=["f1", "fW", "G1_r5"]).sort_values("t").reset_index(drop=True)
        FE = [c for c in D.columns if c[:2] in ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9") and c[2] == "_"]
        FC = [c for c in FE if c[:2] in ("G1", "G7")]
        for ycol, H in (("f1", 1), ("fW", 6)):
            for plc in (False, True):
                jobs.append((s, ycol, H, plc, D[FE].values.astype(float), D[FC].values.astype(float), D[ycol].values, D.t.values))
    P(f"작업 {len(jobs)}개, 특징 {len(FE)} (캔들+교차 {len(FC)}) ({time.time()-t_:.0f}s)")
    res = {}
    with Pool(2) as pool:
        for ycol, plc, df in pool.imap_unordered(job, jobs):
            res.setdefault((ycol, plc), []).append(df); P(f"  완료 {ycol} {'위약' if plc else '실제'} {len(df):,} ({time.time()-t_:.0f}s)")
    for ycol, H, tname in (("f1", 1, "다음 4시간"), ("fW", 6, "다음 24시간")):
        Rl = pd.concat(res[(ycol, False)]); Pl = pd.concat(res[(ycol, True)])
        for d_ in (Rl, Pl): d_["t"] = pd.to_datetime(d_.t, utc=True)
        days = (Rl.t.max() - Rl.t.min()).days
        P(f"\n===== 목표 {tname} (매 4시간 최근 180봉 재학습) — 거래당 비용 후 bp")
        P(f"  모델 θ    | {'실제':>32s} | {'위약':>16s} | 연도(실제)")
        cand = []
        for m in ("M1", "M2", "M3", "M4"):
            for th in THETA:
                def sel(d):
                    side = np.where(d[m] >= th, 1, np.where(d[m] <= 1 - th, -1, 0)); k = side != 0
                    return pd.DataFrame({"t": d.t.values[k], "side": side[k], "fwd": d.fwd.values[k]})
                S, Sp = sel(Rl), sel(Pl)
                r, rp = S.side * S.fwd - COST, Sp.side * Sp.fwd - COST
                yr = r.groupby(S.t.dt.year).mean() * 1e4
                P(f"  {m} {th:.2f} | {r.mean()*1e4:+7.1f}bp 적중 {(S.side*S.fwd>0).mean():.3f} 하루 {len(S)/days:5.2f} | {rp.mean()*1e4:+7.1f}bp 하루 {len(Sp)/days:5.2f} | " +
                  " ".join(f"{k%100}:{v:+.0f}" for k, v in yr.items()))
                if len(S) >= 300: cand.append((r.mean(), m, th, S))
        cand.sort(key=lambda z: z[0], reverse=True)
        for rm, m, th, S in cand[:3]:
            line = " / ".join(f"{nm} 월 {cm:+.2f}% MDD {dd:.0f}% 흑자달 {wp:.0%}" for (cm, dd, wp), nm in
                              ((account(S, H, lev), nm) for lev, nm in ((1, "무레버"), (3, "3배"), (5, "5배"))))
            P(f"  계좌 [{m} θ={th}, 거래당 {rm*1e4:+.1f}bp] {line}")
    (R / "logs" / "r72.summary").write_text("R72 매 4시간 최근 180봉 재학습 완료. 세부 logs/r72.log\n")
