"""R115 — 바이비트 vs 바이낸스 괴리(가격 선행·OI 변화·거래 비중·펀딩): 독립 방향 신호, z0.33 돌파 필터, z0.31 딥 필터.

선행 조사: 차익거래로 가격·펀딩은 붙어 있지만(펀딩 상관 0.74) 유동성이 얇은 쪽에서 수요 불균형이 더 큰 가격·펀딩 반응을 만든다.
바이비트는 개인 비중이 높은 거래소 — OI가 바이낸스와 다르게 움직이면 레버리지 쏠림 신호일 수 있다.
특징(t분까지, 종목별, 5분 OI는 봉 끝 +5분부터 사용):
  lead_k : 바이비트 k분 수익 − 바이낸스 k분 수익 (k 1·5·15)          — 바이비트가 먼저 움직였나
  bas    : log(바이비트 종가) − log(바이낸스 종가), dbas_k 변화 (k 15·60)
  oiB_k  : 바이비트 OI k분 로그 변화, oiN_k 바이낸스 OI(metrics_5m) 같은 것, doi_k = oiB_k − oiN_k (k 15·60·240)
  vsh_k  : 바이비트 거래대금 / (바이비트 + 바이낸스) − 7일 평균 (k 15·60)
  dfund  : 직전 확정 펀딩 차이(바이비트 − 바이낸스)
A/B/C 방식은 R113과 같다(학습 2020~23 / 평가 2024~26, 거래 왕복 14bp, BH; 필터는 학습 |ρ|≥0.9 & 평가 같은 부호).
사용: python research/r115_bybit_divergence.py"""
import sys, warnings, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr, norm
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
import r102_joint_account as J, r103_breakout_cost as K, r109_dip_exact_mtm as M, r113_spot_perp_flow as F13
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)


def utc(i):
    i = pd.DatetimeIndex(i); return i.tz_convert("UTC") if i.tz is not None else i.tz_localize("UTC")


def feats(sym):
    n = store.load("klines_1m", sym).set_index("ts"); b = store.load("bybit_1m", sym).set_index("ts")
    n.index = utc(n.index); b.index = utc(b.index)
    idx = pd.date_range(max(n.index[0], b.index[0]), min(n.index[-1], b.index[-1]), freq="1min")
    n = n.reindex(idx); b = b.reindex(idx); stale = n.close.isna() | b.close.isna()
    ln, lb = np.log(n.close.ffill()), np.log(b.close.ffill())
    Fd = {}
    for k in (1, 5, 15): Fd[f"lead{k}"] = (lb - lb.shift(k)) - (ln - ln.shift(k))
    bas = lb - ln; Fd["bas"] = bas
    for k in (15, 60): Fd[f"dbas{k}"] = bas - bas.shift(k)
    oB = store.load("bybit_oi5m", sym).set_index("ts").oi; oB.index = utc(oB.index) + pd.Timedelta(minutes=5)
    oN = store.load("metrics_5m", sym).set_index("ts").sum_open_interest; oN.index = utc(oN.index) + pd.Timedelta(minutes=5)
    oB = np.log(oB[~oB.index.duplicated()].reindex(idx, method="ffill", limit=15)); oN = np.log(oN[~oN.index.duplicated()].reindex(idx, method="ffill", limit=15))
    for k in (15, 60, 240):
        Fd[f"oiB{k}"] = oB - oB.shift(k); Fd[f"oiN{k}"] = oN - oN.shift(k); Fd[f"doi{k}"] = Fd[f"oiB{k}"] - Fd[f"oiN{k}"]
    vb = b.turnover.fillna(0); vn = n.quote_volume.fillna(0)
    for k in (15, 60):
        s_ = vb.rolling(k).sum() / (vb.rolling(k).sum() + vn.rolling(k).sum()).replace(0, np.nan)
        Fd[f"vsh{k}"] = s_ - s_.rolling(10080, min_periods=1440).mean()
    fB = store.load("bybit_funding", sym).set_index("ts").rate; fB.index = utc(fB.index)
    fN = store.load("funding", sym).set_index("ts").rate; fN.index = utc(fN.index)
    Fd["dfund"] = fB[~fB.index.duplicated()].reindex(idx, method="ffill") - fN[~fN.index.duplicated()].reindex(idx, method="ffill")
    F = pd.DataFrame(Fd).astype("float32")
    return F, ln, stale.rolling(240, min_periods=1).max().astype(bool)


if __name__ == "__main__":
    t_ = time.time()
    rows = []; FE = {}
    for sym in ("BTCUSDT", "ETHUSDT"):
        Fm, lc, bad = feats(sym); FE[sym] = Fm
        P(f"{sym} 특징 {Fm.shape}, 기간 {Fm.index[0]} ~ {Fm.index[-1]}, 결측률 " + " ".join(f"{c}:{Fm[c].isna().mean()*100:.0f}%" for c in ("oiB60", "vsh60", "dfund")) + f" ({time.time()-t_:.0f}s)")
        for h in F13.HS:
            y = (lc.shift(-(1 + h)) - lc.shift(-1)).values; sel = np.arange(0, len(Fm), h)
            for fn in Fm.columns:
                d = pd.DataFrame(dict(f=Fm[fn].values[sel], y=y[sel], bad=bad.values[sel]), index=Fm.index[sel]).dropna(); d = d[~d.bad]
                tr, te = d[d.index < F13.SPLIT], d[d.index >= F13.SPLIT]
                if len(tr) < 200 or len(te) < 100: continue
                ic_tr = spearmanr(tr.f, tr.y)[0]; ic_te = spearmanr(te.f, te.y)[0]
                th = tr.f.abs().quantile(0.9); sg = np.sign(ic_tr)
                tt = te[te.f.abs() >= th]; ret = sg * np.sign(tt.f) * (np.exp(tt.y) - 1) - F13.COST
                trr = tr[tr.f.abs() >= th]; rtr = sg * np.sign(trr.f) * (np.exp(trr.y) - 1) - F13.COST
                yr = d.groupby(d.index.year).apply(lambda g: spearmanr(g.f, g.y)[0])
                rows.append(dict(sym=sym[:3], h=h, feat=fn, ic_tr=ic_tr, ic_te=ic_te, t_te=ic_te * np.sqrt(len(te)), same=np.sign(ic_tr) == np.sign(ic_te),
                                 yr_pos=(np.sign(yr) == sg).mean(), bp_tr=rtr.mean() * 1e4, bp_te=ret.mean() * 1e4, n_te=len(tt)))
    O = pd.DataFrame(rows); pv = 2 * (1 - norm.cdf(np.abs(O.t_te.values))); o = np.argsort(pv); m = len(pv)
    ok = pv[o] <= 0.05 * np.arange(1, m + 1) / m; kmax = np.nonzero(ok)[0].max() + 1 if ok.any() else 0; bh = np.zeros(m, bool); bh[o[:kmax]] = True; O["bh"] = bh
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 100)
    P(f"\n[A] 검정 {len(O)}개 · 평가 |t|≥3 {(O.t_te.abs()>=3).sum()} · BH {O.bh.sum()} · 부호 일치 {O.same.mean():.0%}")
    P(O.reindex(O.t_te.abs().sort_values(ascending=False).index).head(15).round(3).to_string(index=False))
    P("\n[A] 학습·평가 모두 거래 양수 (비용 후):"); P(O[(O.bp_tr > 0) & (O.bp_te > 0)].sort_values("bp_te", ascending=False).head(10).round(2).to_string(index=False))
    T8 = J.add_funding(J.breakout_trades(J.TOP8)); T8["lf"] = K.last_funding(T8); BK = K.events(T8, False, fmax=1e-4).drop(columns="lf")
    D, _, _, W = M.dip_with_px()
    for nm, df in (("B 돌파", BK), ("C 딥", D)):
        X = df[["ret", "e", "s"]].reset_index(drop=True).copy()
        for s_, sym in ((0, "BTCUSDT"), (1, "ETHUSDT")):
            mk = (X.s == s_).values; ts = pd.DatetimeIndex(J.D0 + pd.to_timedelta(X.loc[mk, "e"] - 1, unit="min"))
            for c in FE[sym].columns: X.loc[mk, c] = FE[sym][c].reindex(ts).values
        ts = J.D0 + pd.to_timedelta(X.e, unit="min"); tr = (ts < F13.SPLIT).values
        P(f"\n[{nm}] 거래 {len(X):,} (학습 {tr.sum():,}) — 5분위 거래당 bp (학습 | 평가)")
        for f in FE["BTCUSDT"].columns:
            if X[f].notna().mean() < 0.6: continue
            ra, rb, a, b = F13.quint(nm, X[X[f].notna()], f, tr[X[f].notna().values])
            mark = " ◀ 채택 후보" if abs(ra) >= 0.9 and np.sign(ra) == np.sign(rb) else ""
            P(f"  {f:7s} " + " ".join(f"{v*1e4:+6.1f}" for v in a.values) + f" (ρ{ra:+.1f}) | " + " ".join(f"{v*1e4:+6.1f}" for v in b.values) + f" (ρ{rb:+.1f}){mark}")
    O.to_parquet(R / "data" / "r115_ic.parquet", index=False)
    P(f"완료 ({time.time()-t_:.0f}s)")
