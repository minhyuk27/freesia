"""R113 — 현물 vs 선물 테이커 흐름(바이낸스 현물 1분 spot_1m, 2020~): 누가 움직임을 주도하는가.

실무 가설(선행 조사): 현물이 이끄는 상승은 실수요라 지속되고, 선물(레버리지)이 이끄는 상승·하락은 되돌려진다.
가격 발견 연구(Cosenza·Stalder 2024): 바이낸스 현물·선물이 주 가격 발견처.
특징(t분 종가까지, 종목별):
  순매수 비율 net_k = Σ(2·테이커매수대금 − 거래대금) / Σ거래대금 (최근 k분), 현물 snet_k · 선물 pnet_k, 차이 div_k = snet_k − pnet_k (k 5·15·60·240)
  현물 대금 비중 share_k = Σ현물대금 / Σ(현물+선물 대금) − 7일 평균
  베이시스 변화 db_k = Δ(log 선물 − log 현물) (k 15·60·240)
A. 독립 방향 신호: 목표 = 1분 지연 뒤 선물 수익 h ∈ {15, 60, 240, 1440}분. 겹치지 않는 표본 IC, 학습 2020~23 / 평가 2024~26,
   거래 = 학습 |특징| 상위 10% 문턱·학습 IC 부호, 왕복 14bp. |t|≥3 + BH.
B. z0.33 돌파(TOP8 펀딩≤1bp) 필터: 진입 직전 분 특징의 5분위별 거래당 수익(학습 2020~23 / 평가 2024~26) — 학습 단조(|ρ|≥0.9)만 채택.
C. z0.31 딥 필터: 같은 방식(학습 2022~23 / 평가 2024~26). 가설: 선물 주도 매도(div 큼) 딥이 더 잘 되돌림.
사용: python research/r113_spot_perp_flow.py"""
import sys, warnings, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr, norm
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
import r102_joint_account as J, r103_breakout_cost as K, r109_dip_exact_mtm as M
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
SPLIT = pd.Timestamp("2024-01-01", tz="UTC"); COST = 14e-4; HS = (15, 60, 240, 1440)


def feats(sym):
    f = store.load("klines_1m", sym).set_index("ts"); s = store.load("spot_1m", sym).set_index("ts")
    for d in (f, s): d.index = pd.DatetimeIndex(d.index).tz_convert("UTC") if pd.DatetimeIndex(d.index).tz is not None else pd.DatetimeIndex(d.index).tz_localize("UTC")
    idx = pd.date_range(max(f.index[0], s.index[0]), min(f.index[-1], s.index[-1]), freq="1min")
    f = f.reindex(idx); s = s.reindex(idx)
    stale = f.close.isna() | s.close.isna()
    fq, fb = f.quote_volume.fillna(0), f.taker_buy_quote_volume.fillna(0); sq, sb = s.quote_volume.fillna(0), s.taker_buy_quote_volume.fillna(0)
    F = {}
    for k in (5, 15, 60, 240):
        sn = (2 * sb - sq).rolling(k).sum() / sq.rolling(k).sum().replace(0, np.nan)
        pn = (2 * fb - fq).rolling(k).sum() / fq.rolling(k).sum().replace(0, np.nan)
        F[f"snet{k}"] = sn; F[f"pnet{k}"] = pn; F[f"div{k}"] = sn - pn
        sh = sq.rolling(k).sum() / (sq.rolling(k).sum() + fq.rolling(k).sum()).replace(0, np.nan)
        F[f"share{k}"] = sh - sh.rolling(10080, min_periods=1440).mean()
    b = np.log(f.close.ffill()) - np.log(s.close.ffill())
    for k in (15, 60, 240): F[f"db{k}"] = b - b.shift(k)
    F = pd.DataFrame(F).astype("float32")
    bad = stale.rolling(240, min_periods=1).max().astype(bool)
    return F, np.log(f.close.ffill()), bad


def quint(name, df, f, tr):
    edges = df.loc[tr, f].quantile([.2, .4, .6, .8]).values; q = np.searchsorted(edges, df[f].values) + 1
    a = df[tr].groupby(q[tr]).ret.mean(); b = df[~tr].groupby(q[~tr]).ret.mean()
    ra = spearmanr(a.index, a.values)[0]; rb = spearmanr(b.index, b.values)[0]
    return ra, rb, a, b


if __name__ == "__main__":
    t_ = time.time()
    rows = []; FE = {}
    for sym in ("BTCUSDT", "ETHUSDT"):
        F, lc, bad = feats(sym); FE[sym] = F
        P(f"{sym} 특징 {F.shape} ({time.time()-t_:.0f}s)")
        for h in HS:
            y = (lc.shift(-(1 + h)) - lc.shift(-1)).values; sel = np.arange(0, len(F), h)
            for fn in F.columns:
                d = pd.DataFrame(dict(f=F[fn].values[sel], y=y[sel], bad=bad.values[sel]), index=F.index[sel]).dropna(); d = d[~d.bad]
                tr, te = d[d.index < SPLIT], d[d.index >= SPLIT]
                if len(tr) < 200 or len(te) < 100: continue
                ic_tr = spearmanr(tr.f, tr.y)[0]; ic_te = spearmanr(te.f, te.y)[0]
                th = tr.f.abs().quantile(0.9); sg = np.sign(ic_tr)
                tt = te[te.f.abs() >= th]; ret = sg * np.sign(tt.f) * (np.exp(tt.y) - 1) - COST
                trr = tr[tr.f.abs() >= th]; rtr = sg * np.sign(trr.f) * (np.exp(trr.y) - 1) - COST
                yr = d.groupby(d.index.year).apply(lambda g: spearmanr(g.f, g.y)[0])
                rows.append(dict(sym=sym[:3], h=h, feat=fn, ic_tr=ic_tr, ic_te=ic_te, t_te=ic_te * np.sqrt(len(te)), same=np.sign(ic_tr) == np.sign(ic_te),
                                 yr_pos=(np.sign(yr) == sg).mean(), bp_tr=rtr.mean() * 1e4, bp_te=ret.mean() * 1e4, n_te=len(tt)))
    O = pd.DataFrame(rows); pv = 2 * (1 - norm.cdf(np.abs(O.t_te.values))); o = np.argsort(pv); m = len(pv)
    ok = pv[o] <= 0.05 * np.arange(1, m + 1) / m; kmax = np.nonzero(ok)[0].max() + 1 if ok.any() else 0; bh = np.zeros(m, bool); bh[o[:kmax]] = True; O["bh"] = bh
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 100)
    P(f"\n[A] 검정 {len(O)}개 · 평가 |t|≥3 {(O.t_te.abs()>=3).sum()} · BH {O.bh.sum()} · 부호 일치 {O.same.mean():.0%}")
    P(O.reindex(O.t_te.abs().sort_values(ascending=False).index).head(15).round(3).to_string(index=False))
    P("\n[A] 학습·평가 모두 거래 양수:"); P(O[(O.bp_tr > 0) & (O.bp_te > 0)].sort_values("bp_te", ascending=False).head(10).round(2).to_string(index=False))
    # B·C 필터
    def attach(df, ecol):
        out = pd.DataFrame(index=df.index, columns=list(FE["BTCUSDT"].columns), dtype="float32")
        for s_, sym in ((0, "BTCUSDT"), (1, "ETHUSDT")):
            mk = (df.s == s_).values; ts = J.D0 + pd.to_timedelta(df.loc[mk, ecol] - 1, unit="min")
            out.loc[mk] = FE[sym].reindex(pd.DatetimeIndex(ts)).values
        return out
    T8 = J.add_funding(J.breakout_trades(J.TOP8)); T8["lf"] = K.last_funding(T8); BK = K.events(T8, False, fmax=1e-4).drop(columns="lf")
    D, _, _, W = M.dip_with_px()
    for nm, df, trcut in (("B 돌파", BK, SPLIT), ("C 딥", D, SPLIT)):
        A = attach(df, "e"); X = pd.concat([df[["ret", "e"]].reset_index(drop=True), A.reset_index(drop=True)], axis=1)
        ts = J.D0 + pd.to_timedelta(X.e, unit="min"); tr = (ts < trcut).values
        P(f"\n[{nm}] 거래 {len(X):,} (학습 {tr.sum():,}) — 5분위 거래당 bp (학습 | 평가), 학습 |ρ|≥0.9 표시")
        for f in A.columns:
            if X[f].notna().mean() < 0.8: continue
            ra, rb, a, b = quint(nm, X, f, tr)
            mark = " ◀ 채택 후보" if abs(ra) >= 0.9 and np.sign(ra) == np.sign(rb) else ""
            P(f"  {f:8s} " + " ".join(f"{v*1e4:+6.1f}" for v in a.values) + f" (ρ{ra:+.1f}) | " + " ".join(f"{v*1e4:+6.1f}" for v in b.values) + f" (ρ{rb:+.1f}){mark}")
    O.to_parquet(R / "data" / "r113_ic.parquet", index=False)
    P(f"완료 ({time.time()-t_:.0f}s)")
