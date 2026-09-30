"""R104 — 코인베이스 현물 → 바이낸스 선물 선행: 미국 현물 프리미엄(수준·변화)이 바이낸스 다음 수익을 예측하는가.

프리미엄 p_t = log(코인베이스 BTCUSD/ETHUSD 종가) − log(바이낸스 USDⓈ-M 종가) (USDT/USD 괴리 포함 — BTC·ETH 공통 성분도 따로 본다).
특징(t분 종가까지): dp_k = p_t − p_{t−k} (k 5·15·60·240분), 수준 z(W) = (p_t − 평균_W)/표준편차_W (W 1일·7일, 과거만),
  공통 성분 제거판 dpx_k = BTC와 ETH 프리미엄 변화의 차이가 아니라, 각 종목 프리미엄 − 두 종목 평균(달러 스테이블 요인 제거).
목표: 1분 지연 후 바이낸스 수익 log c_{t+1+h} − log c_{t+1}, h ∈ {15, 60, 240, 1440}분.
검정(사전 고정 40여 개): 겹치지 않는 표본(간격 h)으로 순위 상관 IC, 학습 2020~23 / 평가 2024~26, 연도별 부호.
거래: 학습 구간 |특징| 상위 10% 문턱(방향 = 학습 IC 부호), 한 번에 한 포지션, h 보유, 왕복 14bp. 평가 거래당 bp·일간 샤프.
다중검정: 평가 t ≥ 3, BH(5%). 통과 후보는 z0.31·z0.33과 일간 상관 보고.
사용: python research/r104_coinbase_lead.py"""
import sys, warnings, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R))
from freesia import store
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
COST = 14e-4
SPLIT = pd.Timestamp("2024-01-01", tz="UTC")
HS = (15, 60, 240, 1440)


def load(bn, cb):
    k = store.load("klines_1m", bn); k["ts"] = pd.DatetimeIndex(k.ts)
    c = store.load("coinbase_1m", cb); c["ts"] = pd.DatetimeIndex(c.ts)
    c["ts"] = c.ts.dt.tz_localize("UTC") if c.ts.dt.tz is None else c.ts.dt.tz_convert("UTC")
    k["ts"] = k.ts.dt.tz_localize("UTC") if k.ts.dt.tz is None else k.ts.dt.tz_convert("UTC")
    idx = pd.date_range(max(k.ts.min(), c.ts.min()), min(k.ts.max(), c.ts.max()), freq="1min")
    b = k.set_index("ts").close.reindex(idx); q = c.set_index("ts").close.reindex(idx)
    stale = q.isna() | b.isna()
    return np.log(b.ffill()), np.log(q.ffill()), stale


if __name__ == "__main__":
    t_ = time.time()
    DAT = {s: load(s + "USDT", s + "USD") for s in ("BTC", "ETH")}
    idx = DAT["BTC"][0].index.intersection(DAT["ETH"][0].index)
    PR = {s: (DAT[s][1] - DAT[s][0]).reindex(idx) for s in DAT}
    common = (PR["BTC"] + PR["ETH"]) / 2
    P(f"기간 {idx[0]} ~ {idx[-1]} ({len(idx):,}분), 결측 BTC {DAT['BTC'][2].mean()*100:.1f}% ETH {DAT['ETH'][2].mean()*100:.1f}% ({time.time()-t_:.0f}s)")
    for s in PR: P(f"  {s} 프리미엄 bp: 중앙 {PR[s].median()*1e4:+.1f} · 1%/99% {PR[s].quantile(.01)*1e4:+.1f}/{PR[s].quantile(.99)*1e4:+.1f}")
    rows = []; SIG = {}
    for s in ("BTC", "ETH"):
        lb = DAT[s][0].reindex(idx); p = PR[s]; px = p - common; stale = DAT[s][2].reindex(idx).fillna(True)
        F = {}
        for kk in (5, 15, 60, 240):
            F[f"dp{kk}"] = p - p.shift(kk); F[f"dpx{kk}"] = px - px.shift(kk)
        for W, nm in ((1440, "1d"), (10080, "7d")):
            F[f"z{nm}"] = (p - p.rolling(W, min_periods=W // 2).mean()) / p.rolling(W, min_periods=W // 2).std()
            F[f"zx{nm}"] = (px - px.rolling(W, min_periods=W // 2).mean()) / px.rolling(W, min_periods=W // 2).std()
        F["cz1d"] = (common - common.rolling(1440, min_periods=720).mean()) / common.rolling(1440, min_periods=720).std()
        bad = stale.rolling(240, min_periods=1).max().astype(bool)          # 최근 4시간 결측·정체가 있으면 제외
        for h in HS:
            y = lb.shift(-(1 + h)) - lb.shift(-1)
            sel = np.arange(0, len(idx), h)
            for fn, f in F.items():
                d = pd.DataFrame(dict(f=f.values[sel], y=y.values[sel], bad=bad.values[sel]), index=idx[sel]).dropna()
                d = d[~d.bad]
                tr, te = d[d.index < SPLIT], d[d.index >= SPLIT]
                if len(tr) < 200 or len(te) < 100: continue
                ic_tr = spearmanr(tr.f, tr.y)[0]; ic_te = spearmanr(te.f, te.y)[0]
                yr = d.groupby(d.index.year).apply(lambda g: spearmanr(g.f, g.y)[0])
                # 거래: 학습 상위 10% |f| 문턱, 방향 = 학습 IC 부호
                th = tr.f.abs().quantile(0.9); sgn = np.sign(ic_tr)
                tt = te[te.f.abs() >= th]; ret = sgn * np.sign(tt.f) * (np.exp(tt.y) - 1) - COST
                trr = tr[tr.f.abs() >= th]; ret_tr = sgn * np.sign(trr.f) * (np.exp(trr.y) - 1) - COST
                dd = ret.groupby(ret.index.normalize()).sum().reindex(pd.date_range(SPLIT, idx[-1].normalize(), freq="D"), fill_value=0)
                rows.append(dict(sym=s, h=h, feat=fn, n_te=len(te), ic_tr=ic_tr, ic_te=ic_te, t_te=ic_te * np.sqrt(len(te)),
                                 same=np.sign(ic_tr) == np.sign(ic_te), yr_pos=(np.sign(yr) == np.sign(ic_tr)).mean(),
                                 bp_tr=ret_tr.mean() * 1e4, bp_te=ret.mean() * 1e4, n_tr_te=len(tt), sh_te=dd.mean() / dd.std() * np.sqrt(365) if dd.std() > 0 else np.nan))
                SIG[(s, h, fn)] = dd
        P(f"  {s} 완료 ({time.time()-t_:.0f}s)")
    O = pd.DataFrame(rows); pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    from statsmodels.stats.multitest import multipletests
    from scipy.stats import norm
    pv = 2 * (1 - norm.cdf(np.abs(O.t_te))); O["bh"] = multipletests(pv, 0.05, "fdr_bh")[0]
    P(f"\n검정 {len(O)}개. 평가 |t|≥3: {(O.t_te.abs()>=3).sum()} · BH 통과 {O.bh.sum()} · 학습·평가 IC 부호 일치 {O.same.mean():.0%}")
    P("\n===== 평가 |t| 상위 25"); P(O.reindex(O.t_te.abs().sort_values(ascending=False).index).head(25).round(3).to_string(index=False))
    P("\n===== 평가 거래 bp 상위 15 (학습에서도 양수인 것)"); P(O[O.bp_tr > 0].sort_values("bp_te", ascending=False).head(15).round(2).to_string(index=False))
    O.to_parquet(R / "data" / "r104_ic.parquet", index=False)
    pd.to_pickle(SIG, R / "data" / "r104_daily.pkl")
    P(f"완료 ({time.time()-t_:.0f}s)")
