"""R117 — 15분 단기 신호 결합 × 메이커(지정가) 체결: R104·R113·R116의 작지만 매우 유의한 15분 예측 신호를 하나로 묶으면 메이커 비용을 넘는 독립 고빈도 엣지가 되는가.

신호(t분 종가까지, 종목별):
  현물·선물 흐름(R113): snet·pnet·div 5·15·60, share15 / 김치 프리미엄(R116): dkp5·15·60, kpz, ulead1·5, usurge
  코인베이스 프리미엄(R104): dp15·dp60·z1d / 가격: 1·5·15·60분 수익, 60분 변동성
목표: 1분 지연 뒤 15분 수익 log c_{t+16} − log c_{t+1}.
모델: HistGB 회귀, 연 단위 워크포워드(시험 연도 Y는 Y 이전으로 학습, 경계 1일 제외, 학습 표본 3분 간격), 2021~ 표본 밖 예측.
거래(종목별 한 번에 한 포지션):
  예측이 학습 구간 예측 분포의 상위 q(롱)·하위 q(숏)이면, t분 종가 기준 δ만큼 유리한 가격에 지정가(δ ∈ {0, 2bp}),
  다음 3분 안에 가격이 지정가를 1bp 넘게 관통하면 체결(불리 선택 자연 반영), 체결 뒤 15분째 종가 청산.
  비용 C1 = 메이커 2bp + 테이커 5bp + 청산 슬리피지 1bp, C2 = 메이커·메이커 4bp(청산도 지정가가 된다는 낙관).
  q ∈ {1%, 2%, 5%, 10%} (학습 연도 예측 분위로 문턱 고정).
보고: 연도별 표본 밖 IC, 체결률, 거래당 bp(총·순), 하루 거래 수, 무레버 일간 샤프, z0.31·z0.33과 일간 상관.
사용: python research/r117_short_horizon_maker.py"""
import sys, warnings, time
from pathlib import Path
import numpy as np, pandas as pd, numba
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
import r102_joint_account as J, r103_breakout_cost as K, r113_spot_perp_flow as F13, r116_kimchi_premium as F16, r104_coinbase_lead as F04, r115_bybit_divergence as F15
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
H = 15; WAIT = 3; PEN = 1e-4


def build(sym):
    Fs, lc, bad = F13.feats(sym)
    Fs = Fs[["snet5", "snet15", "snet60", "pnet5", "pnet15", "pnet60", "div5", "div15", "div60", "share15"]]
    Fk = F16.feats(sym)[0][["dkp5", "dkp15", "dkp60", "kpz", "ulead1", "ulead5", "usurge"]]
    b, q, st = F04.load(sym, sym.replace("USDT", "USD"))
    p = (q - b)
    Fc = pd.DataFrame({"cdp15": p - p.shift(15), "cdp60": p - p.shift(60),
                       "cz1d": (p - p.rolling(1440, min_periods=720).mean()) / p.rolling(1440, min_periods=720).std()}).astype("float32")
    k = store.load("klines_1m", sym).set_index("ts"); k.index = F15.utc(k.index)
    idx = Fs.index; k = k.reindex(idx)
    ln = np.log(k.close.ffill())
    Fp = pd.DataFrame({f"r{m}": ln - ln.shift(m) for m in (1, 5, 15, 60)}); Fp["vol60"] = (ln - ln.shift(1)).rolling(60).std()
    X = pd.concat([Fs, Fk.reindex(idx), Fc.reindex(idx), Fp.astype("float32")], axis=1)
    y = (ln.shift(-(1 + H)) - ln.shift(-1)).astype("float32")
    return X, y, k[["open", "high", "low", "close"]], bad | k.close.isna()


@numba.njit(cache=True)
def trade(sig, o, h, l, c, d_, wait, hold, pen):
    """sig: +1/−1/0 (t분 종가에 판정). 지정가 = c[t]·(1 ∓ δ), t+1..t+wait 안에 관통하면 체결, 체결 분 + hold 종가 청산. 한 번에 한 포지션."""
    n = len(sig); ent = np.full(n, -1); ext = np.full(n, -1); px = np.zeros(n); xp = np.zeros(n); sd = np.zeros(n); k = 0; t = 0; busy = -1
    while t < n - wait - hold - 1:
        if t <= busy or sig[t] == 0: t += 1; continue
        s = sig[t]; lim = c[t] * (1 - s * d_); f = -1
        for j in range(t + 1, t + 1 + wait):
            if (s > 0 and l[j] < lim * (1 - pen)) or (s < 0 and h[j] > lim * (1 + pen)): f = j; break
        if f < 0: t += 1; continue
        e = min(f + hold, n - 1)
        ent[k] = f; ext[k] = e; px[k] = lim; xp[k] = c[e]; sd[k] = s; k += 1; busy = e; t = e + 1
    return ent[:k], ext[:k], px[:k], xp[:k], sd[:k]


if __name__ == "__main__":
    t_ = time.time()
    DAILY = {}; rows = []
    for sym in ("BTCUSDT", "ETHUSDT"):
        X, y, OHLC, bad = build(sym)
        P(f"{sym} 특징 {X.shape} ({time.time()-t_:.0f}s)")
        pred = pd.Series(np.nan, index=X.index, dtype="float64"); trq = {}
        Xv = X.values; yv = y.values; ok = ~bad.values & np.isfinite(yv)
        for Y in range(2021, 2027):
            cut = pd.Timestamp(f"{Y}-01-01", tz="UTC"); te = (X.index >= cut) & (X.index < pd.Timestamp(f"{Y+1}-01-01", tz="UTC"))
            tr = np.where((X.index < cut - pd.Timedelta(days=1)) & ok)[0][::3]
            if te.sum() == 0: continue
            m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=500, l2_regularization=1.0, random_state=0)
            m.fit(Xv[tr], yv[tr]); pr_tr = m.predict(Xv[tr[::5]]); pred[te] = m.predict(Xv[te])
            trq[Y] = np.quantile(pr_tr, [0.01, 0.02, 0.05, 0.10, 0.90, 0.95, 0.98, 0.99])
            sel = np.where(te & ok)[0][::H]
            P(f"  {sym} {Y}: 표본 밖 IC {spearmanr(pred.values[sel], yv[sel])[0]:+.4f} (n {len(sel):,}) ({time.time()-t_:.0f}s)")
        o, h, l, c = (OHLC[x].ffill().values for x in ("open", "high", "low", "close"))
        yr = X.index.year.values
        for qi, q in enumerate((0.01, 0.02, 0.05, 0.10)):
            lo_th = np.array([trq.get(Y, [np.nan] * 8)[qi] for Y in yr]); hi_th = np.array([trq.get(Y, [np.nan] * 8)[7 - qi] for Y in yr])
            pv = pred.values
            sig = np.where(pv >= hi_th, 1, np.where(pv <= lo_th, -1, 0)).astype(np.int64); sig[bad.values | np.isnan(pv)] = 0
            for d_ in (0.0, 2e-4):
                en, ex, px, xp, sd = trade(sig, o, h, l, c, d_, WAIT, H, PEN)
                gross = sd * (xp / px - 1)
                for cn, cost in (("C1 메이커+테이커", 2e-4 + 5e-4 + 1e-4), ("C2 메이커·메이커", 4e-4)):
                    net = gross - cost; ts = X.index[en]
                    dd = pd.Series(net, index=ts.normalize()).groupby(level=0).sum()
                    key = (sym, q, d_, cn); DAILY[key] = dd
                    te24 = ts >= F13.SPLIT
                    ndays = (X.index[-1] - X.index[0]).days
                    nsig = (sig != 0).sum()
                    sh = lambda z: z.mean() / z.std() * np.sqrt(365) if len(z) > 10 and z.std() > 0 else np.nan
                    full = dd.reindex(pd.date_range(dd.index.min(), dd.index.max(), freq="D"), fill_value=0)
                    rows.append(dict(sym=sym[:3], q=q, δbp=d_ * 1e4, 비용=cn, 신호분=nsig, 체결=len(en), 하루=len(en) / ndays, 총bp=gross.mean() * 1e4,
                                     순bp_21_23=net[~te24].mean() * 1e4, 순bp_24_26=net[te24].mean() * 1e4, 승률=(net[te24] > 0).mean() * 100,
                                     샤프_21_23=sh(full[:"2023"]), 샤프_24_26=sh(full["2024":]),
                                     연도=" ".join(f"{a%100}:{b*1e4:+.1f}" for a, b in pd.Series(net, index=ts.year).groupby(level=0).mean().items())))
        del X, y
    O = pd.DataFrame(rows); pd.set_option("display.width", 260); pd.set_option("display.max_rows", 100)
    P("\n===== 결과 (표본 밖 2021~, 순bp = 비용 후 거래당)"); P(O.round(2).to_string(index=False))
    # 최선(2021~23 샤프 기준 선택) → 평가, z0.31·z0.33과 상관
    b = O.sort_values("샤프_21_23", ascending=False).iloc[0]
    P(f"\n2021~23 샤프 최선: {b.sym} q{b.q} δ{b.δbp} {b.비용} → 2024~26 순 {b.순bp_24_26:+.1f}bp, 샤프 {b.샤프_24_26:.2f}")
    try:
        T8 = J.add_funding(J.breakout_trades(J.TOP8)); T8["lf"] = K.last_funding(T8); BK = K.events(T8, False, fmax=1e-4)
        z33 = pd.Series(BK.ret.values * BK.mult.values / 48, index=(J.D0 + pd.to_timedelta(BK.e, unit="min")).dt.normalize()).groupby(level=0).sum()
        for (sym, q, d_, cn), dd in DAILY.items():
            if q == b.q and d_ == b.δbp / 1e4 and cn == b.비용:
                I = dd.index.intersection(z33.index); P(f"  {sym} 일간 상관 z0.33 {np.corrcoef(dd.reindex(I), z33.reindex(I))[0,1]:+.2f}")
    except Exception as e:
        P(f"상관 계산 실패: {e}")
    O.to_parquet(R / "data" / "r117.parquet", index=False)
    P(f"완료 ({time.time()-t_:.0f}s)")
