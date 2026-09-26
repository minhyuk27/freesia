"""R77 — 변동성·거래량·관심도(구글 트렌드·위키)·파생·내재변동성의 '변화 방향과 엇갈림'으로 다음 24시간 국면·방향 예측.
학습은 짧게(최대 3~6개월) + 최근 1달 가중치 크게, 매일 재학습.

시점: 4시간봉 종가(BTC·ETH). 모든 값은 그 시점까지 알려진 것만(일간 외부 데이터는 다음 날부터).
목표: T1 = 다음 24시간 추세(직전 24시간 방향 +0.5σ 이상)/반전(−0.5σ 이하)/횡보 3분류(R76 정의), T2 = 다음 24시간 방향(상승/하락).
변수 그룹 (각 기본 계열 X마다 수준 z(30일), 변화 Δ4h·Δ24h·Δ72h, 가속 = Δ24h − 이전 Δ24h)
  PRICE 가격: 4·24·72시간 수익/σ, 24시간 효율비, 범위 안 위치
  VOLA 변동성: 실현변동성(1분 수익) 4·24·72시간, 4시간 범위
  VOLU 거래량: 거래대금, 체결 수, 평균 체결 크기
  FLOW 체결 흐름: 테이커 매수 비율, 대량(평균 크기) 추세
  DERIV 파생: OI, 펀딩, 프리미엄 지수, 코인베이스 프리미엄
  IV 내재변동성: DVOL, DVOL − 실현변동성 스프레드
  ATTN 관심도: 구글 트렌드(bitcoin·ethereum·crypto·buy bitcoin·bitcoin price), 위키 조회수(Bitcoin·Ethereum·Cryptocurrency), 공포탐욕
  DIV 엇갈림: 거래량↑·변동성↓ / 거래량↑·가격 제자리 / OI↑·가격↓ / 관심↑·거래량↓ / 내재↑·실현↓ / 매수체결↑·가격↓ /
             스퀴즈(볼린저 폭 120봉 최저권)·거래량↑ — 각 4상태 코드 + 연속 곱
학습 창(매일 00:00 UTC 재학습, 라벨이 재학습 시각 전에 확정된 표본만):
  W90w 90일·최근 30일 가중 3 / W180w 180일·최근 30일 가중 3 / W180 180일 균등 / EXP 확장창(2020~, 월별 재학습, 기준)
모델: HistGB(작게: 150회, 잎 15, 최소 30, L2 1) / 로지스틱(C=0.05)
평가(2022-01~2026-09): T1 정확도·로그손실(학습 창 빈도 대비)·클래스별 확신 상위 10% 적중. T2 적중·확신 상위 10/20% 적중·24시간 보유 순수익(14bp).
그룹 넣고 빼기: W180w·HistGB·주별 재학습으로 ALL / PRICE+그룹 단독 / ALL−그룹.
사용: python research/r77_dynamics.py [build|main|ablate]"""
import sys, warnings, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
COST = 0.0014; EXT = R / "data" / "ext"


def zc(s, w=180):                                                  # 30일(4시간 180개) z
    return (s - s.rolling(w, min_periods=60).mean()) / s.rolling(w, min_periods=60).std()


def dyn(F, name, s, grp, steps=(1, 6, 18)):
    """기본 계열 s(4시간 격자, 로그 또는 비율)에 수준 z, 변화, 가속 추가."""
    F[f"{grp}|{name}_z"] = zc(s)
    for k in steps: F[f"{grp}|{name}_d{k}"] = s - s.shift(k)
    F[f"{grp}|{name}_acc"] = (s - s.shift(6)) - (s.shift(6) - s.shift(12))


def daily_asof(ser, idx, lag_days=1):
    """일간 계열(날짜 = 그날 하루) → 다음 날 00:00 UTC부터 사용."""
    s = ser.copy(); s.index = pd.DatetimeIndex(s.index).tz_localize("UTC") if s.index.tz is None else s.index
    s.index = s.index.normalize() + pd.Timedelta(days=lag_days)
    s = s[~s.index.duplicated()].sort_index()
    return s.reindex(s.index.union(idx)).ffill().reindex(idx)


def build_sym(sym, other):
    from freesia import store
    k = store.load("klines_1m", sym).set_index("ts")
    r = np.log(k.close).diff()
    g = k.resample("4h").agg({"open": "first", "high": "max", "low": "min", "close": "last", "quote_volume": "sum", "trades": "sum",
                              "taker_buy_quote_volume": "sum"}).dropna()
    end = g.index + pd.Timedelta("4h"); g.index = end                # 시점 = 봉 종가 시각
    F = pd.DataFrame(index=end); lc = np.log(g.close)
    rv4 = np.log(r.resample("4h").std().reindex(end - pd.Timedelta("4h")).values + 1e-9); rv4 = pd.Series(rv4, index=end)
    sig24 = (lc - lc.shift(6)).rolling(360, min_periods=120).std()
    # PRICE
    for k_ in (1, 6, 18): F[f"PRICE|r{k_}"] = (lc - lc.shift(k_)) / sig24 * np.sqrt(6 / k_)
    F["PRICE|er24"] = (lc - lc.shift(6)).abs() / lc.diff().abs().rolling(6).sum()
    hi6, lo6 = g.high.rolling(6).max(), g.low.rolling(6).min(); F["PRICE|pos24"] = (g.close - lo6) / (hi6 - lo6)
    # VOLA
    dyn(F, "rv4", rv4, "VOLA")
    rv24 = 0.5 * np.log((np.exp(2 * rv4)).rolling(6).mean()); dyn(F, "rv24", rv24, "VOLA", (6, 18))
    rv72 = 0.5 * np.log((np.exp(2 * rv4)).rolling(18).mean()); dyn(F, "rv72", rv72, "VOLA", (6, 18))
    rngl = np.log((g.high - g.low) / g.close); dyn(F, "rng", rngl, "VOLA")
    # VOLU
    lv = np.log(g.quote_volume); dyn(F, "vol", lv, "VOLU"); dyn(F, "vol24", np.log(g.quote_volume.rolling(6).sum()), "VOLU", (6, 18))
    lt = np.log(g.trades); dyn(F, "trades", lt, "VOLU"); dyn(F, "size", lv - lt, "VOLU")
    # FLOW
    tbr = g.taker_buy_quote_volume / g.quote_volume - 0.5; dyn(F, "tbr", tbr, "FLOW")
    dyn(F, "tbr24", g.taker_buy_quote_volume.rolling(6).sum() / g.quote_volume.rolling(6).sum() - 0.5, "FLOW", (6, 18))
    # DERIV
    m5 = store.load("metrics_5m", sym).set_index("ts")
    oi = np.log(m5.sum_open_interest_value); oi.index = oi.index + pd.Timedelta("5min")
    oi = oi[~oi.index.duplicated()].sort_index().reindex(oi.index.union(end)).ffill().reindex(end); dyn(F, "oi", oi, "DERIV")
    fu = store.load("funding", sym).set_index("ts").rate; fu = fu[~fu.index.duplicated()].sort_index()
    fa = fu.reindex(fu.index.union(end)).ffill().reindex(end); dyn(F, "fund", fa, "DERIV", (6, 18))
    pr = store.load("premium_1m", sym).set_index("ts").close.resample("4h").mean(); pr.index = pr.index + pd.Timedelta("4h")
    dyn(F, "prem", pr.reindex(end), "DERIV")
    cb = store.load("coinbase_1m", "BTCUSD" if sym == "BTCUSDT" else "ETHUSD").set_index("ts").close.resample("4h").last()
    cb.index = cb.index + pd.Timedelta("4h"); cbp = np.log(cb.reindex(end) / g.close); dyn(F, "cbp", cbp, "DERIV")
    # IV
    dv = pd.read_parquet(EXT / f"dvol_{'btc' if sym == 'BTCUSDT' else 'eth'}_1h.parquet").close
    dv.index = dv.index + pd.Timedelta("1h"); dv = dv[~dv.index.duplicated()].sort_index()
    dva = np.log(dv.reindex(dv.index.union(end)).ffill().reindex(end)); dyn(F, "dvol", dva, "IV")
    rvann = np.log(np.exp(rv24) * np.sqrt(525600) * 100); dyn(F, "ivrv", dva - rvann, "IV", (6, 18))
    # ATTN (다음 날부터)
    gt = pd.read_parquet(EXT / "gtrends.parquet").set_index("date"); wk = pd.read_parquet(EXT / "wiki.parquet").set_index("date")
    fng = pd.read_parquet(EXT / "fng.parquet").value
    for c in gt.columns: dyn(F, f"gt_{c.replace(' ', '_')}", np.log(daily_asof(gt[c], end) + 1), "ATTN", (6, 42))
    for c in wk.columns: dyn(F, f"wk_{c}", np.log(daily_asof(wk[c], end) + 1), "ATTN", (6, 42))
    dyn(F, "fng", daily_asof(fng, end), "ATTN", (6, 42))
    # DIV (엇갈림)
    def combo(nm, a, b):
        F[f"DIV|{nm}_code"] = (np.sign(a) * 2 + np.sign(b)).fillna(0); F[f"DIV|{nm}_prod"] = zc(a) * zc(b)
    combo("volUp_rvDn", lv.rolling(6).sum().diff(6), rv24.diff(6))
    F["DIV|vol_flat"] = zc(lv.rolling(6).sum().diff(6)) * (1 - F["PRICE|r6"].abs().clip(upper=3) / 3)
    combo("oi_px", oi.diff(6), lc.diff(6))
    combo("attn_vol", daily_asof(gt[gt.columns[0 if sym == "BTCUSDT" else 1]], end).pipe(np.log1p).diff(42), lv.rolling(6).sum().diff(6))
    combo("iv_rv", dva.diff(6), rv24.diff(6))
    combo("tb_px", tbr.rolling(6).mean().diff(6), lc.diff(6))
    bbw = g.close.rolling(20).std() / g.close.rolling(20).mean(); sq = (bbw <= bbw.rolling(120, min_periods=60).quantile(0.1)).astype(float)
    F["DIV|squeeze_volUp"] = sq * zc(lv.rolling(6).sum().diff(6)); F["DIV|squeeze"] = sq
    # 라벨
    fut = lc.shift(-6) - lc; d = np.sign(lc - lc.shift(6)); z = d * fut / sig24
    F["y1"] = np.where(z >= 0.5, 0, np.where(z <= -0.5, 1, 2)); F.loc[z.isna() | (d == 0), "y1"] = -1
    F["y2"] = (fut > 0).astype(int); F.loc[fut.isna(), "y2"] = -1
    F["fut"] = fut; F["d"] = d; F["sym"] = sym
    return F.replace([np.inf, -np.inf], np.nan)


def fit_predict(args):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    tag, win, wt, model, feats, target, X, y, t, days, retrain = args
    out = []
    ts = pd.DatetimeIndex(t); tv = ts.values
    for T in days:
        if retrain == "W" and T.weekday() != 0: continue
        if retrain == "M" and T.day != 1: continue
        lo = (T - pd.Timedelta(days=win)) if win else pd.Timestamp("2020-01-01", tz="UTC")
        tr = (ts >= lo) & (ts + pd.Timedelta("24h") <= T) & (y >= 0)
        tend = T + (pd.offsets.MonthBegin(1) if retrain == "M" else pd.Timedelta(days={"D": 1, "W": 7}[retrain]))
        te = (ts >= T) & (ts < tend) & (y >= 0)
        if tr.sum() < 300 or te.sum() == 0 or len(np.unique(y[tr])) < 2: continue
        w = np.where(ts[tr] >= T - pd.Timedelta(days=30), wt, 1.0)
        Xtr = X[tr][:, feats]; Xte = X[te][:, feats]
        if model == "GB":
            m = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=30, l2_regularization=1.0, random_state=0)
            m.fit(Xtr, y[tr], sample_weight=w); p = m.predict_proba(Xte); cls = m.classes_
        else:
            med = np.nanmedian(Xtr, 0); med = np.where(np.isnan(med), 0, med)
            A = np.where(np.isnan(Xtr), med, Xtr); mu, sd = A.mean(0), A.std(0); sd[sd == 0] = 1
            B = (np.where(np.isnan(Xte), med, Xte) - mu) / sd
            m = LogisticRegression(C=0.05, max_iter=500).fit(np.clip((A - mu) / sd, -6, 6), y[tr], sample_weight=w)
            p = m.predict_proba(np.clip(B, -6, 6)); cls = m.classes_
        K = 3 if target == "y1" else 2; full = np.full((len(p), K), 1e-4)
        for j, c in enumerate(cls): full[:, int(c)] = p[:, j]
        prior = np.bincount(y[tr].astype(int), weights=w, minlength=K) / w.sum()
        idx = np.where(te)[0]
        for q, i in enumerate(idx): out.append((i, *full[q], *prior))
    cols = ["i"] + [f"p{c}" for c in range(3 if target == "y1" else 2)] + [f"b{c}" for c in range(3 if target == "y1" else 2)]
    return tag, target, pd.DataFrame(out, columns=cols)


def evaluate(D, res, tag, target):
    from sklearn.metrics import log_loss
    E = res.merge(D[["y1", "y2", "fut", "d", "t"]], left_on="i", right_index=True)
    K = 3 if target == "y1" else 2; y = E[target].values.astype(int)
    p = np.clip(E[[f"p{c}" for c in range(K)]].values, 1e-4, 1); p /= p.sum(1, keepdims=True)
    b = np.clip(E[[f"b{c}" for c in range(K)]].values, 1e-4, 1); b /= b.sum(1, keepdims=True)
    acc = (p.argmax(1) == y).mean(); bacc = (b.argmax(1) == y).mean()
    ll, bll = log_loss(y, p, labels=list(range(K))), log_loss(y, b, labels=list(range(K)))
    yr = E.t.dt.year.values
    if target == "y1":
        conf = []
        for c in range(3):
            top = p[:, c] >= np.quantile(p[:, c], 0.9); conf.append(f"{['추세','반전','횡보'][c]} {(y[top] == c).mean():.3f}({(y == c).mean():.3f})")
        byyr = " ".join(f"{k%100}:{(p.argmax(1)[yr == k] == y[yr == k]).mean() - (b.argmax(1)[yr == k] == y[yr == k]).mean():+.3f}" for k in np.unique(yr))
        return f"{tag:24s} T1 정확도 {acc:.3f} (기준 {bacc:.3f}) 로그손실 {ll:.4f} (기준 {bll:.4f}) | 상위10% " + " ".join(conf) + f" | 연도별 정확도차 {byyr}"
    s = np.abs(p[:, 1] - 0.5); side = np.where(p[:, 1] > 0.5, 1, -1)
    line = f"{tag:24s} T2 적중 {(side == np.where(E.fut > 0, 1, -1)).mean():.3f} 로그손실 {ll:.4f} (기준 {bll:.4f})"
    for q in (0.9, 0.8):
        top = s >= np.quantile(s, q); r = side[top] * E.fut.values[top] - COST
        yrr = " ".join(f"{k%100}:{r[yr[top] == k].mean()*1e4:+.0f}" for k in np.unique(yr[top]))
        line += f" | 상위{int((1-q)*100)}% 적중 {(r + COST > 0).mean():.3f} {r.mean()*1e4:+.1f}bp ({yrr})"
    return line


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "main"; t_ = time.time()
    if mode == "build":
        from freesia import store
        cl = {s: store.load("klines_1m", s).set_index("ts").close.resample("4h").last() for s in ("BTCUSDT", "ETHUSDT")}
        D = pd.concat([build_sym(s, cl["ETHUSDT" if s == "BTCUSDT" else "BTCUSDT"]).reset_index(names="t") for s in ("BTCUSDT", "ETHUSDT")], ignore_index=True)
        D.to_parquet(R / "data" / "r77_feat.parquet", index=False)
        G = pd.Series([c.split("|")[0] for c in D.columns if "|" in c]).value_counts()
        P(f"표본 {len(D):,}, 특징 {sum(G)}: " + " ".join(f"{k}:{v}" for k, v in G.items()) + f" ({time.time()-t_:.0f}s)"); sys.exit(0)
    D = pd.read_parquet(R / "data" / "r77_feat.parquet").sort_values("t").reset_index(drop=True)
    FE = [c for c in D.columns if "|" in c]; GR = sorted({c.split("|")[0] for c in FE})
    X = D[FE].values.astype(np.float32); t = D.t.values
    days = pd.date_range("2022-01-01", D.t.max().floor("D"), freq="D", tz="UTC")
    idx = {c: j for j, c in enumerate(FE)}; allf = list(range(len(FE)))
    jobs = []
    if mode == "main":
        for target in ("y1", "y2"):
            y = D[target].values.astype(int)
            for tag, win, wt, rt in (("W90w(90일·최근1달×3)", 90, 3.0, "D"), ("W180w(180일·최근1달×3)", 180, 3.0, "D"),
                                     ("W180(180일·균등)", 180, 1.0, "D"), ("EXP(확장창·월별)", None, 1.0, "M")):
                for model in ("GB", "LR"):
                    jobs.append((f"{tag} {model}", win, wt, model, allf, target, X, y, t, days, rt))
    else:
        base = [idx[c] for c in FE if c.startswith("PRICE|")]
        for target in ("y1", "y2"):
            y = D[target].values.astype(int)
            cfgs = [("ALL", allf)] + [(f"PRICE+{g}", base + [idx[c] for c in FE if c.startswith(g + "|")]) for g in GR if g != "PRICE"] + \
                   [(f"ALL−{g}", [idx[c] for c in FE if not c.startswith(g + "|")]) for g in GR]
            for nm, fs in cfgs: jobs.append((nm, 180, 3.0, "GB", fs, target, X, y, t, days, "W"))
    P(f"{mode}: 작업 {len(jobs)}개, 특징 {len(FE)}, 평가 {days[0]:%Y-%m-%d}~{days[-1]:%Y-%m-%d}")
    lines = {}
    with Pool(2) as pool:
        for tag, target, res in pool.imap_unordered(fit_predict, jobs):
            lines[(target, tag)] = evaluate(D, res, tag, target); P(f"  완료 {target} {tag} ({time.time()-t_:.0f}s)")
    for target in ("y1", "y2"):
        P(f"\n===== {'T1 다음 24시간 추세/반전/횡보' if target == 'y1' else 'T2 다음 24시간 방향'} ({mode})")
        for (tg, tag), ln in sorted(lines.items(), key=lambda z: z[0][1]):
            if tg == target: P("  " + ln)
    (R / "logs" / f"r77_{mode}.summary").write_text(f"R77 {mode} 완료. 세부 logs/r77_{mode}.log\n")
