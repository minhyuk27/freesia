"""R97 — 방향 예측 대규모 연구 → 변동성 예보 p(R90)와 결합.

1) build: 기존 68특징(R50 52 + R54 옵션·코인베이스 16)에 새 방향 가족 추가(전부 인과: 분 데이터는 t−1 종가까지, 일 데이터 1일 지연, 시간 데이터는 완료된 시간봉만)
   TSM   시계열 모멘텀 3·7·30일 수익, 변동성 나눔
   MA    이동평균 괴리 1·7·30일, 1일 이동평균 기울기
   RNG   7·30일 범위 내 위치, 30일 고점·저점 거리, 30일 고점/저점 이후 경과
   SEMI  상승/하락 반분산 비(4h·24h), 24h 실현 왜도
   VDIR  테이커 매수 비 24h, 상승 분 거래량 비중 24h
   XA    다른 종목 수익(1h·4h·24h), 상대 수익(자기−다른), ETH/BTC 비 30일 z
   IVX   DVOL 수준·24h 변화, DVOL − 실현변동성
   BAS   분기물 연율 베이시스·7일 변화
   FLOW  스테이블 7·30일 증가 z, 거래소 순유출 z, MVRV z, 활성 주소 z, 공포·탐욕·7일 변화
   MAC   VIX·5일 변화, 달러지수 5·20일 변화, S&P·나스닥 1·5일 수익, 하이일드 스프레드 5일 변화, 10년물 5일 변화 (미국 일자료, 1일 지연)
   KIM   업비트 김치 프리미엄·24h 변화
   ATT   구글·위키 7일 변화 z
   SEAS  월중 일, 주말, 미국 장중
2) study: 특징 하나씩 — 연도별 확장창(학습 < Y)에서 상·하위 10% 문턱과 방향 부호를 정하고, 평가 연도 Y(2022~26)에서
   극단 표본의 4시간 부호 수익(bp)과 큰 움직임 결판 적중률을 '전체'와 '고변동 예보(pr ≥ 0.8)'로 나눠 측정.
3) combine: 방향 분류기(큰 움직임 결판 표본 학습, 전체 특징 vs 기존 특징만) → pr·확신 문턱으로 진입,
   청산 E1 ±2% 대칭 / E2 목표 2%·손절 1% / E3 4시간 보유. 2022~23 선택 → 2024~26 검증, 3단 레버리지.
사용: python research/r97_direction.py build|study|combine"""
import sys, warnings, time
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
EXT = R / "data" / "ext"; OUT = R / "data" / "r97_feat.parquet"
SYMS = ("BTCUSDT", "ETHUSDT")


def utc(ix):
    ix = pd.DatetimeIndex(ix); return ix.tz_localize("UTC") if ix.tz is None else ix.tz_convert("UTC")


def z(x, n): return (x - x.rolling(n, min_periods=n // 2).mean()) / x.rolling(n, min_periods=n // 2).std()


def daily_asof(s, ts, lag_days=1):
    """일 자료 s(날짜 인덱스, 그날 값) → 시점 ts에서 쓸 수 있는 값(날짜 + lag_days 00:00 부터)."""
    s = s.copy(); s.index = utc(s.index).normalize() + pd.Timedelta(days=lag_days); s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.reindex(s.index.union(ts)).ffill().reindex(ts).values


def build():
    t_ = time.time()
    from r50_direction_model import D as D0
    G = D0[["sym", "t", "ts"]].copy(); G["ts"] = utc(G.ts)
    K = {s: store.load("klines_1m", s) for s in SYMS}
    C = {s: pd.Series(K[s].close.values, index=utc(K[s].ts)) for s in SYMS}
    out = []
    for s in SYMS:
        g = G[G.sym == s].copy(); k = K[s]; ti = g.t.values - 1                   # t−1 분까지
        c = k.close; lr = np.log(c).diff(); f = {}
        for d, n in ((3, 4320), (7, 10080), (30, 43200)):
            r = np.log(c / c.shift(n)); f[f"TSM|r{d}d"] = r.values[ti]; f[f"TSM|r{d}d_v"] = (r / (lr.rolling(n).std() * np.sqrt(n))).values[ti]
        for d, n in ((1, 1440), (7, 10080), (30, 43200)):
            ma = c.rolling(n).mean(); f[f"MA|d{d}d"] = (c / ma - 1).values[ti]
            if d == 1: f["MA|slope1d"] = (ma / ma.shift(1440) - 1).values[ti]
        for d, n in ((7, 10080), (30, 43200)):
            hi = k.high.rolling(n).max(); lo = k.low.rolling(n).min(); f[f"RNG|pos{d}d"] = ((c - lo) / (hi - lo)).values[ti]
            if d == 30:
                f["RNG|dhi30"] = (c / hi - 1).values[ti]; f["RNG|dlo30"] = (c / lo - 1).values[ti]
        up = (lr.clip(lower=0) ** 2); dn = (lr.clip(upper=0) ** 2)
        for h, n in (("4h", 240), ("24h", 1440)):
            f[f"SEMI|ud{h}"] = np.log((up.rolling(n).sum() + 1e-12) / (dn.rolling(n).sum() + 1e-12)).values[ti]
        f["SEMI|skew24h"] = lr.rolling(1440).skew().values[ti]
        tb = k.taker_buy_volume / k.volume.replace(0, np.nan)
        f["VDIR|tbr24h"] = (k.taker_buy_volume.rolling(1440).sum() / k.volume.rolling(1440).sum()).values[ti]
        f["VDIR|upvol24h"] = ((k.volume * (lr > 0)).rolling(1440).sum() / k.volume.rolling(1440).sum()).values[ti]
        o = [x for x in SYMS if x != s][0]; co = C[o].reindex(utc(k.ts)).ffill().values; co = pd.Series(co)
        for h, n in (("1h", 60), ("4h", 240), ("24h", 1440)):
            ro = np.log(co / co.shift(n)); rs = np.log(c / c.shift(n))
            f[f"XA|oth{h}"] = ro.values[ti]; f[f"XA|rel{h}"] = (rs - ro).values[ti]
        eb = np.log(pd.Series(C["ETHUSDT"].reindex(utc(k.ts)).ffill().values) / pd.Series(C["BTCUSDT"].reindex(utc(k.ts)).ffill().values))
        f["XA|ethbtc_z"] = z(eb, 43200).values[ti]
        rv24 = lr.rolling(1440).std() * np.sqrt(525600)
        ts = utc(g.ts)
        dv = pd.read_parquet(EXT / f"dvol_{s[:3].lower()}_1h.parquet").close; dv.index = utc(dv.index) + pd.Timedelta(hours=1)
        dvv = dv.reindex(dv.index.union(ts)).ffill().reindex(ts).values
        f["IVX|dvol"] = dvv; dv24 = dv.shift(24); f["IVX|ddvol24"] = dvv - dv24.reindex(dv24.index.union(ts)).ffill().reindex(ts).values
        f["IVX|dvol_rv"] = dvv / 100 - rv24.values[ti]
        b = pd.read_parquet(EXT / f"basis_{s[:3].lower()}.parquet").basis_ann
        f["BAS|basis"] = daily_asof(b, ts); f["BAS|dbasis7"] = daily_asof(b.diff(7), ts)
        st = pd.read_parquet(EXT / "stablecoins.parquet"); cm = pd.read_parquet(EXT / f"cm_{s[:3].lower()}.parquet")
        f["FLOW|st7"] = daily_asof(z(np.log(st.stable_total).diff(7), 365), ts); f["FLOW|st30"] = daily_asof(z(np.log(st.stable_total).diff(30), 365), ts)
        nf = -(cm.FlowInExUSD - cm.FlowOutExUSD).rolling(7).sum() / cm.SplyExUSD
        f["FLOW|netout"] = daily_asof(z(nf, 365), ts); f["FLOW|mvrv"] = daily_asof(z(cm.CapMVRVCur, 365), ts)
        f["FLOW|adr"] = daily_asof(z(np.log(cm.AdrActCnt.rolling(7).mean()).diff(30), 365), ts)
        fg = pd.read_parquet(EXT / "fng.parquet").value.astype(float); f["FLOW|fng"] = daily_asof(fg, ts); f["FLOW|dfng7"] = daily_asof(fg.diff(7), ts)
        fr = pd.read_parquet(EXT / "fred.parquet").ffill()
        f["MAC|vix"] = daily_asof(fr.vix, ts); f["MAC|dvix5"] = daily_asof(fr.vix.diff(5), ts)
        f["MAC|usd5"] = daily_asof(np.log(fr.usd_broad).diff(5), ts); f["MAC|usd20"] = daily_asof(np.log(fr.usd_broad).diff(20), ts)
        f["MAC|spx1"] = daily_asof(np.log(fr.spx).diff(1), ts); f["MAC|spx5"] = daily_asof(np.log(fr.spx).diff(5), ts)
        f["MAC|ndq5"] = daily_asof(np.log(fr.nasdaq).diff(5), ts); f["MAC|dhy5"] = daily_asof(fr.hy_spread.diff(5), ts); f["MAC|dy10_5"] = daily_asof(fr.y10.diff(5), ts)
        ub = pd.read_parquet(EXT / "upbit_1h.parquet"); ub.index = utc(ub.ts) + pd.Timedelta(hours=1)
        kr = pd.read_parquet(EXT / "usdkrw.parquet").set_index("date").krw; kr.index = utc(kr.index) + pd.Timedelta(days=1)
        kr = kr[~kr.index.duplicated()].reindex(kr.index.union(ub.index)).ffill().reindex(ub.index)
        bn = C[s].reindex(ub.index - pd.Timedelta(minutes=1)).values
        kim = pd.Series(ub[s[:3].lower()].values / (bn * kr.values) - 1, index=ub.index)
        kv = kim.reindex(kim.index.union(ts)).ffill().reindex(ts).values
        f["KIM|prem"] = kv; k24 = kim.shift(24); f["KIM|dprem24"] = kv - k24.reindex(k24.index.union(ts)).ffill().reindex(ts).values
        gt = pd.read_parquet(EXT / "gtrends.parquet").set_index("date"); wk = pd.read_parquet(EXT / "wiki.parquet").set_index("date")
        gcol = "bitcoin" if s == "BTCUSDT" else "ethereum"; wcol = "Bitcoin" if s == "BTCUSDT" else "Ethereum"
        f["ATT|gt7"] = daily_asof(z(np.log(gt[gcol].clip(lower=0.1)).diff(7), 180), ts, 2); f["ATT|wk7"] = daily_asof(z(np.log(wk[wcol].clip(lower=1)).diff(7), 180), ts, 2)
        f["SEAS|dom"] = ts.day.values; f["SEAS|wkend"] = (ts.dayofweek >= 5).astype(float); f["SEAS|us"] = ((ts.hour >= 13) & (ts.hour < 21)).astype(float)
        F = pd.DataFrame(f, index=g.index)
        F["sym"] = s; F["t"] = g.t.values
        out.append(F.astype({c: "float32" for c in F.columns if c not in ("sym", "t")})); P(f"  {s} 특징 {F.shape[1]-2} ({time.time()-t_:.0f}s)")
    pd.concat(out).replace([np.inf, -np.inf], np.nan).to_parquet(OUT, index=False)


@numba.njit(cache=True)
def first_side(t, h, l, c, W, thr):
    y = np.zeros(len(t), np.int8)
    for q in range(len(t)):
        i = t[q]; ref = c[i - 1]
        for j in range(i, min(len(c), i + W)):
            u = h[j] >= ref * (1 + thr); d = l[j] <= ref * (1 - thr)
            if u and d: break
            if u: y[q] = 1; break
            if d: y[q] = -1; break
    return y


def frame():
    from r50_direction_model import D as D0, FEAT as F0
    import r92_vol_timing as V
    DF = pd.read_parquet(R / "data" / "r54_dfeat.parquet"); OLD = [c for c in F0 if c not in ("bslope", "bimb")] + [c for c in DF.columns if c not in ("sym", "t")]
    D = D0.merge(DF, on=["sym", "t"], how="left").merge(pd.read_parquet(OUT), on=["sym", "t"], how="left")
    NEW = [c for c in pd.read_parquet(OUT).columns if c not in ("sym", "t")]
    Wp = V.load_p()[["sym", "t", "p", "pr"]]; D = D.merge(Wp, on=["sym", "t"], how="left")
    D["side"] = 0; D["fr"] = np.nan
    for s in SYMS:
        m = D.sym == s; k = store.load("klines_1m", s); t = D.loc[m, "t"].values.astype(np.int64)
        D.loc[m, "side"] = first_side(t, k.high.values, k.low.values, k.close.values, 240, 0.02)
        c = k.close.values; ok = t + 239 < len(c); fr = np.full(len(t), np.nan); fr[ok] = c[t[ok] + 239] / c[t[ok] - 1] - 1; D.loc[m, "fr"] = fr
    D["ts"] = utc(D.ts)
    return D.replace([np.inf, -np.inf], np.nan), OLD, NEW


def fam(c):
    return c.split("|")[0] if "|" in c else {"rr25": "OPT", "atm": "OPT", "pcv1": "OPT", "pcv4": "OPT", "pbuy": "OPT", "drr4": "OPT", "drr24": "OPT", "datm1": "OPT",
                                              "datm4": "OPT", "rrz": "OPT", "cbp": "CB", "cbp15": "CB", "cbp60": "CB", "dcbp60": "CB", "cbpz": "CB"}.get(c, "BASE")


def study():
    D, OLD, NEW = frame(); t_ = time.time(); rows = []
    for c in OLD + NEW:
        res = []
        for Y in range(2022, 2027):
            tr = D[(D.yr < Y)][[c, "fr"]].dropna(); te = D[D.yr == Y]
            if len(tr) < 5000 or te[c].notna().sum() < 500: continue
            lo, hi = tr[c].quantile([0.1, 0.9]); sg = np.sign(tr.fr[tr[c] >= hi].mean() - tr.fr[tr[c] <= lo].mean())
            pred = np.where(te[c] >= hi, sg, np.where(te[c] <= lo, -sg, 0))
            res.append(te.assign(pred=pred)[pred != 0])
        if not res: continue
        X = pd.concat(res); X = X.dropna(subset=["fr"]); H = X[X.pr >= 0.8]; Hd = H[H.side != 0]; Ad = X[X.side != 0]
        sr = X.pred * X.fr; shr = H.pred * H.fr
        by = (X.pred * X.fr).groupby(X.yr).mean() * 1e4
        rows.append(dict(fam=fam(c), f=c, n=len(X), bp=sr.mean() * 1e4, t=sr.mean() / sr.std() * np.sqrt(len(X) / 16), hit_all=(Ad.pred == Ad.side).mean(),
                         nH=len(H), bpH=shr.mean() * 1e4, tH=shr.mean() / shr.std() * np.sqrt(max(len(H), 1) / 16), hitH=(Hd.pred == Hd.side).mean(), nHd=len(Hd),
                         pos_yrs=int((by > 0).sum()), yrs=" ".join(f"{y%100}:{v:+.0f}" for y, v in by.items())))
    O = pd.DataFrame(rows); pd.set_option("display.width", 260); pd.set_option("display.max_rows", 300)
    P(f"특징 {len(O)}개 단독 검정 (t는 4시간 겹침 보정 √(n/16)) ({time.time()-t_:.0f}s)")
    P("\n===== 가족별 평균: 전체 거래당 bp / 고변동 예보 bp / 고변동 적중")
    P(O.groupby("fam")[["bp", "t", "bpH", "tH", "hitH"]].agg(["mean", "max"]).round(2).to_string())
    P("\n===== 고변동 예보(pr ≥ 0.8) 기준 상위 30 (tH)")
    P(O.sort_values("tH", ascending=False).head(30).round(3).to_string())
    P("\n===== 전체 기준 상위 20 (t)")
    P(O.sort_values("t", ascending=False).head(20).round(3).to_string())
    O.to_parquet(R / "data" / "r97_study.parquet", index=False)


@numba.njit(cache=True)
def trade(t, side, o, h, l, c, tp, sl, H):
    n = len(c); R_ = np.full(len(t), np.nan); busy = -1
    for q in range(len(t)):
        i = t[q]
        if side[q] == 0 or i <= busy or i + H >= n: continue
        e = o[i]; s = side[q]; r = np.nan; j = i
        while j < i + H:
            if sl > 0 and ((s > 0 and l[j] <= e * (1 - sl)) or (s < 0 and h[j] >= e * (1 + sl))): r = -sl - 14e-4; break
            if tp > 0 and ((s > 0 and h[j] >= e * (1 + tp)) or (s < 0 and l[j] <= e * (1 - tp))): r = tp - 9e-4; break
            j += 1
        if np.isnan(r): r = s * (c[min(j, n - 1)] / e - 1) - 14e-4
        R_[q] = r; busy = j
    return R_


def combine():
    from sklearn.ensemble import HistGradientBoostingClassifier
    D, OLD, NEW = frame(); t_ = time.time()
    K = {s: store.load("klines_1m", s) for s in SYMS}
    days = {"tr": 730, "te": 993}
    for tag, FE in (("기존 특징", OLD), ("기존+새 가족", OLD + NEW)):
        out = []
        for Y in range(2022, 2027):
            tr = D[(D.yr < Y) & (D.side != 0)]; te = D[D.yr == Y].copy(); fu = [c for c in FE if tr[c].notna().sum() > 3000]
            ps = [HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=300, l2_regularization=1.0, random_state=sd)
                  .fit(tr[fu].values, (tr.side.values == 1).astype(int)).predict_proba(te[fu].values)[:, 1] for sd in range(3)]
            te["pu"] = np.mean(ps, 0); out.append(te)
        W = pd.concat(out); W["pred"] = np.where(W.pu > 0.5, 1, -1); W["conf"] = (W.pu - 0.5).abs()
        dec = W[W.side != 0]; P(f"\n[{tag}] 큰 움직임 결판 방향 적중 {(dec.pred == dec.side).mean():.3f} (n={len(dec):,}) ({time.time()-t_:.0f}s)")
        for th in (0.0, 0.8, 0.95):
            x = dec[dec.pr >= th] if th > 0 else dec
            P(f"   pr ≥ {th:.2f}: 적중 {(x.pred == x.side).mean():.3f} (n={len(x):,}) | 확신 상위 25% {(x[x.conf >= x.conf.quantile(.75)].pred == x[x.conf >= x.conf.quantile(.75)].side).mean():.3f} | 연도별 " +
              " ".join(f"{y%100}:{(g.pred == g.side).mean():.2f}" for y, g in x.groupby("yr")))
        rows = []
        for th in (0.0, 0.5, 0.8, 0.95):
            for cq in (0.0, 0.5, 0.75, 0.9):
                for ex, (tp, sl, H) in (("E1 ±2%", (0.02, 0.02, 240)), ("E2 +2/−1", (0.02, 0.01, 240)), ("E3 4h", (0, 0, 240)), ("E4 +1/−1", (0.01, 0.01, 240))):
                    parts = []
                    for s in SYMS:
                        g = W[(W.sym == s) & (W.pr >= th)].sort_values("t")
                        cth = W[W.yr < 2024].conf.quantile(cq) if cq > 0 else 0          # 확신 문턱은 학습 구간 분포
                        g = g[g.conf >= cth]; k = K[s]
                        r = trade(g.t.values.astype(np.int64), g.pred.values.astype(np.int64), k.open.values, k.high.values, k.low.values, k.close.values, tp, sl, H)
                        parts.append(pd.DataFrame({"ts": g.ts.values, "yr": g.yr.values, "r": r}))
                    T = pd.concat(parts).dropna(); T["ts"] = utc(T.ts)
                    a = T[T.yr < 2024]; b = T[T.yr >= 2024]
                    dd = lambda x, s0, s1: x.groupby(x.ts.dt.floor("D")).r.sum().reindex(pd.date_range(s0, s1, freq="D", tz="UTC")).fillna(0)
                    da, db = dd(a, "2022-01-01", "2023-12-31"), dd(b, "2024-01-01", "2026-09-19")
                    rows.append(dict(th=th, cq=cq, ex=ex, na=len(a) / days["tr"], bpa=a.r.mean() * 1e4, sha=da.mean() / da.std() * np.sqrt(365),
                                     nb=len(b) / days["te"], bpb=b.r.mean() * 1e4, shb=db.mean() / db.std() * np.sqrt(365), wb=(b.r > 0).mean(), T=T))
        O = pd.DataFrame([{k: v for k, v in r.items() if k != "T"} for r in rows])
        P(f"   매매 격자 (학습 22~23 → 평가 24~26), 학습 샤프 상위 12:")
        P(O.sort_values("sha", ascending=False).head(12).round(2).to_string())
        best = rows[int(O.sha.idxmax())]; T = best["T"]
        P(f"   학습 최선 {best['th'], best['cq'], best['ex']} 평가 3단(명목 = 자본 × L, 종목별):")
        d = T[T.yr >= 2024]; d = d.groupby(d.ts.dt.floor("D")).r.sum().reindex(pd.date_range("2024-01-01", "2026-09-19", freq="D", tz="UTC")).fillna(0)
        for L in (1, 3, 10):
            eq = (1 + (d * L).clip(lower=-0.99)).cumprod(); yrs = len(d) / 365.25
            P(f"     ×{L:2d}: 월 {(eq.iloc[-1]**(1/12/yrs)-1)*100:+.2f}% MDD {(eq/eq.cummax()-1).min()*100:.0f}%")
        W[["sym", "t", "ts", "yr", "pu", "pr", "side", "fr"]].to_parquet(R / "data" / f"r97_wf_{'new' if len(FE) > len(OLD) else 'old'}.parquet", index=False)


if __name__ == "__main__":
    m = sys.argv[1]
    {"build": build, "study": study, "combine": combine}[m]()
    (R / "logs" / f"r97_{m}.summary").write_text("done\n")
