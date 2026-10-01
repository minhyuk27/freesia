"""R122-2 — 엣지 공장 2단계: 메커니즘 110개 × 조건 42개 × 구간 4개(하위 20%·30%, 상위 30%·20%) 스캔, 3단 분할 + 위약.

조건(진입 분 − 1까지, 종목별): R119 새 특징 24개 + 가격·변동성(1h·4h·1d·7d 수익, 20일선 괴리, 1h 변동성, 1h/1d 변동성 비, 1d 범위 내 위치)
  + 변동성 국면(30일/365일, 전일) + 파생(직전 펀딩, 프리미엄 지수·60분 변화, OI 60·240분 변화, 상위 트레이더 롱숏비, 계정 롱숏비, 테이커 롱숏비)
  + 일간(스테이블코인 7일 증가, 공포탐욕 — 모두 다음 날부터).
구간 경계 = 학습(2020~23) 거래의 분위(인과적 고정). 통계 = 진입일별 평균 수익의 일간 평균·t(같은 날 거래 군집 보정).
3단 분할: 학습 2020~23 / 검증 2024 / 시험 2025~2026-09.
사전 선택 규칙: 학습 일수 ≥ 100, 학습 t ≥ max(3, 위약 95%), 검증 평균 > 0 & 검증 t ≥ 1.5. 시험은 보고만(선택에 쓰지 않음).
위약: 메커니즘마다 학습 거래의 조건 행을 무작위로 섞어(조건 간 상관 유지) 같은 스캔 10회 → 최대 t의 95% 분위.
출력: data/r122_scan.parquet(전 후보), data/r122_feats.parquet(거래별 조건 — 3단계용)
사용: python research/r122_scan.py"""
import sys, warnings, time, gc
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
import r119_feats as NF, r115_bybit_divergence as F15, r102_joint_account as J
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
TR_END, VA_END = 2024, 2025          # 학습 < 2024, 검증 2024, 시험 2025~
D_TR = (pd.Timestamp("2024-01-01", tz="UTC") - J.D0).days; D_VA = (pd.Timestamp("2025-01-01", tz="UTC") - J.D0).days


def panel(sym):
    F = NF.newfeats(sym); idx = F.index
    k = store.load("klines_1m", sym).set_index("ts"); k.index = F15.utc(k.index); k = k.reindex(idx)
    lc = np.log(k.close.ffill()); r1 = lc.diff()
    X = {f"ret{m}": lc - lc.shift(m) for m in (60, 240, 1440, 10080)}
    X["vol60"] = r1.rolling(60).std(); X["volr"] = X["vol60"] / r1.rolling(1440, min_periods=720).std()
    hi = k.high.rolling(1440, min_periods=720).max(); lo = k.low.rolling(1440, min_periods=720).min(); X["pos1d"] = (k.close - lo) / (hi - lo)
    dc = k.close.resample("D").last(); dr = np.log(dc).diff()
    ma20 = dc.rolling(20).mean().shift(1).reindex(idx, method="ffill"); X["ma20dev"] = k.close / ma20 - 1
    v30 = dr.rolling(30).std(); X["vreg"] = (v30 / v30.rolling(365, min_periods=120).median()).shift(1).reindex(idx, method="ffill")
    fu = store.load("funding", sym).set_index("ts").rate; fu.index = F15.utc(fu.index); X["fund"] = fu[~fu.index.duplicated()].reindex(idx, method="ffill")
    pr = store.load("premium_1m", sym).set_index("ts").close; pr.index = F15.utc(pr.index); pr = pr[~pr.index.duplicated()].reindex(idx).ffill()
    X["prem"] = pr; X["dprem60"] = pr - pr.shift(60)
    mt = store.load("metrics_5m", sym).set_index("ts"); mt.index = F15.utc(mt.index) + pd.Timedelta(minutes=5); mt = mt[~mt.index.duplicated()].reindex(idx, method="ffill", limit=15)
    oi = np.log(mt.sum_open_interest); X["oi60"] = oi - oi.shift(60); X["oi240"] = oi - oi.shift(240)
    X["tlsr"] = mt.sum_toptrader_long_short_ratio; X["lsr"] = mt.count_long_short_ratio; X["tklsr"] = mt.sum_taker_long_short_vol_ratio
    st = pd.read_parquet(R / "data" / "ext" / "stablecoins.parquet")
    st.index = F15.utc(st.index if not "date" in st.columns else st.date)
    s7 = (st.stable_total / st.stable_total.shift(7) - 1); s7.index = s7.index + pd.Timedelta(days=1); X["stab7"] = s7[~s7.index.duplicated()].reindex(idx, method="ffill")
    fg = pd.read_parquet(R / "data" / "ext" / "fng.parquet"); fg.index = F15.utc(fg.index if not "date" in fg.columns else fg.date)
    fgv = fg.value.copy(); fgv.index = fgv.index + pd.Timedelta(days=1); X["fng"] = fgv[~fgv.index.duplicated()].reindex(idx, method="ffill")
    Xd = pd.DataFrame(X).astype("float32")
    return pd.concat([F, Xd], axis=1)


def daystats(day, ret, mask, nd_min=30):
    """진입일별 평균 → (일간 평균, t, 일수) for 학습/검증/시험."""
    out = []
    for lo_, hi_ in ((0, D_TR), (D_TR, D_VA), (D_VA, 10 ** 9)):
        m = mask & (day >= lo_) & (day < hi_)
        if m.sum() == 0: out += [np.nan, np.nan, 0]; continue
        d = day[m]; s = np.bincount(d - lo_, weights=ret[m]); n = np.bincount(d - lo_); ok = n > 0
        v = s[ok] / n[ok]; nd = len(v)
        out += [v.mean(), v.mean() / v.std() * np.sqrt(nd) if nd > 2 and v.std() > 0 else np.nan, nd]
    return out


if __name__ == "__main__":
    t_ = time.time()
    T = pd.read_parquet(R / "data" / "r122_trades.parquet")
    P(f"거래 {len(T):,}, 메커니즘 {T.mech.nunique()} ({time.time()-t_:.0f}s)")
    feats = None; FX = np.full((len(T), 0), np.nan, dtype="float32")
    parts = []
    for sym in ("BTCUSDT", "ETHUSDT"):
        Pn = panel(sym); feats = list(Pn.columns)
        m = (T.sym == sym).values
        ts = J.D0 + pd.to_timedelta(T.e.values[m] - 1, unit="m")
        parts.append((np.where(m)[0], Pn.reindex(pd.DatetimeIndex(ts)).values.astype("float32")))
        del Pn; gc.collect()
        P(f"  {sym} 조건 패널 결합 ({time.time()-t_:.0f}s)")
    FX = np.full((len(T), len(feats)), np.nan, dtype="float32")
    for ix, v in parts: FX[ix] = v
    del parts; gc.collect()
    pd.DataFrame(FX, columns=feats).to_parquet(R / "data" / "r122_feats.parquet", index=False)
    day = (T.e.values // 1440).astype(np.int64); ret = T.ret.values.astype(float)
    rows = []; rng = np.random.default_rng(0)
    BUCK = (("Q1", 0.2, "lo"), ("L30", 0.3, "lo"), ("H30", 0.7, "hi"), ("Q5", 0.8, "hi"))
    for mi_, (mech, g) in enumerate(T.groupby("mech", sort=True)):
        ix = g.index.values; d = day[ix]; r = ret[ix]; X = FX[ix]; tr = d < D_TR
        base = daystats(d, r, np.ones(len(ix), bool))
        rows.append(dict(mech=mech, feat="(없음)", buck="전체", n=len(ix), tr_mu=base[0], tr_t=base[1], tr_nd=base[2], va_mu=base[3], va_t=base[4], te_mu=base[6], te_t=base[7], te_nd=base[8]))
        edges = {}
        for j, f in enumerate(feats):
            x = X[tr, j]; x = x[np.isfinite(x)]
            if len(x) < 300: continue
            edges[j] = {b: np.quantile(x, q) for b, q, _ in BUCK}
        def scan(Xs):
            res = []
            for j, eq in edges.items():
                col = Xs[:, j]
                for b, q, side in BUCK:
                    mk = (col <= eq[b]) if side == "lo" else (col >= eq[b])
                    res.append((j, b, mk))
            return res
        cand = []
        for j, b, mk in scan(X):
            st = daystats(d, r, mk)
            cand.append(dict(mech=mech, feat=feats[j], buck=b, n=int(mk.sum()), tr_mu=st[0], tr_t=st[1], tr_nd=st[2], va_mu=st[3], va_t=st[4], va_nd=st[5],
                             te_mu=st[6], te_t=st[7], te_nd=st[8]))
        # 위약: 학습 거래의 조건 행을 섞는다 → 최대 학습 t
        nulls = []
        trix = np.where(tr)[0]
        for _ in range(10):
            Xp = X.copy(); Xp[trix] = X[rng.permutation(trix)]
            mx = -np.inf
            for j, b, mk in scan(Xp):
                m2 = mk & tr
                if m2.sum() == 0: continue
                dd = d[m2]; s = np.bincount(dd, weights=r[m2]); n = np.bincount(dd); ok = n > 0; v = s[ok] / n[ok]
                if len(v) > 30 and v.std() > 0: mx = max(mx, v.mean() / v.std() * np.sqrt(len(v)))
            nulls.append(mx)
        null95 = float(np.quantile(nulls, 0.95))
        for c in cand: c["null95"] = null95
        rows += cand
        if mi_ % 10 == 0: P(f"  [{mi_+1}/{T.mech.nunique()}] {mech}: 기준 학습 {base[0]*1e4:+.1f}bp t{base[1]:+.1f}, 위약95 t {null95:.2f} ({time.time()-t_:.0f}s)")
    O = pd.DataFrame(rows)
    O["pass"] = (O.feat != "(없음)") & (O.tr_nd >= 100) & (O.tr_t >= np.maximum(3.0, O.null95)) & (O.va_mu > 0) & (O.va_t >= 1.5)
    O.to_parquet(R / "data" / "r122_scan.parquet", index=False)
    pd.set_option("display.width", 260); pd.set_option("display.max_rows", 200)
    C = O[O.feat != "(없음)"]
    P(f"\n후보 {len(C):,} · 학습 t≥3 {(C.tr_t >= 3).sum():,} · 위약 문턱 통과 {(C.tr_t >= np.maximum(3, C.null95)).sum():,} · 검증까지 통과 {C['pass'].sum():,}")
    S = O[O["pass"]].copy(); S["bp_tr"] = S.tr_mu * 1e4; S["bp_va"] = S.va_mu * 1e4; S["bp_te"] = S.te_mu * 1e4
    S["fam"] = S.mech.str.split().str[0] + " " + S.mech.str[-1]
    P("통과 수(계열·방향별): " + ", ".join(f"{k} {v}" for k, v in S.fam.value_counts().items()))
    P("\n===== 통과 후보 (시험 2025~26은 보고만) — 메커니즘·방향별 학습 t 상위")
    P(S.sort_values("tr_t", ascending=False).groupby("fam").head(12)[["mech", "feat", "buck", "n", "tr_nd", "bp_tr", "tr_t", "null95", "bp_va", "va_t", "bp_te", "te_t", "te_nd"]].round(2).to_string(index=False))
    P(f"\n통과 후보의 시험 구간: 양수 비율 {(S.te_mu > 0).mean()*100:.0f}% (n {len(S)}), 시험 t≥2 비율 {(S.te_t >= 2).mean()*100:.0f}%")
    for fam, g in S.groupby("fam"):
        P(f"  {fam}: 통과 {len(g)}, 시험 양수 {(g.te_mu > 0).mean()*100:.0f}%, 시험 거래당 중앙 {g.bp_te.median():+.1f}bp")
    P(f"완료 ({time.time()-t_:.0f}s)")
