"""새 데이터·지표를 붙인 엣지 탐색 (BTC·ETH 일봉, 1배, 비용 6bp, 펀딩, 연도별 워크포워드).
A) 개별 신호 단독 타이밍  B) 추세 앙상블에 필터로 얹기  C) ML 직접 예측 / 메타라벨링(+라벨 섞기 대조군).
외부 데이터는 1일 지연(보수적)."""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from freesia import store  # noqa: E402
from freesia.bt import core, ind  # noqa: E402
import screen as S  # noqa: E402
import combo as CB  # noqa: E402

EXT = Path(__file__).resolve().parents[1] / "data" / "ext"


def ld(name):
    p = EXT / f"{name}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def build(sym):
    x = S.Ctx(sym, "1d"); idx = x.idx
    b = x.b; c = b.close; ret = c.pct_change()
    F = pd.DataFrame(index=idx)
    F["ret1"], F["ret7"], F["ret30"], F["ret90"] = ret, c.pct_change(7), c.pct_change(30), c.pct_change(90)
    F["rsi14"] = ind.rsi(c, 14)
    m, sg, h = ind.macd(c, 12, 26, 9); F["macdh"] = h / c
    F["adx14"] = ind.adx(b.high, b.low, c, 14)[2]
    rv = ret.rolling(20).std() * np.sqrt(365); F["rv20"] = rv
    F["rv_ratio"] = ret.rolling(5).std() / ret.rolling(30).std()
    F["dist_sma200"] = c / c.rolling(200).mean() - 1
    F["don_pos"] = (c - c.rolling(120).min()) / (c.rolling(120).max() - c.rolling(120).min())
    F["ens"] = CB.trend_exposure(x, "ALL", None)
    tbr = b.taker_buy_quote_volume / b.quote_volume.replace(0, np.nan); F["tbr"] = tbr.rolling(3).mean()
    F["vol_z"] = (b.quote_volume - b.quote_volume.rolling(30).mean()) / b.quote_volume.rolling(30).std()
    f = store.load("funding", sym).set_index("ts").rate
    fd = f.resample("1D").sum().reindex(idx); F["fund"] = fd; F["fund_z"] = (fd - fd.rolling(90).mean()) / fd.rolling(90).std()
    pr = store.load("premium_1m", sym).set_index("ts").close.resample("1D").last().reindex(idx); F["prem"] = pr
    met = store.load("metrics_5m", sym)
    if met is not None:
        md = met.set_index("ts").resample("1D").last().reindex(idx)
        F["oi_chg7"] = md["sum_open_interest_value"].pct_change(7); F["tls"] = md["sum_toptrader_long_short_ratio"]
    ex = {}
    bs = ld(f"basis_{sym[:3].lower()}")
    if bs is not None:
        bs = bs.reindex(idx); ex["basis_ann"] = bs.basis_ann; ex["basis_slope"] = bs.next_ann - bs.basis_ann
    fg = ld("fng")
    if fg is not None:
        v_ = fg.value.reindex(idx); ex["fng"] = v_; ex["fng_d7"] = v_.diff(7)
    fr = ld("fred")
    if fr is not None:
        fr = fr.reindex(idx, method="ffill", limit=6)
        ex["vix"] = fr.vix; ex["vix_d5"] = fr.vix.pct_change(5); ex["usd_d20"] = fr.usd_broad.pct_change(20); ex["y10_d20"] = fr.y10.diff(20)
        ex["nasdaq_r20"] = fr.nasdaq.pct_change(20); ex["nasdaq_a200"] = fr.nasdaq / fr.nasdaq.rolling(200).mean() - 1
        ex["spx_r20"] = fr.spx.pct_change(20)
        if "hy_spread" in fr: ex["hy"] = fr.hy_spread
        ex["fed_d28"] = fr.fed_assets.pct_change(28)
        if "gold" in fr: ex["gold_r20"] = fr.gold.pct_change(20)
    st = ld("stablecoins")
    if st is not None:
        st = st.reindex(idx, method="ffill"); ex["stable_g30"] = st.stable_total.pct_change(30); ex["usdt_g30"] = st.usdt.pct_change(30)
    cm = ld(f"cm_{sym[:3].lower()}")
    if cm is not None:
        cm = cm.reindex(idx)
        for k, nm in (("CapMVRVCur", "mvrv"),):
            if k in cm: ex[nm] = cm[k]
        if "AdrActCnt" in cm: ex["adr_g30"] = cm.AdrActCnt.pct_change(30)
        if "TxCnt" in cm: ex["tx_g30"] = cm.TxCnt.pct_change(30)
        if "NVTAdj" in cm: ex["nvt"] = cm.NVTAdj
        if "FeeTotUSD" in cm: ex["fee_g30"] = cm.FeeTotUSD.pct_change(30)
        if "HashRate" in cm: ex["hash_g30"] = cm.HashRate.pct_change(30)
        if "FlowInExUSD" in cm and "FlowOutExUSD" in cm:
            net = (cm.FlowInExUSD - cm.FlowOutExUSD)
            ex["ex_net7"] = net.rolling(7).sum() / cm.CapMrktCurUSD; ex["ex_net30"] = net.rolling(30).sum() / cm.CapMrktCurUSD
            ex["ex_in_z"] = (cm.FlowInExUSD.rolling(7).mean() / cm.CapMrktCurUSD)
        if "SplyExUSD" in cm and "CapMrktCurUSD" in cm:
            ex["ex_sply_share"] = cm.SplyExUSD / cm.CapMrktCurUSD; ex["ex_sply_d30"] = (cm.SplyExUSD / cm.CapMrktCurUSD).diff(30)
        if "IssTotUSD" in cm: ex["iss_ratio"] = cm.IssTotUSD.rolling(30).sum() / cm.CapMrktCurUSD
        if "ROI30d" in cm: ex["roi30"] = cm.ROI30d
    cb = ld(f"cb_{sym[:3].lower()}_1d")
    if cb is not None:
        spot = core.bars(sym, "1d").close  # 선물 종가 대신 현물이 이상적이나 여기선 선물 종가 사용(근사)
        ex["cb_prem"] = (cb.close.reindex(idx) / spot - 1)
    dv = ld(f"dvol_{sym[:3].lower()}_1h")
    if dv is not None:
        d1 = dv.close.resample("1D").last().reindex(idx); ex["dvol"] = d1; ex["vrp"] = d1 / 100 - rv
    for k, s in ex.items():
        F[k] = s.shift(1)      # 외부 데이터는 1일 지연
    F["dow"] = idx.dayofweek
    return x, F


def zexp(s, minp=365):
    return (s - s.expanding(minp).mean()) / s.expanding(minp).std()


def cond_variants(z):
    zz = z.to_numpy()
    out = {}
    for k in (0.0, 0.5, 1.0):
        out[f"z>{k}"] = np.nan_to_num(zz > k).astype(float)
        out[f"z<-{k}"] = np.nan_to_num(zz < -k).astype(float)
    return out


def wf(R, idx, first=2022, minyr=1.5):
    oos, used, picks = core.walk_forward(R, idx, "1d", first, minyr)
    if used.sum() < 365: return None
    ro = oos[used]; m = core.metrics(ro, "1d")
    yrs = idx.year.to_numpy()[used]
    return ro, used, {"sh": round(m["sharpe"], 2), "ann": round(m["ann"], 3), "mdd": round(m["mdd"], 3),
                      "yrs+": int(sum(ro[yrs == Y].mean() > 0 for Y in set(yrs))), "yrs": len(set(yrs))}


def main():
    out = {"A": {}, "B": {}, "C": {}}
    for sym in S.SYMS:
        x, F = build(sym)
        idx, C, fund = x.idx, x.C.to_numpy(), x.fund
        base_e = F["ens"].to_numpy()
        Rbase = core.run(base_e, C, fund)
        bh = x.r.to_numpy()
        # ---- A, B: 특징별 단독 / 필터
        feats = [c for c in F.columns if c not in ("ens", "dow", "ret1")]
        rows = []
        for f in feats:
            z = zexp(F[f]); cv = cond_variants(z)
            names = list(cv)
            Ra = np.array([core.run(cv[n], C, fund) for n in names]).T                  # A: 단독 롱플랫
            Rb = np.array([core.run(base_e * cv[n], C, fund) for n in names]).T         # B: 추세 * 조건
            ra, rb = wf(Ra, idx), wf(Rb, idx)
            if ra is None: continue
            m = ra[1]
            base_m = core.metrics(Rbase[m], "1d"); bh_m = core.metrics(bh[m], "1d")
            rows.append({"feature": f, "avail_from": str(z.first_valid_index().date()) if z.first_valid_index() is not None else None,
                         "A": ra[2], "B": rb[2] if rb else None, "base_sh": round(base_m["sharpe"], 2), "base_mdd": round(base_m["mdd"], 3),
                         "bh_sh": round(bh_m["sharpe"], 2), "A_pos_frac": round(float(((Ra[m].mean(0) / np.where(Ra[m].std(0) > 0, Ra[m].std(0), 1)) > 0).mean()), 2),
                         "B_pos_vs_base": round(float(((Rb[m].mean(0) / np.where(Rb[m].std(0) > 0, Rb[m].std(0), 1) * np.sqrt(365)) > base_m["sharpe"]).mean()), 2)})
        out["A"][sym] = rows
        print(sym, "A/B done", len(rows), flush=True)
        # ---- C: ML
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        X = F.drop(columns=["ens", "dow"]).copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        # 순위 변환(확장 창)으로 스케일 문제 회피: 단순히 expanding z
        Xz = X.apply(lambda s: zexp(s, 180)).clip(-5, 5)
        Xz["dow"] = F["dow"]
        nxt = pd.Series(C).pct_change().shift(-1).to_numpy()   # 내일 수익률
        y = (nxt > 0).astype(float); y[np.isnan(nxt)] = np.nan
        yrs = idx.year.to_numpy()
        res = {}
        prim = (base_e > 0.5)
        h5 = pd.Series(C).pct_change(5).shift(-5).to_numpy()   # 향후 5일 수익률
        y_meta = (h5 > 0).astype(float); y_meta[np.isnan(h5)] = np.nan

        def run_model(label, mask_train_fn, target, shuffle=False, seed=0, model="hgb"):
            pos = np.zeros(len(idx)); rng = np.random.default_rng(seed)
            for Y in range(2022, yrs.max() + 1):
                tr = (yrs < Y) & mask_train_fn & ~np.isnan(target) & Xz.notna().sum(axis=1).to_numpy().astype(bool)
                te = yrs == Y
                if tr.sum() < 300: continue
                Xt = Xz[tr].fillna(0).to_numpy(); yt = target[tr].copy()
                if shuffle: yt = rng.permutation(yt)
                mdl = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=120, min_samples_leaf=40, l2_regularization=5.0, random_state=seed) \
                    if model == "hgb" else LogisticRegression(C=0.05, max_iter=500)
                mdl.fit(Xt, yt)
                pos[te] = mdl.predict_proba(Xz[te].fillna(0).to_numpy())[:, 1]
            return pos

        allm = np.ones(len(idx), bool)
        for model in ("hgb", "logit"):
            p_direct = run_model("direct", allm, y, model=model)
            p_meta = run_model("meta", prim, y_meta, model=model)
            for tau in (0.50, 0.52, 0.55):
                pd_ = (p_direct > tau).astype(float)              # 직접 예측 롱플랫
                pm_ = base_e * (p_meta > tau)                     # 메타라벨링: 추세 노출 * 2차 모델 통과
                used = yrs >= 2022
                for nm, pp in (("직접예측", pd_), ("메타라벨링", pm_)):
                    Rr = core.run(pp, C, fund)
                    res[f"{model} {nm} τ={tau}"] = {**{k: round(v, 3) for k, v in core.metrics(Rr[used], "1d").items()},
                                                   "trades": int((np.abs(np.diff(pp, prepend=0)) >= 0.25).sum() / ((used.sum()) / 365))}
        bm = core.metrics(Rbase[yrs >= 2022], "1d")
        res["기준: 추세 앙상블(필터 없음)"] = {k: round(v, 3) for k, v in bm.items()}
        res["기준: 매수후보유"] = {k: round(v, 3) for k, v in core.metrics(bh[yrs >= 2022], "1d").items()}
        # 라벨 섞기 대조군 (hgb 메타라벨링/직접, τ=0.52)
        null_d, null_m = [], []
        for sd in range(12):
            pd_ = run_model("direct", allm, y, shuffle=True, seed=sd); pm_ = run_model("meta", prim, y_meta, shuffle=True, seed=sd)
            used = yrs >= 2022
            null_d.append(core.metrics(core.run((pd_ > 0.52).astype(float), C, fund)[used], "1d")["sharpe"])
            null_m.append(core.metrics(core.run(base_e * (pm_ > 0.52), C, fund)[used], "1d")["sharpe"])
        res["대조군(라벨섞기) 직접예측 Sharpe 분포"] = {"mean": round(float(np.mean(null_d)), 2), "p95": round(float(np.percentile(null_d, 95)), 2), "max": round(float(np.max(null_d)), 2)}
        res["대조군(라벨섞기) 메타라벨링 Sharpe 분포"] = {"mean": round(float(np.mean(null_m)), 2), "p95": round(float(np.percentile(null_m, 95)), 2), "max": round(float(np.max(null_m)), 2)}
        out["C"][sym] = res
        print(sym, "C done", flush=True)
    Path("logs/edge2.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
    for sym in S.SYMS:
        print("=====", sym, "A: 단독 타이밍 / B: 추세×필터  (표본 밖 2022~, WF)")
        for r in sorted(out["A"][sym], key=lambda r: -(r["B"]["sh"] if r["B"] else -9)):
            a, b = r["A"], r["B"]
            print(f"{r['feature']:14} 시작 {r['avail_from']} | A Sh {a['sh']:5} 양수연도 {a['yrs+']}/{a['yrs']} 고원 {r['A_pos_frac']:4} | B(추세×필터) Sh {b['sh'] if b else None} mdd {b['mdd'] if b else None} 기준추세 Sh {r['base_sh']} mdd {r['base_mdd']} 필터가 기준 초과한 파라미터 비율 {r['B_pos_vs_base']}")
        print("--- C: ML")
        for k, v in out["C"][sym].items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
