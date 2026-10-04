"""R132 1단 — "구할 수 있는 지표는 전부" 패널 빌더. r132_indicators 라이브러리로 세 결정-행 패널
(D1 창 시작·D2 첫 이탈·D3 창 내부 5분격자)의 특징을 채운다.

결정 규칙(엄수): 결정은 분 m의 "종가 시점". 상위 시간대·15분 격자·일 데이터 조인 규칙은 전부
  r132_indicators.py 상단 docstring과 동일(gather_bar/grid_asof_idx/daily_asof).

패널:
  D1 "창 시작": r90_wf p≥th30, 종목별 240분 겹침 제거(R129.dedup_time) — 창마다 1행, 결정분 m=t−1.
    라벨: side2pct(ref 대비 ±2% 240분 내 첫 터치 방향), bar05/bar10(ref 기준 ±0.5%/±1% 480분 배리어).
  D2 "첫 이탈": r131_events.parquet 사건 키(sym,t,f,s,y,X,cont) 재사용 — 결정분 m=f. SIGNED 특징은
    s로 부호 정렬한 _s 열을 추가로 둔다(원본·정렬판 둘 다 보존).
  D3 "창 내부 5분격자": 같은 창에서 m = t−1, t+4, t+9, … < t+240 (240분/5 = 48칸 간격, 49행/창).
    라벨: bar05/bar10 (close[m] 기준 480분 배리어). window_id = (sym,t) 로 군집 가능.

사용(oracle1 전용): nohup .venv/bin/python research/r132_build.py > logs/r132_build.log 2>&1 &"""
import sys, time, json, gc, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
import r129_onset_anatomy as R129
import r129_feats as RF
import r132_indicators as I
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)

SYMS = ("BTCUSDT", "ETHUSDT")
ROUND_STEPS = {"BTCUSDT": (1000.0, 500.0, 100.0), "ETHUSDT": (100.0, 50.0, 10.0)}
W, H = 240, 480
MICRO_LB = (1, 5, 10, 30, 60)
SEED = 132

# EXT(r97_feat/r54_dfeat/r119_feats) 방향성 있는 부분집합 — "변화·수익·스큐·상대" 류만 signed로 표시.
EXT_SIGNED_SHORT = {
    "TSM_r3d", "TSM_r7d", "TSM_r30d", "TSM_r3d_v", "TSM_r7d_v", "TSM_r30d_v", "MA_d1d", "MA_d7d", "MA_d30d",
    "MA_slope1d", "RNG_dhi30", "RNG_dlo30", "SEMI_skew24h", "VDIR_tbr24h", "XA_oth1h", "XA_rel1h", "XA_oth4h",
    "XA_rel4h", "XA_oth24h", "XA_rel24h", "XA_ethbtc_z", "IVX_ddvol24", "IVX_dvol_rv", "BAS_dbasis7",
    "FLOW_netout", "FLOW_mvrv", "FLOW_adr", "FLOW_dfng7", "MAC_dvix5", "MAC_usd5", "MAC_usd20", "MAC_spx1",
    "MAC_spx5", "MAC_ndq5", "MAC_dhy5", "MAC_dy10_5", "KIM_prem", "KIM_dprem24",
    "drr4", "drr24", "datm1", "datm4", "rrz", "dcbp60", "cbpz",
    "NEWF_dkp5", "NEWF_dkp15", "NEWF_dkp60", "NEWF_cz1d", "NEWF_pnet5", "NEWF_pnet15", "NEWF_pnet60",
    "NEWF_snet5", "NEWF_snet15", "NEWF_snet60", "NEWF_div5", "NEWF_div15", "NEWF_div60", "NEWF_ulead5",
    "NEWF_cdp15", "NEWF_cdp60", "NEWF_doi60", "NEWF_dfund", "etf_flow_z90",
}


def utc(ix):
    ix = pd.DatetimeIndex(ix)
    return ix.tz_convert("UTC") if ix.tz is not None else ix.tz_localize("UTC")


@numba.njit(cache=True)
def first_touch(entry_idx, h, l, c, X, Wn):
    """entry_idx[q] 분 종가(=진입가) 기준, 다음 분부터 Wn분 안에 ±X% 중 어느 쪽을 먼저 터치하는가.
    +1=위 먼저, −1=아래 먼저, 0=같은 분에 양쪽 다/Wn분 안에 미터치/데이터 부족(미결판 — 전부 0)."""
    n = len(c); nQ = len(entry_idx)
    out = np.zeros(nQ, np.int8)
    for q in range(nQ):
        i = entry_idx[q]
        if i < 0 or i + 1 + Wn > n:
            out[q] = 0
            continue
        entry = c[i]
        up_th = entry * (1 + X); dn_th = entry * (1 - X)
        res = 0
        for j in range(i + 1, i + 1 + Wn):
            up = h[j] >= up_th; dn = l[j] <= dn_th
            if up and dn:
                res = 0; break
            if up:
                res = 1; break
            if dn:
                res = -1; break
        out[q] = res
    return out


# ============================================================ 결정-행 테이블 ============================================================

def build_windows():
    """D1/D3 공통 창: r90_wf p≥th30, 종목별 240분 겹침 제거(시간순 그리디, R129.dedup_time 재사용, 새로 안 만듦)."""
    T = pd.read_parquet(R / "data" / "r90_wf.parquet")
    ev = T[T.p >= T.th30].sort_values(["sym", "t"]).copy()
    ev = R129.dedup_time(ev, W)
    ev["tier"] = np.where(ev.p >= ev.th10, "T10", np.where(ev.p >= ev.th20, "T20", "T30"))
    return ev[["sym", "t", "ts", "yr", "p", "tier"]].reset_index(drop=True)


def build_D1(WIN, K):
    rows = []
    for sym, g in WIN.groupby("sym"):
        k = K[sym]; h, l, c = k.high.values, k.low.values, k.close.values
        m = g.t.values.astype(np.int64) - 1
        d = g.copy()
        d["m"] = m
        d["ts_dec"] = k.ts.values[m]
        d["side2pct"] = first_touch(m, h, l, c, 0.02, W)
        d["bar05"] = first_touch(m, h, l, c, 0.005, H)
        d["bar10"] = first_touch(m, h, l, c, 0.01, H)
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def build_D2(K):
    EV = pd.read_parquet(R / "data" / "r131_events.parquet")
    D2 = EV[["sym", "t", "yr", "f", "s", "y", "X", "cont"]].rename(columns={"f": "m"}).copy()
    for sym, g in D2.groupby("sym"):
        k = K[sym]
        D2.loc[g.index, "ts_dec"] = k.ts.values[g.m.values.astype(np.int64)]
    return D2


def build_D3(WIN, K):
    offsets = -1 + 5 * np.arange(49)  # t−1, t+4, …, t+239 (49개, 전부 < t+240)
    rows = []
    for sym, g in WIN.groupby("sym"):
        t = g.t.values.astype(np.int64)
        m_grid = t[:, None] + offsets[None, :]
        rows.append(pd.DataFrame({"sym": sym, "t": np.repeat(t, len(offsets)), "yr": np.repeat(g.yr.values, len(offsets)),
                                   "tier": np.repeat(g.tier.values, len(offsets)), "m": m_grid.ravel()}))
    D3 = pd.concat(rows, ignore_index=True)
    D3["bar05"] = np.int8(0); D3["bar10"] = np.int8(0); D3["ts_dec"] = pd.NaT; D3["valid"] = False
    for sym, g in D3.groupby("sym"):
        k = K[sym]; h, l, c = (k[col].values.astype(np.float64) for col in ("high", "low", "close"))
        m = g.m.values.astype(np.int64)
        ok = (m >= 0) & (m < len(c))
        idx_ok = g.index[ok]; m_ok = m[ok]
        D3.loc[idx_ok, "bar05"] = first_touch(m_ok, h, l, c, 0.005, H)
        D3.loc[idx_ok, "bar10"] = first_touch(m_ok, h, l, c, 0.01, H)
        D3.loc[idx_ok, "ts_dec"] = k.ts.values[m_ok]
        D3.loc[idx_ok, "valid"] = True
    D3 = D3[D3.valid].drop(columns=["valid"]).reset_index(drop=True)
    return D3


# ============================================================ 특징 — 심볼당 유니크 결정분에서 한 번만 계산 ============================================================

def _rsi14_macdhist(close):
    """교차종목(family8)용 — RSI14·MACD 히스토그램만 가볍게 재계산(family1의 1m 블록과 별도 — 교차 쌍을 먼저 만들기 위해)."""
    C = pd.Series(close)
    diff = C.diff(); gain = diff.clip(lower=0); loss = (-diff).clip(lower=0)
    ag = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    al = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rsi = 100 - 100 / (1 + ag / (al + 1e-9))
    ema12 = C.ewm(span=12, adjust=False, min_periods=12).mean(); ema26 = C.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = (ema12 - ema26) / C; sig = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    return rsi.values.astype(np.float32), (macd - sig).values.astype(np.float32)


def compute_features(sym, minutes, K, FAM, SIGNED, t0):
    """sym 의 유니크 결정분(minutes, 정수 배열)에서 family 1·2·3·4·5·6·7 특징을 전부 계산해 반환.
    FAM/SIGNED는 호출부가 들고 있는 dict/set 에 바로 채운다(열이름→family, signed 열이름 집합)."""
    k = K[sym]; n = len(k)
    o, h, l, c, v = (k[col].values.astype(np.float64) for col in ("open", "high", "low", "close", "volume"))
    ts_full = utc(k.ts)
    minutes = np.asarray(minutes, dtype=np.int64)
    cols = {}

    # ---- family1: 고전 기술 지표, 6개 시간대 ----
    for tf, Pp in I.TIMEFRAMES:
        oo, hh, ll, cc, vv = I.bars_from_1m(o, h, l, c, v, Pp)
        blk = I.classical_indicators(oo, hh, ll, cc, vv)
        for short, arr in blk.items():
            name = f"TA_{tf}_{short}"
            cols[name] = I.gather_bar(arr, Pp, minutes)
            FAM[name] = f"TA_{tf}"
            if short in I.TA_SIGNED: SIGNED.add(name)
        del blk, oo, hh, ll, cc, vv
        gc.collect()
    P(f"    {sym} family1(TA, 6개 시간대) 완료 ({time.time()-t0:.0f}s)")

    # ---- family2: 레벨 ----
    lvl = I.level_features(ts_full, o, h, l, c, v, ROUND_STEPS[sym])
    for short, arr in lvl.items():
        name = f"LVL_{short}"
        cols[name] = arr[minutes]
        FAM[name] = "LVL"
        if short in I.LVL_SIGNED: SIGNED.add(name)
    del lvl; gc.collect()
    P(f"    {sym} family2(LVL) 완료 ({time.time()-t0:.0f}s)")

    # ---- family3: 미시구조(R129 재사용, 룩백 1/5/10/30/60) ----
    panel = RF.minute_panel(sym, minutes, lookbacks=MICRO_LB)
    for col in panel.columns:
        name = f"MS_{col}"
        cols[name] = panel[col].values.astype(np.float32)
        FAM[name] = "MS"
        base = col.rsplit("_", 1)[0]
        if base in RF.SIGNED: SIGNED.add(name)
    del panel; gc.collect()
    P(f"    {sym} family3(MS) 완료 ({time.time()-t0:.0f}s)")

    # ---- family4: 파생상품·포지셔닝 ----
    drv = I.derivatives_features(sym, ts_full)
    for short, arr in drv.items():
        name = f"DRV_{short}"
        cols[name] = arr[minutes]
        FAM[name] = "DRV"
        if short in I.DRV_SIGNED: SIGNED.add(name)
    del drv; gc.collect()
    P(f"    {sym} family4(DRV) 완료 ({time.time()-t0:.0f}s)")

    # ---- family5: 변동성(실현변동성·vol-of-vol·R90 p/pr) ----
    rv = I.realized_vol_features(c)
    for short, arr in rv.items():
        cols[f"VOL_{short}"] = arr[minutes]; FAM[f"VOL_{short}"] = "VOL"
    del rv
    t90, c90, v90 = I.load_r90_pr(sym)
    g90 = I.gather_grid(t90, v90, minutes)
    for j, short in enumerate(c90):
        cols[f"VOL_r90_{short}"] = g90[:, j]; FAM[f"VOL_r90_{short}"] = "VOL"
    del g90, v90
    gc.collect()
    P(f"    {sym} family5(VOL) 완료 ({time.time()-t0:.0f}s)")

    # ---- family6: r97_feat + r54_dfeat(15분 격자, t_g≤m+1) + r119 newfeats(분 해상도) + ETF(1일 지연) ----
    t97, c97, v97 = I.load_grid_table(R / "data" / "r97_feat.parquet", sym)
    g97 = I.gather_grid(t97, v97, minutes)
    for j, short0 in enumerate(c97):
        short = short0.replace("|", "_")
        name = f"EXT_{short}"
        cols[name] = g97[:, j]; FAM[name] = "EXT"
        if short in EXT_SIGNED_SHORT: SIGNED.add(name)
    del g97, v97, t97
    t54, c54, v54 = I.load_grid_table(R / "data" / "r54_dfeat.parquet", sym)
    g54 = I.gather_grid(t54, v54, minutes)
    for j, short in enumerate(c54):
        name = f"EXT_{short}"
        cols[name] = g54[:, j]; FAM[name] = "EXT"
        if short in EXT_SIGNED_SHORT: SIGNED.add(name)
    del g54, v54, t54
    gc.collect()
    cnew, vnew = I.newfeats_grid(sym, ts_full)
    for j, short in enumerate(cnew):
        name = f"EXT_NEWF_{short}"
        cols[name] = vnew[minutes, j]; FAM[name] = "EXT"
        if f"NEWF_{short}" in EXT_SIGNED_SHORT: SIGNED.add(name)
    del vnew; gc.collect()
    etf = I.load_etf_flows()
    col97 = "btc_total" if sym == "BTCUSDT" else "eth_total"
    s_etf = etf[col97].astype(float)
    cols["EXT_etf_flow"] = I.daily_asof(s_etf, ts_full)[minutes]; FAM["EXT_etf_flow"] = "EXT"
    sz = (s_etf - s_etf.rolling(90, min_periods=10).mean()) / (s_etf.rolling(90, min_periods=10).std() + 1e-9)
    cols["EXT_etf_flow_z90"] = I.daily_asof(sz, ts_full)[minutes]; FAM["EXT_etf_flow_z90"] = "EXT"; SIGNED.add("EXT_etf_flow_z90")
    P(f"    {sym} family6(EXT) 완료 ({time.time()-t0:.0f}s)")

    # ---- family7: 계절성 ----
    seas = I.seasonality_features(ts_full[minutes])
    for short, arr in seas.items():
        cols[f"SEAS_{short}"] = arr; FAM[f"SEAS_{short}"] = "SEAS"

    for key in list(cols.keys()):
        cols[key] = np.asarray(cols[key], dtype=np.float32)
    return cols


def build_cross(sym, minutes, CROSS, FAM, SIGNED):
    oth = "ETHUSDT" if sym == "BTCUSDT" else "BTCUSDT"
    blk = I.cross_asset_features(CROSS[oth]["close"], CROSS[oth]["rsi14"], CROSS[oth]["macd_hist"])
    out = {}
    for short, arr in blk.items():
        name = f"CROSS_{short}"
        out[name] = arr[minutes]; FAM[name] = "CROSS"
        if short in I.CROSS_SIGNED: SIGNED.add(name)
    out["CROSS_ethbtc_ratio_z"] = CROSS["ratioz"][minutes]; FAM["CROSS_ethbtc_ratio_z"] = "CROSS"; SIGNED.add("CROSS_ethbtc_ratio_z")
    return out


# ============================================================ 룩어헤드 검증 ============================================================

LOOKAHEAD_LOOKBACK = 760000  # 가장 긴 롤링(봉해상도 ~500봉 × 1d=1440분) 을 전부 덮는 여유 — 1440의 배수로 내림(모든 시간대 봉 경계 정렬 유지)


def lookahead_check(K, FEATALL, n_sample=50, seed=SEED):
    """결정분 m 이후 데이터를 전부 잘라내고(배열을 길이 LOOKAHEAD_LOOKBACK 안에서 m까지만 절단) family1(6개
    시간대)·family2(레벨) 을 다시 계산해, 운영 계산값(FEATALL)과 동일한지 확인. (sym,m) 무작위 50개.
    절단 시작점을 1440의 배수로 내림해 모든 시간대의 봉 경계(1440으로 전부 나눠짐)가 운영과 그대로 맞도록 한다."""
    pool = FEATALL[["sym", "m"]].drop_duplicates()
    pool = pool.sample(n=min(n_sample, len(pool)), random_state=seed).reset_index(drop=True)
    n_fail = 0; checked_cols = 0; fail_detail = []
    for row in pool.itertuples():
        sym, m = row.sym, int(row.m)
        k = K[sym]
        if m < 2000 or m + 1 > len(k):  # 워밍업 구간은 스킵(둘 다 NaN이라 비교 의미 없음)
            continue
        start = max(0, ((m - LOOKAHEAD_LOOKBACK) // 1440) * 1440)
        local_m = m - start
        sl = slice(start, m + 1)
        o = k.open.values[sl].astype(np.float64); h = k.high.values[sl].astype(np.float64)
        l = k.low.values[sl].astype(np.float64); c = k.close.values[sl].astype(np.float64)
        v = k.volume.values[sl].astype(np.float64); ts_trunc = utc(k.ts)[sl]
        prod = FEATALL[(FEATALL.sym == sym) & (FEATALL.m == m)].iloc[0]

        for tf, Pp in I.TIMEFRAMES:
            oo, hh, ll, cc, vv = I.bars_from_1m(o, h, l, c, v, Pp)
            blk = I.classical_indicators(oo, hh, ll, cc, vv)
            for short, arr in blk.items():
                name = f"TA_{tf}_{short}"
                if name not in prod.index: continue
                got = I.gather_bar(arr, Pp, np.array([local_m]))[0]
                checked_cols += 1
                a, b = got, prod[name]
                if not (np.isnan(a) and np.isnan(b)) and not np.isclose(a, b, rtol=1e-3, atol=1e-4):
                    n_fail += 1; fail_detail.append((sym, m, name, float(a), float(b)))
            del blk

        lvl = I.level_features(ts_trunc, o, h, l, c, v, ROUND_STEPS[sym])
        for short, arr in lvl.items():
            name = f"LVL_{short}"
            if name not in prod.index: continue
            checked_cols += 1
            a, b = arr[local_m], prod[name]
            if not (np.isnan(a) and np.isnan(b)) and not np.isclose(a, b, rtol=1e-3, atol=1e-4):
                n_fail += 1; fail_detail.append((sym, m, name, float(a), float(b)))
        del lvl; gc.collect()

    P(f"\n===== 룩어헤드 검증: {len(pool)}개 (sym,m) 표본, 특징값 비교 {checked_cols}개, 불일치 {n_fail}개")
    if fail_detail:
        for d in fail_detail[:15]: P("  불일치:", d)
        P("룩어헤드 검증: FAIL")
    else:
        P("룩어헤드 검증: PASS")
    return n_fail == 0


# ============================================================ 저장·보고 ============================================================

def drop_bad_cols(df, feat_cols, tag):
    const = []; highnan = []
    for c in feat_cols:
        s = df[c]
        nan_rate = s.isna().mean()
        if nan_rate > 0.95:
            highnan.append(c); continue
        if s.dropna().nunique() <= 1:
            const.append(c)
    drop = const + highnan
    if drop:
        P(f"  [{tag}] 상수 열 {len(const)}개, 결측>95% 열 {len(highnan)}개 → 제거: " + ", ".join(drop[:40]) + (" ..." if len(drop) > 40 else ""))
    return df.drop(columns=drop), [c for c in feat_cols if c not in drop]


def coverage_report(df, feat_cols, fam_of, tag):
    fams = sorted(set(fam_of.get(c, "?") for c in feat_cols))
    rows = []
    yrs = sorted(df.yr.dropna().unique()) if "yr" in df.columns else []
    for fam in fams:
        cc = [c for c in feat_cols if fam_of.get(c) == fam]
        row = {"family": fam, "n_feat": len(cc)}
        for y in yrs:
            sub = df[df.yr == y]
            row[f"cov_{int(y)}"] = round(100 * (1 - sub[cc].isna().mean().mean()), 1) if len(sub) else np.nan
        rows.append(row)
    T = pd.DataFrame(rows)
    P(f"\n===== [{tag}] family별 특징 수 · 연도별 비결측률(%)")
    pd.set_option("display.width", 220)
    P(T.to_string(index=False))


def main():
    t0 = time.time()
    K = {s: store.load("klines_1m", s) for s in SYMS}
    P(f"klines 로드 완료 ({time.time()-t0:.0f}s): " + " ".join(f"{s}={len(K[s]):,}" for s in SYMS))

    WIN = build_windows()
    P(f"창(window) {len(WIN)}개(심볼별): " + " ".join(f"{s}={ (WIN.sym==s).sum()}" for s in SYMS))
    D1 = build_D1(WIN, K); P(f"D1 {len(D1)}행 ({time.time()-t0:.0f}s)")
    D2 = build_D2(K); P(f"D2 {len(D2)}행 ({time.time()-t0:.0f}s)")
    D3 = build_D3(WIN, K); P(f"D3 {len(D3)}행 ({time.time()-t0:.0f}s)")

    CLOSE = {s: K[s].close.values.astype(np.float64) for s in SYMS}
    RSI = {}; MACDH = {}
    for s in SYMS:
        RSI[s], MACDH[s] = _rsi14_macdhist(CLOSE[s])
    CROSS = {s: {"close": CLOSE[s], "rsi14": RSI[s], "macd_hist": MACDH[s]} for s in SYMS}
    CROSS["ratioz"] = I.ethbtc_ratio_z(CLOSE["BTCUSDT"], CLOSE["ETHUSDT"])
    P(f"교차종목 보조열 준비 완료 ({time.time()-t0:.0f}s)")

    FAM = {}; SIGNED = set()
    FEAT_PARTS = []
    for sym in SYMS:
        mins = np.unique(np.concatenate([
            D1.loc[D1.sym == sym, "m"].values.astype(np.int64),
            D2.loc[D2.sym == sym, "m"].values.astype(np.int64),
            D3.loc[D3.sym == sym, "m"].values.astype(np.int64),
        ]))
        P(f"\n[{sym}] 유니크 결정분 {len(mins):,}개 특징 계산 시작 ({time.time()-t0:.0f}s)")
        cols = compute_features(sym, mins, K, FAM, SIGNED, t0)
        cross = build_cross(sym, mins, CROSS, FAM, SIGNED)
        cols.update(cross)
        dfp = pd.DataFrame(cols); dfp.insert(0, "m", mins); dfp.insert(0, "sym", sym)
        FEAT_PARTS.append(dfp)
        del cols, cross
        gc.collect()
        P(f"[{sym}] 특징 {dfp.shape[1]-2}열 완료 ({time.time()-t0:.0f}s)")

    FEATALL = pd.concat(FEAT_PARTS, ignore_index=True)
    del FEAT_PARTS; gc.collect()
    FEAT_COLS = [c for c in FEATALL.columns if c not in ("sym", "m")]
    P(f"\n전체 특징 {len(FEAT_COLS)}개(가족 {len(set(FAM.values()))}개) ({time.time()-t0:.0f}s)")

    ok = lookahead_check(K, FEATALL, n_sample=50)

    PANELS = {}
    for tag, dd in (("D1", D1), ("D2", D2), ("D3", D3)):
        P(f"\n[{tag}] 결정-행과 특징 결합 중 ({time.time()-t0:.0f}s)")
        merged = dd.merge(FEATALL, on=["sym", "m"], how="left")
        if tag == "D2":
            sgn = merged.s.values.astype(np.float32)
            for c in sorted(SIGNED & set(FEAT_COLS)):
                merged[f"{c}_s"] = merged[c].values * sgn
        merged, kept = drop_bad_cols(merged, [c for c in merged.columns if c in FEAT_COLS or c.endswith("_s")], tag)
        PANELS[tag] = (merged, kept)
        P(f"[{tag}] {len(merged):,}행 x {len(kept)}특징(결정행렬 포함 전체 {merged.shape[1]}열) ({time.time()-t0:.0f}s)")
        coverage_report(merged, [c for c in kept if c in FAM], FAM, tag)

    for tag in ("D1", "D2", "D3"):
        merged, kept = PANELS[tag]
        out = merged.copy()
        for c in out.columns:
            if out[c].dtype == np.float64:
                out[c] = out[c].astype(np.float32)
        out.to_parquet(R / "data" / f"r132_{tag}.parquet", index=False)
        P(f"저장: data/r132_{tag}.parquet ({len(out):,}행 x {out.shape[1]}열) ({time.time()-t0:.0f}s)")

    ALL_KEPT = sorted(set().union(*[set(kept) for _, kept in PANELS.values()]))
    FAMILIES = {}
    for c in ALL_KEPT:
        fam = FAM.get(c.removesuffix("_s") if c.endswith("_s") else c, "?")
        FAMILIES.setdefault(fam, []).append(c)
    (R / "data" / "r132_families.json").write_text(json.dumps(FAMILIES, indent=1, ensure_ascii=False))
    P(f"저장: data/r132_families.json (family {len(FAMILIES)}개, 특징 합계 {sum(len(v) for v in FAMILIES.values())}개)")

    P(f"\n===== 요약")
    for tag in ("D1", "D2", "D3"):
        merged, kept = PANELS[tag]
        P(f"  {tag}: {len(merged):,}행, 특징 {len(kept)}개")
    P(f"  룩어헤드 검증: {'PASS' if ok else 'FAIL'}")
    P(f"총 실행 시간 {time.time()-t0:.0f}s")
    (R / "logs" / "r132.summary").write_text(
        f"R132 1단 완료 ({time.time()-t0:.0f}s). D1={len(PANELS['D1'][0])} D2={len(PANELS['D2'][0])} D3={len(PANELS['D3'][0])} "
        f"lookahead={'PASS' if ok else 'FAIL'}\n")


if __name__ == "__main__":
    main()
