"""R127 — 주간 단위 '꾸준하고 큰 수익' 증분: 어떤 후보 엣지를 베이스라인에 더하면 주간 수익 분포가 더 꾸준해지는가.

베이스라인(공유 계좌 엔진 R118, 딥 = R119 v2 모델 pc10bp, 돌파 = z0.33 TOP8 pnet15):
  D = r119_dip_model_v2.dip_events(W, 0.001)[COLS], B = std_breakout(), base = concat(D, B)
  d, pk = run_multi(base, 3, 12, {1: 2/48})
먼저 이 베이스라인을 재현해(2025~26 주간 양수 67%·중앙 +2.3%·25%분위 −1.5%·최악 −40%·샤프 0.34,
2022~24 양수 65%·중앙 +4.9%·샤프 0.32) 확인한다.

주간 수익 = 일간 계좌 수익을 복리해 W-SUN 리샘플. 구간: 2022~24 = [2022-01-01, 2024-01-01), 2025~26 = [2025-01-01, 2026-09-23).

Part A: R122/R123 엣지 공장 후보(학습 t≥2.5 & 검증(2024) 평균>0, 두 파일 모두 같은 규칙, 학습 t 상위 150 cap) 각각을
  R123 증분 검정과 같은 방식(메커니즘 + 특징 구간, 학습 분위 경계)으로 거래를 만들고, 베이스라인 딥 거래와 겹치는
  (종목, 진입분)을 뺀 뒤 채널 k=5로 비중 Wn ∈ {0.25,0.5,1,2} 추가. Wn 선택 = 2022~24 주간 샤프 최대
  (단, 2022~24 주간 양수 비율을 베이스라인보다 낮추면 안 됨). 이후 전진 선택(최대 5개, 채널 k=5,6,7…),
  기준 = 2022~24 주간 샤프 개선폭 ≥0.02.
Part B(사전 등록, 결과를 본 뒤 후보를 추가하지 않음): 베이스라인의 약점인 저변동 국면(딥 거래 희소)을 겨냥한
  새 메커니즘 3개(B1 얕은 딥 지정가, B2 같은 것 + TP0.5, B3 24h VWAP 레인지 평균회귀) — 저변동 국면 = 30일/365일
  변동성 비 <0.8(전일까지만 사용). 각 메커니즘을 단독 성과(연도별 거래당 bp·승률)와 베이스라인 증분(Part A와
  같은 절차)으로 평가.

출력: data/r127_A.parquet(Part A 후보 전수), data/r127_B.parquet(Part B 메커니즘 요약),
      data/r127_selected_events.parquet(최종 채택분 사건 — 표준 엔진 열 + label)
사용: python research/r127_consistency_increment.py"""
import sys, warnings, time
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store, costs
import r118_engine as G, r119_dip_model_v2 as V, r102_joint_account as J, r122_scan as SC, r26_hf_twosided as H26
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
t_ = time.time()

TR_LO, TR_HI = "2022-01-01", "2024-01-01"
TE_LO, TE_HI = "2025-01-01", "2026-09-23"
WN_GRID = (0.25, 0.5, 1.0, 2.0)
MAKER, TAKER = costs.MAKER, costs.TAKER


def weekly_metrics(d, lo, hi):
    """d: 일간 계좌 수익률 시계열(tz-UTC). 주간 = 일간 복리 후 W-SUN 리샘플. 샤프 = 평균/표준편차(연환산 없음)."""
    x = d[(d.index >= pd.Timestamp(lo, tz="UTC")) & (d.index < pd.Timestamp(hi, tz="UTC"))]
    if len(x) == 0:
        return dict(pos=np.nan, med=np.nan, p25=np.nan, worst=np.nan, sharpe=np.nan, posm=np.nan, medm=np.nan, nw=0)
    w = (1 + x).resample("W-SUN").prod() - 1
    m = (1 + x).resample("ME").prod() - 1
    return dict(pos=(w > 0).mean() * 100, med=w.median() * 100, p25=w.quantile(.25) * 100, worst=w.min() * 100,
                sharpe=(w.mean() / w.std() if w.std() > 0 else np.nan), posm=(m > 0).mean() * 100, medm=m.median() * 100, nw=len(w))


def fmt(m):
    return f"양수{m['pos']:.0f}% 중앙{m['med']:+.1f}% 25%분위{m['p25']:+.1f}% 최악{m['worst']:+.1f}% 샤프{m['sharpe']:.2f} | 월양수{m['posm']:.0f}% 월중앙{m['medm']:+.1f}%"


# ===== 0. 베이스라인 재현 =====
W32 = pd.read_parquet(R / "data" / "r32_wf.parquet"); W32["ts"] = pd.DatetimeIndex(W32.ts)
DIP = V.dip_events(W32, 0.001)[G.COLS]
BRK = G.std_breakout()
BASE = pd.concat([DIP, BRK], ignore_index=True)
d_base, pk_base = G.run_multi(BASE, 3, 12, {1: 2 / 48})
assert d_base is not None, "베이스라인 파산 — 재현 실패"
m_base_tr = weekly_metrics(d_base, TR_LO, TR_HI); m_base_te = weekly_metrics(d_base, TE_LO, TE_HI)
P(f"[재현 점검] 딥 사건 {len(DIP):,}, 돌파 사건 {len(BRK):,}, 최대 명목 {pk_base:.1f}배")
P(f"  2022~24: {fmt(m_base_tr)}  (기지값: 양수65% 중앙+4.9% 샤프0.32)")
P(f"  2025~26: {fmt(m_base_te)}  (기지값: 양수67% 중앙+2.3% 25%분위-1.5% 최악-40% 샤프0.34)")
ck = set(zip(DIP.s.values, DIP.e.values))   # 베이스라인 딥 거래(종목, 진입분) — 신규 후보와 겹침 제거용


def run_portfolio(channels):
    """channels: [(df(k 포함 전, COLS), k, Wn), ...]. 반환 (일간 수익 또는 None, 최대 명목)."""
    parts = [BASE]
    Wk = {1: 2 / 48}
    for df, k, w in channels:
        parts.append(df.assign(k=k)[G.COLS]); Wk[k] = w
    E = pd.concat(parts, ignore_index=True)
    return G.run_multi(E, 3, 12, Wk)


def pick_weight(cand_df, fixed_channels, next_k, base_pos_tr):
    """Wn 격자에서 2022~24 주간 샤프 최대(주간 양수 비율이 base_pos_tr보다 낮아지면 제외)."""
    best = None
    for Wn in WN_GRID:
        d, pk = run_portfolio(fixed_channels + [(cand_df, next_k, Wn)])
        if d is None: continue
        mt = weekly_metrics(d, TR_LO, TR_HI)
        if not np.isfinite(mt["sharpe"]): continue
        if mt["pos"] < base_pos_tr - 1e-9: continue
        if best is None or mt["sharpe"] > best[1]:
            best = (Wn, mt["sharpe"], mt["pos"], d)
    return best


# ===== Part A =====
P(f"\n===== Part A: 엣지 공장 후보 ({time.time()-t_:.0f}s)")
POOL = {}
for pfx in ("r122", "r123"):
    T = pd.read_parquet(R / "data" / f"{pfx}_trades.parquet")
    FX = pd.read_parquet(R / "data" / f"{pfx}_feats.parquet")
    S = pd.read_parquet(R / "data" / f"{pfx}_scan.parquet")
    C = S[(S.feat != "(없음)") & (S.tr_t >= 2.5) & (S.va_mu > 0)].copy()
    C["prefix"] = pfx
    POOL[pfx] = dict(T=T, FX=FX)
    P(f"  {pfx}: 거래 {len(T):,}, 통과 후보(t≥2.5,검증>0) {len(C):,}")
    C_all = C if pfx == "r122" else pd.concat([C_all, C], ignore_index=True)
C_all = C_all.sort_values("tr_t", ascending=False).head(150).reset_index(drop=True)
P(f"  학습 t 상위 150 cap 적용 → 후보 {len(C_all)}개 (최저 tr_t {C_all.tr_t.min():.2f})")

day_cache = {}
for pfx in ("r122", "r123"):
    day_cache[pfx] = POOL[pfx]["T"].e.values // 1440


def build_candidate(pfx, mech, feat, buck):
    T = POOL[pfx]["T"]; FX = POOL[pfx]["FX"]; day = day_cache[pfx]; tr = day < SC.D_TR
    m = (T.mech == mech).values; x = FX[feat].values
    q = {"Q1": .2, "L30": .3, "H30": .7, "Q5": .8}[buck]
    e_ = np.nanquantile(x[m & tr], q)
    sel = m & ((x <= e_) if buck in ("Q1", "L30") else (x >= e_))
    N = T[sel].copy(); N["s"] = (N.sym == "ETHUSDT").astype(int)
    N = N[[(s, e) not in ck for s, e in zip(N.s.values, N.e.values)]].copy()
    N["k"] = 5; N["mult"] = 1.0
    return N[G.COLS].reset_index(drop=True)


rows_A = []
cand_trades = {}
for i, c in enumerate(C_all.itertuples()):
    N = build_candidate(c.prefix, c.mech, c.feat, c.buck)
    lab = f"{c.prefix}:{c.mech}|{c.feat}|{c.buck}"
    r = dict(cand=lab, prefix=c.prefix, mech=c.mech, feat=c.feat, buck=c.buck, tr_t=c.tr_t, va_mu=c.va_mu, te_mu=c.te_mu, n_trades=len(N))
    if len(N) < 20:
        r.update(ok=False, reason="거래부족"); rows_A.append(r); continue
    best = pick_weight(N, [], 5, m_base_tr["pos"])
    if best is None:
        r.update(ok=False, reason="적합Wn없음"); rows_A.append(r); continue
    Wn, sh, pos, d = best
    r.update(ok=True, Wn=Wn, sharpe_alone_2224=sh, pos_alone_2224=pos)
    rows_A.append(r); cand_trades[lab] = N
    if (i + 1) % 25 == 0: P(f"  후보 {i+1}/{len(C_all)} 처리 ({time.time()-t_:.0f}s)")
O_A = pd.DataFrame(rows_A)
P(f"  유효 후보(Wn 선택됨) {int(O_A.get('ok', pd.Series(dtype=bool)).sum())}/{len(O_A)} ({time.time()-t_:.0f}s)")

# 전진 선택
pool_valid = O_A[O_A.ok].sort_values("sharpe_alone_2224", ascending=False).reset_index(drop=True)
selected = []  # (label, df, k, Wn)
cur_sharpe = m_base_tr["sharpe"]
greedy_log = []
avail = list(pool_valid.cand.values)
for rnd in range(5):
    best = None
    for lab in avail:
        row = pool_valid[pool_valid.cand == lab].iloc[0]
        k = 5 + len(selected)
        trial = [(df, kk, w) for _, df, kk, w in selected] + [(cand_trades[lab], k, row.Wn)]
        d, pk = run_portfolio(trial)
        if d is None: continue
        mt = weekly_metrics(d, TR_LO, TR_HI)
        if not np.isfinite(mt["sharpe"]): continue
        if best is None or mt["sharpe"] > best[1]:
            best = (lab, mt["sharpe"], row.Wn, d)
    if best is None: break
    lab, sh, Wn, d = best
    improve = sh - cur_sharpe
    P(f"  [전진선택 {rnd+1}] 최선 후보 {lab} Wn{Wn} → 2022~24 샤프 {sh:.3f} (개선 {improve:+.3f}) ({time.time()-t_:.0f}s)")
    if improve < 0.02:
        P(f"    개선폭 <0.02 → 중단"); break
    k = 5 + len(selected)
    selected.append((lab, cand_trades[lab], k, Wn))
    avail.remove(lab)
    cur_sharpe = sh
    mt_tr = weekly_metrics(d, TR_LO, TR_HI); mt_te = weekly_metrics(d, TE_LO, TE_HI)
    greedy_log.append(dict(step=rnd + 1, cand=lab, Wn=Wn, k=k, tr=mt_tr, te=mt_te))
    P(f"    채택 k={k}: 2022~24 {fmt(mt_tr)}")
    P(f"             2025~26 {fmt(mt_te)}")
if not selected:
    P("  전진 선택: 채택된 후보 없음(모두 개선폭 <0.02)")

O_A.to_parquet(R / "data" / "r127_A.parquet", index=False)
P(f"Part A 저장: data/r127_A.parquet ({len(O_A)}행), 채택 {len(selected)}개")

# ===== Part B =====
P(f"\n===== Part B: 저변동 국면 메커니즘 ({time.time()-t_:.0f}s)")
_REG = {}


def vol_regime(sym):
    """30일/365일 실현변동성 비(전일까지), r119_dip_model_v2.dip_events와 동일 정의."""
    if sym in _REG: return _REG[sym]
    k = store.load("klines_1m", sym)
    c = k.set_index("ts").close.resample("D").last(); r = np.log(c).diff()
    v30 = r.rolling(30).std(); reg = (v30 / v30.rolling(365, min_periods=120).median()).shift(1)
    _REG[sym] = reg
    return reg


def tag_regime(sym, ts):
    """ts: tz 유무 상관없이 받아 UTC로 맞춘 뒤 날짜 단위로 reg(전일까지 30/365일 변동성 비)를 붙인다.
    (.values로 꺼낸 배열은 tz 정보가 사라져 tz-aware reg.index와 바로 reindex하면 전부 NaN이 되는 버그가 있었다 — 수정.)"""
    reg = vol_regime(sym); day = pd.DatetimeIndex(ts)
    day = day.tz_localize("UTC") if day.tz is None else day.tz_convert("UTC")
    return reg.reindex(day.normalize()).values


def mk_B12(mode, g, H, label):
    """B1(mode0)·B2(mode1 g0.5 H0): r26 롱 전용 지정가 딥, 저변동 국면만."""
    out = []
    for sym in ("BTCUSDT", "ETHUSDT"):
        d = H26.D[sym]; mi = J.mins(d["idx"]); o = d["o"]; rv1 = d["rv1"]
        for B in (5, 15, 60):
            for kx in (1.5, 2.0, 2.5):
                a, b, r, m, yr = H26.events(sym, B, 1, kx, mode=mode, g=g, H=H)
                if len(a) == 0: continue
                ts = d["idx"].values[a]; rv = tag_regime(sym, ts)
                keep = np.isfinite(rv) & (rv < 0.8)
                if keep.sum() == 0: continue
                s0 = (a[keep] // B) * B
                px = o[s0] * (1 - kx * rv1[s0] * np.sqrt(B))  # r122_mechanisms.py와 같은 지정가 기준가(롱, 진입가 평가용)
                out.append(pd.DataFrame(dict(mech=f"{label} B{B} k{kx:g}", sym=sym, e=mi[a[keep]], x=mi[b[keep]] + 1,
                                             ret=r[keep].astype("float32"), mae=m[keep].astype("float32"),
                                             s=(1 if sym == "ETHUSDT" else 0), px=px, side=1, yr=yr[keep])))
    T = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["mech", "sym", "e", "x", "ret", "mae", "s", "px", "side", "yr"])
    T["k"] = 5; T["mult"] = 1.0
    return T


@numba.njit(cache=True)
def mr24_scan(o, h, l, c, vwap, thr, Hi, reg_ok, pen, touch_eps, hold):
    """pen = 진입 관통(불리한 방향, 1bp), touch_eps = VWAP 터치 확인 여유(2bp, r26 EPS와 동일 관례)."""
    n = len(Hi)
    EI = np.full(n, -1, np.int64); XI = np.full(n, -1, np.int64); SD = np.zeros(n, np.int64)
    PX = np.zeros(n); XP = np.zeros(n); AD = np.zeros(n); MKX = np.zeros(n, np.bool_)
    N = len(c); kk = 0
    for i in range(n):
        t = Hi[i]
        if not reg_ok[i] or t == 0 or t + hold >= N: continue
        vw = vwap[t]; th = thr[t]
        if not (np.isfinite(vw) and np.isfinite(th)) or th <= 0: continue
        up = vw * (1 + th); dn = vw * (1 - th)
        side = 0
        if o[t] > up: side = -1
        elif o[t] < dn: side = 1
        if side == 0: continue
        lvl = up if side < 0 else dn
        px = lvl * (1 - side * pen)   # 관통(매수는 더 비싸게, 매도는 더 싸게 — 불리한 방향)
        ad = px; xi = t + hold; xp = c[t + hold]; mkx = False
        for j in range(t + 1, t + hold + 1):
            if side == 1 and l[j] < ad: ad = l[j]
            if side == -1 and h[j] > ad: ad = h[j]
            if side == 1 and h[j] >= vw * (1 + touch_eps):
                xi = j; xp = vw; mkx = True; break
            if side == -1 and l[j] <= vw * (1 - touch_eps):
                xi = j; xp = vw; mkx = True; break
        EI[kk] = t; XI[kk] = xi; SD[kk] = side; PX[kk] = px; XP[kk] = xp; AD[kk] = abs(ad / px - 1); MKX[kk] = mkx
        kk += 1
    return EI[:kk], XI[:kk], SD[:kk], PX[:kk], XP[:kk], AD[:kk], MKX[:kk]


def mk_B3():
    """B3: 24h VWAP 레인지 평균회귀. 진입 메이커 2bp(1bp 관통), 청산 VWAP 터치 메이커 2bp 또는 4h 뒤 테이커 5bp+2bp 슬리피지."""
    out = []
    for sym in ("BTCUSDT", "ETHUSDT"):
        k = store.load("klines_1m", sym); k["ts"] = pd.DatetimeIndex(k.ts); mi = J.mins(k.ts)
        o, h, l, c = k.open.values, k.high.values, k.low.values, k.close.values
        qv_roll = pd.Series(k.quote_volume.values).rolling(1440).sum().shift(1).values
        v_roll = pd.Series(k.volume.values).rolling(1440).sum().shift(1).values
        vwap = qv_roll / v_roll
        lr = np.r_[np.nan, np.diff(np.log(c))]
        sig1 = pd.Series(lr).rolling(1440).std().shift(1).values
        thr = 1.5 * sig1 * np.sqrt(60)
        minute_of_day = (k.ts.dt.hour * 60 + k.ts.dt.minute).values
        Hi = np.where(minute_of_day % 60 == 0)[0].astype(np.int64); Hi = Hi[Hi >= 1440]
        rv = tag_regime(sym, k.ts.values[Hi]); reg_ok = np.isfinite(rv) & (rv < 0.8)
        EI, XI, SD, PX, XP, AD, MKX = mr24_scan(o, h, l, c, vwap, thr, Hi, reg_ok, 1e-4, 2e-4, 240)
        if len(EI) == 0: continue
        exv = np.where(MKX, XP, XP * (1 - SD * 2e-4))   # 메이커 터치는 VWAP 그대로, 테이커 종료는 2bp 슬리피지
        fee_out = np.where(MKX, MAKER, TAKER)
        ret = SD * (exv / PX - 1) - MAKER - fee_out
        yr = k.ts.dt.year.values[EI]
        out.append(pd.DataFrame(dict(mech="B3 VWAP레인지", sym=sym, e=mi[EI], x=mi[XI] + 1, ret=ret.astype("float32"),
                                     mae=AD.astype("float32"), s=(1 if sym == "ETHUSDT" else 0), px=PX, side=SD, yr=yr)))
    T = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["mech", "sym", "e", "x", "ret", "mae", "s", "px", "side", "yr"])
    Tf = J.add_funding(T[["sym", "e", "x"]].copy())
    T["ret"] = (T.ret.values - T.side.values * Tf.fund.values).astype("float32")
    T["k"] = 5; T["mult"] = 1.0
    return T


B_defs = {"B1": lambda: mk_B12(0, 1.0, 0, "B1"), "B2": lambda: mk_B12(1, 0.5, 0, "B2"), "B3": mk_B3}
rows_B = []
B_trades = {}
for lab, fn in B_defs.items():
    T = fn()
    T = T[[(s, e) not in ck for s, e in zip(T.s.values, T.e.values)]].copy()
    B_trades[lab] = T
    P(f"  {lab}: 거래 {len(T):,} ({time.time()-t_:.0f}s)")
    if len(T) > 0:
        yr_stats = T.groupby("yr").apply(lambda g: pd.Series(dict(n=len(g), bp=g.ret.mean() * 1e4, hit=(g.ret > 0).mean() * 100)))
        P("    " + " | ".join(f"{int(y)}:n{int(r.n)} {r.bp:+.1f}bp 승{r.hit:.0f}%" for y, r in yr_stats.iterrows()))
        for y, r in yr_stats.iterrows():
            rows_B.append(dict(mech=lab, yr=int(y), n=int(r.n), bp=r.bp, hit=r.hit))
    if len(T) < 20:
        P(f"    {lab}: 거래 부족, 증분 평가 생략"); continue
    N = T[G.COLS].reset_index(drop=True)
    best = pick_weight(N, [], 5, m_base_tr["pos"])
    if best is None:
        P(f"    {lab}: 적합 Wn 없음(양수비율 제약 불충족)"); continue
    Wn, sh, pos, d = best
    mt_tr = weekly_metrics(d, TR_LO, TR_HI); mt_te = weekly_metrics(d, TE_LO, TE_HI)
    P(f"    {lab} 증분(Wn{Wn}): 2022~24 {fmt(mt_tr)}")
    P(f"                 2025~26 {fmt(mt_te)}")
    rows_B.append(dict(mech=lab, yr=-1, n=len(T), bp=T.ret.mean() * 1e4, hit=(T.ret > 0).mean() * 100,
                       Wn=Wn, sharpe_2224=mt_tr["sharpe"], pos_2224=mt_tr["pos"], med_2224=mt_tr["med"],
                       sharpe_2526=mt_te["sharpe"], pos_2526=mt_te["pos"], med_2526=mt_te["med"], worst_2526=mt_te["worst"]))

O_B = pd.DataFrame(rows_B)
O_B.to_parquet(R / "data" / "r127_B.parquet", index=False)
P(f"Part B 저장: data/r127_B.parquet ({len(O_B)}행)")

# ===== 최종 채택분 사건 저장 =====
sel_parts = []
for lab, df, k, Wn in selected:
    sp = df.copy(); sp["k"] = k; sp["label"] = f"A:{lab}|Wn{Wn}"; sel_parts.append(sp[G.COLS + ["label"]])
next_k = 5 + len(selected)
for lab, T in B_trades.items():
    if len(T) < 20: continue
    sp = T[G.COLS].copy(); sp["k"] = next_k; sp["label"] = f"{lab}(참고, 포트폴리오 미편입)"; sel_parts.append(sp[G.COLS + ["label"]])
    next_k += 1
if sel_parts:
    SEL = pd.concat(sel_parts, ignore_index=True)
    SEL.to_parquet(R / "data" / "r127_selected_events.parquet", index=False)
    P(f"최종 채택 사건 저장: data/r127_selected_events.parquet ({len(SEL):,}행, label {SEL.label.nunique()}종)")
else:
    P("최종 채택 사건 없음(Part A 전진선택 0건, Part B 거래부족)")

P(f"\n완료 ({time.time()-t_:.0f}s)")
