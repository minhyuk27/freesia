"""R131 — R90 큰 움직임 예보 창 안에서, 가격이 기준가에서 "첫 작은 이탈"을 보이는 순간에
그 움직임이 "계속될지" 조기에 높은 정확도로 판단할 수 있는가(조기 포착).

사건 창: R90 p ≥ th30(T10/T20/T30), 종목별 240분 겹침 제거(시간순 그리디) — R129의
  dedup_time/W를 그대로 재사용(새로 안 만듦). 기준가 ref = c[t−1]. 연도 2022–2026.

첫 이탈(first excursion) f: t부터 240분 안에서 ref·(1±y)를 처음 벗어나는 분, y ∈ {0.1,0.2,0.3,0.5}%.
  같은 분에 위·아래 둘 다 닿으면(모호) 버림. 방향 s=±1. 결정 시점 = f의 종가, 진입가 = close[f]
  (테이커 가정, 비용은 아래 참고).

라벨(ENTRY 기준 대칭 → 무작위보행 기저율이 정확히 50%가 되어 선택 편향이 없음): f+1분부터
  entry·(1+s·X) 와 entry·(1−s·X) 중 어느 쪽에 480분 안에 먼저 닿는가, X ∈ {0.5%,1.0%}.
  s방향(연속) 먼저 닿으면 cont=1, 반대(반전) 먼저 닿으면 cont=0. 같은 분에 둘 다/480분 안에 미결판
  → 버림(버림율 보고). 이 cont의 무조건부(모델 없이) 평균이 "그냥 따라가기" 기저선(R91≈50%).

비용: 진입 테이커 7bp, 목표(TP) 지정가 청산 2bp → 적중 시 X−9bp. 반전(SL) 테이커 청산 7bp →
  오판 시 −X−14bp. (설계상 480분 내 미결판은 라벨 자체에서 버리므로 타임아웃 거래는 실질적으로
  발생하지 않음 — 혹시 남더라도 보수적으로 미결판과 함께 버림 처리.)

특징(전부 f의 "종가 시점"까지만 — 미래 참조 없음):
  - r129_feats.minute_panel(sym,[f]) 룩백 1/5/10분, 부호있는 특징(SIGNED)은 s로 부호 정렬.
  - 속도: f−t(분) · 이탈 오버슈트 (close[f]/ref−1)·s · f분봉 범위 (h[f]−l[f])/ref
  - p(R90), 등급(tier_rank: T10=3/T20=2/T30=1), f의 UTC 시(hour)
  - 다른 종목 동행: s·(log othclose[f] − log othclose[t−1])  (BTC·ETH 상호 참조, r129_feats 정렬 재사용)
  - 24h 고/저 거리(부호): s·(close[f]/rolling1440max[f]−1), s·(close[f]/rolling1440min[f]−1)
  - 직전 4시간 수익(부호): s·log(close[f]/close[f−240])
  - osc: t부터 f까지(=인과적, R129의 false_starts와 달리 미래 극값 bm을 쓰지 않음) y/2 문턱을
    히스테리시스(y/4)로 왕복한 횟수 — "돌파 전 흔들림" 인과 버전.
  agg1s·bookDepth 유래 특징(ofi/lti/maxtrade/bdimb/dbdimb)은 2023-01 이전 NaN(HistGB가 처리).

모델: HistGradientBoostingClassifier(max_iter=300, lr=0.04, max_leaf_nodes=15, min_samples_leaf=200,
  l2=1.0), (y,X)쌍마다 1개, 종목·등급 풀링, 연도별 확장창 워크포워드(학습 연도<Y, 평가 Y=2023..2026,
  2022은 학습 전용). 확신 문턱(상위 50/20/10%)은 "그 Y의 학습 예측" 분포의 |p̂−0.5| 분위수로 정해
  평가 연도에 그대로 적용(미래 분포를 보지 않음) — p̂>0.5면 추세추종(follow), p̂<0.5면 역추세(fade).

평가: 2023–24 / 2025–26 / 전체(2023–26), 확신 구간별 정확도(매매 방향 적중)·n·이진검정 z(대 50%)·
  거래당 순bp·하루 거래수. 치환검정: 학습 연도 내부에서만 cont 라벨을 섞어(연도별 독립 셔플) 같은
  파이프라인으로 2025–26 상위20% 정확도의 무(null) 분포 20회 생성 → 실측값의 백분위 보고.
  (y,X)별 연도별 AUC, 최고 (y,X)의 2025년 표본 순열중요도(permutation importance) 상위 10개.

저장: data/r131_events.parquet(사건+특징+라벨+표본외 예측). 데이터는 git에 올리지 않음.
사용(원격에서만 실행, oracle1): python research/r131_early_judge.py"""
import sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
import r129_feats as RF
import r129_onset_anatomy as R129
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)

SYMS = ("BTCUSDT", "ETHUSDT")
W = 240                       # R90 예보 창(=첫 이탈을 찾는 범위), R129와 동일
H = 480                       # 라벨(계속/반전 판정) 창
YS = (0.001, 0.002, 0.003, 0.005)
XS2 = (0.005, 0.01)
SEED = 131
CT, CM = 7e-4, 2e-4            # 진입/반전청산 테이커, 목표청산 지정가(R91과 동일 상수)
TIER_RANK = {"T10": 3, "T20": 2, "T30": 1}


# ---------- 1) 사건(=R129 재사용) ----------

def build_events():
    """R90 p≥th30, 종목별 240분 겹침 제거 — R129.dedup_time을 그대로 호출(새로 안 만듦)."""
    T = pd.read_parquet(R / "data" / "r90_wf.parquet")
    ev = T[T.p >= T.th30].sort_values(["sym", "t"]).copy()
    ev = R129.dedup_time(ev, W)
    ev["tier"] = np.where(ev.p >= ev.th10, "T10", np.where(ev.p >= ev.th20, "T20", "T30"))
    return ev[["sym", "t", "ts", "yr", "p", "tier"]].reset_index(drop=True)


# ---------- 2) 첫 이탈 + 인과적 흔들림(osc) ----------

@numba.njit(cache=True)
def first_excursion(t, c, h, l, W_, X):
    n = len(c); nQ = len(t)
    f = np.full(nQ, -1, np.int64); s = np.zeros(nQ, np.int8); osc = np.zeros(nQ, np.int32)
    band = X / 2.0; reset = X / 4.0
    for q in range(nQ):
        i = t[q]
        if i < 1 or i + W_ >= n: continue
        ref = c[i - 1]; armed = True; cnt = 0; found = -1; d = 0
        for j in range(i, i + W_):
            devp = (h[j] - ref) / ref; devn = (ref - l[j]) / ref
            dev = devp if devp > devn else devn
            if dev >= band and armed: cnt += 1; armed = False
            if dev < reset: armed = True
            up = h[j] >= ref * (1 + X); dn = l[j] <= ref * (1 - X)
            if up and dn: found = j; d = 0; break
            if up: found = j; d = 1; break
            if dn: found = j; d = -1; break
        f[q] = found; s[q] = d; osc[q] = cnt
    return f, s, osc


# ---------- 3) 라벨(계속 vs 반전, 480분) ----------

@numba.njit(cache=True)
def label_barrier(f, s, h, l, c, X, H_):
    n = len(c); nQ = len(f)
    out = np.full(nQ, -1, np.int8)   # -1=미결판(버림), -2=같은분 모호(버림), 0/1=결판
    for q in range(nQ):
        ff = f[q]; sd = s[q]
        if ff < 0 or sd == 0 or ff + 1 + H_ >= n: continue
        entry = c[ff]
        if sd == 1: cl = entry * (1 + X); rl = entry * (1 - X)
        else: cl = entry * (1 - X); rl = entry * (1 + X)
        res = -1
        for j in range(ff + 1, ff + 1 + H_):
            if sd == 1: hc = h[j] >= cl; hr = l[j] <= rl
            else: hc = l[j] <= cl; hr = h[j] >= rl
            if hc and hr: res = -2; break
            if hc: res = 1; break
            if hr: res = 0; break
        out[q] = res
    return out


# ---------- 4) 특징(인과) ----------

def sym_arrays(sym, k):
    close = k.close.values.astype(np.float64); n = len(close)
    hour = pd.DatetimeIndex(k.ts).hour.values.astype(np.float64)
    oth = RF._aligned(sym)["other"].astype(np.float64)
    cs = pd.Series(close)
    rmax = cs.rolling(1440, min_periods=1440).max().values
    rmin = cs.rolling(1440, min_periods=1440).min().values
    ret4h = np.log(close) - np.log(cs.shift(240).values)
    return dict(n=n, close=close, high=k.high.values.astype(np.float64), low=k.low.values.astype(np.float64),
                hour=hour, oth=oth, rmax=rmax, rmin=rmin, ret4h=ret4h)


def add_features(sym, df, A):
    """df: sym 하나에 대한 (t,f,s,osc,...) 행 — f시점까지만 쓰는 특징을 붙인다."""
    fidx = df.f.values.astype(np.int64); tt = df.t.values.astype(np.int64); sd = df.s.values.astype(np.float64)
    ref = A["close"][tt - 1]
    out = pd.DataFrame(index=df.index)
    out["minutes_to_f"] = (fidx - tt).astype(np.float64)
    out["overshoot_s"] = (A["close"][fidx] / ref - 1) * sd
    out["bar_range"] = (A["high"][fidx] - A["low"][fidx]) / ref
    out["hour"] = A["hour"][fidx]
    out["oth_move_s"] = sd * (np.log(A["oth"][fidx]) - np.log(A["oth"][tt - 1]))
    out["d24_hi_s"] = sd * (A["close"][fidx] / A["rmax"][fidx] - 1)
    out["d24_lo_s"] = sd * (A["close"][fidx] / A["rmin"][fidx] - 1)
    out["ret4h_s"] = sd * A["ret4h"][fidx]
    out["osc"] = df.osc.values.astype(np.float64)
    panel = RF.minute_panel(sym, fidx)
    for fn in RF.FEAT_NAMES:
        for L in RF.LOOKBACKS:
            col = f"{fn}_{L}"
            if fn in RF.SIGNED: panel[col] = panel[col].values * sd
    panel.index = df.index
    return pd.concat([out, panel], axis=1)


PANEL_COLS = [f"{fn}_{L}" for fn in RF.FEAT_NAMES for L in RF.LOOKBACKS]
EXTRA_COLS = ["osc", "minutes_to_f", "overshoot_s", "bar_range", "hour", "oth_move_s", "d24_hi_s", "d24_lo_s",
              "ret4h_s", "p", "tier_rank"]
FCOLS = PANEL_COLS + EXTRA_COLS


# ---------- 5) 사건+특징+라벨 테이블 구성 ----------

def build_ev(EVTS, K, ARR):
    rows = []; drop_fe = []; drop_lab = []
    for y in YS:
        parts = []
        for s in SYMS:
            g = EVTS[EVTS.sym == s]
            f, sd, osc = first_excursion(g.t.values.astype(np.int64), ARR[s]["close"], ARR[s]["high"], ARR[s]["low"], W, y)
            gg = g.copy(); gg["f"] = f; gg["s"] = sd; gg["osc"] = osc
            parts.append(gg)
        BASE = pd.concat(parts)
        n0 = len(BASE); BASE = BASE[BASE.s != 0].copy()
        drop_fe.append(dict(y=y, n_sample=n0, drop_no_excursion=1 - len(BASE) / n0))
        fparts = [add_features(s, BASE[BASE.sym == s], ARR[s]) for s in SYMS if (BASE.sym == s).any()]
        FEAT = pd.concat(fparts)
        BASE = BASE.join(FEAT)
        for X in XS2:
            parts2 = []
            for s in SYMS:
                gg = BASE[BASE.sym == s]
                if len(gg) == 0: continue
                out = label_barrier(gg.f.values.astype(np.int64), gg.s.values.astype(np.int8),
                                     ARR[s]["high"], ARR[s]["low"], ARR[s]["close"], X, H)
                gg2 = gg.copy(); gg2["cont_raw"] = out
                parts2.append(gg2)
            LB = pd.concat(parts2); n1 = len(LB)
            drop_lab.append(dict(y=y, X=X, n_events=n1, drop_amb=(LB.cont_raw == -2).mean(),
                                  drop_undecided=(LB.cont_raw == -1).mean()))
            LB2 = LB[LB.cont_raw.isin([0, 1])].copy()
            LB2["cont"] = LB2.cont_raw.astype(int); LB2["y"] = y; LB2["X"] = X
            rows.append(LB2.drop(columns=["cont_raw"]))
    EV = pd.concat(rows, ignore_index=True)
    EV["tier_rank"] = EV.tier.map(TIER_RANK).astype(float)
    EV["period"] = np.where(EV.yr <= 2024, "22-24", "25-26")
    return EV, pd.DataFrame(drop_fe), pd.DataFrame(drop_lab)


# ---------- 6) 모델: 연도별 확장창 워크포워드 ----------

def hgb(seed=0):
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.04, max_leaf_nodes=15,
                                           min_samples_leaf=200, l2_regularization=1.0, random_state=seed)


def net_bp(cont, pred_dir, X):
    """pred_dir=+1(추세추종, s방향에 베팅)/−1(역추세, 반전에 베팅). cont는 이미 s방향 기준이라 s는 불필요."""
    correct = (pred_dir == 1) & (cont == 1) | (pred_dir == -1) & (cont == 0)
    return np.where(correct, X - CT - CM, -X - 2 * CT), correct


def walk_forward(EV, y, X):
    sub = EV[(EV.y == y) & (EV.X == X)].copy()
    sub["pred"] = np.nan; sub["conf"] = np.nan; sub["pred_dir"] = 0
    sub["in_top50"] = False; sub["in_top20"] = False; sub["in_top10"] = False
    auc_rows = []
    for Y in (2023, 2024, 2025, 2026):
        tr = sub[sub.yr < Y]; te = sub[sub.yr == Y]
        if len(tr) < 100 or len(te) < 10: continue
        m = hgb(SEED).fit(tr[FCOLS].values, tr.cont.values)
        p_tr = m.predict_proba(tr[FCOLS].values)[:, 1]
        p_te = m.predict_proba(te[FCOLS].values)[:, 1]
        q50, q20, q10 = np.quantile(np.abs(p_tr - 0.5), [0.5, 0.8, 0.9])
        idx = te.index
        sub.loc[idx, "pred"] = p_te
        sub.loc[idx, "conf"] = np.abs(p_te - 0.5)
        sub.loc[idx, "pred_dir"] = np.where(p_te > 0.5, 1, -1)
        sub.loc[idx, "in_top50"] = np.abs(p_te - 0.5) >= q50
        sub.loc[idx, "in_top20"] = np.abs(p_te - 0.5) >= q20
        sub.loc[idx, "in_top10"] = np.abs(p_te - 0.5) >= q10
        if te.cont.nunique() > 1:
            auc_rows.append(dict(y=y, X=X, Y=Y, n=len(te), auc=roc_auc_score(te.cont.values, p_te)))
    nb, corr = net_bp(sub.cont.values, sub.pred_dir.values, X)
    sub["net_bp"] = nb; sub["correct"] = corr
    return sub, pd.DataFrame(auc_rows)


def summarize(sub, days_by_period):
    rows = []
    for period, yrs in (("2023-24", (2023, 2024)), ("2025-26", (2025, 2026)), ("전체(23-26)", (2023, 2024, 2025, 2026))):
        base = sub[sub.yr.isin(yrs) & sub.pred.notna()]
        for bucket, flag in (("전체", np.ones(len(base), bool)), ("상위50%", base.in_top50.values),
                              ("상위20%", base.in_top20.values), ("상위10%", base.in_top10.values)):
            d = base[flag]; n = len(d)
            if n < 10: continue
            acc = d.correct.mean(); z = (acc - 0.5) / np.sqrt(0.25 / n)
            rows.append(dict(period=period, bucket=bucket, n=n, acc=acc, z=z, net_bp=d.net_bp.mean() * 1e4,
                              trades_per_day=n / days_by_period.get(period, np.nan)))
    return pd.DataFrame(rows)


# ---------- 7) 치환검정(permutation null) ----------

def permute_null(EV, y, X, n_perm=20):
    sub = EV[(EV.y == y) & (EV.X == X)].copy()
    accs = []
    for perm in range(n_perm):
        rng = np.random.default_rng(SEED * 1000 + int(y * 1e4) * 10 + int(X * 1e4) + perm)
        pooled = []
        for Y in (2025, 2026):
            tr = sub[sub.yr < Y].copy(); te = sub[sub.yr == Y]
            if len(tr) < 100 or len(te) < 10: continue
            shuf = tr.cont.values.copy()
            for yr_ in tr.yr.unique():
                m = (tr.yr.values == yr_)
                shuf[m] = rng.permutation(shuf[m])
            mdl = hgb(SEED + perm + 1).fit(tr[FCOLS].values, shuf)
            p_tr = mdl.predict_proba(tr[FCOLS].values)[:, 1]
            p_te = mdl.predict_proba(te[FCOLS].values)[:, 1]
            q20 = np.quantile(np.abs(p_tr - 0.5), 0.8)
            sel = np.abs(p_te - 0.5) >= q20
            pred_dir = np.where(p_te > 0.5, 1, -1)
            nb, corr = net_bp(te.cont.values, pred_dir, X)
            pooled.append(pd.DataFrame({"correct": corr[sel]}))
        P_ = pd.concat(pooled) if pooled else pd.DataFrame({"correct": []})
        if len(P_) >= 10: accs.append(P_.correct.mean())
    return np.array(accs)


# ---------- main ----------

if __name__ == "__main__":
    t0 = time.time()
    K = {s: store.load("klines_1m", s) for s in SYMS}
    ARR = {s: sym_arrays(s, K[s]) for s in SYMS}
    EVTS = build_events()
    P(f"사건 {len(EVTS)}개 (T10/T20/T30) ({time.time()-t0:.0f}s)")

    EV, drop_fe, drop_lab = build_ev(EVTS, K, ARR)
    P(f"\n===== 첫 이탈 미발생(240분 안에 ±y 안 닿음) 비율")
    P(drop_fe.round(4).to_string(index=False))
    P(f"\n===== 라벨 버림율(480분 내 모호/미결판) — 사건·y·X별")
    P(drop_lab.round(4).to_string(index=False))
    P(f"\n이벤트+특징+라벨 테이블 {len(EV)}행 ({time.time()-t0:.0f}s)")

    P("\n===== 무조건부 계속 비율(cont 평균, 모델 없음) — y/X/등급/기간별 (R91 기저선 ≈0.5 검증)")
    base_tbl = EV.groupby(["y", "X", "tier", "period"]).cont.agg(["mean", "size"]).reset_index()
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 300)
    P(base_tbl.round(3).to_string(index=False))
    overall = EV.groupby(["y", "X"]).cont.agg(["mean", "size"]).reset_index()
    P("\n-- y/X별 전체 풀링 무조건부 계속 비율 --")
    P(overall.round(3).to_string(index=False))

    days_by_period = {}
    for period, yrs in (("2023-24", (2023, 2024)), ("2025-26", (2025, 2026)), ("전체(23-26)", (2023, 2024, 2025, 2026))):
        ts = EVTS[EVTS.yr.isin(yrs)].ts
        days_by_period[period] = max((ts.max() - ts.min()).days, 1) if len(ts) else np.nan

    all_sub = []; all_auc = []; summ_rows = []
    for y in YS:
        for X in XS2:
            sub, auc = walk_forward(EV, y, X)
            all_sub.append(sub); all_auc.append(auc)
            S = summarize(sub, days_by_period); S["y"] = y; S["X"] = X
            summ_rows.append(S)
            P(f"\n===== y={y*100:.1f}% X={X*100:.1f}% 모델 평가 (n={len(sub)})")
            P(S[["period", "bucket", "n", "acc", "z", "net_bp", "trades_per_day"]].round(3).to_string(index=False))
    SUMM = pd.concat(summ_rows, ignore_index=True)
    AUC = pd.concat(all_auc, ignore_index=True) if all_auc else pd.DataFrame()
    EV_OOS = pd.concat(all_sub, ignore_index=True)
    P(f"\n({time.time()-t0:.0f}s) 모든 (y,X) 워크포워드 완료")

    P("\n===== AUC — (y,X)·평가연도별")
    if len(AUC): P(AUC.round(3).to_string(index=False))

    P("\n===== 치환검정(permutation null, 20회) — 2025-26 상위20% 정확도")
    perm_rows = []
    best_key = None; best_acc = -1
    for y in YS:
        for X in XS2:
            obs = SUMM[(SUMM.y == y) & (SUMM.X == X) & (SUMM.period == "2025-26") & (SUMM.bucket == "상위20%")]
            if len(obs) == 0: continue
            obs_acc = obs.acc.values[0]
            if obs_acc > best_acc: best_acc = obs_acc; best_key = (y, X)
            null = permute_null(EV, y, X, 20)
            if len(null) == 0: continue
            pct = (null <= obs_acc).mean() * 100
            perm_rows.append(dict(y=y, X=X, obs_acc=obs_acc, null_mean=null.mean(), null_std=null.std(),
                                   n_perm=len(null), percentile=pct))
            P(f"  y={y*100:.1f}% X={X*100:.1f}%: 실측 상위20% 정확도 {obs_acc:.3f} vs null 평균 {null.mean():.3f}±{null.std():.3f} "
              f"(n={len(null)}) → 백분위 {pct:.0f}% ({time.time()-t0:.0f}s)")
    PERM = pd.DataFrame(perm_rows)

    P(f"\n===== 최고 조합 y={best_key[0]*100:.1f}% X={best_key[1]*100:.1f}% — 2025년 표본 순열중요도 상위 10개")
    by, bx = best_key
    bsub = EV[(EV.y == by) & (EV.X == bx)]
    btr = bsub[bsub.yr < 2025]; bte = bsub[bsub.yr == 2025]
    if len(btr) >= 100 and len(bte) >= 10:
        bm = hgb(SEED).fit(btr[FCOLS].values, btr.cont.values)
        pim = permutation_importance(bm, bte[FCOLS].values, bte.cont.values, n_repeats=5, random_state=SEED, scoring="roc_auc")
        imp = pd.DataFrame({"feature": FCOLS, "importance": pim.importances_mean}).sort_values("importance", ascending=False)
        P(imp.head(10).round(4).to_string(index=False))
    else:
        P("  2025년 표본 부족 — 중요도 생략")

    P("\n===== 검증(룩어헤드 의심 신호): 무조건부 계속 비율은 0.5 근방이어야 하고, 상위10% 정확도가 0.65를 "
      "크게 넘으면 특징·라벨 경계(f+1 사용 여부, ref 정렬)를 다시 확인할 것")
    P(f"  전체 풀링 무조건부 계속 비율 평균: {EV.cont.mean():.3f} (표본 {len(EV)})")
    hi = SUMM[(SUMM.bucket == "상위10%") & (SUMM.acc > 0.65)]
    if len(hi): P("  ⚠ 상위10% 정확도 0.65 초과 조합:\n" + hi.round(3).to_string(index=False))
    else: P("  상위10% 정확도 0.65 초과 조합 없음(룩어헤드 의심 신호 없음)")

    EV_OOS.to_parquet(R / "data" / "r131_events.parquet", index=False)
    P(f"\n저장: data/r131_events.parquet ({len(EV_OOS)}행) ({time.time()-t0:.0f}s)")
    (R / "logs" / "r131.summary").write_text(
        f"R131 완료 ({time.time()-t0:.0f}s). 최고 조합 y={by*100:.1f}% X={bx*100:.1f}% 2025-26 상위20% 정확도 {best_acc:.3f}\n")
    P(f"\n전체 완료 ({time.time()-t0:.0f}s)")
