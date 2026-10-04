"""R132 2단 — 1단(r132_build)이 만든 D1/D2/D3 패널에서 "구할 수 있는 지표는 전부" 스캔해
TRAIN(2022-2024)/TEST(2025-2026) 분할로 생존자(survivor)가 있는지 검증한다.

패널·타깃(모두 "drop 0" = 미결판/모호 행 제거 후 ±1 라벨):
  D1_side2pct : ref 대비 첫 ±2% 터치 방향(240분), 각 행이 곧 자기 군집(창마다 1행, 겹침 이미 제거됨).
  D1_bar10    : entry(=ref) 대칭 ±1% 480분 배리어(=R131과 같은 X=1% 배리어 정의).
  D2_y{y}_X{X}: R131 cont(계속=+1/반전=-1), y·X 8조합 각각 — 조합마다 (sym,t) 사건이 유일해 행=군집.
  D3_bar05/10 : 창 내부 5분격자 entry 대칭 ±0.5%/±1% 480분 배리어, 군집=window_id(sym_t), 창당 최대49행.

통계(전부 house 관행 — r122_scan.daystats/r131_early_judge와 동일한 군집-강건 t):
  correct_i = 1{sgn_y_i == d}, d=TRAIN에서 정한 예측방향(sign of TRAIN 평균 라벨).
  t = (mean(correct)-0.5) / se,  se = sqrt(sum_c(sum_{i in c}(correct_i-mean))^2) / n (CR1, 군집=1행이면 보통 se로 수렴).
  비용(R131과 동일): 적중 시 X-9bp(테이커진입+지정가청산), 오판 시 -X-14bp(테이커진입+테이커반전청산).

A. 단변량: 특징×버킷(TRAIN 분위 bottom10/20·top20/10)×(연도별 Spearman IC 동봉).
   BH(5%, A 전체 테스트 풀) + 기대 |t|>2 개수 + 생존 기준(TRAIN t≥3, TEST t≥2, BH유의, TEST hit≥0.55,
   TEST net bp>0, 2025·2026 개별 hit>0.5) + 블록단위(연도 내 군집) 라벨 셔플 위약(5회, 특징 20% 표본).
B. 쌍별: 그룹별 TRAIN|t| 상위40 전부쌍(780) × 2×2 극단버킷(상·하위30%) — 같은 기준/BH(B 전용).
C. ML: HistGB·L1로지스틱(all/top50-IC/family별), 고정분할 + 연도별 워크포워드, 치환검정(20회).
   (시간·메모리 제약으로 축소한 부분은 각 섹션 주석에 명시 — oracle1 2코어·여유 RAM 제한.)
D. 리포트 + data/r132_scan.parquet(A), r132_pairs.parquet(B), r132_ml.parquet(C) 저장.

사용(oracle1 전용): nohup .venv/bin/python research/r132_scan.py > logs/r132_scan.log 2>&1 &
빠른 점검: .venv/bin/python research/r132_scan.py --smoke"""
import sys, json, time, warnings, gc
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R))
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)

SEED = 132
CT, CM = 7e-4, 2e-4                      # R131과 동일: 적중 X-(CT+CM)=X-9bp, 오판 -X-2*CT=-X-14bp
TRAIN_YR = (2022, 2023, 2024); TEST_YR = (2025, 2026)
SMOKE = "--smoke" in sys.argv             # 빠른 점검 모드(특징·반복수 축소)

META_COLS = {
    "D1": {"sym", "t", "ts", "yr", "p", "tier", "m", "ts_dec", "side2pct", "bar05", "bar10"},
    "D2": {"sym", "t", "yr", "m", "s", "y", "X", "cont", "ts_dec"},
    "D3": {"sym", "t", "yr", "tier", "m", "bar05", "bar10", "ts_dec"},
}


def _rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return -1.0


def _trim():
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


# ============================================================ 패널·그룹 구성 ============================================================

def load_panels():
    D1 = pd.read_parquet(R / "data" / "r132_D1.parquet")
    D2 = pd.read_parquet(R / "data" / "r132_D2.parquet")
    D3 = pd.read_parquet(R / "data" / "r132_D3.parquet")
    return D1, D2, D3


def _window_id_codes(df):
    wid = df.sym.astype(str).values + "_" + df.t.astype(str).values
    codes, _ = pd.factorize(wid, sort=False)
    return codes.astype(np.int64)


def make_groups(D1, D2, D3):
    """그룹 dict: df, feats(특징 열 이름 목록), sgn(±1 신호 라벨), X(배리어 폭, bp 계산용),
    yr, cluster(군집 코드 배열 또는 None=행 자신이 군집), panel 태그."""
    groups = {}
    for label_col, Xw in (("side2pct", 0.02), ("bar10", 0.01)):
        d = D1[D1[label_col] != 0].reset_index(drop=True)
        feats = [c for c in d.columns if c not in META_COLS["D1"]]
        groups[f"D1_{label_col}"] = dict(df=d, feats=feats, sgn=d[label_col].values.astype(np.float64),
                                          X=Xw, yr=d.yr.values.astype(int), cluster=None, panel="D1")
    for label_col, Xw in (("bar05", 0.005), ("bar10", 0.01)):
        d = D3[D3[label_col] != 0].reset_index(drop=True)
        feats = [c for c in d.columns if c not in META_COLS["D3"]]
        codes = _window_id_codes(d)
        groups[f"D3_{label_col}"] = dict(df=d, feats=feats, sgn=d[label_col].values.astype(np.float64),
                                          X=Xw, yr=d.yr.values.astype(int), cluster=codes, panel="D3")
    feats_d2 = [c for c in D2.columns if c not in META_COLS["D2"]]
    for (y_, X_), g in D2.groupby(["y", "X"]):
        d = g.reset_index(drop=True)
        groups[f"D2_y{y_}_X{X_}"] = dict(df=d, feats=feats_d2, sgn=(2 * d.cont.values - 1).astype(np.float64),
                                          X=float(X_), yr=d.yr.values.astype(int), cluster=None, panel="D2",
                                          y=float(y_), Xc=float(X_))
    return groups, feats_d2


# ============================================================ 군집-강건 통계(house 관행) ============================================================

def cluster_stat(correct, codes=None, ncl_global=None):
    """correct: 0/1 적중 지시자. codes=None이면 '행=군집'(일반 표준오차로 수렴). 반환: hit, t(대 0.5), n, 군집수."""
    n = len(correct)
    if n < 5:
        return np.nan, np.nan, n, 0
    mu = correct.mean()
    resid = correct - mu
    if codes is None:
        se = resid.std(ddof=0) / np.sqrt(n)
        ncl = n
    else:
        cs = np.bincount(codes, weights=resid, minlength=ncl_global)
        cnt = np.bincount(codes, minlength=ncl_global)
        ncl = int((cnt > 0).sum())
        se = np.sqrt((cs ** 2).sum()) / n
    if se <= 0 or ncl <= 2:
        return mu, np.nan, n, ncl
    return mu, (mu - 0.5) / se, n, ncl


def tstat_pvalue(t, df):
    if not np.isfinite(t) or df <= 1:
        return np.nan
    return float(2 * stats.t.sf(abs(t), df))


def bh_flag(pvals, alpha=0.05):
    """Benjamini-Hochberg, 유한 p값에만 적용."""
    p = np.asarray(pvals, dtype=float)
    out = np.zeros(len(p), dtype=bool)
    idx = np.where(np.isfinite(p))[0]
    if len(idx) == 0:
        return out
    pp = p[idx]; n = len(pp)
    order = np.argsort(pp)
    ranked = pp[order]
    thresh = alpha * (np.arange(1, n + 1) / n)
    passed = ranked <= thresh
    if not passed.any():
        return out
    cutoff = ranked[np.max(np.where(passed)[0])]
    out[idx[pp <= cutoff]] = True
    return out


def add_survivor(df):
    """BH + 생존 기준(자체 테스트 풀 안에서 BH 재계산 — A/B 각자 전용)."""
    if len(df) == 0:
        df = df.copy(); df["bh"] = False; df["survivor"] = False; return df
    df = df.copy()
    df["bh"] = bh_flag(df.p_te.values)
    df["survivor"] = ((df.t_tr >= 3) & (df.t_te >= 2) & df.bh & (df.hit_te >= 0.55) & (df.net_bp_te > 0) &
                       (df.hit25 > 0.5) & (df.hit26 > 0.5))
    return df


def _rank(x):
    return pd.Series(x).rank().values


def _spearman_fast(x, y):
    rx = _rank(x); ry = _rank(y)
    rx = rx - rx.mean(); ry = ry - ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den > 0 else np.nan


# ============================================================ A. 단변량 스캔 ============================================================

def univariate_scan(name, grp, feat_list=None):
    df = grp["df"]; feats = feat_list or grp["feats"]; sgn = grp["sgn"]; yr = grp["yr"]
    codes = grp["cluster"]; ncl_global = int(codes.max()) + 1 if codes is not None else None
    tr0 = np.isin(yr, TRAIN_YR); te0 = np.isin(yr, TEST_YR)
    years = sorted(set(yr.tolist()))
    win_bp = grp["X"] * 1e4 - 9; loss_bp = -grp["X"] * 1e4 - 14
    rows = []
    for feat in feats:
        x = df[feat].values.astype(np.float64)
        finite = np.isfinite(x)
        xt = x[finite & tr0]
        if len(xt) < 100:
            continue
        p10, p20, p80, p90 = np.nanquantile(xt, [0.1, 0.2, 0.8, 0.9])
        if p10 == p90:
            continue
        buckets = (("bottom10", x <= p10), ("bottom20", x <= p20), ("top20", x >= p80), ("top10", x >= p90))
        ics = {}
        for y_ in years:
            m = finite & (yr == y_)
            ics[y_] = _spearman_fast(x[m], sgn[m]) if m.sum() >= 30 else np.nan
        for bname, bmask in buckets:
            m = bmask & finite
            m_tr = m & tr0; m_te = m & te0
            n_tr = int(m_tr.sum()); n_te = int(m_te.sum())
            if n_tr < 30 or n_te < 10:
                continue
            mean_tr = sgn[m_tr].mean()
            d = 1.0 if mean_tr > 0 else (-1.0 if mean_tr < 0 else 0.0)
            if d == 0:
                continue
            correct_tr = (sgn[m_tr] == d).astype(float)
            correct_te = (sgn[m_te] == d).astype(float)
            codes_tr = codes[m_tr] if codes is not None else None
            codes_te = codes[m_te] if codes is not None else None
            hit_tr, t_tr, _, _ = cluster_stat(correct_tr, codes_tr, ncl_global)
            hit_te, t_te, _, ncl_te = cluster_stat(correct_te, codes_te, ncl_global)
            net_bp_te = float(np.where(sgn[m_te] == d, win_bp, loss_bp).mean())
            yr_te = yr[m_te]
            m25 = yr_te == 2025; m26 = yr_te == 2026
            hit25 = correct_te[m25].mean() if m25.sum() >= 10 else np.nan
            hit26 = correct_te[m26].mean() if m26.sum() >= 10 else np.nan
            row = dict(group=name, feature=feat, bucket=bname, direction=d, n_tr=n_tr, n_te=n_te,
                       hit_tr=hit_tr, t_tr=t_tr, hit_te=hit_te, t_te=t_te,
                       p_te=tstat_pvalue(t_te, max(ncl_te - 1, 1)), net_bp_te=net_bp_te,
                       hit25=hit25, hit26=hit26)
            for y_ in (2022, 2023, 2024, 2025, 2026):
                row[f"ic_{y_}"] = ics.get(y_, np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def block_shuffle(sgn, codes, yr, rng):
    """연도 내부에서 군집(window) 블록 단위로 라벨을 섞는다(행=군집이면 보통 행 셔플로 수렴).
    블록 길이가 다르면(창 경계 절단 등 드문 경우) 타일링으로 맞춘다."""
    out = sgn.copy()
    for y_ in np.unique(yr):
        idx = np.where(yr == y_)[0]
        if codes is None:
            out[idx] = sgn[rng.permutation(idx)]
            continue
        codes_y = codes[idx]
        uniq = np.unique(codes_y)
        perm = rng.permutation(uniq)
        mapping = dict(zip(uniq.tolist(), perm.tolist()))
        order = {c: idx[codes_y == c] for c in uniq}
        for c in uniq:
            dst = order[c]; src = order[mapping[c]]
            if len(src) == 0 or len(dst) == 0:
                continue
            reps = int(np.ceil(len(dst) / len(src)))
            out[dst] = np.tile(sgn[src], reps)[:len(dst)]
    return out


def run_null(groups, n_shuffle=5, frac=0.2):
    """위약: 연도 내 군집 블록 라벨 셔플 n_shuffle회 × 특징 frac 무작위 표본 → 재스캔."""
    n_shuffle = 2 if SMOKE else n_shuffle
    rng = np.random.default_rng(SEED + 999)
    rows = []
    for name, grp in groups.items():
        feats = grp["feats"]
        k = min(len(feats), 5) if SMOKE else max(1, int(len(feats) * frac))
        sel = list(rng.choice(feats, size=k, replace=False))
        for rep in range(n_shuffle):
            g2 = dict(grp); g2["sgn"] = block_shuffle(grp["sgn"], grp["cluster"], grp["yr"], rng)
            res = univariate_scan(name, g2, feat_list=sel)
            if len(res) == 0:
                continue
            res = add_survivor(res)
            rows.append(dict(group=name, rep=rep, n_feat=k, max_abs_t_te=float(res.t_te.abs().max()),
                              n_survivors=int(res.survivor.sum()), n_tests=len(res)))
    return pd.DataFrame(rows)


# ============================================================ B. 쌍별 상호작용 ============================================================

def pairwise_scan(name, grp, top_feats):
    df = grp["df"]; sgn = grp["sgn"]; yr = grp["yr"]; codes = grp["cluster"]
    ncl_global = int(codes.max()) + 1 if codes is not None else None
    tr0 = np.isin(yr, TRAIN_YR); te0 = np.isin(yr, TEST_YR)
    win_bp = grp["X"] * 1e4 - 9; loss_bp = -grp["X"] * 1e4 - 14
    X = {f: df[f].values.astype(np.float64) for f in top_feats}
    thr = {}
    for f in top_feats:
        xt = X[f][tr0]; xt = xt[np.isfinite(xt)]
        if len(xt) >= 50:
            thr[f] = tuple(np.nanquantile(xt, [0.3, 0.7]))
    vf = [f for f in top_feats if f in thr]
    rows = []
    for i in range(len(vf)):
        f1 = vf[i]; p30_1, p70_1 = thr[f1]; x1 = X[f1]
        for j in range(i + 1, len(vf)):
            f2 = vf[j]; p30_2, p70_2 = thr[f2]; x2 = X[f2]
            finite = np.isfinite(x1) & np.isfinite(x2)
            combos = (("TT", (x1 >= p70_1) & (x2 >= p70_2)), ("TB", (x1 >= p70_1) & (x2 <= p30_2)),
                      ("BT", (x1 <= p30_1) & (x2 >= p70_2)), ("BB", (x1 <= p30_1) & (x2 <= p30_2)))
            for cname, cmask in combos:
                m = cmask & finite
                m_tr = m & tr0; m_te = m & te0
                n_tr = int(m_tr.sum()); n_te = int(m_te.sum())
                if n_tr < 30 or n_te < 10:
                    continue
                mean_tr = sgn[m_tr].mean()
                d = 1.0 if mean_tr > 0 else (-1.0 if mean_tr < 0 else 0.0)
                if d == 0:
                    continue
                correct_tr = (sgn[m_tr] == d).astype(float)
                correct_te = (sgn[m_te] == d).astype(float)
                codes_tr = codes[m_tr] if codes is not None else None
                codes_te = codes[m_te] if codes is not None else None
                hit_tr, t_tr, _, _ = cluster_stat(correct_tr, codes_tr, ncl_global)
                hit_te, t_te, _, ncl_te = cluster_stat(correct_te, codes_te, ncl_global)
                net_bp_te = float(np.where(sgn[m_te] == d, win_bp, loss_bp).mean())
                yr_te = yr[m_te]
                m25 = yr_te == 2025; m26 = yr_te == 2026
                hit25 = correct_te[m25].mean() if m25.sum() >= 10 else np.nan
                hit26 = correct_te[m26].mean() if m26.sum() >= 10 else np.nan
                rows.append(dict(group=name, feature=f"{f1} x {f2}", feat1=f1, feat2=f2, bucket=cname,
                                  direction=d, n_tr=n_tr, n_te=n_te, hit_tr=hit_tr, t_tr=t_tr, hit_te=hit_te,
                                  t_te=t_te, p_te=tstat_pvalue(t_te, max(ncl_te - 1, 1)), net_bp_te=net_bp_te,
                                  hit25=hit25, hit26=hit26))
    return pd.DataFrame(rows)


# ============================================================ C. ML ============================================================

def train_ic(x_tr, y_tr):
    finite = np.isfinite(x_tr)
    if finite.sum() < 100:
        return 0.0
    return abs(_spearman_fast(x_tr[finite], y_tr[finite]) or 0.0)


def top50_by_ic(df, feats, sgn, yr, force=()):
    tr = np.isin(yr, TRAIN_YR)
    y_tr = sgn[tr]
    ics = {f: train_ic(df.loc[tr, f].values.astype(np.float64), y_tr) for f in feats}
    ranked = sorted(ics, key=lambda f: -ics.get(f, 0.0))
    out = list(force)
    for f in ranked:
        if len(out) >= 50:
            break
        if f not in out:
            out.append(f)
    return out[:50]


def hgb_model(seed, min_leaf):
    return HistGradientBoostingClassifier(max_iter=400 if not SMOKE else 30, learning_rate=0.03,
                                           max_leaf_nodes=15, min_samples_leaf=min_leaf, l2_regularization=1.0,
                                           random_state=seed, early_stopping=False)


def l1_model(seed):
    return LogisticRegression(penalty="l1", solver="liblinear", C=1.0, max_iter=200, random_state=seed)


def _prep_l1(df, feats, tr_mask, te_mask):
    Xtr = df.loc[tr_mask, feats].values.astype(np.float64)
    Xte = df.loc[te_mask, feats].values.astype(np.float64)
    nan_tr = ~np.isfinite(Xtr); nan_te = ~np.isfinite(Xte)
    med = np.nanmedian(np.where(nan_tr, np.nan, Xtr), axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Xtr_f = np.where(nan_tr, med, Xtr); Xte_f = np.where(nan_te, med, Xte)
    sc = StandardScaler().fit(Xtr_f)
    Xtr_s = sc.transform(Xtr_f); Xte_s = sc.transform(Xte_f)
    has_nan = nan_tr.any(axis=0)
    if has_nan.any():
        Xtr_s = np.hstack([Xtr_s, nan_tr[:, has_nan].astype(np.float64)])
        Xte_s = np.hstack([Xte_s, nan_te[:, has_nan].astype(np.float64)])
    return Xtr_s, Xte_s


def fit_predict(df, feats, label01, tr_mask, te_mask, kind, seed=SEED, min_leaf=300):
    ytr = label01[tr_mask]
    if kind == "hgb":
        Xtr = df.loc[tr_mask, feats].values.astype(np.float32)
        Xte = df.loc[te_mask, feats].values.astype(np.float32)
        m = hgb_model(seed, min_leaf).fit(Xtr, ytr)
    else:
        Xtr, Xte = _prep_l1(df, feats, tr_mask, te_mask)
        m = l1_model(seed).fit(Xtr, ytr)
    p_tr = m.predict_proba(Xtr)[:, 1]; p_te = m.predict_proba(Xte)[:, 1]
    del Xtr, Xte; _trim()
    return p_tr, p_te


def eval_buckets(p_tr, p_te, label01_te, Xw, days_test):
    conf_tr = np.abs(p_tr - 0.5); q50, q20, q10 = np.quantile(conf_tr, [0.5, 0.8, 0.9])
    conf_te = np.abs(p_te - 0.5)
    pred_dir = np.where(p_te > 0.5, 1, -1); sgn_te = 2 * label01_te - 1
    correct = (pred_dir == sgn_te).astype(float)
    win = Xw * 1e4 - 9; loss = -Xw * 1e4 - 14
    net_bp = np.where(correct == 1, win, loss)
    auc = roc_auc_score(label01_te, p_te) if len(np.unique(label01_te)) > 1 else np.nan
    rows = []
    for bname, thr_ in (("전체", -1.0), ("상위50%", q50), ("상위20%", q20), ("상위10%", q10)):
        sel = conf_te >= thr_ if thr_ >= 0 else np.ones(len(conf_te), bool)
        n = int(sel.sum())
        if n < 10:
            continue
        rows.append(dict(bucket=bname, n=n, acc=float(correct[sel].mean()), net_bp=float(net_bp[sel].mean()),
                          trades_per_day=n / days_test if days_test else np.nan, auc=auc))
    return pd.DataFrame(rows), correct, conf_te, q20


def cluster_bootstrap_ci(correct, codes_te, sel, n_boot=200, seed=0):
    rng = np.random.default_rng(seed)
    idx_sel = np.where(sel)[0]
    if len(idx_sel) == 0:
        return np.nan, np.nan
    codes_sel = codes_te[idx_sel]
    uniq = np.unique(codes_sel)
    if len(uniq) < 5:
        return np.nan, np.nan
    groups_idx = {c: idx_sel[codes_sel == c] for c in uniq}
    accs = []
    for _ in range(n_boot):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([groups_idx[c] for c in samp])
        accs.append(correct[rows].mean())
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def days_span(df, mask):
    if "ts_dec" not in df.columns or mask.sum() == 0:
        return np.nan
    ts = df.loc[mask, "ts_dec"]
    d = (ts.max() - ts.min()).days
    return max(d, 1)


def _xw_slice(Xw, mask):
    """Xw: 스칼라(배리어 폭 고정) 또는 df와 같은 길이의 배열(D2 풀링처럼 행마다 X가 다른 경우) — mask로 슬라이스."""
    return Xw[mask] if isinstance(Xw, np.ndarray) else Xw


def run_config(panel, target, cfg_name, feats, df, label01, yr, Xw, kind, codes=None, min_leaf=300,
                walk_forward=False, boot_ci=False):
    rows = []
    tr0 = np.isin(yr, TRAIN_YR); te0 = np.isin(yr, TEST_YR)
    p_tr, p_te = fit_predict(df, feats, label01, tr0, te0, kind, min_leaf=min_leaf)
    dtest = days_span(df, te0)
    B, correct, conf_te, q20 = eval_buckets(p_tr, p_te, label01[te0], _xw_slice(Xw, te0), dtest)
    B["panel"] = panel; B["target"] = target; B["config"] = cfg_name; B["model"] = kind; B["split"] = "fixed"
    if boot_ci and codes is not None:
        codes_te = codes[te0]
        sel20 = conf_te >= q20
        lo, hi = cluster_bootstrap_ci(correct, codes_te, sel20, n_boot=50 if SMOKE else 200, seed=SEED)
        B.loc[B.bucket == "상위20%", "ci_lo"] = lo; B.loc[B.bucket == "상위20%", "ci_hi"] = hi
    rows.append(B)
    if walk_forward:
        for Y in (2023, 2024, 2025, 2026):
            trY = yr < Y; teY = yr == Y
            if trY.sum() < 100 or teY.sum() < 10:
                continue
            p_trY, p_teY = fit_predict(df, feats, label01, trY, teY, kind, min_leaf=min_leaf)
            dY = days_span(df, teY)
            BY, _, _, _ = eval_buckets(p_trY, p_teY, label01[teY], _xw_slice(Xw, teY), dY)
            BY["panel"] = panel; BY["target"] = target; BY["config"] = cfg_name; BY["model"] = kind
            BY["split"] = f"wf_{Y}"
            rows.append(BY)
    return pd.concat(rows, ignore_index=True)


def permute_null_ml(df, feats, label01, yr, Xw, kind, codes, min_leaf, n_perm=20, seed=SEED):
    """R131 관행 재사용: TRAIN 라벨만 연도블록 단위로 섞어 학습 → 실제 TEST로 평가, 상위20% 정확도 분포."""
    n_perm = 3 if SMOKE else n_perm
    tr0 = np.isin(yr, TRAIN_YR); te0 = np.isin(yr, TEST_YR)
    sgn = (2 * label01 - 1).astype(np.float64)
    accs = []
    rng = np.random.default_rng(seed)
    for rep in range(n_perm):
        # TRAIN 구간만 블록 셔플(연도 내부), TEST는 실제값 유지
        tr_idx = np.where(tr0)[0]
        shuf_full = sgn.copy()
        shuf_full[tr_idx] = block_shuffle(sgn[tr_idx], codes[tr_idx] if codes is not None else None,
                                           yr[tr_idx], rng)
        shuf01 = ((shuf_full + 1) / 2)
        p_tr, p_te = fit_predict(df, feats, shuf01, tr0, te0, kind, seed=seed + rep + 1, min_leaf=min_leaf)
        conf_tr = np.abs(p_tr - 0.5); q20 = np.quantile(conf_tr, 0.8)
        conf_te = np.abs(p_te - 0.5)
        sel = conf_te >= q20
        if sel.sum() < 10:
            continue
        pred_dir = np.where(p_te[sel] > 0.5, 1, -1)
        real_sgn_te = sgn[te0][sel]
        accs.append(float((pred_dir == real_sgn_te).mean()))
    return np.array(accs)


# ============================================================ main ============================================================

def main():
    t0 = time.time()
    FAMILIES = json.loads((R / "data" / "r132_families.json").read_text())
    D1, D2, D3 = load_panels()
    P(f"패널 로드: D1={len(D1):,} D2={len(D2):,} D3={len(D3):,} ({time.time()-t0:.0f}s, RSS={_rss_mb():.0f}MB)")
    groups, feats_d2 = make_groups(D1, D2, D3)
    del D1, D2, D3; _trim()
    P("그룹 " + str(len(groups)) + "개: " + ", ".join(groups.keys()))

    # ---------------- A. 단변량 ----------------
    A_parts = []
    for name, grp in groups.items():
        feats = grp["feats"] if not SMOKE else grp["feats"][:30]
        res = univariate_scan(name, grp, feat_list=feats)
        A_parts.append(res)
        P(f"  [A] {name}: 특징 {len(feats)}개 → 테스트 {len(res)}행 ({time.time()-t0:.0f}s, RSS={_rss_mb():.0f}MB)")
    A = pd.concat(A_parts, ignore_index=True)
    A = add_survivor(A)
    n_expect = len(A) * 2 * stats.norm.sf(2)
    P(f"\n===== A. 단변량 스캔 요약: 총 테스트 {len(A):,}개, BH(5%) 유의 {A.bh.sum():,}개, "
      f"survivor {A.survivor.sum()}개 ({time.time()-t0:.0f}s)")
    P(f"무작위 기대 |t_te|>2 개수: {n_expect:.1f} (실측 {(A.t_te.abs()>2).sum():,})")
    if A.survivor.sum() > 0:
        P("\n----- survivor 상세 -----")
        cols = ["group", "feature", "bucket", "direction", "n_tr", "n_te", "hit_tr", "t_tr", "hit_te", "t_te",
                "net_bp_te", "hit25", "hit26"]
        P(A[A.survivor][cols].round(4).to_string(index=False))
    P("\n----- TEST |t| 상위 20 (survivor 여부 무관) -----")
    top20 = A.reindex(A.t_te.abs().sort_values(ascending=False).index).head(20)
    P(top20[["group", "feature", "bucket", "t_tr", "t_te", "hit_tr", "hit_te", "net_bp_te", "bh", "survivor"]]
      .round(4).to_string(index=False))

    NULL = run_null(groups)
    P(f"\n===== 위약(라벨 블록 셔플, 그룹별 특징 20% 표본) — {NULL.rep.nunique() if len(NULL) else 0}회 "
      f"({time.time()-t0:.0f}s)")
    if len(NULL):
        agg = NULL.groupby("group")[["max_abs_t_te", "n_survivors", "n_tests"]].agg(
            ["mean", "max"]).round(3)
        P(agg.to_string())
        P(f"전체 위약 max|t_te| 분포: 평균 {NULL.max_abs_t_te.mean():.2f}, 최댓값 {NULL.max_abs_t_te.max():.2f}, "
          f"survivor 합계 {NULL.n_survivors.sum()} (실측 테스트 수 대비 참고용)")

    # ---------------- B. 쌍별 ----------------
    B_parts = []
    for name, grp in groups.items():
        sub = A[A.group == name].copy(); sub["abst"] = sub.t_tr.abs()
        top_n = 6 if SMOKE else 40
        top_feats = sub.sort_values("abst", ascending=False).feature.drop_duplicates().head(top_n).tolist()
        res = pairwise_scan(name, grp, top_feats)
        B_parts.append(res)
        P(f"  [B] {name}: top{len(top_feats)} → 페어테스트 {len(res)}행 ({time.time()-t0:.0f}s, RSS={_rss_mb():.0f}MB)")
    B = pd.concat(B_parts, ignore_index=True) if B_parts else pd.DataFrame()
    B = add_survivor(B)
    P(f"\n===== B. 쌍별 상호작용 요약: 총 테스트 {len(B):,}개, BH(5%) 유의 {int(B.bh.sum()) if len(B) else 0}개, "
      f"survivor {int(B.survivor.sum()) if len(B) else 0}개 ({time.time()-t0:.0f}s)")
    if len(B) and B.survivor.sum() > 0:
        cols = ["group", "feat1", "feat2", "bucket", "direction", "n_tr", "n_te", "hit_tr", "t_tr", "hit_te",
                "t_te", "net_bp_te", "hit25", "hit26"]
        P(B[B.survivor][cols].round(4).to_string(index=False))
    if len(B):
        P("\n----- B. TEST |t| 상위 20 -----")
        top20b = B.reindex(B.t_te.abs().sort_values(ascending=False).index).head(20)
        P(top20b[["group", "feat1", "feat2", "bucket", "t_tr", "t_te", "hit_te", "net_bp_te", "bh", "survivor"]]
          .round(4).to_string(index=False))

    A.to_parquet(R / "data" / "r132_scan.parquet", index=False)
    B.to_parquet(R / "data" / "r132_pairs.parquet", index=False)
    P(f"\n저장: data/r132_scan.parquet({len(A):,}행), data/r132_pairs.parquet({len(B):,}행) ({time.time()-t0:.0f}s)")
    del A_parts, B_parts; _trim()

    # ---------------- C. ML ----------------
    # 시간/메모리 제약(oracle1 2코어)으로: 패밀리별 모델은 HGB만, D3 연도별 워크포워드는 HGB만,
    # 치환검정(20회)의 "최고 후보" 탐색은 HGB(all/top50)만 대상으로 한다(L1은 고정분할에서만 비교).
    D1, D2, D3 = load_panels()
    groups2, _ = make_groups(D1, D2, D3)
    ML_rows = []

    def run_panel_target(gname, grp, min_leaf, do_wf, do_family, boot_ci):
        df = grp["df"]; feats_all = grp["feats"]; sgn = grp["sgn"]; yr = grp["yr"]; Xw = grp["X"]
        codes = grp["cluster"]; label01 = ((sgn + 1) / 2)
        top50 = top50_by_ic(df, feats_all, sgn, yr)
        configs = [("all", feats_all), ("top50_ic", top50)]
        out = []
        for cfg_name, feats in configs:
            for kind in (("hgb", "l1") if not (gname.startswith("D3") and cfg_name == "all") else ("hgb",)):
                wf = do_wf and not (gname.startswith("D3") and kind == "l1")
                res = run_config(gname.split(":")[0], gname.split(":")[1], cfg_name, feats, df, label01, yr,
                                  Xw, kind, codes=codes, min_leaf=min_leaf, walk_forward=wf,
                                  boot_ci=(boot_ci and cfg_name == "all" and kind == "hgb"))
                out.append(res)
                P(f"    [C] {gname} {cfg_name}/{kind}: AUC(fixed)={res[res.split=='fixed'].auc.iloc[0]:.3f} "
                  f"({time.time()-t0:.0f}s, RSS={_rss_mb():.0f}MB)")
        if do_family:
            panel_feats = set(feats_all)
            for fam, cols in FAMILIES.items():
                ff = [c for c in cols if c in panel_feats]
                if len(ff) < 2:
                    continue
                res = run_config(gname.split(":")[0], gname.split(":")[1], f"fam_{fam}", ff, df, label01,
                                  yr, Xw, "hgb", codes=codes, min_leaf=min_leaf, walk_forward=False,
                                  boot_ci=False)
                out.append(res)
        return pd.concat(out, ignore_index=True), top50

    best_cfg_cache = {}
    for label_col in ("side2pct", "bar10"):
        gname = f"D1:{label_col}"
        res, top50 = run_panel_target(gname, groups2[f"D1_{label_col}"], 300, True, True, False)
        ML_rows.append(res)
        best_cfg_cache[gname] = (groups2[f"D1_{label_col}"], top50, 300, None)

    for label_col in ("bar05", "bar10"):
        gname = f"D3:{label_col}"
        grp = groups2[f"D3_{label_col}"]
        res, top50 = run_panel_target(gname, grp, 1000, True, True, True)
        ML_rows.append(res)
        best_cfg_cache[gname] = (grp, top50, 1000, grp["cluster"])

    # D2: 조합별(8) + 풀링(y,X를 특징으로 추가)
    D2_combo_groups = {k: v for k, v in groups2.items() if k.startswith("D2_")}
    for gkey, grp in D2_combo_groups.items():
        gname = f"D2combo:{gkey}"
        df = grp["df"]; feats_all = grp["feats"]; sgn = grp["sgn"]; yr = grp["yr"]; Xw = grp["X"]
        label01 = (sgn + 1) / 2
        top50 = top50_by_ic(df, feats_all, sgn, yr)
        for cfg_name, feats in (("all", feats_all), ("top50_ic", top50)):
            for kind in ("hgb", "l1"):
                res = run_config("D2", gkey, cfg_name, feats, df, label01, yr, Xw, kind, codes=None,
                                  min_leaf=300, walk_forward=False, boot_ci=False)
                ML_rows.append(res)
        P(f"    [C] {gname} 완료 ({time.time()-t0:.0f}s)")

    D2_full = pd.concat([g["df"].assign(y=g["y"], X=g["Xc"]) for g in D2_combo_groups.values()],
                         ignore_index=True)
    sgn_pool = np.concatenate([g["sgn"] for g in D2_combo_groups.values()])
    yr_pool = np.concatenate([g["yr"] for g in D2_combo_groups.values()])
    label01_pool = (sgn_pool + 1) / 2
    Xw_pool = D2_full["X"].values.astype(np.float64)  # 행마다 배리어 폭이 다름(풀링) — net bp를 행별로 계산
    feats_pool = feats_d2 + ["y", "X"]
    top50_pool = top50_by_ic(D2_full, feats_pool, sgn_pool, yr_pool, force=("y", "X"))
    for cfg_name, feats in (("pooled_all", feats_pool), ("pooled_top50_ic", top50_pool)):
        for kind in ("hgb", "l1"):
            wf = not (kind == "l1")
            res = run_config("D2", "pooled", cfg_name, feats, D2_full, label01_pool, yr_pool, Xw_pool, kind,
                              codes=None, min_leaf=300, walk_forward=wf, boot_ci=False)
            ML_rows.append(res)
    for fam, cols in FAMILIES.items():
        ff = [c for c in cols if c in feats_d2]
        if len(ff) < 2:
            continue
        res = run_config("D2", "pooled", f"fam_{fam}", ff, D2_full, label01_pool, yr_pool, Xw_pool, "hgb",
                          codes=None, min_leaf=300, walk_forward=False, boot_ci=False)
        ML_rows.append(res)
    best_cfg_cache["D2:pooled"] = (dict(df=D2_full, feats=feats_pool, sgn=sgn_pool, yr=yr_pool, X=Xw_pool.mean(),
                                         cluster=None), top50_pool, 300, None)
    P(f"  D2 풀링 모델 완료 ({time.time()-t0:.0f}s, RSS={_rss_mb():.0f}MB)")
    del D2_full; _trim()

    ML = pd.concat(ML_rows, ignore_index=True)
    ML.to_parquet(R / "data" / "r132_ml.parquet", index=False)
    P(f"\n저장: data/r132_ml.parquet ({len(ML):,}행) ({time.time()-t0:.0f}s)")

    P("\n===== C. ML 요약(고정분할, 상위20% 버킷) =====")
    fx = ML[(ML.split == "fixed") & (ML.bucket == "상위20%")]
    P(fx[["panel", "target", "config", "model", "n", "acc", "auc", "net_bp", "trades_per_day"]]
      .sort_values(["panel", "target", "config", "model"]).round(4).to_string(index=False))
    P("\n----- family별 최고 AUC(상위20% 버킷, fam_* config) -----")
    famtab = ML[(ML.split == "fixed") & (ML.bucket == "상위20%") & (ML.config.str.startswith("fam_"))]
    if len(famtab):
        best_fam = famtab.sort_values("auc", ascending=False).groupby(["panel", "target"]).head(3)
        P(best_fam[["panel", "target", "config", "auc", "acc"]].round(4).to_string(index=False))

    # ---------------- C-치환검정: 패널별 최고(HGB all/top50, TRAIN 내부 검증) 후보로 20회 ----------------
    P("\n===== 치환검정(라벨 블록 셔플, 20회) — 패널별 최고 HGB 구성의 2025-26 상위20% 정확도 백분위 =====")
    perm_rows = []
    for gname, (grp, top50, min_leaf, codes) in best_cfg_cache.items():
        df = grp["df"]; sgn = grp["sgn"]; yr = grp["yr"]; Xw = grp["X"]
        label01 = (sgn + 1) / 2
        tr0 = np.isin(yr, TRAIN_YR); te0 = np.isin(yr, TEST_YR)
        va = (yr == 2024); trv = (yr < 2024)
        best_auc = -1; best_feats = None
        for cfg_name, feats in (("all", grp["feats"]), ("top50_ic", top50)):
            if trv.sum() < 100 or va.sum() < 10:
                best_feats = feats; continue
            p_trv, p_va = fit_predict(df, feats, label01, trv, va, "hgb", min_leaf=min_leaf)
            auc_v = roc_auc_score(label01[va], p_va) if len(np.unique(label01[va])) > 1 else -1
            if auc_v > best_auc:
                best_auc = auc_v; best_feats = feats
        if best_feats is None:
            continue
        accs = permute_null_ml(df, best_feats, label01, yr, Xw if np.isfinite(Xw) else 0.0075, "hgb",
                                codes, min_leaf, n_perm=20)
        pfx = "pooled_" if gname == "D2:pooled" else ""
        cfg_match = f"{pfx}all" if best_feats is grp["feats"] else f"{pfx}top50_ic"
        obs_row = ML[(ML.split == "fixed") & (ML.bucket == "상위20%") & (ML.model == "hgb") &
                     (ML.panel == gname.split(":")[0]) & (ML.target == gname.split(":")[1]) &
                     (ML.config == cfg_match)]
        obs = float(obs_row.acc.iloc[0]) if len(obs_row) else np.nan
        pct = float((accs <= obs).mean() * 100) if len(accs) and np.isfinite(obs) else np.nan
        perm_rows.append(dict(panel_target=gname, best_val_auc=best_auc, obs_top20_acc=obs,
                               null_mean=accs.mean() if len(accs) else np.nan,
                               null_std=accs.std() if len(accs) else np.nan, n_perm=len(accs), percentile=pct))
        P(f"  {gname}: 검증AUC={best_auc:.3f} 실측 상위20% 정확도={obs:.3f} vs null "
          f"{(accs.mean() if len(accs) else np.nan):.3f}±{(accs.std() if len(accs) else np.nan):.3f} "
          f"(n={len(accs)}) → 백분위 {pct:.0f}% ({time.time()-t0:.0f}s)")
    PERM = pd.DataFrame(perm_rows)

    # ---------------- 룩어헤드 재검증(사후) ----------------
    P("\n===== 룩어헤드 재점검(사후) =====")
    susp = ML[(ML.split.isin(["fixed"])) & (ML.bucket.isin(["상위10%", "상위20%"]))]
    flagged = susp[(susp.acc > 0.62) | (susp.auc > 0.60)]
    if len(flagged):
        P(f"⚠ TEST 정확도>0.62 또는 AUC>0.60 조합 {len(flagged)}개 — 조사 필요:")
        P(flagged[["panel", "target", "config", "model", "bucket", "n", "acc", "auc"]].round(4).to_string(index=False))
    else:
        P("상위10/20% 정확도가 0.62, AUC가 0.60을 넘는 ML 조합 없음(의심 신호 없음).")
    hiA = A[(A.hit_te > 0.62)]
    if len(hiA):
        P(f"⚠ 단변량(A) TEST hit>0.62 버킷 {len(hiA)}개(survivor 여부 무관) — 상위 5개:")
        P(hiA.sort_values("t_te", ascending=False).head(5)[["group", "feature", "bucket", "hit_tr", "hit_te", "t_te"]]
          .round(4).to_string(index=False))

    P(f"\n전체 완료 ({time.time()-t0:.0f}s), RSS={_rss_mb():.0f}MB")
    (R / "logs" / "r132_scan.summary").write_text(
        f"R132 2단(scan) 완료 ({time.time()-t0:.0f}s). A테스트={len(A)} survivor={int(A.survivor.sum())} "
        f"B테스트={len(B)} survivor={int(B.survivor.sum()) if len(B) else 0} ML행={len(ML)}\n")


if __name__ == "__main__":
    main()
