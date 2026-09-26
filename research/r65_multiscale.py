"""R65 — 차트 선 × 호가·체결을 1초부터 4시간까지 9개 시간 단위로 같은 틀에서 비교 (R62~R64 확장).

시간 단위: 1s 5s 15s 30s (agg1s 1초 체결에서) · 1min 5min 15min 1h 4h (1분봉 + 테이커 매수 대금에서)
기간: 1분 이상 = 학습 2023~24 / 평가 2025-01~2026-09. 초 단위 = 데이터 크기 때문에 학습 2024-11~12 / 평가 2026-07~08.
선·사건: R62와 같은 정의(피벗 k=5, δ=0.25 ATR, 반등 = 1 ATR, 돌파 = 종가 선−δ 밖, 수평 H + 사선 T, 지지·저항, 중복 제거).
위약 선: 실제 선의 위치만 ±0.5~1.5 ATR 이동.
특징: 접근봉 j의 **직전 봉 j−1 종가까지** (그 시점에 주문을 걸어두는 매매). 전부 해당 시간 단위의 봉 격자에서 계산.
  flow5/flow30 (선 쪽 테이커 순체결 비율), vrel5 (5봉 거래대금 / 1440봉 중앙값), speed15 (15봉 선 쪽 이동 / ATR),
  wall1 (선 쪽 1% 잔량 / 반대쪽, bookDepth 30초 as-of), dwall30 (선 쪽 1% 잔량 30봉 변화) + 맥락(nprev, age, slope, kind)
주문 (체결·청산은 더 촘촘한 기준 격자: 초 단위는 1초, 그 외는 1분): 주문은 j 시작에 걸고 1봉 대기, 보유 12봉.
  E1 반등 지정가 @선 (관통 min(5bp, 0.25 ATR)), 손절 선−0.5A, 목표 선+0.5A
  E3 스윕 지정가 @선−0.5A, 손절 선−1.25A, 목표 선
  E2 돌파 스톱 @선+δ(돌파 방향), 손절 선 안쪽 0.25A, 목표 선+1A
  비용: 메이커 2bp, 테이커 5+2bp. 비용 전(총) 수익도 함께 보고해 '신호는 있는데 비용에 먹히는지'를 가른다.
필터: HistGB(맥락+호가·체결, 시드 3) 학습 → 평가 반등확률. E1·E3 상위 30%, E2 하위 30%.
사용: python research/r65_multiscale.py [시간단위 ...]"""
import sys, warnings, glob, time
from pathlib import Path
import numpy as np, pandas as pd, numba
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R))
from freesia import store, costs
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
K, DLT, TO, AGE, WIN, SLOPE = 5, 0.25, 48, 500, 150, 0.3
CM = costs.MAKER; CT = costs.TAKER + 2e-4
SUB = ["1s", "5s", "15s", "30s"]; MIN = ["1min", "5min", "15min", "1h", "4h"]
W_SUB = {"tr": ("2024-11-01", "2025-01-01"), "te": ("2026-07-01", "2026-09-01")}
W_MIN = {"tr": ("2023-01-01", "2025-01-01"), "te": ("2025-01-01", "2026-09-25")}
SYMS = ("BTCUSDT", "ETHUSDT")
FEAT = ["flow5", "flow30", "vrel5", "speed15", "wall1", "dwall30"]
CTX = ["nprev", "age", "slope", "kind"]


# ---------- 기준 격자 ----------
def base_1s(sym, a, b):
    a, b = pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC"); parts = []
    for ym in pd.period_range(a, b - pd.Timedelta(seconds=1), freq="M"):
        fs = glob.glob(str(store.PQ / "agg1s" / sym / f"{ym}*.parquet"))
        if fs: parts.append(pd.read_parquet(fs[0], columns=["ts", "o", "h", "l", "c", "bq", "sq"]))
    x = pd.concat(parts).drop_duplicates("ts").set_index("ts").sort_index()
    x = x[(x.index >= a) & (x.index < b)]
    g = pd.date_range(a, b, freq="1s", inclusive="left", tz="UTC")
    x = x.reindex(g); x["c"] = x.c.ffill()
    for k in ("o", "h", "l"): x[k] = x[k].fillna(x.c)
    x[["bq", "sq"]] = x[["bq", "sq"]].fillna(0.0)
    return x.dropna()


def base_1m(sym, a, b):
    k = store.load("klines_1m", sym).set_index("ts")
    k = k[(k.index >= pd.Timestamp(a, tz="UTC")) & (k.index < pd.Timestamp(b, tz="UTC"))]
    return pd.DataFrame({"o": k.open, "h": k.high, "l": k.low, "c": k.close,
                         "bq": k.taker_buy_quote_volume, "sq": k.quote_volume - k.taker_buy_quote_volume})


def tf_bars(base, tf, book):
    g = base.resample(tf).agg({"o": "first", "h": "max", "l": "min", "c": "last", "bq": "sum", "sq": "sum"}).dropna()
    lh, ll, lc = np.log(g.h), np.log(g.l), np.log(g.c)
    tr = pd.concat([lh - ll, (lh - lc.shift()).abs(), (ll - lc.shift()).abs()], axis=1).max(axis=1)
    g["atr"] = tr.rolling(14).mean()
    v = g.bq + g.sq
    g["f5"] = (g.sq - g.bq).rolling(5).sum() / v.rolling(5).sum().clip(lower=1)
    g["f30"] = (g.sq - g.bq).rolling(30).sum() / v.rolling(30).sum().clip(lower=1)
    v5 = v.rolling(5).sum()
    g["vrel5"] = np.log(v5.clip(lower=1) / v5.rolling(1440, min_periods=100).median().clip(lower=1))
    g["r15"] = lc - lc.shift(15)
    end = g.index + pd.Timedelta(tf)                              # 봉 종가 시각 기준 as-of 호가
    bk = pd.merge_asof(pd.DataFrame({"t": end.astype("datetime64[ns, UTC]")}), book, left_on="t", right_on="ts", direction="backward",
                       tolerance=pd.Timedelta("2min"))
    g["wall1"] = np.log(bk["n-1"].values / bk["n+1"].values)
    g["nb1"] = np.log(bk["n-1"].values); g["na1"] = np.log(bk["n+1"].values)
    g["dwb30"] = g.nb1 - g.nb1.shift(30); g["dwa30"] = g.na1 - g.na1.shift(30)
    g["lh"], g["ll"], g["lc"] = lh, ll, lc
    return g.dropna(subset=["atr"])


# ---------- 선 (numba) ----------
@numba.njit(cache=True)
def pivots_low(x, k):
    n = len(x); out = []
    for i in range(k, n - k):
        v = x[i]; ok = True
        for j in range(i - k, i + k + 1):
            if x[j] < v: ok = False; break
        if ok: out.append(i)
    return np.array(out, dtype=np.int64)


@numba.njit(cache=True)
def lines_H(pl, l, atr, n):
    m = len(pl); L = np.zeros((m, 7)); c = 0
    for p in pl:
        i0 = p + K
        if i0 >= n: continue
        L[c, 0] = i0; L[c, 1] = min(n, i0 + AGE); L[c, 2] = p; L[c, 3] = l[p]; L[c, 4] = 0.0; L[c, 5] = DLT * atr[p]; L[c, 6] = 1
        c += 1
    return L[:c]


@numba.njit(cache=True)
def lines_T(pl, l, atr, n):
    m = len(pl); L = np.zeros((m, 7)); c = 0
    for q in range(m):
        p2 = pl[q]; i0 = p2 + K
        if i0 >= n: continue
        best = -1; bs = 0.0
        for r in range(q - 1, -1, -1):
            p1 = pl[r]
            if p2 - p1 > WIN: break
            if p2 - p1 < 10: continue
            s = (l[p2] - l[p1]) / (p2 - p1)
            if abs(s) > SLOPE * atr[p2]: continue
            ok = True
            for t in range(p1, p2 + 1):
                if l[t] - (l[p1] + s * (t - p1)) < -0.1 * atr[p2]: ok = False; break
            if ok: best = p1; bs = s
        if best < 0: continue
        n0 = 2
        for r in range(q - 1, -1, -1):
            p = pl[r]
            if p <= best: break
            if l[p] - (l[best] + bs * (p - best)) < DLT * atr[p2]: n0 += 1
        L[c, 0] = i0; L[c, 1] = min(n, i0 + min(AGE, 2 * (p2 - best) + 50)); L[c, 2] = p2; L[c, 3] = l[p2]
        L[c, 4] = bs; L[c, 5] = DLT * atr[p2]; L[c, 6] = n0
        c += 1
    return L[:c]


@numba.njit(cache=True)
def scan_all(Ls, h, l, c, dedup):
    """각 선을 추적해 사건을 모은다. 반환 ev[:, (j, res, m, nprev, L_j, d, i0, slope/atr)], kept 마스크."""
    n = len(c); cap = len(Ls) * 4 + 16; ev = np.zeros((cap, 8)); ne = 0
    kept = np.zeros(len(Ls), np.bool_); death = np.zeros(len(Ls))
    for q in range(len(Ls)):
        i0 = int(Ls[q, 0]); iend = int(Ls[q, 1]); ia = int(Ls[q, 2]); a = Ls[q, 3]; s = Ls[q, 4]; d = Ls[q, 5]; n0 = Ls[q, 6]
        B = d / DLT
        # 먼저 추적(소멸 시각 필요)
        tmp = np.zeros((64, 5)); e = 0; armed = False; nb = 0; j = i0; dth = iend
        while j < min(iend, n):
            Lj = a + s * (j - ia)
            if not armed:
                if c[j] < Lj - d: dth = j; break
                if l[j] > Lj + d: armed = True
                j += 1; continue
            if l[j] <= Lj + d:
                res = 0; m = j
                while m < n and m - j <= TO:
                    Lm = a + s * (m - ia)
                    if c[m] < Lm - d: res = -1; break
                    if m > j and h[m] > Lm + B: res = 1; break
                    m += 1
                m = min(m, n - 1)
                if e < 64:
                    tmp[e, 0] = j; tmp[e, 1] = res; tmp[e, 2] = m; tmp[e, 3] = nb + n0; tmp[e, 4] = Lj; e += 1
                if res == -1: dth = m; break
                if res == 1: nb += 1
                armed = False; j = m + 1; continue
            j += 1
        death[q] = dth
        if dedup:
            L0 = a + s * (i0 - ia); dup = False
            r = q - 1; cnt = 0
            while r >= 0 and cnt < 400:
                if kept[r] and Ls[r, 0] < i0 and death[r] > i0:
                    Lr = Ls[r, 3] + Ls[r, 4] * (i0 - Ls[r, 2])
                    if abs(Lr - L0) < d: dup = True; break
                r -= 1; cnt += 1
            if dup: continue
        kept[q] = True
        for k in range(e):
            if ne >= cap: break
            ev[ne, 0] = tmp[k, 0]; ev[ne, 1] = tmp[k, 1]; ev[ne, 2] = tmp[k, 2]; ev[ne, 3] = tmp[k, 3]
            ev[ne, 4] = tmp[k, 4]; ev[ne, 5] = d; ev[ne, 6] = i0; ev[ne, 7] = s / (d / DLT); ne += 1
    return ev[:ne], kept


def events(g, rng):
    """한 시간 단위·한 종목: 지지/저항 × H/T × 실제/위약 사건 DataFrame."""
    out = []; n = len(g); atr = g.atr.values
    for side, sg in (("지지", 1), ("저항", -1)):
        h, l, c = (g.lh.values, g.ll.values, g.lc.values) if sg == 1 else (-g.ll.values, -g.lh.values, -g.lc.values)
        pl = pivots_low(l, K)
        for kind in ("H", "T"):
            Ls = lines_H(pl, l, atr, n) if kind == "H" else lines_T(pl, l, atr, n)
            if len(Ls) == 0: continue
            ev, kept = scan_all(Ls, h, l, c, True)
            Lk = Ls[kept].copy()
            u = rng.uniform(0.5, 1.5, len(Lk)) * (Lk[:, 5] / DLT) * rng.choice([-1, 1], len(Lk))
            lvl0 = Lk[:, 3] + u + Lk[:, 4] * (Lk[:, 0] - Lk[:, 2])
            above = lvl0 >= c[Lk[:, 0].astype(int)] - Lk[:, 5]
            u = np.where(above, -np.abs(u), u); Lk[:, 3] = Lk[:, 3] + u
            evp, _ = scan_all(Lk, h, l, c, False)
            for E, plc in ((ev, False), (evp, True)):
                df = pd.DataFrame(E, columns=["j", "res", "m", "nprev", "L", "d", "i0", "slope"])
                df["side"] = side; df["sg"] = sg; df["kind"] = int(kind == "T"); df["plc"] = plc
                df["lvl"] = sg * df.L
                out.append(df)
    D = pd.concat(out, ignore_index=True)
    D[["j", "m", "i0"]] = D[["j", "m", "i0"]].astype(int)
    D = D[(D.res != 0) & (D.j >= 31)].reset_index(drop=True)
    r = g.iloc[D.j.values - 1]; sg = D.sg.values                  # 직전 봉 종가까지
    D["t"] = g.index[D.j.values]; D["atr"] = g.atr.values[D.j.values - 1]
    D["flow5"] = sg * r.f5.values; D["flow30"] = sg * r.f30.values; D["vrel5"] = r.vrel5.values
    D["speed15"] = -sg * r.r15.values / D.atr.values; D["wall1"] = sg * r.wall1.values
    D["dwall30"] = np.where(sg == 1, r.dwb30.values, r.dwa30.values)
    D["age"] = D.j - D.i0; D["y"] = (D.res == 1).astype(int)
    return D


# ---------- 주문 시뮬레이션 (기준 격자) ----------
@numba.njit(cache=True)
def sim(t0, wait, hold, ori, ye, ys, yt, se, pen, lh, ll, lc, cm, ct):
    n = len(lc); out = np.full((len(t0), 2), np.nan)
    for e in range(len(t0)):
        s = ori[e]; fill = -1
        for i in range(t0[e], min(n, t0[e] + wait[e])):
            hi_ = lh[i] if s == 1 else -ll[i]; lo_ = ll[i] if s == 1 else -lh[i]
            if se:
                if hi_ >= ye[e]: fill = i; break
            elif lo_ <= ye[e] - pen[e]: fill = i; break
        if fill < 0: out[e, 0] = 0; continue
        res = np.nan; j = fill
        while j < min(n, fill + hold[e]):
            hi_ = lh[j] if s == 1 else -ll[j]; lo_ = ll[j] if s == 1 else -lh[j]
            if lo_ <= ys[e]: res = ys[e] - ye[e] - ct; break
            if j > fill and hi_ >= yt[e]: res = yt[e] - ye[e] - cm; break
            j += 1
        if np.isnan(res): res = lc[min(j, n - 1)] * s - ye[e] - ct
        out[e, 0] = 1; out[e, 1] = res - (ct if se else cm)
    return out


def trade(D, base, tf, rule, cost=True):
    lh, ll, lc = np.log(base.h.values), np.log(base.l.values), np.log(base.c.values)
    step = pd.Timedelta(tf) // (base.index[1] - base.index[0])
    t0 = base.index.searchsorted(pd.DatetimeIndex(D.t))
    ori = D.sg.values.copy(); A = D.atr.values; d = D.d.values; yL = ori * D.lvl.values
    wait = np.full(len(D), step); hold = 12 * wait; pen = np.minimum(5e-4, 0.25 * A)
    if rule == "E1": ye, ys, yt, se = yL, yL - 0.5 * A, yL + 0.5 * A, False
    elif rule == "E3": ye, ys, yt, se = yL - 0.5 * A, yL - 1.25 * A, yL, False
    else: ori = -ori; zL = -yL; ye, ys, yt, se = zL + d, zL - 0.25 * A, zL + A, True
    cm, ct = (CM, CT) if cost else (0.0, 0.0)
    return sim(t0, wait, hold, ori, ye, ys, yt, se, pen, lh, ll, lc, cm, ct)


def model_p(tr, te, feats):
    p = np.mean([HistGradientBoostingClassifier(max_iter=250, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=200,
                                                 l2_regularization=1.0, random_state=sd).fit(tr[feats].values, tr.y.values)
                 .predict_proba(te[feats].values)[:, 1] for sd in range(3)], axis=0)
    return p


if __name__ == "__main__":
    TFS = sys.argv[1:] or SUB + MIN
    rng = np.random.default_rng(0)
    books = {}
    for s_ in SYMS:
        b_ = store.load("bookdepth", s_)[["ts", "n-1", "n+1"]].sort_values("ts"); b_["ts"] = b_.ts.astype("datetime64[ns, UTC]"); books[s_] = b_
    rows = []
    for tf in TFS:
        t_ = time.time(); W = W_SUB if tf in SUB else W_MIN; bf = base_1s if tf in SUB else base_1m
        Ds, bases, days = [], {}, 0
        for sym in SYMS:
            for w in ("tr", "te"):
                base = bf(sym, *W[w]); g = tf_bars(base, tf, books[sym])
                D = events(g, rng).assign(sym=sym, w=w); Ds.append(D)
                if w == "te": bases[sym] = base
            days = (pd.Timestamp(W["te"][1]) - pd.Timestamp(W["te"][0])).days
        D = pd.concat(Ds, ignore_index=True).dropna(subset=FEAT)
        te_all = D[D.w == "te"]
        atr_bp = te_all[~te_all.plc].atr.median() * 1e4
        # 반등률·AUC
        br = {plc: te_all[te_all.plc == plc].y.mean() for plc in (False, True)}
        br1 = {plc: te_all[(te_all.plc == plc) & (te_all.nprev == 1) & (te_all.kind == 0)].y.mean() for plc in (False, True)}
        auc = {}
        for plc in (False, True):
            S = D[D.plc == plc]; tr, te = S[S.w == "tr"], S[S.w == "te"]
            p = model_p(tr, te, CTX + FEAT); auc[plc] = roc_auc_score(te.y, p)
            D.loc[te.index, "p"] = p
        D.loc[D.w == "te", "q"] = D[D.w == "te"].groupby("plc").p.rank(pct=True)
        uni = []
        for c_ in ("speed15", "vrel5", "wall1", "flow30"):
            vals = []
            for plc in (False, True):
                s = te_all[te_all.plc == plc][[c_, "y"]].dropna(); qq = pd.qcut(s[c_].rank(method="first"), 5, labels=False)
                m = s.groupby(qq).y.mean(); vals.append((m.iloc[-1] - m.iloc[0]) * 100)
            uni.append(f"{c_} {vals[0]:+.1f}/{vals[1]:+.1f}")
        # 주문
        tr_res = {}
        for rule in ("E1", "E3", "E2"):
            for plc in (False, True):
                for filt in (False, True):
                    S = D[(D.w == "te") & (D.plc == plc)]
                    if filt: S = S[S.q >= 0.7] if rule != "E2" else S[S.q <= 0.3]
                    net, gross = [], []
                    for sym in SYMS:
                        ev = S[S.sym == sym]
                        if len(ev) == 0: continue
                        o1 = trade(ev, bases[sym], tf, rule, True); o0 = trade(ev, bases[sym], tf, rule, False)
                        f = o1[:, 0] == 1; net.append(o1[f, 1]); gross.append(o0[f, 1])
                    net = np.concatenate(net); gross = np.concatenate(gross)
                    tr_res[(rule, plc, filt)] = (net.mean() * 1e4, gross.mean() * 1e4, len(net) / days,
                                                 net.std() / np.sqrt(max(len(net), 1)) * 1e4)
        P(f"\n===== {tf}  (평가 {W['te'][0]}~{W['te'][1]}, ATR 중앙 {atr_bp:.1f}bp, 실제 사건 {(~te_all.plc).sum():,} / 하루 {(~te_all.plc).sum()/days:.0f}, {time.time()-t_:.0f}s)")
        P(f"  반등률 실제/위약 {br[False]:.3f}/{br[True]:.3f} | 수평 첫 접근 {br1[False]:.3f}/{br1[True]:.3f} | AUC 실제/위약 {auc[False]:.3f}/{auc[True]:.3f}")
        P(f"  단변량 Q5−Q1 반등률 %p (실제/위약): " + " | ".join(uni))
        for rule in ("E1", "E3", "E2"):
            a, b = tr_res[(rule, False, False)], tr_res[(rule, True, False)]; fa, fb = tr_res[(rule, False, True)], tr_res[(rule, True, True)]
            P(f"  {rule}: 실제 총 {a[1]:+6.2f} 순 {a[0]:+6.2f}bp (±{a[3]:.2f}, 하루 {a[2]:.0f}) | 위약 총 {b[1]:+6.2f} 순 {b[0]:+6.2f} | "
              f"필터 실제 총 {fa[1]:+6.2f} 순 {fa[0]:+6.2f} (하루 {fa[2]:.0f}) / 위약 총 {fb[1]:+6.2f}")
            rows.append((tf, atr_bp, rule, a[1], a[0], b[1], b[0], fa[1], fa[0], fb[1], fb[0], a[2], fa[2]))
        rows.append((tf, atr_bp, "반등률", br[False], br[True], br1[False], br1[True], auc[False], auc[True], np.nan, np.nan, np.nan, np.nan))
        del Ds, D, bases
    O = pd.DataFrame(rows, columns=["tf", "atr_bp", "항목", "실제총", "실제순", "위약총", "위약순", "필터실제총", "필터실제순", "필터위약총", "필터위약순", "하루", "필터하루"])
    O.to_csv(R / "logs" / f"r65_table{'_' + '_'.join(sys.argv[1:]) if len(sys.argv) > 1 else ''}.csv", index=False)
    (R / "logs" / "r65.summary").write_text("R65 1초~4시간 9개 시간단위 선×호가 비교 완료. 세부 logs/r65.log\n")
