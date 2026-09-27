"""R99 — 변동성 적중 80~92% 시점(R90 타이밍 상위 10%)에서 방향을 맞추는 방법 300가지 이상 조사·검정.

사건: R90 p ≥ th10(학습 문턱, 표본 밖 도달률 80%), 종목별 240분 겹침 제거. 방향 라벨 side = 기준가 c[t−1]에서 ±2% 먼저 닿은 쪽.
학습/평가: 방향 관계·문턱은 **2023년 이전 데이터만**으로 학습(학습 풀 = 2020~23 결판 큰 움직임, 60분 겹침 제거),
           선택 지표 = 학습 풀 z + 학습 사건(2022~23 상위 10%) 적중. 평가 = 2024~26 상위 10% 사건(손대지 않은 구간).
가족
  F1 단일 신호 규칙: R97 119개 + 사건 문맥 특징 약 35개 × 극단 문턱 4(10·20·33·50%) → 방향은 학습 풀에서 결정
  F2 이론 규칙 약 40개: 방향을 이론으로 고정(학습 없음) — 모멘텀·역추세·포지션 쏠림(펀딩·롱숏비)·OI×가격·코인베이스 선행·ETH 선행·옵션 흐름·
     라운드넘버·가까운 극값 돌파·대형 체결 불균형 등
  F3 지연 진입 확인: 사건 후 120분 안에 기준가 ±x(0.25·0.5·0.75%) 첫 이탈 → 이탈 구간의 확인 신호(테이커 매수비·거래량·속도·코인베이스/ETH 동행·
     대형 체결·꼬리)로 추종/역행 결정(학습 = 2022~23 상위 30% 사건). 성과 = 진입가 ±1.5% 먼저 닿기.
  F4 상위 단일 규칙 20개의 쌍 합의(190)   F5 ML(HistGB·로지스틱·ExtraTrees·kNN × 특징 묶음 × 확신 필터)   F6 투표(상위 k)
다중검정: 평가 z = (적중 − 예측 방향 구성에 맞춘 기대 적중)/표준오차. 전체 방법의 평가 z 분포 vs 정규 기대, BH 5%.
매매: 평가 상위 후보 — 정적은 다음 분 시가 진입 ±2%/4h, 지연은 확인 다음 분 시가 ±1.5%. 비용 테이커 7bp·목표 지정가 2bp. 3단 레버리지.
사용: python research/r99_dir300.py"""
import sys, warnings, time, glob
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
import r97_direction as Q
from freesia import store
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
SYMS = Q.SYMS


def minute_extra(s, K, CB, AG):
    """분 단위 추가 특징(값은 분 i 종가까지) → 배열 dict."""
    k = K[s]; c = k.close; v = k.volume; tb = k.taker_buy_volume; n = len(k); f = {}
    for m in (1, 15, 30): f[f"EV|tbr{m}"] = (tb.rolling(m).sum() / v.rolling(m).sum()).values
    f["EV|volz15"] = (v.rolling(15).sum() / (v.rolling(1440).sum() / 96)).values
    for m in (1, 30, 120): f[f"EV|ret{m}"] = np.log(c / c.shift(m)).values
    f["EV|pos15"] = ((c - k.low.rolling(15).min()) / (k.high.rolling(15).max() - k.low.rolling(15).min())).values
    rv = np.log(c).diff().rolling(300).std() * np.sqrt(240)
    hi = k.high.rolling(1440).max(); lo = k.low.rolling(1440).min()
    f["EV|dhi24"] = ((hi / c - 1) / rv).values; f["EV|dlo24"] = ((1 - lo / c) / rv).values; f["EV|near24"] = f["EV|dlo24"] - f["EV|dhi24"]
    step = 1000.0 if s == "BTCUSDT" else 50.0
    up = np.ceil(c / step) * step / c - 1; dn = 1 - np.floor(c / step) * step / c; f["EV|round"] = ((up - dn) / (up + dn + 1e-12)).values
    r240 = np.log(c / c.shift(240)); f["EV|big24"] = (r240.rolling(1440).max() + r240.rolling(1440).min()).values
    ts = pd.DatetimeIndex(k.ts); day0 = pd.Series(c.values, index=ts).groupby(ts.normalize()).transform("first").values
    f["EV|dayopen"] = c.values / day0 - 1
    wk = ts.normalize() - pd.to_timedelta(ts.dayofweek, unit="D"); w0 = pd.Series(c.values, index=ts).groupby(wk).transform("first").values
    f["EV|weekopen"] = c.values / w0 - 1
    cb = CB[s].reindex(ts).ffill().values; cbs = pd.Series(cb)
    for m in (5, 15, 60): f[f"EV|cblead{m}"] = (np.log(cbs / cbs.shift(m)) - np.log(c / c.shift(m))).values
    o = [x for x in SYMS if x != s][0]; co = pd.Series(K[o].close.values, index=pd.DatetimeIndex(K[o].ts)).reindex(ts).ffill().values; co = pd.Series(co)
    for m in (5, 15): f[f"EV|oth{m}"] = np.log(co / co.shift(m)).values
    if AG is not None:
        a = AG[s].reindex(ts)
        for m in (5, 15, 60):
            lb = a.lbq.rolling(m, min_periods=1).sum(); ls = a.lsq.rolling(m, min_periods=1).sum()
            f[f"EV|lti{m}"] = ((lb - ls) / (lb + ls + 1)).where(a.lbq.notna()).values
    return f


def load_agg():
    out = {}
    for s in SYMS:
        parts = []
        for f in sorted(glob.glob(str(R / "data/parquet/agg1s" / s / "*.parquet"))):
            d = pd.read_parquet(f, columns=["ts", "bq", "sq", "lbq", "lsq"]); d["ts"] = pd.DatetimeIndex(d.ts).floor("min")
            parts.append(d.groupby("ts")[["bq", "sq", "lbq", "lsq"]].sum())
        out[s] = pd.concat(parts); out[s] = out[s][~out[s].index.duplicated()]
    return out


def load_cb():
    out = {}
    for s in SYMS:
        d = pd.concat([pd.read_parquet(f, columns=["ts", "close"]) for f in sorted(glob.glob(str(R / "data/parquet/coinbase_1m" / (s[:3] + "USD") / "*.parquet")))])
        d = d.drop_duplicates("ts").set_index("ts").close; d.index = Q.utc(d.index); out[s] = d.sort_index()
    return out


@numba.njit(cache=True)
def delayed(t, o, h, l, c, v, tb, cbc, oc, lb, ls, x, W, B):
    """사건마다 첫 ±x 이탈 → (분 j, 방향 d, 확인 특징 8개, 추종 결과 y(1/−1/0), 추종 수익, 역행 수익)."""
    n = len(c); out = np.full((len(t), 13), np.nan)
    for q in range(len(t)):
        i = t[q]; ref = c[i - 1]; j0 = -1; d = 0
        if i + W + 240 >= n: continue
        for j in range(i, i + W):
            u = h[j] >= ref * (1 + x); dn = l[j] <= ref * (1 - x)
            if u and dn: break
            if u: j0 = j; d = 1; break
            if dn: j0 = j; d = -1; break
        if j0 < 0: continue
        vs = 0.0; ts_ = 0.0; lbs = 0.0; lss = 0.0
        for j in range(i, j0 + 1):
            vs += v[j]; ts_ += tb[j]
            if not np.isnan(lb[j]): lbs += lb[j]; lss += ls[j]
        pre = 0.0
        for j in range(i - 60, i): pre += v[j]
        span = j0 - i + 1
        out[q, 0] = j0; out[q, 1] = d
        out[q, 2] = d * (ts_ / vs - 0.5) if vs > 0 else np.nan            # 이탈 방향 쪽 테이커 우위
        out[q, 3] = (vs / span) / (pre / 60 + 1e-9)                        # 거래량 가속
        out[q, 4] = span                                                   # 속도(분)
        out[q, 5] = d * (cbc[j0] / cbc[i - 1] - 1) / x if cbc[i - 1] > 0 else np.nan   # 코인베이스 동행
        out[q, 6] = d * (oc[j0] / oc[i - 1] - 1) / x if oc[i - 1] > 0 else np.nan      # 다른 종목 동행
        out[q, 7] = d * (lbs - lss) / (lbs + lss + 1) if lbs + lss > 0 else np.nan    # 대형 체결 불균형
        out[q, 8] = d * (c[j0] / ref - 1) / x                              # 이탈 봉 종가 유지(꼬리)
        e = o[j0 + 1]; tp = e * (1 + d * B); sl = e * (1 - d * B); y = 0; rr = np.nan
        for j in range(j0 + 1, j0 + 241):
            if (d > 0 and l[j] <= sl) or (d < 0 and h[j] >= sl): y = -1; break
            if (d > 0 and h[j] >= tp) or (d < 0 and l[j] <= tp): y = 1; break
        if y == 1: rr = B
        elif y == -1: rr = -B
        else: rr = d * (c[j0 + 240] / e - 1)
        out[q, 9] = y; out[q, 10] = rr - (9e-4 if y == 1 else 14e-4); out[q, 11] = -rr - (9e-4 if y == -1 else 14e-4)
        out[q, 12] = pre
    return out


def dedup(g, gap):
    keep = []; last = {}
    for s, t in zip(g.sym.values, g.t.values):
        ok = t >= last.get(s, -10**12) + gap; keep.append(ok)
        if ok: last[s] = t
    return g[np.array(keep)]


def zhit(pred, side):
    pred = np.nan_to_num(np.asarray(pred, float)); side = np.asarray(side)
    m = (pred != 0) & (side != 0); pr_ = pred[m]; sd = side[m]; n = m.sum()
    if n < 15: return np.nan, np.nan, n
    hit = (pr_ == sd).mean(); b = (sd == 1).mean(); pu = (pr_ == 1).mean(); e = pu * b + (1 - pu) * (1 - b)
    return hit, (hit - e) / np.sqrt(e * (1 - e) / n), n


if __name__ == "__main__":
    t_ = time.time()
    D, OLD, NEW = Q.frame()
    T90 = pd.read_parquet(R / "data" / "r90_wf.parquet")[["sym", "t", "th10", "th20", "th30"]]
    D = D.merge(T90, on=["sym", "t"], how="left")
    K = {s: store.load("klines_1m", s) for s in SYMS}; CB = load_cb(); AG = load_agg(); P(f"데이터 준비 ({time.time()-t_:.0f}s)")
    EXF = None
    for s in SYMS:
        f = minute_extra(s, K, CB, AG); m = (D.sym == s).values; ti = D.loc[m, "t"].values - 1
        for kk, arr in f.items():
            if kk not in D: D[kk] = np.nan
            D.loc[m, kk] = arr[ti]
        EXF = list(f.keys())
    D = D.replace([np.inf, -np.inf], np.nan)
    FE = OLD + NEW + EXF; P(f"특징 {len(FE)}개 ({time.time()-t_:.0f}s)")
    D = D.sort_values(["sym", "t"]).reset_index(drop=True)
    TP = dedup(D[(D.yr <= 2023) & (D.side != 0)], 60)                          # 학습 풀
    hp = D[(D.p >= D.th10)]; EV = dedup(hp, 240)
    TRE = EV[EV.yr <= 2023]; TE = EV[EV.yr >= 2024]
    E30 = dedup(D[(D.p >= D.th30)], 240); TR30 = E30[E30.yr <= 2023]
    P(f"학습 풀 {len(TP):,} · 학습 사건(상위10%) {len(TRE)} (결판 {(TRE.side!=0).sum()}) · 평가 사건 {len(TE)} (결판 {(TE.side!=0).sum()}, 상승 비율 {(TE.side[TE.side!=0]==1).mean():.3f})")
    rows = []; PRED = {}

    def add(fam, name, ptp, ptre, pte):
        ptp, ptre, pte = (None if v is None else np.nan_to_num(np.asarray(v, float)) for v in (ptp, ptre, pte))
        a = zhit(ptp, TP.side.values) if ptp is not None else (np.nan, np.nan, 0)
        b = zhit(ptre, TRE.side.values); c_ = zhit(pte, TE.side.values)
        rows.append(dict(fam=fam, m=name, tp_hit=a[0], tp_z=a[1], tr_hit=b[0], tr_z=b[1], tr_n=b[2], te_hit=c_[0], te_z=c_[1], te_n=c_[2]))
        PRED[(fam, name)] = pte

    # F1 단일 신호
    for c in FE:
        x = TP[c].dropna()
        if len(x) < 2000: continue
        for q in (0.10, 0.20, 0.33, 0.50):
            lo, hi = x.quantile([q, 1 - q]) if q < 0.5 else (x.median(), x.median())
            up_hi = (TP.side[TP[c] >= hi] == 1).mean(); up_lo = (TP.side[TP[c] <= lo] == 1).mean()
            sg = 1 if up_hi >= up_lo else -1
            fn = lambda G: np.where(G[c] >= hi, sg, np.where(G[c] <= lo, -sg, 0)) * G[c].notna()
            add("F1", f"{c}@{int(q*100)}", fn(TP), fn(TRE), fn(TE))
    P(f"F1 {sum(r['fam']=='F1' for r in rows)} ({time.time()-t_:.0f}s)")
    # F2 이론 규칙(방향 고정)
    med = lambda c: TP[c].median()
    TH = {
        "모멘텀 15분": lambda G: np.sign(G.ret15), "모멘텀 1시간": lambda G: np.sign(G.ret60), "모멘텀 4시간": lambda G: np.sign(G.ret240),
        "모멘텀 1일": lambda G: np.sign(G.ret1440), "역추세 1시간": lambda G: -np.sign(G.ret60), "역추세 1일": lambda G: -np.sign(G.ret1440),
        "모멘텀 1분": lambda G: np.sign(G["EV|ret1"]), "모멘텀 2시간": lambda G: np.sign(G["EV|ret120"]),
        "압축 위치 돌파(4h 범위 위쪽→상승)": lambda G: np.sign(G.pos240 - 0.5), "압축 위치 돌파(1일)": lambda G: np.sign(G.pos1440 - 0.5),
        "7일 범위 위치 추종": lambda G: np.sign(G.pos7d - 0.5), "가까운 24h 극값 쪽 돌파": lambda G: np.sign(G["EV|near24"]),
        "먼 24h 극값 쪽(되돌림)": lambda G: -np.sign(G["EV|near24"]), "라운드넘버 가까운 쪽": lambda G: -np.sign(G["EV|round"]),
        "펀딩 역발상": lambda G: -np.sign(G.fund - med("fund")), "롱숏비 역발상": lambda G: -np.sign(G.lsr - med("lsr")),
        "상위트레이더 롱숏 추종": lambda G: np.sign(G.tlsr - med("tlsr")),
        "OI↑·가격↑=상승, OI↑·가격↓=하락": lambda G: np.where(G.oi60 > 0, np.sign(G.ret60), 0),
        "OI↓(청산)·가격 반대": lambda G: np.where(G.oi60 < 0, -np.sign(G.ret60), 0),
        "과열 롱(펀딩↑)+하락 시작=하락 캐스케이드": lambda G: np.where((G.fund > TP.fund.quantile(.8)) & (G.ret15 < 0), -1, np.where((G.fund < TP.fund.quantile(.2)) & (G.ret15 > 0), 1, 0)),
        "코인베이스 프리미엄": lambda G: np.sign(G.cbp), "코인베이스 프리미엄 변화": lambda G: np.sign(G.dcbp60),
        "코인베이스 선행 15분": lambda G: np.sign(G["EV|cblead15"]), "코인베이스 선행 5분": lambda G: np.sign(G["EV|cblead5"]),
        "다른 종목 선행 15분": lambda G: np.sign(G["EV|oth15"] - G.ret15),
        "테이커 매수 우위 15분": lambda G: np.sign(G["EV|tbr15"] - 0.5), "테이커 매수 우위 24시간": lambda G: np.sign(G["VDIR|tbr24h"] - 0.5),
        "CVD 30분": lambda G: np.sign(G.cvd30), "대형 체결 불균형 15분": lambda G: np.sign(G["EV|lti15"]).fillna(0),
        "대형 체결 불균형 60분": lambda G: np.sign(G["EV|lti60"]).fillna(0),
        "옵션 매수 주도 콜>풋": lambda G: np.sign(G.pbuy - 0.5).fillna(0), "리스크 리버설(콜 비쌈→상승)": lambda G: np.sign(G.rr25).fillna(0),
        "리스크 리버설 4h 변화": lambda G: np.sign(G.drr4).fillna(0), "선물 프리미엄": lambda G: np.sign(G.prem - med("prem")),
        "프리미엄 5분 변화": lambda G: np.sign(G.dprem5), "스테이블 7일 증가": lambda G: np.sign(G["FLOW|st7"]),
        "김프 24h 변화": lambda G: np.sign(G["KIM|dprem24"]), "당일 시가 대비 추종": lambda G: np.sign(G["EV|dayopen"]),
        "주 시가 회귀": lambda G: -np.sign(G["EV|weekopen"]), "직전 큰 4h 움직임 지속": lambda G: np.sign(G["EV|big24"]),
        "직전 큰 4h 움직임 반전": lambda G: -np.sign(G["EV|big24"]), "상승 반분산 우위": lambda G: np.sign(G["SEMI|ud4h"]),
        "30일 이평 위=상승": lambda G: np.sign(G["MA|d30d"]), "7일 이평 위=상승": lambda G: np.sign(G["MA|d7d"]),
        "30일 모멘텀": lambda G: np.sign(G["TSM|r30d"]), "DVOL 상승=하락": lambda G: -np.sign(G["IVX|ddvol24"]).fillna(0),
    }
    for nm, fn in TH.items():
        g = lambda G: np.nan_to_num(np.asarray(fn(G), float))
        add("F2", nm, g(TP), g(TRE), g(TE))
    P(f"F2 {len(TH)} ({time.time()-t_:.0f}s)")
    # F4 쌍 합의 · F6 투표 (학습 풀 z + 학습 사건 z 합으로 상위 20 단일 규칙)
    O = pd.DataFrame(rows); O["score"] = O.tp_z.fillna(0) + O.tr_z.fillna(0)
    top = O[O.fam.isin(["F1", "F2"])].sort_values("score", ascending=False)
    top = top[~top.m.str.split("@").str[0].duplicated()].head(20)
    def pv(fam, nm, G):
        if fam == "F2": return np.nan_to_num(np.asarray(TH[nm](G), float))
        c, q = nm.split("@"); q = int(q) / 100; x = TP[c].dropna()
        lo, hi = x.quantile([q, 1 - q]) if q < 0.5 else (x.median(), x.median())
        sg = 1 if (TP.side[TP[c] >= hi] == 1).mean() >= (TP.side[TP[c] <= lo] == 1).mean() else -1
        return np.where(G[c] >= hi, sg, np.where(G[c] <= lo, -sg, 0)) * G[c].notna()
    TOPP = {(r.fam, r.m): {nm: pv(r.fam, r.m, G) for nm, G in (("tp", TP), ("tre", TRE), ("te", TE))} for r in top.itertuples()}
    keys = list(TOPP)
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            pa, pb = TOPP[keys[a]], TOPP[keys[b]]
            f = lambda k: np.where((pa[k] == pb[k]) & (pa[k] != 0), pa[k], 0)
            add("F4", f"{keys[a][1]} & {keys[b][1]}", f("tp"), f("tre"), f("te"))
    for kN in (3, 5, 7, 10, 15, 20):
        for need in (0.5, 0.7):
            f = lambda k: (lambda s_, n_: np.where(np.abs(s_) >= need * np.maximum(n_, 1), np.sign(s_), 0))(sum(TOPP[x][k] for x in keys[:kN]), sum((TOPP[x][k] != 0).astype(float) for x in keys[:kN]))
            add("F6", f"상위{kN} 투표 합의≥{int(need*100)}%", f("tp"), f("tre"), f("te"))
    P(f"F4·F6 ({time.time()-t_:.0f}s)")
    # F5 ML
    from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    fams = {}
    for c in FE: fams.setdefault(Q.fam(c) if not c.startswith("EV|") else "EV", []).append(c)
    single_rank = O[O.fam == "F1"].assign(f=lambda x: x.m.str.split("@").str[0]).groupby("f").score.max().sort_values(ascending=False)
    sets = {"전체": FE, "상위30": single_rank.index[:30].tolist(), "상위10": single_rank.index[:10].tolist()}
    sets.update({f"가족:{k}": v for k, v in fams.items() if len(v) >= 3})
    y = (TP.side.values == 1).astype(int)
    for sn, cols in sets.items():
        med_ = TP[cols].median()
        Xtp, Xtre, Xte = (G[cols].fillna(med_).values for G in (TP, TRE, TE))
        models = {"HGB": lambda: HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200, l2_regularization=1.0, random_state=0)}
        if sn in ("전체", "상위30", "상위10"):
            models["LR"] = lambda: LogisticRegression(C=0.1, max_iter=2000)
            models["ET"] = lambda: ExtraTreesClassifier(n_estimators=300, min_samples_leaf=100, max_features=0.3, n_jobs=4, random_state=0)
            if sn == "상위10": models["kNN"] = lambda: KNeighborsClassifier(n_neighbors=200)
        for mn, mk in models.items():
            mu, sd = Xtp.mean(0), Xtp.std(0) + 1e-9
            sc = (lambda X: (X - mu) / sd) if mn in ("LR", "kNN") else (lambda X: X)
            m = mk().fit(sc(Xtp), y); pt, pr_, pe = (m.predict_proba(sc(X))[:, 1] for X in (Xtp, Xtre, Xte))
            for cf in (0.0, 0.5):
                thr = np.quantile(np.abs(pt - 0.5), cf) if cf > 0 else -1
                g = lambda p_: np.where(np.abs(p_ - 0.5) > thr, np.where(p_ > 0.5, 1, -1), 0)
                add("F5", f"{mn}[{sn}] 확신{'상위50%' if cf else '전체'}", g(pt), g(pr_), g(pe))
    P(f"F5 ({time.time()-t_:.0f}s)")
    # F3 지연 진입 확인
    DL = {}
    for x in (0.0025, 0.005, 0.0075):
        parts = []
        for nmE, G in (("tr30", TR30), ("te", TE), ("tre", TRE)):
            for s in SYMS:
                g = G[G.sym == s]; k = K[s]; ts = pd.DatetimeIndex(k.ts)
                cbc = CB[s].reindex(ts).ffill().values; o = [z for z in SYMS if z != s][0]
                oc = pd.Series(K[o].close.values, index=pd.DatetimeIndex(K[o].ts)).reindex(ts).ffill().values
                a = AG[s].reindex(ts); lb = a.lbq.values.astype(float); ls = a.lsq.values.astype(float)
                out = delayed(g.t.values.astype(np.int64), k.open.values, k.high.values, k.low.values, k.close.values, k.volume.values, k.taker_buy_volume.values,
                              cbc, oc, lb, ls, x, 120, 0.015)
                F = pd.DataFrame(out, columns=["j", "d", "tbr", "vacc", "span", "cbco", "othco", "lti", "hold", "y", "rf", "rv", "pre"])
                F["sym"] = s; F["set"] = nmE; F["ts"] = g.ts.values; F["yr"] = g.yr.values; parts.append(F)
        DL[x] = pd.concat(parts).dropna(subset=["j"])
    for x, F in DL.items():
        tr = F[F.set == "tr30"]; te = F[F.set == "te"]; tre = F[F.set == "tre"]
        for c in ("tbr", "vacc", "span", "cbco", "othco", "lti", "hold", None):
            for q in (0.2, 0.33):
                if c is None:
                    rules = {"항상 추종": lambda G: np.ones(len(G)), "항상 역행": lambda G: -np.ones(len(G))}
                else:
                    xx = tr[c].dropna()
                    if len(xx) < 100: continue
                    lo, hi = xx.quantile([q, 1 - q]); fh = tr[tr[c] >= hi].y; fl = tr[tr[c] <= lo].y
                    ah = 1 if (fh == 1).mean() >= (fh == -1).mean() else -1; al = 1 if (fl == 1).mean() >= (fl == -1).mean() else -1
                    rules = {f"{c}@{int(q*100)}": (lambda G, lo=lo, hi=hi, ah=ah, al=al, c=c: np.where(G[c] >= hi, ah, np.where(G[c] <= lo, al, 0)) * G[c].notna())}
                for rn, fn in rules.items():
                    a_ = fn(tr); b_ = fn(tre); e_ = fn(te)
                    # 여기서 '적중' = 행동(추종 1/역행 −1)이 결과 y와 일치
                    ha = zhit(a_, tr.y.values); hb = zhit(b_, tre.y.values); he = zhit(e_, te.y.values)
                    ret = np.where(e_ == 1, te.rf.values, np.where(e_ == -1, te.rv.values, np.nan))
                    rows.append(dict(fam="F3", m=f"x{x*100:.2f}% {rn}", tp_hit=ha[0], tp_z=ha[1], tr_hit=hb[0], tr_z=hb[1], tr_n=hb[2], te_hit=he[0], te_z=he[1], te_n=he[2],
                                     te_bp=np.nanmean(ret) * 1e4 if np.isfinite(ret).any() else np.nan))
                if c is None: break
    P(f"F3 ({time.time()-t_:.0f}s)")
    O = pd.DataFrame(rows); O["score"] = O.tp_z.fillna(0) + O.tr_z.fillna(0)
    pd.set_option("display.width", 280); pd.set_option("display.max_rows", 400); pd.set_option("display.max_colwidth", 60)
    P(f"\n===== 방법 총 {len(O)}개 — 가족별")
    P(O.groupby("fam").agg(n=("m", "size"), te_hit_mean=("te_hit", "mean"), te_hit_max=("te_hit", "max"), te_z_mean=("te_z", "mean"), z_gt2=("te_z", lambda z: (z > 1.96).sum()),
                         z_lt2=("te_z", lambda z: (z < -1.96).sum())).round(3).to_string())
    zz = O.te_z.dropna(); P(f"  평가 z > 1.96: {(zz>1.96).sum()} / {len(zz)} (정규 기대 {0.025*len(zz):.0f}), z < −1.96: {(zz<-1.96).sum()}")
    from scipy.stats import norm
    pv_ = 1 - norm.cdf(zz.values); srt = np.sort(pv_); bh = srt <= 0.05 * np.arange(1, len(srt) + 1) / len(srt)
    P(f"  BH 5% 통과: {bh.nonzero()[0].max() + 1 if bh.any() else 0}개")
    P("\n===== 학습 점수(학습 풀 z + 학습 사건 z) 상위 30 → 평가 (선택 편향 없는 판정)")
    P(O.sort_values("score", ascending=False).head(30).round(3).to_string())
    P("\n===== 평가 적중 상위 20 (참고 — 사후 선택, 믿지 말 것)")
    P(O[O.te_n >= 50].sort_values("te_hit", ascending=False).head(20).round(3).to_string())
    O.to_parquet(R / "data" / "r99_methods.parquet", index=False)
    # 매매: 학습 점수 상위 정적 방법 5개(F1·F2·F4·F5·F6)
    P("\n===== 매매 (학습 점수 상위 정적 5개, 평가 2024~26, 다음 분 시가 진입)")
    days = (TE.ts.max() - TE.ts.min()).days
    for r in O[O.fam != "F3"].sort_values("score", ascending=False).head(5).itertuples():
        pe = np.asarray(PRED[(r.fam, r.m)], float)
        for ex, (tp, sl) in (("±2%", (0.02, 0.02)), ("4h", (0, 0))):
            parts = []
            for s in SYMS:
                m = (TE.sym == s).values & (pe != 0); g = TE[m]; k = K[s]
                rr = Q.trade(g.t.values.astype(np.int64), pe[m].astype(np.int64), k.open.values, k.high.values, k.low.values, k.close.values, tp, sl, 240)
                parts.append(pd.DataFrame({"ts": Q.utc(g.ts.values), "r": rr}))
            T = pd.concat(parts).dropna()
            d = T.groupby(T.ts.dt.floor("D")).r.sum().reindex(pd.date_range("2024-01-01", "2026-09-19", freq="D", tz="UTC")).fillna(0)
            tiers = []
            for L in (1, 3, 10):
                eq = (1 + (d * L).clip(lower=-0.99)).cumprod(); yrs = len(d) / 365.25
                tiers.append(f"×{L} 월 {(eq.iloc[-1]**(1/12/yrs)-1)*100:+.2f}% MDD {(eq/eq.cummax()-1).min()*100:.0f}%")
            P(f"  [{r.fam}] {r.m[:50]:50s} {ex:4s}: {len(T)}건(하루 {len(T)/days:.2f}) 거래당 {T.r.mean()*1e4:+.1f}bp 승률 {(T.r>0).mean():.3f} | " + " | ".join(tiers))
    (R / "logs" / "r99.summary").write_text("done\n")
