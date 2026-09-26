"""R68 — 1시간봉 5개(5시간)·4시간봉 6개(24시간) 패턴 + 지표 그룹 넣고 빼기 → 다음 방향 예측, 분기 무작위 분할 검증.

표본: 격자 봉 t의 종가 시점(모든 봉), BTC·ETH 합산. 특징은 t 종가까지의 정보만.
목표: t 종가 → t+h 종가 로그수익의 부호. h = 1봉, 그리고 창 길이(1시간 격자 5봉, 4시간 격자 6봉).
지표 그룹 (창 W = 5 또는 6)
  G1 캔들: 창 안 봉별 수익/ATR·몸통비·윗꼬리·아랫꼬리·범위/ATR, 창 누적수익, 창 범위 안 종가 위치, 양봉 수
  G2 추세: 20·60·120봉 수익/ATR, EMA 9·21·55·200 거리/ATR, MACD 히스토그램/ATR, ADX14
  G3 오실레이터: RSI14, 스토캐스틱 %K14, CCI20, MFI14
  G4 변동성: ATR 500봉 백분위, 볼린저(20) 폭·%b, 실현변동성 W/50 비율
  G5 거래량·체결: 창 봉별 거래량(로그, 100봉 중앙값 대비)·테이커 매수 비율, OBV 20봉 기울기, 24시간 VWAP 거리/ATR
  G6 파생: 펀딩(최근·변화), 프리미엄 지수(봉 평균·W봉 변화), OI 변화(1봉·W봉), 상위 트레이더 롱숏비, 전체 롱숏비, 테이커 롱숏 체결비 (2021-12~, 없으면 결측)
  G7 교차: 상대 코인 1봉·W봉 수익/ATR, 코인베이스 프리미엄(수준·W봉 변화), ETH/BTC 비율 W봉 변화
  G8 시간: 시간대, 요일, 종목
  G9 차트 선 거리: 전일 고·저·종가, 클래식 피벗 P, 일간 VWAP, 스윙 고·저, 20일 고·저까지 거리/ATR (R66 선 생성기)
구성 27개: 그룹 단독 9, 전체 1, 전체−그룹 9, 캔들+그룹 8. 전체 구성은 라벨 셔플 위약 추가.
검증: 27개 분기를 학습 16 / 테스트 11로 무작위 8회. 분기 마지막 2일 표본 제외. HistGB 분류기.
평가(테스트): AUC, 예측 상위 10%(학습 예측 문턱) 롱·하위 10% 숏, 비용 왕복 14bp(테이커 5+2bp ×2) 후 거래당 순수익·하루 건수·적중률.
사용: python research/r68_bar_patterns.py <1h|4h>"""
import sys, warnings, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store, costs
from r66_lines_all import grid, lines_period, lines_swing
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
COST = 2 * (costs.TAKER + 2e-4)
NSPLIT, NTR = 8, 16


def load_1m(sym):
    k = store.load("klines_1m", sym).set_index("ts")
    return pd.DataFrame({"o": k.open, "h": k.high, "l": k.low, "c": k.close, "v": k.quote_volume, "tb": k.taker_buy_quote_volume})


def asof(src, idx, delay="0min"):
    s = src.copy(); s.index = s.index + pd.Timedelta(delay)
    return s[~s.index.duplicated()].sort_index().reindex(s.index.union(idx)).ffill().reindex(idx)


def build(sym, tf, W, other_close):
    b = load_1m(sym)
    g = b.resample(tf).agg({"o": "first", "h": "max", "l": "min", "c": "last", "v": "sum", "tb": "sum"}).dropna()
    gg = grid(b[["o", "h", "l", "c", "v"]], tf)
    end = g.index + pd.Timedelta(tf)                              # 봉 종가 시각
    lc = np.log(g.c); tr = pd.concat([g.h - g.l, (g.h - g.c.shift()).abs(), (g.l - g.c.shift()).abs()], axis=1).max(axis=1)
    atrp = tr.rolling(14).mean(); A = np.log(1 + atrp / g.c)      # 로그 ATR (종가 시점)
    r1 = lc.diff(); rng_ = (g.h - g.l).replace(0, np.nan)
    F = {}
    # G1 캔들
    for k in range(W):
        F[f"G1_r{k}"] = r1.shift(k) / A; F[f"G1_body{k}"] = ((g.c - g.o) / rng_).shift(k)
        F[f"G1_uw{k}"] = ((g.h - np.maximum(g.o, g.c)) / rng_).shift(k); F[f"G1_lw{k}"] = ((np.minimum(g.o, g.c) - g.l) / rng_).shift(k)
        F[f"G1_hl{k}"] = (rng_ / atrp).shift(k)
    hiW = g.h.rolling(W).max(); loW = g.l.rolling(W).min()
    F["G1_cum"] = (lc - lc.shift(W)) / A; F["G1_pos"] = (g.c - loW) / (hiW - loW).replace(0, np.nan)
    F["G1_nup"] = (r1 > 0).astype(float).rolling(W).sum()
    # G2 추세
    for n in (20, 60, 120): F[f"G2_ret{n}"] = (lc - lc.shift(n)) / (A * np.sqrt(n))
    for n in (9, 21, 55, 200): F[f"G2_ema{n}"] = np.log(g.c / g.c.ewm(span=n, adjust=False).mean()) / A
    macd = g.c.ewm(span=12, adjust=False).mean() - g.c.ewm(span=26, adjust=False).mean()
    F["G2_macdh"] = (macd - macd.ewm(span=9, adjust=False).mean()) / atrp
    up = g.h.diff(); dn = -g.l.diff(); pdm = np.where((up > dn) & (up > 0), up, 0.0); ndm = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr14 = tr.ewm(alpha=1 / 14, adjust=False).mean()
    pdi = 100 * pd.Series(pdm, index=g.index).ewm(alpha=1 / 14, adjust=False).mean() / atr14
    ndi = 100 * pd.Series(ndm, index=g.index).ewm(alpha=1 / 14, adjust=False).mean() / atr14
    F["G2_adx"] = (100 * (pdi - ndi).abs() / (pdi + ndi)).ewm(alpha=1 / 14, adjust=False).mean(); F["G2_di"] = pdi - ndi
    # G3 오실레이터
    d = g.c.diff(); au = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean(); ad = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    F["G3_rsi"] = 100 - 100 / (1 + au / ad)
    F["G3_stoch"] = (g.c - g.l.rolling(14).min()) / (g.h.rolling(14).max() - g.l.rolling(14).min())
    tp = (g.h + g.l + g.c) / 3; F["G3_cci"] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True))
    mf = tp * g.v; pos = mf.where(tp > tp.shift(), 0).rolling(14).sum(); neg = mf.where(tp < tp.shift(), 0).rolling(14).sum()
    F["G3_mfi"] = 100 - 100 / (1 + pos / neg)
    # G4 변동성
    F["G4_atrp"] = A.rolling(500, min_periods=100).rank(pct=True)
    sd = g.c.rolling(20).std(); ma = g.c.rolling(20).mean()
    F["G4_bbw"] = 4 * sd / ma; F["G4_pctb"] = (g.c - (ma - 2 * sd)) / (4 * sd)
    F["G4_rvr"] = r1.rolling(W).std() / r1.rolling(50).std()
    # G5 거래량·체결
    vm = g.v.rolling(100, min_periods=20).median()
    for k in range(W):
        F[f"G5_v{k}"] = np.log(g.v / vm).shift(k); F[f"G5_tb{k}"] = (g.tb / g.v - 0.5).shift(k)
    obv = (np.sign(r1) * g.v).cumsum(); F["G5_obv"] = (obv - obv.shift(20)) / g.v.rolling(20).sum()
    nb = {"1h": 24, "4h": 6}[tf]
    vw = (tp * g.v).rolling(nb).sum() / g.v.rolling(nb).sum(); F["G5_vwap"] = np.log(g.c / vw) / A
    # G6 파생
    fu = store.load("funding", sym).set_index("ts").rate
    fa = asof(fu, end); F["G6_fund"] = fa.values; F["G6_dfund"] = (fa - asof(fu.shift(1), end)).values
    pr = store.load("premium_1m", sym).set_index("ts").close.resample(tf).mean()
    F["G6_prem"] = pr.reindex(g.index); F["G6_dprem"] = F["G6_prem"] - F["G6_prem"].shift(W)
    m5 = store.load("metrics_5m", sym).set_index("ts")
    oi = asof(np.log(m5.sum_open_interest_value), end, "5min").values; oi = pd.Series(oi, index=g.index)
    F["G6_oi1"] = oi.diff(); F["G6_oiW"] = oi.diff(W)
    for c_, nm in (("sum_toptrader_long_short_ratio", "G6_top"), ("count_long_short_ratio", "G6_ls"), ("sum_taker_long_short_vol_ratio", "G6_tk")):
        F[nm] = np.log(asof(m5[c_], end, "5min").values)
    # G7 교차
    oc = np.log(other_close.reindex(g.index))
    F["G7_o1"] = oc.diff() / A; F["G7_oW"] = oc.diff(W) / A
    cbs = "BTCUSD" if sym == "BTCUSDT" else "ETHUSD"
    cb = store.load("coinbase_1m", cbs).set_index("ts").close
    cbp = np.log(asof(cb, end, "1min").values / g.c.values); F["G7_cbp"] = cbp; F["G7_dcbp"] = pd.Series(cbp, index=g.index).diff(W)
    F["G7_eb"] = (lc - oc).diff(W) * (1 if sym == "ETHUSDT" else -1)
    # G8 시간
    F["G8_hour"] = end.hour; F["G8_wday"] = end.weekday; F["G8_sym"] = int(sym == "ETHUSDT")
    # G9 차트 선 거리 (다음 봉 시작 시점 값 = t 종가에 알려진 값)
    S = {}; S.update(lines_period(b, gg, sym, None)); S.update(lines_swing(gg))
    for nm in ("PDH", "PDL", "PDC", "CL_P", "VW", "SW_H", "SW_L", "D20H", "D20L"):
        v = pd.Series(np.asarray(S[nm], float), index=gg.index).reindex(g.index).shift(-1)
        F[f"G9_{nm}"] = np.log(g.c / v) / A
    X = pd.DataFrame(F, index=g.index)
    fw = {h: lc.shift(-h) - lc for h in (1, W)}
    X["f1"] = fw[1]; X["fW"] = fw[W]; X["t"] = end
    return X


def run_cfg(D, feats, ycol, splits, shuffle=False):
    out = []
    for trq in splits:
        tr = D.q.isin(trq).values; te = ~tr
        y = (D[ycol].values > 0).astype(int)
        yt = y[tr].copy()
        if shuffle: yt = np.random.default_rng(0).permutation(yt)
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
                                           l2_regularization=1.0, random_state=0).fit(D[feats].values[tr], yt)
        ptr = m.predict_proba(D[feats].values[tr])[:, 1]; hi, lo = np.quantile(ptr, 0.9), np.quantile(ptr, 0.1)
        p = m.predict_proba(D[feats].values[te])[:, 1]; fr = D[ycol].values[te]
        sg = np.where(p >= hi, 1, np.where(p <= lo, -1, 0)); k = sg != 0
        pnl = sg[k] * fr[k] - COST
        days = D.day[te].nunique()
        out.append((roc_auc_score(y[te], p), pnl.mean() * 1e4, (sg[k] * fr[k]).mean() * 1e4, (sg[k] * fr[k] > 0).mean(), k.sum() / days))
    return np.array(out)


if __name__ == "__main__":
    tf = sys.argv[1]; W = {"1h": 5, "4h": 6}[tf]; t_ = time.time()
    closes = {s: store.load("klines_1m", s).set_index("ts").close.resample(tf).last() for s in ("BTCUSDT", "ETHUSDT")}
    D = pd.concat([build(s, tf, W, closes["ETHUSDT" if s == "BTCUSDT" else "BTCUSDT"]) for s in ("BTCUSDT", "ETHUSDT")], ignore_index=True)
    D = D.replace([np.inf, -np.inf], np.nan).dropna(subset=["f1", "fW", "G1_r0"])
    t = D.t; D["q"] = (t.dt.year * 4 + (t.dt.month - 1) // 3).astype(int); D["day"] = t.dt.floor("D")
    qend = t.dt.to_period("Q").dt.end_time.dt.tz_localize("UTC"); D = D[(qend - t) > pd.Timedelta(days=2)].reset_index(drop=True)
    G = {g_: [c for c in D.columns if c.startswith(g_ + "_")] for g_ in ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9")}
    ALL = sum(G.values(), [])
    NAME = {"G1": "캔들", "G2": "추세", "G3": "오실레이터", "G4": "변동성", "G5": "거래량·체결", "G6": "파생", "G7": "교차", "G8": "시간", "G9": "차트선거리"}
    Q = np.sort(D.q.unique())
    splits = [set(np.random.default_rng(s).choice(Q, NTR, replace=False)) for s in range(NSPLIT)]
    P(f"{tf}: 표본 {len(D):,}, 특징 {len(ALL)} ({' '.join(f'{k}:{len(v)}' for k, v in G.items())}), 분기 {len(Q)}, 분할 {NSPLIT} ({time.time()-t_:.0f}s)")
    CFG = [(f"{NAME[k]} 단독", v) for k, v in G.items()] + [("전체", ALL)] + \
          [(f"전체−{NAME[k]}", [c for c in ALL if c not in v]) for k, v in G.items()] + \
          [(f"캔들+{NAME[k]}", G["G1"] + v) for k, v in G.items() if k != "G1"]
    rows = []
    for ycol, hname in (("f1", "다음 1봉"), ("fW", f"다음 {W}봉")):
        P(f"\n===== {tf} 목표: {hname} 방향 (기저 상승 비율 {(D[ycol] > 0).mean():.3f}) — 분할 {NSPLIT}회 평균 [최소~최대]")
        P(f"  {'구성':18s} {'AUC':>6s} | {'순수익bp':>18s} | {'총bp':>6s} {'적중':>6s} {'하루':>5s} | 흑자분할")
        for name, feats in CFG + [("전체 (라벨 셔플 위약)", ALL)]:
            r = run_cfg(D, feats, ycol, splits, shuffle=name.startswith("전체 (라벨"))
            P(f"  {name:18s} {r[:,0].mean():.3f} | {r[:,1].mean():+6.1f} [{r[:,1].min():+5.1f}~{r[:,1].max():+5.1f}] | {r[:,2].mean():+6.1f} {r[:,3].mean():.3f} {r[:,4].mean():5.1f} | {(r[:,1] > 0).sum()}/{NSPLIT}  ({time.time()-t_:.0f}s)")
            rows.append((tf, hname, name, *r.mean(0), r[:, 1].min(), r[:, 1].max(), (r[:, 1] > 0).sum()))
    O = pd.DataFrame(rows, columns=["tf", "목표", "구성", "auc", "순bp", "총bp", "적중", "하루", "순최소", "순최대", "흑자분할"])
    O.to_csv(R / "logs" / f"r68_{tf}.csv", index=False)
    best = O[~O.구성.str.contains("위약")].sort_values("순bp", ascending=False).head(3)
    (R / "logs" / f"r68_{tf}.summary").write_text(f"R68 {tf} 상위:\n" + best[["목표", "구성", "auc", "순bp", "흑자분할"]].round(3).to_string(index=False) + "\n")
