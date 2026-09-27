"""R85 — z0.31 거래당 엣지 올리기(기존 격자): A 초 단위 특징, B 청산 변형.

A 초 단위 특징(agg1s, 2023~): 주문을 켜는 분 f 직전 5분(초 단위, 분 f−1 종료까지)
  s60_sell 매도 비율, s60_big 10만$+ 대량 체결 비중, s60_bigsell 대량 중 매도 비중, s60_mx 최대 단일 체결/거래대금,
  s10_ret·s30_ret·s60_ret 초 단위 수익/rv, s_decel 최근 20초 − 이전 40초 절반(하락 감속), s10_vshare 마지막 10초 거래대금 비중,
  s60_active 체결 있는 초 비율, s300_rv 1초 수익 표준편차(5분)/분 rv, s60_dd 60초 고점 대비 낙폭/rv, s10_sellacc 마지막 10초 매도/60초 평균
  비교(공정): 2023~ 데이터로만 학습 — 기존 50특징 vs 50 + 초 단위 13, 평가 2024~2026 연도별 확장창.
B 청산: 봉 종가(기준) / 1·2·3봉 연장 종가 / 봉 시가 회복 지정가(메이커) 미체결 시 봉 종가 / 지정가~시가 중간 회복 지정가 미체결 시 봉 종가.
  같은 선택(R83 예측 > 25bp), 비용 메이커 진입 + (지정가 청산 메이커 | 시장가 테이커 + 지연 1.2bp).
평가: 일간 연 샤프(V3·w1N4), R40 엔진 최대 성장(w 1~4, N = 4w) 월 복리·MDD, 2024~26.
사용: python research/r85_edge.py [feat|walk|exits]"""
import sys, warnings, time, glob
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
EPOCH = pd.Timestamp("2020-01-01", tz="UTC")
S1 = ["s60_sell", "s60_big", "s60_bigsell", "s60_mx", "s10_ret", "s30_ret", "s60_ret", "s_decel", "s10_vshare", "s60_active", "s300_rv", "s60_dd", "s10_sellacc"]


def load_events():
    E = pd.read_parquet(R / "data" / "r32_events.parquet"); E["ts"] = pd.DatetimeIndex(E.ts); E["yr"] = E.ts.dt.year
    E["isbtc"] = (E.sym == "BTCUSDT").astype(float); return E.replace([np.inf, -np.inf], np.nan)


def feat():
    """월 단위로 agg1s를 읽어(메모리 절약) 그 달 이벤트의 직전 5분 초 단위 특징 계산."""
    E = load_events(); E = E[E.yr >= 2023].copy(); t_ = time.time(); out = []
    E["mon"] = E.ts.dt.strftime("%Y-%m")
    for s in ("BTCUSDT", "ETHUSDT"):
        for mon, Es in E[E.sym == s].groupby("mon"):
            prev = (pd.Period(mon, "M") - 1).strftime("%Y-%m")
            fs = [f for mm in (prev, mon) for f in glob.glob(str(store.PQ / "agg1s" / s / f"{mm}*.parquet"))]
            if not fs: continue
            A = pd.concat([pd.read_parquet(f, columns=["ts", "c", "h", "bq", "sq", "mx", "lbq", "lsq"]) for f in fs]).drop_duplicates("ts").set_index("ts").sort_index()
            sec = ((A.index - EPOCH) // pd.Timedelta("1s")).values.astype(np.int64)
            c, h, bq, sq, mx, lbq, lsq = (A[k].values for k in ("c", "h", "bq", "sq", "mx", "lbq", "lsq"))
            rows = []
            for f in Es.f.values:
                end = int(f) * 60
                i0, i1 = np.searchsorted(sec, end - 300), np.searchsorted(sec, end)
                if i1 - i0 < 5: rows.append([np.nan] * len(S1)); continue
                ss = sec[i0:i1]; cc = c[i0:i1]; vv = bq[i0:i1] + sq[i0:i1]; sq_ = sq[i0:i1]; lb, ls = lbq[i0:i1], lsq[i0:i1]
                w60 = ss >= end - 60; w10 = ss >= end - 10; w30 = ss >= end - 30; w20 = ss >= end - 20
                v60 = vv[w60].sum() + 1e-9; last = cc[-1]
                lr = np.diff(np.log(cc)); rv1 = lr.std() if len(lr) > 3 else np.nan
                pref = lambda m: cc[np.argmax(m)] if m.any() else last
                r = lambda m: np.log(last / pref(m))
                rows.append([sq_[w60].sum() / v60, (lb[w60].sum() + ls[w60].sum()) / v60, ls[w60].sum() / (lb[w60].sum() + ls[w60].sum() + 1e-9),
                             (mx[i0:i1][w60].max() / v60) if w60.any() else np.nan, r(w10), r(w30), r(w60),
                             r(w20) - np.log(pref(w20) / pref(w60)) / 2, vv[w10].sum() / v60, w60.sum() / 60.0, rv1,
                             np.log(last / h[i0:i1][w60].max()) if w60.any() else 0.0, (sq_[w10].sum() / 10) / (sq_[w60].sum() / 60 + 1e-9)])
            out.append(pd.DataFrame(rows, columns=S1, index=Es.index))
            del A
        P(f"  {s} 완료 ({time.time()-t_:.0f}s)")
    X = pd.concat(out)
    for k in ("s10_ret", "s30_ret", "s60_ret", "s_decel", "s60_dd"): X[k] = X[k] / (X["s300_rv"] * np.sqrt(60) + 1e-9)
    X.to_parquet(R / "data" / "r85_s1.parquet"); P(f"저장 data/r85_s1.parquet {X.shape}")


def acct(S):
    import r40_regime_leverage as RL, r80_max_return as R80
    RL.MMR = 0.0065
    S = S.sort_values(["f", "sym"]).reset_index(drop=True)
    mult = np.minimum(np.clip(S.vreg.values ** 1.5, 0.3, 4.0) * np.clip(S.pred.values / 0.001, 0.5, 3.0), 6.0)
    b = RL.sim(S, 1, 4, mult, X=0.03, H=720); sh = np.nan
    if b is not None: d = R80.daily_from_eq(b[0]); sh = d.mean() / d.std() * np.sqrt(365)
    best = None
    for w in (1, 1.5, 2, 2.5, 3, 3.5, 4):
        res = RL.sim(S, w, 4 * w, mult, X=0.03, H=720)
        if res is None: break
        d = R80.daily_from_eq(res[0]); m_ = R80.metrics(d); mt = R80.metrics(d, "2024-01-01", "2027-01-01")
        if best is None or m_["mo"] > best[1]["mo"]: best = (w, m_, mt)
    s = f"샤프(w1N4) {sh:.2f}"
    if best: s += f" | 최대 w{best[0]}N{4*best[0]:g}: 월 {best[1]['mo']:+.1f}% (중앙 {best[1]['med']:+.1f}%) MDD {best[1]['mdd']:.0f}% | 24~26 월 {best[2]['mo']:+.1f}% MDD {best[2]['mdd']:.0f}%"
    return s


def add_vreg(W):
    for s in ("BTCUSDT", "ETHUSDT"):
        k = store.load("klines_1m", s).set_index("ts").close.resample("D").last(); r = np.log(k).diff()
        v30 = r.rolling(30).std(); reg = (v30 / v30.rolling(365, min_periods=120).median()).shift(1)
        m = W.sym == s; W.loc[m, "vreg"] = reg.reindex(pd.DatetimeIndex(W.loc[m, "ts"]).normalize()).values
    W["vreg"] = W.vreg.fillna(1.0); return W


def walk():
    from sklearn.ensemble import HistGradientBoostingRegressor
    E = load_events(); X = pd.read_parquet(R / "data" / "r85_s1.parquet"); E = E.join(X, how="inner")
    LIVE = [c for c in E.columns if c not in {"sym", "f", "x", "ret", "mae", "ts", "yr", "bslope", "bimb"} | set(S1)]
    res = {}
    for tag, feats in (("기존 50특징(2023~ 학습)", LIVE), ("50 + 초 단위 13", LIVE + S1)):
        out = []
        for Y in (2024, 2025, 2026):
            tr = E[E.yr < Y]; te = E[E.yr == Y]
            fu = [c for c in feats if tr[c].notna().sum() > 1000]
            m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=400, l2_regularization=1.0,
                                              random_state=0).fit(tr[fu].values, np.clip(tr.ret.values, -0.02, 0.02))
            out.append(te.assign(pred=m.predict(te[fu].values)))
        res[tag] = pd.concat(out)
    P("\n===== A 초 단위 특징 (평가 2024~2026, 학습은 2023~ 데이터만)")
    for tag, W in res.items():
        W = add_vreg(W.copy()); W["ret"] = W.ret - 1.2e-4
        for th in (0.0015, 0.0025, 0.0035):
            S = W[W.pred > th]
            yr = S.groupby("yr").ret.mean() * 1e4
            P(f"  {tag:22s} 문턱 {th*1e4:.0f}bp: 거래 {len(S)/2.75:6.0f}/년 거래당 {S.ret.mean()*1e4:+6.2f}bp 승률 {(S.ret>0).mean():.3f} | " +
              " ".join(f"{y%100}:{v:+.1f}" for y, v in yr.items()) + " | " + acct(S))
    (R / "logs" / "r85a.summary").write_text("R85A 초 단위 특징 완료. 세부 logs/r85a.log\n")


@numba.njit(cache=True)
def exits(f, B, T, o, h, l, c, cm, ct):
    """반환 열: 0 기준 종가, 1·2·3 연장 1~3봉 종가, 4 시가 회복 지정가, 5 중간 회복 지정가 / 각 순수익과 MAE."""
    n = len(f); R_ = np.zeros((n, 6)); M_ = np.zeros((n, 6)); N = len(c)
    for q in range(n):
        a = f[q]; b = B[q]; s0 = (a // b) * b; e0 = s0 + b; A = o[s0]; t = T[q]
        for v in range(6):
            ext = v if v <= 3 else 0
            e = min(e0 + ext * b, N) ; tp = A if v == 4 else (t + 0.5 * (A - t) if v == 5 else -1.0)
            adv = l[a]; px = -1.0; mk = False
            for j in range(a + 1, e):
                if tp > 0 and h[j] >= tp * 1.0002:
                    px = tp; mk = True; break
                if l[j] < adv: adv = l[j]
            if px < 0: px = c[e - 1]
            R_[q, v] = px / t - 1 - cm - (cm if mk else ct + 1.2e-4)
            M_[q, v] = 1 - adv / t
    return R_, M_


def exit_eval():
    W = pd.read_parquet(R / "data" / "r83_wf_live.parquet"); W["ts"] = pd.DatetimeIndex(W.ts)
    W = W[W.pred > 0.0025].copy(); rows = []
    for s in ("BTCUSDT", "ETHUSDT"):
        k = store.load("klines_1m", s); o, h, l, c = (k[x].values for x in ("open", "high", "low", "close"))
        m = W.sym == s; Ws = W[m]
        s0 = (Ws.f.values // Ws.B.values.astype(int)) * Ws.B.values.astype(int)
        rv1 = pd.Series(np.log(c)).diff().rolling(300).std().shift(1).values
        T = o[s0] * (1 - Ws.k.values * rv1[s0] * np.sqrt(Ws.B.values))
        R_, M_ = exits(Ws.f.values.astype(np.int64), Ws.B.values.astype(np.int64), T, o, h, l, c, 2e-4, 5e-4)
        for v in range(6): W.loc[m, f"r{v}"] = R_[:, v]; W.loc[m, f"m{v}"] = M_[:, v]
    W = add_vreg(W)
    names = ["봉 종가(기준)", "1봉 연장", "2봉 연장", "3봉 연장", "시가 회복 지정가", "중간 회복 지정가"]
    P("\n===== B 청산 변형 (R83 예측 > 25bp, 2022~2026)")
    P(f"  검증: 기준 청산 재계산 거래당 {W.r0.mean()*1e4:+.2f}bp vs 저장값 {(W.ret-1.2e-4).mean()*1e4:+.2f}bp")
    for v, nm in enumerate(names):
        S = W.copy(); S["ret"] = S[f"r{v}"]; S["mae"] = S[f"m{v}"]
        if 1 <= v <= 3: S["x"] = (S.f // S.B.astype(int)) * S.B.astype(int) + S.B.astype(int) * (1 + v) - 1   # 자본 점유 연장
        yr = S.groupby(S.ts.dt.year).ret.mean() * 1e4
        P(f"  {nm:12s}: 거래당 {S.ret.mean()*1e4:+6.2f}bp 승률 {(S.ret>0).mean():.3f} MAE95 {np.percentile(S.mae,95)*100:.2f}% | " +
          " ".join(f"{y%100}:{v_:+.0f}" for y, v_ in yr.items()) + " | " + acct(S))
    (R / "logs" / "r85b.summary").write_text("R85B 청산 변형 완료. 세부 logs/r85b.log\n")


if __name__ == "__main__":
    {"feat": feat, "walk": walk, "exits": exit_eval}[sys.argv[1]]()
