"""R79 — 롱·숏 다중 조각 수익 구조: 약 350개 조각 라이브러리 → 매년 과거만 보고 선택 → 공유 계좌.

조각(전부 비용 포함 거래 단위 수익, 조각·종목 안에서 한 번에 한 포지션)
  DIP   z0.31 딥 매수 롱(r32 워크포워드 예측 > 10bp, 2022~, 청산 지연 1.2bp) — 이미 검증된 핵심 조각
  BRK   R41 돌파(N분 최고/최저 + 거래대금 v배, 손절 ks·σd, 추가 ka, 추적 kt, 최대 U단위) 파라미터 × 롱/숏
  TH    R78 이론 셋업 73개의 원래 방향 + 뒤집은 방향(역추세 → 추종 등)
  TSM   일간 시계열 모멘텀: k일 수익 부호 롱(양수일 때만)/숏(음수일 때만), k = 3·7·14·30·60, 같은 부호 연속일은 한 거래
선택(평가 연도 Y = 2022~2026, 그해 이전에 청산된 거래만): n ≥ 50, 평균 > 0, t ≥ 2, 직전 3년 중 2년 이상 흑자 →
  t 순서로, 이미 고른 조각과 일간 손익 상관 < 0.5 인 것만, 최대 K개(5·10·20).
계좌: 선택 조각 + (옵션) DIP에 자본 균등 배분. 거래 손익 = 배분 × 레버리지 × 수익(두 종목은 각 절반), 청산일에 반영, 일 복리.
구성: 롱+숏 / 롱만 / 숏만 × DIP 포함·미포함 × K. 레버리지 1·3·5.
사용: python research/r79_longshort_book.py [build|run]"""
import sys, warnings, time
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
CT, CM = 7e-4, 2e-4
SYMS = ("BTCUSDT", "ETHUSDT")


@numba.njit(cache=True)
def run_trades_x(ent, side, tg, st, hold_m, o, h, l, c, ct, cm):
    n = len(c); m = len(ent); used = np.zeros(m, np.bool_); ret = np.zeros(m); ex = np.zeros(m, np.int64); busy = -1
    for q in range(m):
        i = ent[q]
        if i <= busy or i >= n - 1 or not (tg[q] > 0) or not (st[q] > 0): continue
        e = o[i]; s = side[q]; up = e * (1 + s * tg[q] / 100); dn = e * (1 - s * st[q] / 100)
        j = i; r = np.nan
        while j < min(n, i + hold_m[q]):
            if s > 0:
                if l[j] <= dn: r = (min(dn, o[j]) if j > i else dn) / e - 1 - 2 * ct; break
                if h[j] >= up: r = up / e - 1 - ct - cm; break
            else:
                if h[j] >= dn: r = -((max(dn, o[j]) if j > i else dn) / e - 1) - 2 * ct; break
                if l[j] <= up: r = -(up / e - 1) - ct - cm; break
            j += 1
        if np.isnan(r):
            j = min(j, n - 1); r = s * (c[j] / e - 1) - 2 * ct
        used[q] = True; ret[q] = r; busy = j; ex[q] = j
    return used, ret, ex


def build():
    import r78_theory_setups as M
    out = []; t_ = time.time()
    K = {s: store.load("klines_1m", s).set_index("ts") for s in SYMS}
    # DIP
    W = pd.read_parquet(R / "data" / "r32_wf.parquet"); W = W[W.pred > 0.001]
    out.append(pd.DataFrame({"sl": "DIP|z0.31", "sym": W.sym.values, "side": 1, "te": pd.DatetimeIndex(W.ts),
                             "tx": pd.DatetimeIndex(W.ts) + pd.to_timedelta((W.x - W.f).values, unit="min"), "r": W.ret.values - 1.2e-4}))
    # BRK
    B = pd.read_parquet(R / "data" / "r41_events.parquet")
    for s in SYMS:
        ts = K[s].index; b = B[B.sym == s]
        out.append(pd.DataFrame({"sl": "BRK|" + b.N.astype(str) + "|" + b.v.astype(str) + "|" + b.ks.astype(str) + "|" + b.ka.astype(str) + "|" +
                                       b.kt.astype(str) + "|" + b.U.astype(str) + "|" + np.where(b.side > 0, "L", "S"),
                                 "sym": s, "side": b.side.values, "te": ts[b.ent.values], "tx": ts[np.minimum(b.ext.values, len(ts) - 1)], "r": b.ret.values}))
    P(f"  DIP·BRK 적재 ({time.time()-t_:.0f}s)")
    # TH (원래 + 뒤집기)
    cl = {s: K[s].close.resample("1h").last() for s in SYMS}
    for s in SYMS:
        F = M.hourly(s, cl["ETHUSDT" if s == "BTCUSDT" else "BTCUSDT"]); k = K[s]
        o, h, l, c = k.open.values, k.high.values, k.low.values, k.close.values
        for code, fam, desc, state, trig, side, tg, st, H in M.setups(F, s):
            sig = (state.fillna(False).astype(bool) & trig.fillna(False).astype(bool)).values
            idx = np.where(sig)[0]
            if len(idx) == 0: continue
            ent = k.index.searchsorted(F.index[idx] + pd.Timedelta("1h")); ok = ent < len(k) - 2; idx, ent = idx[ok], ent[ok]
            arr = lambda x: (x.values[idx] if isinstance(x, pd.Series) else np.full(len(idx), float(x))).astype(np.float64)
            for flip in (False, True):
                sd = np.full(len(idx), -side if flip else side, np.int64)
                u, r, ex = run_trades_x(ent.astype(np.int64), sd, arr(tg) if not flip else arr(st) * 0 + arr(tg), arr(st), np.full(len(idx), H * 60, np.int64),
                                        o, h, l, c, CT, CM)
                out.append(pd.DataFrame({"sl": f"TH|{code}{'F' if flip else ''}|{'L' if (side > 0) != flip else 'S'}", "sym": s, "side": sd[u],
                                         "te": k.index[ent[u]], "tx": k.index[ex[u]], "r": r[u]}))
        P(f"  TH {s} ({time.time()-t_:.0f}s)")
    # TSM
    for s in SYMS:
        d = K[s].close.resample("1D").last(); ld = np.log(d)
        for kk in (3, 7, 14, 30, 60):
            sgn = np.sign(ld - ld.shift(kk))
            for want, tag in ((1, "L"), (-1, "S")):
                on = (sgn == want).astype(int); grp = (on.diff() != 0).cumsum()
                for g, blk in on[on == 1].groupby(grp[on == 1]):
                    t0, t1 = blk.index[0], blk.index[-1] + pd.Timedelta("1D")
                    if t1 not in d.index: continue
                    r_ = want * (d.loc[t1] / d.loc[t0] - 1) - 2 * CT
                    out.append(pd.DataFrame({"sl": [f"TSM|{kk}|{tag}"], "sym": s, "side": want, "te": [t0 + pd.Timedelta("1D")], "tx": [t1 + pd.Timedelta("1D")], "r": [r_]}))
    A = pd.concat(out, ignore_index=True)
    A["te"] = pd.to_datetime(A.te, utc=True); A["tx"] = pd.to_datetime(A.tx, utc=True)
    A.to_parquet(R / "data" / "r79_trades.parquet", index=False)
    P(f"조각 {A.sl.nunique()}개, 거래 {len(A):,} ({time.time()-t_:.0f}s)")


def select(A, Y, K, sides, corr_max=0.5):
    tr = A[(A.tx < pd.Timestamp(f"{Y}-01-01", tz="UTC")) & A.sl.str[:3].ne("DIP")]
    tr = tr[tr.sl.str[-1].isin(sides)]
    g = tr.groupby("sl").r
    st = pd.DataFrame({"n": g.size(), "m": g.mean(), "s": g.std()}); st["t"] = st.m / (st.s / np.sqrt(st.n))
    st = st[(st.n >= 50) & (st.m > 0) & (st.t >= 2)]
    if st.empty: return []
    yrs = tr[tr.sl.isin(st.index)].assign(yr=lambda x: x.tx.dt.year)
    ym = yrs[yrs.yr >= Y - 3].groupby(["sl", "yr"]).r.mean().unstack()
    keep = ym.index[(ym > 0).sum(axis=1) >= 2]
    st = st.loc[st.index.intersection(keep)].sort_values("t", ascending=False)
    daily = tr[tr.sl.isin(st.index)].assign(d=lambda x: x.tx.dt.floor("D")).groupby(["d", "sl"]).r.sum().unstack().fillna(0)
    pick = []
    for sl in st.index:
        if len(pick) >= K: break
        if pick and daily[pick].corrwith(daily[sl]).abs().max() >= corr_max: continue
        pick.append(sl)
    return pick


def book(A, picks_by_year, use_dip, lev):
    rows = []
    for Y, picks in picks_by_year.items():
        sl = list(picks) + (["DIP|z0.31"] if use_dip else [])
        if not sl: continue
        T = A[A.sl.isin(sl) & (A.te >= pd.Timestamp(f"{Y}-01-01", tz="UTC")) & (A.te < pd.Timestamp(f"{Y+1}-01-01", tz="UTC"))].copy()
        T["pnl"] = T.r * lev / len(sl) * 0.5
        rows.append(T)
    if not rows: return None
    T = pd.concat(rows)
    daily = T.groupby(T.tx.dt.floor("D")).pnl.sum()
    idx = pd.date_range("2022-01-01", T.tx.max().floor("D"), freq="D", tz="UTC"); daily = daily.reindex(idx).fillna(0)
    eq = (1 + daily.clip(lower=-0.99)).cumprod()
    mon = eq.resample("ME").last().pct_change().dropna(); yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    yr = eq.resample("YE").last().pct_change(); yr.iloc[0] = eq.resample("YE").last().iloc[0] - 1
    ls = T.groupby(np.where(T.side > 0, "롱", "숏")).pnl.sum()
    return dict(mo=(eq.iloc[-1] ** (1 / 12 / yrs) - 1) * 100, mdd=(eq / eq.cummax() - 1).min() * 100, wm=(mon > 0).mean(),
                yr=" ".join(f"{d.year%100}:{v*100:+.0f}%" for d, v in yr.items()), n=len(T), ls=" ".join(f"{k} {v*100:+.0f}" for k, v in ls.items()))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "build": build(); sys.exit(0)
    A = pd.read_parquet(R / "data" / "r79_trades.parquet")
    P(f"조각 {A.sl.nunique()}개 (DIP 1, BRK {A[A.sl.str.startswith('BRK')].sl.nunique()}, TH {A[A.sl.str.startswith('TH')].sl.nunique()}, TSM {A[A.sl.str.startswith('TSM')].sl.nunique()}), 거래 {len(A):,}")
    P("\n===== 매년 선택된 조각 (롱+숏, K=10)")
    for Y in range(2022, 2027): P(f"  {Y}: " + ", ".join(select(A, Y, 10, ("L", "S"))))
    P("\n===== 계좌 (2022-01~2026-09, 선택은 매년 과거만) — 월 복리 / MDD / 흑자 달 / 연도별 / 롱·숏 손익 기여(자본 %)")
    for sides, sn in ((("L", "S"), "롱+숏"), (("L",), "롱만"), (("S",), "숏만")):
        for K_ in (5, 10, 20):
            picks = {Y: select(A, Y, K_, sides) for Y in range(2022, 2027)}
            for use_dip in (False, True):
                for lev in (1, 3, 5):
                    r = book(A, picks, use_dip, lev)
                    if r is None: continue
                    P(f"  {sn:4s} K={K_:2d} DIP {'포함' if use_dip else '없음'} {lev}배 | 월 {r['mo']:+6.2f}% MDD {r['mdd']:6.1f}% 흑자달 {r['wm']:.0%} | {r['yr']} | 거래 {r['n']} | {r['ls']}")
    P("\n===== 기준: DIP(z0.31) 단독")
    for lev in (1, 3, 5):
        r = book(A, {Y: [] for Y in range(2022, 2027)}, True, lev)
        P(f"  DIP 단독 {lev}배 | 월 {r['mo']:+6.2f}% MDD {r['mdd']:6.1f}% 흑자달 {r['wm']:.0%} | {r['yr']}")
    (R / "logs" / "r79.summary").write_text("R79 롱·숏 다중 조각 수익 구조 완료. 세부 logs/r79.log\n")
