"""R133 — 같은 목표·다른 접근 A: 예정 경제지표 발표(시작 시각이 미리 정해진 큰 움직임)의 첫 반응을 따라가면 이어지는가.

달력: data/ext/macro_calendar.parquet (scripts/get_macro_calendar.py — ALFRED 실제 발표일, 동부 시각 → UTC).
같은 분에 여러 발표가 겹치면 한 '순간'으로 묶고 우선순위가 가장 높은 이름을 붙인다(FOMC > CPI > NFP > PCE > PPI > RETAIL > GDP > JOLTS > CLAIMS > FOMC_PC).
대조: 이벤트가 없는 평일의 같은 동부 시각(08:30·10:00·14:00) — '그 시각이라서'와 '발표라서'를 가른다.
반응 창 k: 1분봉 k = 1·2·5·15분(2020~), 1초봉 k = 10·30초(agg1s, 2023~). 반응 r = 가격(발표 + k) / 가격(발표 직전 분 종가) − 1.
규칙: |r| ≥ 문턱(0·0.1·0.2·0.3%)이면 r 방향으로 진입(추종). 진입가 = 그 시점 가격 + 불리한 슬리피지
  (1분봉: 종가 ± 3bp, 1초봉: 그 1초봉의 고가(매수)·저가(매도) — 보수적). 진입 수수료 테이커 5bp.
청산: 진입가 대칭 ±X(0.5·1·1.5%) 먼저 닿는 쪽(목표 지정가 2bp, 손절 테이커 5+2bp), 최대 H = 720분 뒤 종가(테이커 7bp).
  1분봉 장벽 판정은 진입 다음 분부터(같은 분 양쪽 → 손실로 보수 처리).
평가: 이벤트 종류별·통합, 학습 2020~23 / 시험 2024~26, 적중률(대칭 장벽이라 무작위면 50%)·거래당 순 bp·이항 z. BTC·ETH 따로 + 함께.
사용: python research/r133_macro_events.py"""
import sys, glob, warnings, time
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
PRI = ["FOMC", "CPI", "NFP", "PCE", "PPI", "RETAIL", "GDP", "JOLTS", "CLAIMS", "FOMC_PC"]
TK, MK, SL1 = 5e-4, 2e-4, 3e-4
H = 720


def moments():
    C = pd.read_parquet(R / "data" / "ext" / "macro_calendar.parquet")
    C["r"] = C.ev.map({e: i for i, e in enumerate(PRI)})
    M = C.sort_values("r").groupby("ts").agg(ev=("ev", "first"), all=("ev", lambda s: "+".join(sorted(s)))).reset_index()
    M["ctrl"] = False
    # 대조: 이벤트 없는 평일 같은 동부 시각
    et = M.ts.dt.tz_convert("America/New_York")
    days = pd.date_range("2020-01-01", "2026-09-30", freq="B")
    have = set(et.dt.strftime("%Y-%m-%d %H:%M"))
    rows = []
    for hm, lab in (("08:30", "CTRL_0830"), ("10:00", "CTRL_1000"), ("14:00", "CTRL_1400")):
        for d in days:
            key = f"{d:%Y-%m-%d} {hm}"
            if key in have: continue
            rows.append(dict(ts=pd.Timestamp(key).tz_localize("America/New_York").tz_convert("UTC"), ev=lab, all=lab, ctrl=True))
    return pd.concat([M, pd.DataFrame(rows)], ignore_index=True).sort_values("ts").reset_index(drop=True)


def barrier(h, l, c, j0, px, sd, X):
    """j0 분부터 진입가 대칭 ±X. 반환 (순수익 비용 전 방향 수익, 청산 종류 0 시간·1 목표·2 손절)."""
    up, dn = px * (1 + X), px * (1 - X); n = len(c); j1 = min(n, j0 + H)
    hh, ll = h[j0:j1], l[j0:j1]
    hu = np.flatnonzero(hh >= up); hd = np.flatnonzero(ll <= dn)
    iu = hu[0] if len(hu) else 10**9; idn = hd[0] if len(hd) else 10**9
    if iu == idn == 10**9: return sd * (c[j1 - 1] / px - 1), 0
    if iu == idn: return -X, 2
    win = (iu < idn) == (sd > 0)
    return (X, 1) if win else (-X, 2)


def net(g, kind):
    return g - TK - np.where(kind == 1, MK, np.where(kind == 2, TK + 2e-4, TK + 2e-4))


if __name__ == "__main__":
    t_ = time.time()
    M = moments(); P(f"순간 {len(M)} (이벤트 {(~M.ctrl).sum()}, 대조 {M.ctrl.sum()})")
    out = []
    for sym in ("BTCUSDT", "ETHUSDT"):
        k = store.load("klines_1m", sym); ts = pd.DatetimeIndex(k.ts)
        o, h, l, c = k.open.values, k.high.values, k.low.values, k.close.values
        idx = ts.get_indexer(M.ts)                       # 발표 분(그 분 시가 = 발표 시각)
        # 1초봉: 2023~ 이벤트 주변만
        A = {}
        for f in sorted(glob.glob(str(R / "data" / "parquet" / "agg1s" / sym / "*.parquet"))):
            ym = Path(f).stem[:7]
            mm = M[M.ts.dt.strftime("%Y-%m") == ym]
            if not len(mm): continue
            d = pd.read_parquet(f, columns=["ts", "h", "l", "c"]); d["ts"] = pd.DatetimeIndex(d.ts)
            for t0 in mm.ts:
                A[t0] = d[(d.ts >= t0 - pd.Timedelta("2min")) & (d.ts < t0 + pd.Timedelta("2min"))].set_index("ts")
        for q, e in enumerate(idx):
            if e < 1 or e + H + 20 >= len(c): continue
            m = M.iloc[q]; ref = c[e - 1]; base = dict(sym=sym, ts=m.ts, ev=m.ev, all=m["all"], ctrl=m.ctrl, yr=m.ts.year,
                                                    mv15=max(h[e:e + 15].max() / ref - 1, 1 - l[e:e + 15].min() / ref))
            for kk in (1, 2, 5, 15):
                r = c[e + kk - 1] / ref - 1; sd = 1 if r > 0 else -1
                px = c[e + kk - 1] * (1 + sd * SL1)
                for X in (0.005, 0.01, 0.015):
                    g, kind = barrier(h, l, c, e + kk, px, sd, X)
                    out.append(dict(base, k=f"{kk}m", r=r, X=X, g=g, kind=kind))
            a = A.get(m.ts)
            if a is not None and len(a):
                for ks in (10, 30):
                    tt = m.ts + pd.Timedelta(seconds=ks)
                    b = a[a.index < tt]
                    if not len(b) or b.index[-1] < m.ts - pd.Timedelta("90s"): continue
                    r = b.c.iloc[-1] / ref - 1; sd = 1 if r > 0 else -1
                    cur = b[b.index >= tt - pd.Timedelta("1s")]
                    px = (cur.h.iloc[-1] if sd > 0 else cur.l.iloc[-1]) if len(cur) else b.c.iloc[-1] * (1 + sd * SL1)
                    rest = a[(a.index >= tt) & (a.index < m.ts + pd.Timedelta("60s"))]   # 발표 분의 남은 초
                    for X in (0.005, 0.01, 0.015):
                        up, dn = px * (1 + X), px * (1 - X); done = None
                        hu = np.flatnonzero(rest.h.values >= up); hd = np.flatnonzero(rest.l.values <= dn)
                        if len(hu) or len(hd):
                            iu = hu[0] if len(hu) else 10**9; idn = hd[0] if len(hd) else 10**9
                            done = (-X, 2) if iu == idn else ((X, 1) if (iu < idn) == (sd > 0) else (-X, 2))
                        g, kind = done if done else barrier(h, l, c, e + 1, px, sd, X)
                        out.append(dict(base, k=f"{ks}s", r=r, X=X, g=g, kind=kind))
        P(f"  {sym} 완료 ({time.time()-t_:.0f}s)")
    T = pd.DataFrame(out); T["net"] = net(T.g.values, T.kind.values); T["win"] = T.g > 0
    T.to_parquet(R / "data" / "r133_trades.parquet", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
    # 0) 크기 확인: 첫 15분 최대 이동(중앙값) — 이벤트가 정말 큰 움직임인가
    S0 = T[(T.k == "1m") & (T.X == 0.01)].groupby("ev").agg(n=("mv15", "size"), mv15_med=("mv15", "median"), mv15_p75=("mv15", lambda s: s.quantile(.75)))
    S0[["mv15_med", "mv15_p75"]] *= 100; P("\n===== 0) 발표 뒤 15분 최대 이동(%) — 종목 합산"); P(S0.round(3).to_string())

    def tab(g):
        n = len(g); w = g.win.mean() if n else np.nan
        return pd.Series(dict(n=n, hit=w, z=(w - 0.5) / np.sqrt(0.25 / n) if n else np.nan, net_bp=g.net.mean() * 1e4))
    P("\n===== 1) 추종 적중·순 bp: 이벤트 통합 vs 대조, k × X × 문턱, 학습 20~23 / 시험 24~26")
    rows = []
    for (k_, X), g in T.groupby(["k", "X"]):
        for thr in (0, 0.001, 0.002, 0.003):
            gg = g[g.r.abs() >= thr]
            for lab, sel in (("이벤트", ~gg.ctrl), ("대조", gg.ctrl)):
                for per, ps in (("학습", gg.yr <= 2023), ("시험", gg.yr >= 2024)):
                    s = tab(gg[sel & ps]); rows.append(dict(k=k_, X=X, thr=thr, 집단=lab, 기간=per, **s))
    O = pd.DataFrame(rows)
    W = O.pivot_table(index=["k", "X", "thr", "집단"], columns="기간", values=["n", "hit", "z", "net_bp"])
    P(W.round(3).to_string())
    P("\n===== 2) 이벤트 종류별 (k 5m·15m, X 1%, 문턱 0·0.2%) 학습/시험")
    rows = []
    for (ev, k_, X), g in T[~T.ctrl & T.k.isin(["30s", "5m", "15m"]) & (T.X == 0.01)].groupby(["ev", "k", "X"]):
        for thr in (0, 0.002):
            gg = g[g.r.abs() >= thr]
            a, b = tab(gg[gg.yr <= 2023]), tab(gg[gg.yr >= 2024])
            rows.append(dict(ev=ev, k=k_, thr=thr, n_tr=a.n, hit_tr=a.hit, net_tr=a.net_bp, n_te=b.n, hit_te=b.hit, z_te=b.z, net_te=b.net_bp))
    P(pd.DataFrame(rows).round(3).to_string(index=False))
    P(f"\n완료 ({time.time()-t_:.0f}s)")
