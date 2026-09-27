"""R88 — R87 생존 계열 발전: 변형 격자 → 2020~23으로만 선택 → 2024~26 검증 → z0.31 덧얹기.

계열·변형(전부 롱은 같은 보유 무작위 롱 대비 초과수익으로 판정, 양방향은 원수익)
  V 변동성 압축 양방향 스톱: 압축 문턱 q ∈ {0.4, 0.5, 0.6} × 스톱 거리 x ∈ {0.75, 1.0, 1.25}σ24 × 목표 {2, 3}배 × 보유 48h (18)
  O 옵션 순매수: 창 {12, 24, 48}h × z 문턱 {1.5, 2, 2.5} × 보유 {24, 48}h, 롱 (18)
  S 스테이블코인 증가: 창 {5, 7, 10, 14}일 × 상위 {15, 20, 30}% × 보유 {5, 7}일, 롱 (24)
  K 역프(김프 z < 문턱): 문턱 {−1.5, −2, −2.5} × 보유 {48, 72, 120}h, 롱 (9)
선택(사전 고정, 2020~23만): n ≥ 30, 초과 평균 > 0, 초과 t ≥ 2, 2020~23 중 3년 이상 흑자. 계열마다 이웃 고원 확인을 위해 상위 3개까지.
결합: z0.31(w1N4) + k × 바구니(선택 전체 위험 균등, 2020~23 비중, z0.31 변동성 맞춤), k ∈ {0.25, 0.5, 0.75, 1}, 평가 2024~26.
사용: python research/r88_evolve.py"""
import sys, warnings, itertools
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
import r87_new80 as M
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
IDX = pd.date_range("2020-01-01", "2026-09-25", freq="D", tz="UTC")


def main():
    D = M.Data()
    import glob
    from freesia import store
    base = {}
    for s in M.SYMS:
        c = D.H[s].close
        for h in (24, 48, 72, 120, 168, 240, 336):
            f = np.log(c.shift(-h) / c); base[(s, h)] = f.groupby(f.index.year).mean()
    fams = {}
    # V
    for q, x, tm in itertools.product((0.4, 0.5, 0.6), (0.75, 1.0, 1.25), (2, 3)):
        def f(s, q=q, x=x, tm=tm):
            H = D.H[s].copy(); rv2 = H.rv ** 2 * 60; RVd = rv2.rolling(24).mean(); RVm = rv2.rolling(720).mean(); qq = RVd / RVm; sig = np.sqrt(RVm * 24)
            m = (qq < q) & (qq.shift(1) >= q); T = H[m]; k = D.K[s]; st_ = k.index.searchsorted(pd.DatetimeIndex(T.t)); ok = st_ < len(k) - 2
            st_ = st_[ok]; xx = (x * sig[m].values)[ok]
            ent, sd = M.straddle(st_.astype(np.int64), k.close.values[np.maximum(st_ - 1, 0)], xx, np.full(len(st_), 2880, np.int64), k.high.values, k.low.values)
            g = ent >= 0
            return pd.DataFrame({"i": ent[g], "side": sd[g], "tg": xx[g] * 100 * tm, "st": xx[g] * 100, "hold": np.full(g.sum(), 48 * 60)})
        fams[f"V q{q} x{x} 목표{tm}"] = ("V", f, 48, False)
    # O (옵션 순매수 — R87 계산 재사용 위해 간단히 재구성)
    OFN = {}
    for s in M.SYMS:
        cur = "BTC" if s == "BTCUSDT" else "ETH"; agg = []
        for fp in sorted(glob.glob(str(store.PQ / "deribit_opt" / cur / "*.parquet"))):
            x_ = pd.read_parquet(fp, columns=["ts", "cp", "price", "index_price", "amount", "side"])
            prem = x_.amount * x_.price * x_.index_price; sg = np.where(x_.side == 1, 1, -1) * np.where(x_.cp == 1, 1, -1)
            agg.append(pd.Series(prem * sg).groupby(x_.ts.dt.floor("h").values).sum())
        net = pd.concat(agg).groupby(level=0).sum(); net.index = pd.DatetimeIndex(net.index).tz_localize("UTC") if pd.DatetimeIndex(net.index).tz is None else net.index
        net.index = net.index + pd.Timedelta("1h"); OFN[s] = net.reindex(D.H[s].t).fillna(0).values
    for wdw, zt, hold in itertools.product((12, 24, 48), (1.5, 2.0, 2.5), (24, 48)):
        def f(s, wdw=wdw, zt=zt, hold=hold):
            H = D.H[s]; n = pd.Series(OFN[s], index=H.index).rolling(wdw).sum()
            z = (n - n.rolling(720, min_periods=240).mean()) / n.rolling(720, min_periods=240).std()
            m = (z > zt) & ~(z.shift(1) > zt); T = H[m]
            return M.entries_at(D, s, T.t, 1, M.BIG, M.BIG, hold)
        fams[f"O 창{wdw}h z>{zt} 보유{hold}h"] = ("O", f, hold, True)
    # S
    sc = pd.read_parquet(M.EXT / "stablecoins.parquet").stable_total; sc.index = pd.DatetimeIndex(sc.index)
    sc.index = (sc.index.tz_localize("UTC") if sc.index.tz is None else sc.index) + pd.Timedelta(days=1)
    for look, top, hd in itertools.product((5, 7, 10, 14), (0.15, 0.2, 0.3), (5, 7)):
        def f(s, look=look, top=top, hd=hd):
            g = np.log(sc).diff(look); qq = g.rolling(365, min_periods=120).rank(pct=True)
            m = (qq >= 1 - top) & ~(qq.shift(1) >= 1 - top)
            return M.entries_at(D, s, qq.index[m.values], 1, M.BIG, M.BIG, hd * 24)
        fams[f"S 창{look}일 상위{int(top*100)}% 보유{hd}일"] = ("S", f, hd * 24, True)
    # K
    up = pd.read_parquet(M.EXT / "upbit_1h.parquet").set_index("ts"); fx = pd.read_parquet(M.EXT / "usdkrw.parquet").set_index("date").krw
    fx.index = fx.index + pd.Timedelta(days=1)
    for th, hold in itertools.product((-1.5, -2.0, -2.5), (48, 72, 120)):
        def f(s, th=th, hold=hold):
            H = D.H[s]; u = up["btc" if s == "BTCUSDT" else "eth"].reindex(H.index); f_ = fx.reindex(fx.index.union(H.index)).ffill().reindex(H.index)
            kp = u / (H.close * f_) - 1; z = (kp - kp.rolling(720, min_periods=240).mean()) / kp.rolling(720, min_periods=240).std()
            m = (z < th) & ~(z.shift(1) < th); T = H[m]
            return M.entries_at(D, s, T.t, 1, M.BIG, M.BIG, hold)
        fams[f"K z<{th} 보유{hold}h"] = ("K", f, hold, True)
    rows = []; daily = {}
    for name, (fam, fn, hold, is_long) in fams.items():
        parts = []
        for s in M.SYMS:
            E = fn(s)
            if E is None or len(E) == 0: continue
            T = M.run(D, s, E); T["t"] = pd.to_datetime(T.t, utc=True)
            hb = min(base_keys := (24, 48, 72, 120, 168, 240, 336), key=lambda x: abs(x - hold))
            T["ex"] = T.r - (base[(s, hb)].reindex(T.t.dt.year).values if is_long else 0)
            parts.append(T)
        if not parts: continue
        T = pd.concat(parts); tr = T[T.t < "2024-01-01"]; te = T[T.t >= "2024-01-01"]
        t_tr = tr.ex.mean() / (tr.ex.std() / np.sqrt(len(tr))) if len(tr) > 2 else np.nan
        yrs_pos = (tr.groupby(tr.t.dt.year).ex.mean() > 0).sum()
        rows.append((fam, name, len(tr), tr.ex.mean() * 1e4, t_tr, yrs_pos, len(te), te.ex.mean() * 1e4 if len(te) else np.nan))
        daily[name] = (T.groupby(T.t.dt.floor("D")).r.sum() * 0.5).reindex(IDX).fillna(0)
    O = pd.DataFrame(rows, columns=["계열", "변형", "n학습", "초과bp학습", "t학습", "흑자연도", "n평가", "초과bp평가"])
    O["선택"] = (O.n학습 >= 30) & (O.초과bp학습 > 0) & (O.t학습 >= 2) & (O.흑자연도 >= 3)
    sel = O[O.선택].sort_values("t학습", ascending=False).groupby("계열").head(3)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    P(f"===== 변형 {len(O)}개, 학습(2020~23) 기준 통과 {O.선택.sum()}개, 계열별 상위 3 선택 {len(sel)}개")
    P(O.sort_values(["계열", "t학습"], ascending=[True, False]).round(2).to_string(index=False))
    P("\n  선택된 변형의 평가 구간(2024~26) 초과수익: " + " | ".join(f"{r.변형}: {r.초과bp평가:+.0f}bp(n{r.n평가})" for r in sel.itertuples()))
    import r40_regime_leverage as RL, r80_max_return as R80
    RL.MMR = 0.0065; W = R80.dip_frame(0.0025)
    mult = np.minimum(np.clip(W.vreg.values ** 1.5, 0.3, 4.0) * np.clip(W.pred.values / 0.001, 0.5, 3.0), 6.0)
    dip = R80.daily_from_eq(RL.sim(W, 1, 4, mult, X=0.03, H=720)[0]).reindex(IDX).fillna(0)
    Mx = pd.DataFrame({n: daily[n] for n in sel.변형}); tr = Mx[:"2023-12-31"]
    w = 1 / tr.std(); w = w / w.sum(); bask = (Mx * w).sum(axis=1)
    bask = bask * dip["2022-01-01":"2023-12-31"].std() / bask["2022-01-01":"2023-12-31"].std()
    bt = bask["2024-01-01":]; dt = dip["2024-01-01":]
    P(f"\n===== z0.31 + k × 발전 바구니({len(sel)}개) — 평가 2024~26 표본 밖")
    P(f"  바구니 단독 샤프 {bt.mean()/bt.std()*np.sqrt(365):.2f}, z0.31 상관 {np.corrcoef(bt, dt)[0,1]:+.2f}")
    for k in (0, 0.25, 0.5, 0.75, 1.0):
        r = dt + k * bt; sh = r.mean() / r.std() * np.sqrt(365); lv = []
        for L in (1, 2, 3):
            eq = (1 + (r * L).clip(lower=-0.99)).cumprod(); yrs = (eq.index[-1] - eq.index[0]).days / 365.25
            lv.append(f"×{L} 월 {(eq.iloc[-1]**(1/12/yrs)-1)*100:+.1f}% MDD {(eq/eq.cummax()-1).min()*100:.0f}%")
        P(f"  k={k:.2f}: 샤프 {sh:.2f} | " + " | ".join(lv))
    (R / "logs" / "r88.summary").write_text(f"R88 변형 {len(O)}개, 선택 {len(sel)}개\n")


if __name__ == "__main__":
    main()
