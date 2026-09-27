"""R86 — 다른 방향의 약한 신호를 여럿 모아 상관을 고려해 결합(월 100% = 결합 샤프 4+ 목표).

근거: 독립 신호 결합 샤프 ≈ √Σ S_i². z0.31(2.85) + 상관 없는 샤프 1.2 신호 6개 ≈ 4.1.
라이브러리(일간 손익, 거래 1건 = 명목 1, 종목별 0.5):
  R79 조각(돌파 192, 이론 원·반대 146, TSMOM 10, 하락국면 숏) + z0.31(R81 설정 문턱 25bp·V3, w1N4 일간 수익)
  + 신규 ETH/BTC 페어(시장 중립): 페어 모멘텀 4h·1·3·7·14·30일, 페어 평균회귀 z(20일) ±1.5 진입 / 0.5 청산, 롱·숏 양쪽 다리 비용 반영
결합(평가 연도 Y마다 그 전 데이터만): 후보 = n ≥ 50거래(연속 신호는 가동일 100일) · 평균>0 · 일간 t ≥ 2 · 직전 3년 중 2년 흑자
  방식 ① 위험 균등(1/σ)  ② 최대 샤프(축소 공분산 0.5 · 음수 비중 금지 · 상한 30%)  ③ z0.31 50% + 나머지 위험 균등
평가: 2022~2026 표본 밖 결합 일간 샤프, 실효 독립 수(비중² 기반), z0.31 단독과 비교, 레버리지(일간 수익 배율) 1~8배 월 복리·MDD.
사용: python research/r86_combine.py"""
import sys, warnings, time
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
IDX = pd.date_range("2020-01-01", "2026-09-25", freq="D", tz="UTC")
CT = 7e-4


def pair_sleeves():
    """ETH/BTC 페어: 포지션 p ∈ {−1,0,1} (ETH 롱·BTC 숏 = +1), 일간 수익 = p × (rETH − rBTC)/2 − 전환 비용."""
    out = {}
    for tf, per in (("1D", 1), ("4h", 6)):
        cl = {s: store.load("klines_1m", s).set_index("ts").close.resample(tf).last() for s in ("BTCUSDT", "ETHUSDT")}
        sp = np.log(cl["ETHUSDT"]) - np.log(cl["BTCUSDT"]); dr = sp.diff().shift(-1)            # 다음 구간 스프레드 변화
        looks = ((1, 3, 7, 14, 30) if tf == "1D" else (1, 6))
        for k in looks:
            p = np.sign(sp - sp.shift(k)).fillna(0)
            ret = p * dr / 2 - p.diff().abs().fillna(0) * CT      # 다리 2개 × 절반 명목 → 전환 시 7bp
            out[f"PAIR|MOM{k}{tf}"] = ret.groupby(ret.index.floor("D")).sum()
        z = (sp - sp.rolling(20 * per).mean()) / sp.rolling(20 * per).std(); pos = pd.Series(0.0, index=sp.index); cur = 0.0
        for t, zz in z.items():
            if np.isnan(zz): pos[t] = 0.0; continue
            if cur == 0 and abs(zz) > 1.5: cur = -np.sign(zz)
            elif cur != 0 and abs(zz) < 0.5: cur = 0.0
            pos[t] = cur
        ret = pos * dr / 2 - pos.diff().abs().fillna(0) * CT
        out[f"PAIR|MR{tf}"] = ret.groupby(ret.index.floor("D")).sum()
    return {k: v.reindex(IDX).fillna(0) for k, v in out.items()}


def library():
    A = pd.read_parquet(R / "data" / "r79_trades.parquet")
    d = store.load("klines_1m", "BTCUSDT").set_index("ts").close.resample("1D").last(); reg = (np.log(d).diff(30) < 0).shift(1)
    Sx = A[A.sl.str[-1].eq("S")].copy(); Sx = Sx[reg.reindex(Sx.te.dt.floor("D")).fillna(False).values.astype(bool)]
    A = pd.concat([A[A.sl != "DIP|z0.31"], Sx.assign(sl=lambda x: x.sl.str.replace("|S", "|BEAR|S", regex=False))], ignore_index=True)
    A["d"] = A.tx.dt.floor("D")
    D = A.groupby(["d", "sl"]).r.sum().mul(0.5).unstack().reindex(IDX).fillna(0)
    N = A.groupby(["sl", A.tx.dt.year]).size().unstack().fillna(0)
    # z0.31 (R81 설정, w1N4 일간)
    import r40_regime_leverage as RL, r80_max_return as R80
    RL.MMR = 0.0065; W = R80.dip_frame(0.0025)
    mult = np.minimum(np.clip(W.vreg.values ** 1.5, 0.3, 4.0) * np.clip(W.pred.values / 0.001, 0.5, 3.0), 6.0)
    dip = R80.daily_from_eq(RL.sim(W, 1, 4, mult, X=0.03, H=720)[0]).reindex(IDX).fillna(0)
    D["DIP|z0.31"] = dip
    for k, v in pair_sleeves().items(): D[k] = v
    return D, N


def pick(D, N, Y):
    tr = D[D.index < pd.Timestamp(f"{Y}-01-01", tz="UTC")]
    rows = []
    for c in D.columns:
        x = tr[c]; act = (x != 0).sum()
        if c.startswith("DIP") or c.startswith("PAIR"): ok_n = act >= 100
        else: ok_n = (N.loc[c][N.columns < Y].sum() if c in N.index else 0) >= 50
        if not ok_n or x.std() == 0: continue
        t = x.mean() / x.std() * np.sqrt(len(x))
        yrs = [tr[c][tr.index.year == yy].sum() for yy in (Y - 3, Y - 2, Y - 1)]
        if x.mean() > 0 and t >= 2 and sum(v > 0 for v in yrs) >= 2: rows.append((c, t))
    return [c for c, _ in sorted(rows, key=lambda z: -z[1])]


def weights(tr, cols, how):
    X = tr[cols]; sd = X.std().replace(0, np.nan)
    if how == "rp": w = (1 / sd).fillna(0); return w / w.sum()
    if how == "dip50":
        o = [c for c in cols if c != "DIP|z0.31"]
        w = pd.Series(0.0, index=cols)
        if "DIP|z0.31" in cols: w["DIP|z0.31"] = 0.5
        if o:
            ww = (1 / sd[o]).fillna(0); w[o] = ww / ww.sum() * (0.5 if "DIP|z0.31" in cols else 1.0)
        return w / w.sum()
    mu = X.mean().values; C = X.cov().values; C = 0.5 * C + 0.5 * np.diag(np.diag(C))
    w = np.linalg.pinv(C) @ mu; w = np.clip(w, 0, None)
    if w.sum() == 0: w = np.ones(len(cols))
    w = w / w.sum(); w = np.minimum(w, 0.3); w = w / w.sum()
    return pd.Series(w, index=cols)


def perf(r):
    eq = (1 + r.clip(lower=-0.99)).cumprod()
    mon = eq.resample("ME").last().pct_change().dropna(); yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    return (eq.iloc[-1] ** (1 / 12 / yrs) - 1) * 100 if eq.iloc[-1] > 0 else -100, (eq / eq.cummax() - 1).min() * 100, mon.median() * 100


if __name__ == "__main__":
    t_ = time.time(); D, N = library()
    P(f"라이브러리 {D.shape[1]}개 신호 (페어 {sum(c.startswith('PAIR') for c in D)}) ({time.time()-t_:.0f}s)")
    ev = D[D.index >= "2022-01-01"]
    sh = (ev.mean() / ev.std() * np.sqrt(365)).dropna().sort_values(ascending=False)
    corr_dip = ev.corrwith(ev["DIP|z0.31"])
    P("\n  표본 밖이 아닌 참고: 2022~26 단독 샤프 상위 15 (z0.31 상관)")
    for c in sh.index[:15]: P(f"    {c:40s} 샤프 {sh[c]:5.2f}  z0.31 상관 {corr_dip[c]:+.2f}")
    P("  페어 신호: " + " | ".join(f"{c.split('|')[1]} {sh.get(c, np.nan):.2f}({corr_dip[c]:+.2f})" for c in D.columns if c.startswith("PAIR")))
    res = {}
    for how, nm in (("rp", "위험 균등"), ("ms", "최대 샤프(축소)"), ("dip50", "z0.31 50% + 위험 균등")):
        parts = []; info = []
        for Y in range(2022, 2027):
            cols = pick(D, N, Y)
            if not cols: continue
            tr = D[(D.index < pd.Timestamp(f"{Y}-01-01", tz="UTC")) & (D.index >= pd.Timestamp(f"{Y-3}-01-01", tz="UTC"))]
            w = weights(tr, cols, how)
            te = D[(D.index >= pd.Timestamp(f"{Y}-01-01", tz="UTC")) & (D.index < pd.Timestamp(f"{Y+1}-01-01", tz="UTC"))]
            parts.append((te[cols] * w).sum(axis=1))
            info.append(f"{Y}:{len(cols)}개(실효 {1/(w**2).sum():.1f}, z0.31 {w.get('DIP|z0.31', 0):.2f})")
        r = pd.concat(parts); res[nm] = r
        P(f"\n  [{nm}] 선택: " + " ".join(info))
        P(f"    표본 밖 2022~26 결합 샤프 {r.mean()/r.std()*np.sqrt(365):.2f} (z0.31 단독 {ev['DIP|z0.31'].mean()/ev['DIP|z0.31'].std()*np.sqrt(365):.2f})")
        vol_ratio = ev["DIP|z0.31"].std() / r.std()
        for L in (1, 2, 4, 6, 8):
            mo, mdd, med = perf(r * vol_ratio * L)
            P(f"    z0.31(w1N4)과 같은 변동성 × {L}배: 월 복리 {mo:+.1f}% (중앙 {med:+.1f}%) MDD {mdd:.0f}%")
    dp = ev["DIP|z0.31"]
    P("\n  기준 z0.31 단독(같은 일간 배율): " + " | ".join(f"×{L}: 월 {perf(dp*L)[0]:+.1f}% MDD {perf(dp*L)[1]:.0f}%" for L in (1, 2, 4, 6, 8)))
    (R / "logs" / "r86.summary").write_text("R86 신호 결합 완료. 세부 logs/r86.log\n")
