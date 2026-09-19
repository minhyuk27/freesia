"""분봉 이론 연구(61가지)에서 도출한 측정 9종. 전략이 아니라 통계적 사실의 측정. 5분 봉 기준, 학습 2020~2023 / 테스트 2024~.
사용: python research/minute_stats.py  → logs/minute_stats.out, logs/minute_stats.summary"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freesia import store  # noqa: E402

OUT = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    OUT.append(s)


def load(sym):
    k = store.load("klines_1m", sym).set_index("ts")
    b = k.resample("5min").agg({"open": "first", "high": "max", "low": "min", "close": "last", "quote_volume": "sum",
                                "trades": "sum", "taker_buy_quote_volume": "sum"}).dropna(subset=["close"])
    return k["close"], b


def ac(x, lag):
    x = x.dropna(); return float(np.corrcoef(x.values[:-lag], x.values[lag:])[0, 1])


def ols_r2(X, y, tr, te):
    Xa = np.column_stack([np.ones(len(X)), X])
    beta = np.linalg.lstsq(Xa[tr], y[tr], rcond=None)[0]
    def r2(m):
        e = y[m] - Xa[m] @ beta; return 1 - e.var() / y[m].var()
    return r2(tr), r2(te), beta


SUM = []
for sym in ("BTCUSDT", "ETHUSDT"):
    c1, b = load(sym)
    P(f"\n=== {sym} (5분봉 {len(b):,}개) ===")
    r1 = np.log(c1).diff().dropna()
    r5 = np.log(b.close).diff()
    r15 = np.log(b.close.iloc[::3]).diff()
    yr = b.index.year
    # M4: 자기상관·분산비
    P("[4] 수익률 자기상관(lag1) — 1분/5분/15분, 연도별")
    rows = []
    for y in range(2020, 2027):
        m1 = r1[r1.index.year == y]; m5 = r5[yr == y]; m15 = r15[r15.index.year == y]
        rows.append((y, ac(m1, 1), ac(m5, 1), ac(m15, 1)))
    P(pd.DataFrame(rows, columns=["연", "1분", "5분", "15분"]).round(4).to_string(index=False))
    lp = np.log(b.close)
    vr = {}
    for q in (3, 12, 48):
        rq = lp.diff(q).dropna()[::q]
        vr[q] = float(rq.var() / (q * r5.dropna().var()))
    P("분산비 VR(q) q=3,12,48 (1=랜덤워크):", {k: round(v, 3) for k, v in vr.items()})
    # M5: 테이커 불균형 ACF와 방향 예측력
    imb = (2 * b.taker_buy_quote_volume / b.quote_volume.replace(0, np.nan) - 1)
    P("[5] 테이커 불균형 ACF lag 1,3,6,12,36:", {L: round(ac(imb, L), 3) for L in (1, 3, 6, 12, 36)})
    fwd1 = r5.shift(-1); fwd6 = lp.shift(-6) - lp
    for nm, f in (("다음 5분", fwd1), ("다음 30분", fwd6)):
        d = pd.concat([imb, f], axis=1).dropna(); d.columns = ["i", "f"]
        cor = d.corr().iloc[0, 1]
        hi = d[d.i > d.i.quantile(0.95)].f.mean() * 1e4; lo = d[d.i < d.i.quantile(0.05)].f.mean() * 1e4
        P(f"  불균형→{nm} 상관 {cor:.4f} | 상위5% 다음 평균 {hi:.2f}bp / 하위5% {lo:.2f}bp (비용 12bp)")
    # M1: HAR 변동성 예측
    r2s = r5 ** 2
    rv_f = r2s[::-1].rolling(12).sum()[::-1].shift(-1)          # 다음 1시간
    rv1h = r2s.rolling(12).sum(); rv1d = r2s.rolling(288).sum() / 24; rv1w = r2s.rolling(2016).sum() / 168
    hod = b.index.hour * 12 + b.index.minute // 5
    tr_mask = (yr <= 2023)
    prof = pd.Series(np.log(rv_f.where(tr_mask).replace(0, np.nan))).groupby(hod).mean()
    prof_v = prof.reindex(hod).values
    d = pd.DataFrame({"y": np.log(rv_f.replace(0, np.nan)), "a": np.log(rv1h.replace(0, np.nan)), "d": np.log(rv1d.replace(0, np.nan)),
                      "w": np.log(rv1w.replace(0, np.nan)), "p": prof_v}).replace([np.inf, -np.inf], np.nan)
    ok = d.notna().all(axis=1).values & (yr >= 2020)
    yv = d.y.values
    trm = ok & tr_mask; tem = ok & ~tr_mask
    P("[1] 다음 1시간 log 실현변동성 예측 R² (학습 / 테스트)")
    for nm, cols in (("직전1h", ["a"]), ("HAR(1h,1d,1w)", ["a", "d", "w"]), ("시간대 프로필만", ["p"]), ("HAR+시간대", ["a", "d", "w", "p"])):
        a, t, _ = ols_r2(d[cols].values, yv, trm, tem)
        P(f"  {nm:16s} {a:.3f} / {t:.3f}")
    pv = d.copy(); pv["ok"] = ok
    _, _, beta = ols_r2(d[["a", "d", "w", "p"]].values, yv, trm, tem)
    pred = np.column_stack([np.ones(len(d)), d[["a", "d", "w", "p"]].values]) @ beta
    SUM.append((sym, "HAR+시간대 테스트R²", ols_r2(d[["a", "d", "w", "p"]].values, yv, trm, tem)[1]))
    # M2: 체결 건수 급증
    tz = np.log(b.trades.replace(0, np.nan)) - np.log(b.trades.replace(0, np.nan)).rolling(288).median()
    surge = tz > tz[tr_mask].quantile(0.95)
    base = np.exp(d.y).mean()
    P(f"[2] 체결 건수 급증(상위5%) 뒤 1시간 실현변동성 배수: {np.exp(d.y[surge & pd.Series(ok, index=b.index)]).mean() / base:.2f}배 (무조건 1.0)")
    # M3: 점프 군집
    sig = (r5.abs().rolling(288).median() / 0.6745).shift(1)
    jump = (r5.abs() > 4 * sig)
    jf = jump[::-1].rolling(12).max()[::-1].shift(-1)
    pj = jf[jump].mean(); pu = jf.mean()
    P(f"[3] 4σ 점프 뒤 1시간 내 추가 점프 확률 {pj:.3f} vs 무조건 {pu:.3f} → {pj / pu:.1f}배 (점프 {int(jump.sum())}회)")
    # M6: 예상 변동성 상위 구간에서 방향 규칙의 총 기대값(직전 1시간 방향 추종, 1시간 보유)
    mom = lp - lp.shift(12); fwd12 = lp.shift(-12) - lp
    dd = pd.DataFrame({"pred": pred, "mom": mom, "f": fwd12, "ok": ok}).dropna()
    dd = dd[dd.ok.astype(bool) & (dd.index.year >= 2024)]
    dd["q"] = pd.qcut(dd.pred, 5, labels=False)
    P("[6] 예상 변동성 5분위별 (테스트기간) 1시간 추종 규칙 총 기대값 bp / |1시간 이동| 평균 bp / 비용 12bp")
    rows = []
    for q, g in dd.groupby("q"):
        g = g.iloc[::12]
        pnl = (np.sign(g.mom) * g.f).mean() * 1e4
        rows.append((int(q) + 1, pnl, g.f.abs().mean() * 1e4, len(g)))
    P(pd.DataFrame(rows, columns=["분위", "추종총기대bp", "평균|이동|bp", "n"]).round(2).to_string(index=False))
    # M7: 이벤트 (펀딩 정산·금요일 08 UTC)
    ab = r5.abs()
    prof_h = ab.groupby(b.index.hour).mean(); allm = ab.mean()
    P("[7] 시간대(UTC)별 5분 |수익률| 평균의 배수 — 00/08/16(펀딩)·13/14(미국 개장) 정각 30분 vs 전체:")
    m30 = ab[(b.index.minute < 30)]
    P("  ", {h: round(float(m30[m30.index.hour == h].mean() / allm), 2) for h in (0, 8, 13, 14, 16, 21)})
    fr = b[(b.index.dayofweek == 4) & (b.index.hour == 8)]
    fr_ab = ab.loc[fr.index].mean()
    P(f"  금요일 08시 UTC 1시간 5분|수익률| {fr_ab / allm:.2f}배, 금요일 08~16시 방향 평균 {(lp.shift(-96) - lp).loc[fr.index[fr.index.minute == 0]].mean() * 1e4:.1f}bp")
    # M8: 청산 확률 vs 예상 변동성
    lo12 = b.low[::-1].rolling(12).min()[::-1].shift(-1); hi12 = b.high[::-1].rolling(12).max()[::-1].shift(-1)
    adv_long = (b.close - lo12) / b.close; adv_short = (hi12 - b.close) / b.close
    dq = pd.DataFrame({"pred": pred, "al": adv_long, "as": adv_short, "ok": ok}).dropna()
    dq = dq[dq.ok.astype(bool) & (dq.index.year >= 2024)]
    dq["q"] = pd.qcut(dq.pred, 5, labels=False)
    P("[8] 예상 변동성 5분위별 1시간 내 청산 확률(롱·숏 평균, 청산거리=1/L-0.4%) — 테스트기간")
    rows = []
    for q, g in dq.groupby("q"):
        rows.append((int(q) + 1, ((g.al >= 0.006).mean() + (g["as"] >= 0.006).mean()) / 2 * 100,
                     ((g.al >= 0.016).mean() + (g["as"] >= 0.016).mean()) / 2 * 100, ((g.al >= 0.046).mean() + (g["as"] >= 0.046).mean()) / 2 * 100))
    P(pd.DataFrame(rows, columns=["분위", "100x %", "50x %", "20x %"]).round(2).to_string(index=False))
    # M9: 시간대 프로필 연도 간 안정성 (변동성, Amihud)
    am = (r5.abs() / b.quote_volume.replace(0, np.nan))
    for nm, s in (("변동성", ab), ("Amihud", am)):
        pr = {y: s[yr == y].groupby(b.index[yr == y].hour).mean() for y in range(2021, 2027)}
        cs = [np.corrcoef(pr[y].values, pr[y + 1].values)[0, 1] for y in range(2021, 2026)]
        P(f"[9] {nm} 시간대 프로필 연도간 상관(인접 연도): {np.round(cs, 2).tolist()}")
    h = am.groupby(b.index.hour).mean(); P("  Amihud 최대/최소 시간(UTC):", int(h.idxmax()), int(h.idxmin()), "배수", round(float(h.max() / h.min()), 2))

Path("logs").mkdir(exist_ok=True)
Path("logs/minute_stats.summary").write_text("측정 9종 완료(BTC·ETH). 세부는 logs/minute_stats.out\n" + "\n".join(f"{a} {b} {c:.3f}" for a, b, c in SUM))
