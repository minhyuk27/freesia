"""새 요인 탐색 — 지금까지 안 쓴 데이터(premium, OI, 테이커비율, 펀딩)로 '급함'을 직접 관측.
전부 진입 시점에 알 수 있는 값만 사용(신호는 t분, 진입은 t+1분 시가). 룩어헤드 없음.
사용: python research/newfactors.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
SYM = ("BTCUSDT", "ETHUSDT")

D = {}
for s in SYM:
    k = store.load("klines_1m", s).set_index("ts")
    pr = store.load("premium_1m", s).set_index("ts")
    mk = store.load("mark_1m", s).set_index("ts")
    ix = k.index.intersection(pr.index).intersection(mk.index)
    d = pd.DataFrame(index=ix)
    d["o"], d["h"], d["l"], d["c"] = k.open[ix], k.high[ix], k.low[ix], k.close[ix]
    d["prem"] = pr.close[ix]          # 프리미엄 지수 (무기한 vs 지수 괴리)
    d["mark"] = mk.close[ix]
    m5 = store.load("metrics_5m", s).set_index("ts")
    d = d.join(m5.reindex(ix, method="ffill", limit=10), how="left")
    f = store.load("funding", s).set_index("ts")
    d["fund"] = f.rate.reindex(ix, method="ffill", limit=600)
    D[s] = d
    P(f"{s}: {len(d):,}행  프리미엄 분포 중앙 {d.prem.median()*1e4:+.1f}bp "
      f"1분위 {d.prem.quantile(0.01)*1e4:+.0f}bp 0.1분위 {d.prem.quantile(0.001)*1e4:+.0f}bp "
      f"최소 {d.prem.min()*1e4:+.0f}bp")

HOR = (5, 15, 30, 60, 240, 1440)


def fwd(c, hz):
    return {h: pd.Series(c).shift(-h).values/c - 1.0 for h in hz}


def report(name, sig, s, cond_desc, qs=(0.001, 0.005, 0.01, 0.05)):
    """sig: 낮을수록 극단(예: 프리미엄). 하위 분위별 forward 수익."""
    d = D[s]; c = d.c.values
    F = fwd(c, HOR)
    v = pd.Series(sig).values
    ok = np.isfinite(v)
    rows = []
    for q in qs:
        th = np.nanquantile(v[ok], q)
        sel = ok & (v <= th)
        if sel.sum() < 50: continue
        row = {"조건": f"하위{q*100:g}%", "임계": f"{th:.5g}", "n": int(sel.sum())}
        for h in HOR:
            x = F[h][sel]; x = x[np.isfinite(x)]
            row[f"{h}분"] = f"{np.mean(x)*1e4:+.0f}"
        rows.append(row)
    base = {}
    for h in HOR:
        x = F[h][np.isfinite(F[h])]
        base[f"{h}분"] = f"{np.mean(x)*1e4:+.0f}"
    rows.append({"조건": "전체평균", "임계": "-", "n": int(ok.sum()), **base})
    P(f"\n  ── {s} · {name} ({cond_desc}) ──")
    P(pd.DataFrame(rows).to_string(index=False))


P("\n" + "="*100)
P("[1] 프리미엄 급락 — 무기한이 지수 대비 얼마나 싸게 팔리는가 = '급함의 가격'")
P("="*100)
for s in SYM:
    report("프리미엄 수준", D[s].prem, s, "낮을수록 무기한이 지수보다 쌈")

P("\n" + "="*100)
P("[2] 프리미엄 변화(1·5분) — 급격한 괴리 확대")
P("="*100)
for s in SYM:
    report("프리미엄 5분 변화", D[s].prem.diff(5), s, "5분간 괴리가 급히 벌어짐")

P("\n" + "="*100)
P("[3] 미결제약정 급감 = 강제청산 캐스케이드 (2021-12~)")
P("="*100)
for s in SYM:
    oi = D[s].sum_open_interest
    report("OI 30분 변화율", oi.pct_change(30), s, "OI가 급감 = 포지션 강제 정리")

P("\n" + "="*100)
P("[4] 테이커 매도 비율 극단 = 공격적 시장가 매도")
P("="*100)
for s in SYM:
    report("테이커 롱/숏 비율", D[s].sum_taker_long_short_vol_ratio, s, "낮을수록 시장가 매도 우세")

P("\n" + "="*100)
P("[5] 펀딩 극단 — 음수 펀딩은 숏이 롱에게 지불 = 숏 과밀")
P("="*100)
for s in SYM:
    report("펀딩율", D[s].fund, s, "음수일수록 숏 과밀")

P("\n" + "="*100)
P("[6] 상위 트레이더 롱/숏 비율")
P("="*100)
for s in SYM:
    report("상위 계정 롱숏비", D[s].sum_toptrader_long_short_ratio, s, "낮을수록 고래가 숏")

P("\n" + "="*100)
P("[7] 조합 — 프리미엄 급락 + OI 급감 (진짜 청산 캐스케이드)")
P("="*100)
for s in SYM:
    d = D[s]
    pq = d.prem.rank(pct=True)
    oq = d.sum_open_interest.pct_change(30).rank(pct=True)
    c = d.c.values; F = fwd(c, HOR)
    rows = []
    for pt, ot, lab in ((0.01, 0.05, "프리미엄 하위1% + OI 하위5%"), (0.005, 0.10, "프리미엄 하위0.5% + OI 하위10%"),
                        (0.01, 1.01, "프리미엄 하위1% 단독"), (1.01, 0.01, "OI 하위1% 단독")):
        sel = ((pq <= pt) & (oq <= ot)).values
        if sel.sum() < 30: continue
        row = {"조건": lab, "n": int(sel.sum())}
        for h in HOR:
            x = F[h][sel]; x = x[np.isfinite(x)]
            row[f"{h}분"] = f"{np.mean(x)*1e4:+.0f}"
        rows.append(row)
    P(f"\n  ── {s} ──")
    P(pd.DataFrame(rows).to_string(index=False))
