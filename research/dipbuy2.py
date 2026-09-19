"""1일봉 '시가 -x% 지정가 매수 → 종가 청산'의 정체 규명: 매수보유·시가매수종가청산 대비 초과가 있는가.
연도별·워크포워드·위약까지. 사용: python research/dipbuy2.py"""
import sys, warnings
from pathlib import Path
import numba, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R))
from freesia import store
from dipbuy import scan
TE = pd.Timestamp("2024-01-01", tz="UTC")
rng = np.random.default_rng(151)
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)

for sym in ("BTCUSDT", "ETHUSDT"):
    k = store.load("klines_1m", sym).set_index("ts")
    o, h, l, c = k.open.values, k.high.values, k.low.values, k.close.values
    idx = k.index
    B = 1440
    nb = len(c) // B
    dayO = o[::B][:nb]; dayC = c[B - 1::B][:nb]; dayIdx = idx[::B][:nb]
    oc = (dayC / dayO - 1) * 1e4            # 시가매수 종가청산 (하루 낮 수익)
    o2o = np.concatenate([[np.nan], (dayO[1:] / dayO[:-1] - 1) * 1e4])   # 매수보유
    P("=" * 96 + f"\n### {sym}  1일봉 {nb}개")
    P(f"[기준선] 시가매수→종가청산 평균 {np.nanmean(oc):.2f}bp | 매수보유(시가→시가) 평균 {np.nanmean(o2o):.2f}bp")
    for nm, m in (("학습", dayIdx < TE), ("테스트", dayIdx >= TE)):
        P(f"  {nm}: 시가→종가 {np.nanmean(oc[m]):.2f}bp (승률 {np.nanmean(oc[m]>0):.3f}) | 매수보유 {np.nanmean(o2o[m]):.2f}bp")
    P("\n[전략] 시가 -x% 지정가 매수 → 종가 청산. 메이커 진입 2bp + 테이커 청산 5bp = 7bp")
    rows = []
    for x in (0.001, 0.002, 0.003, 0.005, 0.010):
        st, ret, sgn, tmin = scan(o, h, l, c, B, x, 1)
        ok = ~np.isnan(ret)
        r = ret * 1e4
        for nm, m in (("학습", (dayIdx < TE)[:len(ok)]), ("테스트", (dayIdx >= TE)[:len(ok)])):
            sel = ok & m
            if sel.sum() < 30:
                continue
            # 같은 날의 시가→종가 대비 초과
            exc = r[sel] - oc[:len(ok)][sel]
            rows.append((x * 100, nm, r[sel].mean(), r[sel].mean() - 7, float((r[sel] > 0).mean()),
                         oc[:len(ok)][sel].mean(), exc.mean(), float(sel.sum()), float(sel.sum()) / m.sum()))
    d = pd.DataFrame(rows, columns=["x%", "구간", "총bp", "메이커순bp", "승률", "같은날 시가→종가bp", "초과bp", "일수", "발생률"])
    P(d.round(2).to_string(index=False))
    P("\n  ※ '초과bp' = 지정가 진입이 그날 시가 진입보다 얼마나 더 벌었나 (= 이론상 x + 진입 시점 이득)")
    # 연도별 (x=0.003)
    st, ret, sgn, tmin = scan(o, h, l, c, B, 0.003, 1)
    ok = ~np.isnan(ret); r = ret * 1e4
    P("\n[연도별] x=0.3%")
    rows = []
    for y in range(2020, 2027):
        m = ok & (dayIdx[:len(ok)].year == y)
        if m.sum() < 30:
            continue
        rows.append((y, r[m].mean(), r[m].mean() - 7, float((r[m] > 0).mean()),
                     oc[:len(ok)][m].mean(), (r[m] - oc[:len(ok)][m]).mean(), int(m.sum()),
                     r[m].sum() / 100 - m.sum() * 0.07))
    P(pd.DataFrame(rows, columns=["연", "총bp", "메이커순bp", "승률", "시가→종가bp", "초과bp", "일수", "연누적%p"]).round(2).to_string(index=False))
    # 위약: 같은 날 무작위 분 진입(지정가 아님)
    P("\n[위약] 같은 날 무작위 분 진입 → 종가 청산, 300회")
    pls = []
    sel_all = np.where(ok & (dayIdx[:len(ok)] >= TE))[0]
    for _ in range(300):
        j = rng.integers(0, B, len(sel_all))
        ent = c[sel_all * B + j]
        pls.append(((dayC[sel_all] / ent - 1) * 1e4).mean())
    real = r[ok & (dayIdx[:len(ok)] >= TE)].mean()
    P(f"  위약 평균 {np.mean(pls):.2f}bp, 95% {np.percentile(pls,95):.2f}, 최대 {np.max(pls):.2f} | 실제 {real:.2f} → 백분위 {(np.array(pls)<real).mean()*100:.1f}%")
    P(f"  메이커 비용 7bp 반영 시 실제 {real-7:.2f}bp vs 위약 평균 {np.mean(pls)-7:.2f}bp")
