"""봉 안 역행 되돌림 전략 110종 — 앞선 측정(초과 = 정확히 x, 즉 타이밍 기여 0)에서 출발.
핵심 지표: 초과 − x. 0보다 커야 '진입 조건에 실력이 있다'는 뜻. 0이면 산술 할인일 뿐.
격자: 봉 2종 × 진입깊이 5종 × 조건 11종 = 110. 청산은 종가. 상위 후보는 청산 규칙 변형 추가.
비용: 지정가 진입 2bp + 시장가 청산 5bp = 7bp. 학습 2020~2023 / 테스트 2024~.
사용: python research/intrabar110.py"""
import sys, warnings
from pathlib import Path
import numba, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R))
from freesia import store
TE = pd.Timestamp("2024-01-01", tz="UTC")
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)


@numba.njit(cache=True)
def run(o, h, l, c, B, xarr, cond, exitmode):
    """봉별: 시가 -x 터치 시 지정가 매수. exitmode 0=종가, 1=시가복귀시, 2=목표 2x, 3=봉 절반시점
    반환 (봉인덱스, 수익bp, 같은봉 시가→종가bp, 사용된 x bp, 진입분)"""
    nb = len(c) // B
    ret = np.full(nb, np.nan); oc = np.full(nb, np.nan); xs = np.full(nb, np.nan); tm = np.zeros(nb, np.int32)
    for b in range(nb):
        if not cond[b] or np.isnan(xarr[b]) or xarr[b] <= 0:
            continue
        s = b * B; e = s + B
        O = o[s]; C = c[e - 1]
        x = xarr[b]
        trig = O * (1.0 - x)
        j = -1
        for t in range(s, e):
            if l[t] <= trig:
                j = t; break
        if j < 0:
            continue
        px = C
        if exitmode == 1:
            for t in range(j, e):
                if h[t] >= O:
                    px = O; break
            else:
                px = C
        elif exitmode == 2:
            tgt = trig * (1.0 + 2.0 * x)
            for t in range(j, e):
                if h[t] >= tgt:
                    px = tgt; break
            else:
                px = C
        elif exitmode == 3:
            px = c[min(s + B // 2, e - 1)] if j < s + B // 2 else C
        ret[b] = (px / trig - 1.0) * 1e4
        oc[b] = (C / O - 1.0) * 1e4
        xs[b] = x * 1e4
        tm[b] = j - s
    return ret, oc, xs, tm


for sym in ("BTCUSDT", "ETHUSDT"):
    k = store.load("klines_1m", sym).set_index("ts")
    o, h, l, c, v = k.open.values, k.high.values, k.low.values, k.close.values, k.quote_volume.values
    idx = k.index
    P("=" * 112 + f"\n### {sym}")
    rows = []
    for B, bn in ((60, "1시간"), (240, "4시간")):
        nb = len(c) // B
        bo = o[::B][:nb]; bc = c[B - 1::B][:nb]
        bh = np.array([h[i * B:(i + 1) * B].max() for i in range(nb)])
        bl = np.array([l[i * B:(i + 1) * B].min() for i in range(nb)])
        bv = np.array([v[i * B:(i + 1) * B].sum() for i in range(nb)])
        bi = idx[::B][:nb]
        s = pd.Series
        rng_prev = s((bh - bl) / bo).shift(1)
        body_prev = s((bc - bo) / bo).shift(1)
        lw_prev = s((np.minimum(bo, bc) - bl) / bo).shift(1)
        gap = s(bo).div(s(bc).shift(1)).sub(1)
        rv = s(np.log(bc)).diff().rolling(24).std()
        volz = s(bv) / s(bv).rolling(168).median()
        XS = {"고정0.2%": np.full(nb, 0.002), "고정0.3%": np.full(nb, 0.003), "고정0.5%": np.full(nb, 0.005),
              "직전범위0.5배": (rng_prev * 0.5).values, "직전범위1.0배": (rng_prev * 1.0).values}
        CD = {"전체": np.ones(nb, bool),
              "직전봉 상승": (body_prev > 0).fillna(False).values,
              "직전봉 하락": (body_prev < 0).fillna(False).values,
              "직전봉 아래꼬리 김": (lw_prev > lw_prev.rolling(168).quantile(.7)).fillna(False).values,
              "갭하락 진입": (gap < 0).fillna(False).values,
              "갭상승 진입": (gap > 0).fillna(False).values,
              "저변동": (rv < rv.rolling(720).quantile(.3)).fillna(False).values,
              "고변동": (rv > rv.rolling(720).quantile(.7)).fillna(False).values,
              "거래량급증": (volz > 1.5).fillna(False).values,
              "미국장(13~21)": ((bi.hour >= 13) & (bi.hour < 21)),
              "아시아장(0~8)": ((bi.hour >= 0) & (bi.hour < 8))}
        for xn, xa in XS.items():
            for cn, cd in CD.items():
                ret, oc, xs, tm = run(o, h, l, c, B, xa, cd, 0)
                ok = ~np.isnan(ret)
                if ok.sum() < 150:
                    continue
                for nm, m in (("학습", bi < TE), ("테스트", bi >= TE)):
                    sel = ok & m
                    if sel.sum() < 60:
                        continue
                    exc = ret[sel] - oc[sel]
                    skill = exc - xs[sel]
                    rows.append((bn, xn, cn, nm, ret[sel].mean(), ret[sel].mean() - 7,
                                 oc[sel].mean(), exc.mean(), xs[sel].mean(), skill.mean(),
                                 skill.mean() / (skill.std() / np.sqrt(len(skill))) if skill.std() > 0 else 0,
                                 float((ret[sel] > 0).mean()), int(sel.sum()),
                                 sel.sum() / max((bi[m][-1] - bi[m][0]).days / 7, 1)))
    d = pd.DataFrame(rows, columns=["봉", "깊이", "조건", "구간", "총bp", "메이커순bp", "시가→종가bp",
                                    "초과bp", "x bp", "실력bp", "실력t", "승률", "n", "주당"])
    P(f"전략 {len(d)//2}종 검증")
    te = d[d.구간 == "테스트"]
    P("\n[핵심 지표: 실력bp = 초과 − x. 0보다 커야 진입 조건에 정보가 있다] 테스트 상위 15")
    P(te.sort_values("실력bp", ascending=False).head(15)[["봉", "깊이", "조건", "총bp", "메이커순bp", "x bp", "초과bp", "실력bp", "실력t", "승률", "n", "주당"]].round(2).to_string(index=False))
    P(f"\n  실력bp > 0 인 전략: {int((te.실력bp > 0).sum())} / {len(te)}")
    P(f"  실력 t > 2 인 전략: {int((te.실력t > 2).sum())} / {len(te)}  (110종 기준 다중검정 문턱 t≈3.4)")
    P(f"  실력 t > 3.4 인 전략: {int((te.실력t > 3.4).sum())} / {len(te)}")
    piv = d.pivot_table(index=["봉", "깊이", "조건"], columns="구간", values=["메이커순bp", "실력bp"])
    both = piv[(piv[("메이커순bp", "학습")] > 0) & (piv[("메이커순bp", "테스트")] > 0)]
    P(f"\n  학습·테스트 모두 메이커 순 양수: {len(both)} / {len(piv)}")
    if len(both):
        P(both.sort_values(("메이커순bp", "테스트"), ascending=False).head(10).round(2).to_string())
    P("\n[조건별 평균 실력bp — 테스트]")
    P(te.groupby("조건")[["실력bp", "메이커순bp"]].mean().round(2).sort_values("실력bp", ascending=False).to_string())
    P("\n[깊이별 평균 실력bp — 테스트]")
    P(te.groupby("깊이")[["실력bp", "메이커순bp", "x bp"]].mean().round(2).to_string())
