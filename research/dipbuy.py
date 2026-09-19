"""봉 안 역행 되돌림 전략: 시가 대비 x% 하락하면 지정가 매수 → 그 봉 종가에 청산 (반대도).
비용: 진입 지정가(메이커 2bp) + 청산 시장가(테이커 5bp) = 7bp. 테이커 진입 가정(12bp)도 병기.
학습 2020~2023 / 테스트 2024~, 위약(같은 빈도·보유의 무작위 진입) 비교.
사용: python research/dipbuy.py"""
import sys, warnings
from pathlib import Path
import numba, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R))
from freesia import store
TE = pd.Timestamp("2024-01-01", tz="UTC")
rng = np.random.default_rng(131)
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)


@numba.njit(cache=True)
def scan(o, h, l, c, B, x, side):
    """side=1: 시가 -x% 터치 시 매수→종가 청산. side=-1: +x% 터치 시 매도→종가 청산.
    반환 (봉 시작 인덱스, 수익률, 시가대비종가 부호, 진입까지 분)"""
    n = len(c); nb = n // B
    st = np.full(nb, -1, np.int64); ret = np.full(nb, np.nan); sgn = np.zeros(nb, np.int8); tmin = np.zeros(nb, np.int32)
    for b in range(nb):
        s = b * B; e = s + B
        O = o[s]; C = c[e - 1]
        trig = O * (1.0 - x) if side == 1 else O * (1.0 + x)
        j = -1
        for t in range(s, e):
            if side == 1:
                if l[t] <= trig:
                    j = t; break
            else:
                if h[t] >= trig:
                    j = t; break
        if j < 0:
            continue
        st[b] = s; tmin[b] = j - s
        ret[b] = (C / trig - 1.0) if side == 1 else (1.0 - C / trig)
        sgn[b] = 1 if C > O else (-1 if C < O else 0)
    return st, ret, sgn, tmin


@numba.njit(cache=True)
def rand_ctrl(o, c, B, idxs, side):
    """대조군: 같은 봉에서 무작위 분에 진입해 종가 청산"""
    out = np.full(len(idxs), np.nan)
    for q in range(len(idxs)):
        s = idxs[q]; e = s + B
        if e > len(c):
            continue
        j = s + (q * 7919) % B
        entry = c[j]; C = c[e - 1]
        out[q] = (C / entry - 1.0) if side == 1 else (1.0 - C / entry)
    return out


for sym in ("BTCUSDT", "ETHUSDT"):
    k = store.load("klines_1m", sym).set_index("ts")
    o, h, l, c = k.open.values, k.high.values, k.low.values, k.close.values
    idx = k.index
    P("=" * 110 + f"\n### {sym}  1분봉 {len(k):,}개")
    rows = []
    for B, bn in ((5, "5분"), (15, "15분"), (60, "1시간"), (240, "4시간"), (1440, "1일")):
        for x in (0.001, 0.002, 0.003, 0.005, 0.010):
            for side, sn in ((1, "하락후매수"), (-1, "상승후매도")):
                st, ret, sgn, tmin = scan(o, h, l, c, B, x, side)
                ok = ~np.isnan(ret)
                if ok.sum() < 60:
                    continue
                bt = idx[st[ok]]
                nb = len(ret)
                for nm, m in (("학습", bt < TE), ("테스트", bt >= TE)):
                    r = ret[ok][m]
                    if len(r) < 30:
                        continue
                    sg = sgn[ok][m]
                    wk = len(r) / max((bt[m][-1] - bt[m][0]).days / 7, 1)
                    # 대조군
                    ctrl = rand_ctrl(o, c, B, st[ok][m], side)
                    ctrl = ctrl[~np.isnan(ctrl)]
                    rows.append((bn, x * 100, sn, nm, r.mean() * 1e4, r.mean() * 1e4 - 7, r.mean() * 1e4 - 12,
                                 float((r > 0).mean()), float((sg == side).mean()), np.median(tmin[ok][m]),
                                 len(r), wk, ctrl.mean() * 1e4 if len(ctrl) else np.nan,
                                 float(ok.sum()) / nb))
    d = pd.DataFrame(rows, columns=["봉", "x%", "방향", "구간", "총bp", "메이커순bp", "테이커순bp", "승률",
                                    "종가가 시가반대편", "진입 중앙분", "n", "주당", "대조군bp", "발생률"])
    te = d[d.구간 == "테스트"]
    P("\n[테스트 구간 — 메이커 진입(7bp) 기준 순수익 상위 15]")
    P(te.sort_values("메이커순bp", ascending=False).head(15).round(2).to_string(index=False))
    piv = d.pivot_table(index=["봉", "x%", "방향"], columns="구간", values=["총bp", "메이커순bp", "대조군bp"])
    both = piv[(piv[("메이커순bp", "학습")] > 0) & (piv[("메이커순bp", "테스트")] > 0)]
    P(f"\n학습·테스트 모두 메이커 기준 양수: {len(both)} / {len(piv)}")
    if len(both):
        b2 = both.copy()
        b2["초과(테스트)"] = b2[("총bp", "테스트")] - b2[("대조군bp", "테스트")]
        P(b2.sort_values("초과(테스트)", ascending=False).head(12).round(2).to_string())
    P("\n[대조군 대비 초과 — 같은 봉·무작위 진입 시점 대비, 테스트 구간]")
    te2 = te.assign(초과=te.총bp - te.대조군bp).sort_values("초과", ascending=False)
    P(te2[["봉", "x%", "방향", "총bp", "대조군bp", "초과", "메이커순bp", "승률", "n", "주당"]].head(12).round(2).to_string(index=False))
    P("\n[사용자 질문 직답] 시가 아래로 x% 내려간 뒤 종가가 시가 위인 비율 (테스트 구간)")
    q = te[te.방향 == "하락후매수"].pivot_table(index="x%", columns="봉", values="종가가 시가반대편")
    P(q.round(4).to_string())
    P("  (상승후매도: 시가 위로 x% 오른 뒤 종가가 시가 아래인 비율)")
    q2 = te[te.방향 == "상승후매도"].pivot_table(index="x%", columns="봉", values="종가가 시가반대편")
    P(q2.round(4).to_string())
