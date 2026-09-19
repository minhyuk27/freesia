"""후보 정밀 검증: 4시간봉 · 저변동 구간 · 직전봉 범위 1배 하락 시 지정가 매수 → 종가 청산.
대조군 3종(조건봉 시가진입 / 무작위봉 같은x / 조건 시간회전), 연도별, 워크포워드, 비용 민감도, 청산 변형.
사용: python research/intrabar_v.py"""
import sys, warnings
from pathlib import Path
import numba, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R))
from freesia import store
from intrabar110 import run
TE = pd.Timestamp("2024-01-01", tz="UTC")
rng = np.random.default_rng(173)
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
B = 240

for sym in ("BTCUSDT", "ETHUSDT"):
    k = store.load("klines_1m", sym).set_index("ts")
    o, h, l, c = k.open.values, k.high.values, k.low.values, k.close.values
    idx = k.index; nb = len(c) // B
    bo = o[::B][:nb]; bc = c[B - 1::B][:nb]
    bh = np.array([h[i * B:(i + 1) * B].max() for i in range(nb)])
    bl = np.array([l[i * B:(i + 1) * B].min() for i in range(nb)])
    bi = idx[::B][:nb]
    s = pd.Series
    rng_prev = s((bh - bl) / bo).shift(1)
    rv = s(np.log(bc)).diff().rolling(24).std()
    P("=" * 96 + f"\n### {sym}  4시간봉 {nb}개")
    for qv, qn in ((.2, "하위20%"), (.3, "하위30%"), (.5, "하위50%"), (1.1, "전체")):
        for mult, mn in ((0.75, "0.75배"), (1.0, "1.0배"), (1.25, "1.25배")):
            xa = (rng_prev * mult).values
            cd = (rv < rv.rolling(720).quantile(qv)).fillna(False).values if qv < 1 else np.ones(nb, bool)
            ret, oc, xs, tm = run(o, h, l, c, B, xa, cd, 0)
            ok = ~np.isnan(ret)
            if ok.sum() < 100:
                continue
            a = ok & (bi < TE); b_ = ok & (bi >= TE)
            if a.sum() < 60 or b_.sum() < 40:
                continue
            P(f"  변동성{qn:7s} 깊이{mn:6s} | 학습 {ret[a].mean()-7:7.2f}bp(n={a.sum():5d}) | "
              f"테스트 {ret[b_].mean()-7:7.2f}bp(n={b_.sum():5d}, 주{b_.sum()/((bi[-1]-TE).days/7):.2f}) | "
              f"x평균 {xs[b_].mean():6.1f}bp 승률 {float((ret[b_]>0).mean()):.3f}")
    # 채택 후보 고정
    xa = (rng_prev * 1.0).values
    cd = (rv < rv.rolling(720).quantile(.3)).fillna(False).values
    ret, oc, xs, tm = run(o, h, l, c, B, xa, cd, 0)
    ok = ~np.isnan(ret); te = ok & (bi >= TE); trn = ok & (bi < TE)
    P(f"\n[후보] 저변동 하위30% + 직전범위 1.0배 하락")
    P(f"  대조군1 같은 조건 봉에 시가 진입→종가 청산: 학습 {oc[trn].mean()-10:.2f}bp / 테스트 {oc[te].mean()-10:.2f}bp (테이커 왕복 10bp)")
    # 대조군2: 무작위 봉에서 같은 x 분포로 진입
    pls = []
    cand = np.where((bi >= TE) & ~np.isnan(xa))[0]
    for _ in range(300):
        pick = rng.choice(cand, te.sum(), replace=False)
        cd2 = np.zeros(nb, bool); cd2[pick] = True
        r2, o2, x2, t2 = run(o, h, l, c, B, xa, cd2, 0)
        m2 = ~np.isnan(r2)
        if m2.sum() > 30:
            pls.append(r2[m2].mean())
    P(f"  대조군2 무작위 봉·같은 깊이 300회: 평균 {np.mean(pls):.2f}bp 95% {np.percentile(pls,95):.2f} 최대 {np.max(pls):.2f} | 실제 {ret[te].mean():.2f} → 백분위 {(np.array(pls)<ret[te].mean()).mean()*100:.1f}%")
    # 연도별
    rows = []
    for y in range(2020, 2027):
        m = ok & (bi.year == y)
        if m.sum() < 20:
            continue
        rows.append((y, ret[m].mean(), ret[m].mean() - 7, float((ret[m] > 0).mean()), oc[m].mean(),
                     xs[m].mean(), int(m.sum()), (ret[m].mean() - 7) * m.sum() / 100))
    P("\n[연도별]")
    P(pd.DataFrame(rows, columns=["연", "총bp", "메이커순bp", "승률", "시가→종가bp", "x bp", "거래", "연누적%p"]).round(2).to_string(index=False))
    # 워크포워드
    P("\n[앵커드 워크포워드] 매년 직전까지 학습해 (변동성분위, 깊이배수) 선택 → 그 해 평가")
    wf = []
    for y in range(2022, 2027):
        sD = pd.Timestamp(f"{y}-01-01", tz="UTC"); eD = pd.Timestamp(f"{y+1}-01-01", tz="UTC")
        best = None
        for qv in (.2, .3, .5, 1.1):
            for mult in (0.75, 1.0, 1.25):
                xa2 = (rng_prev * mult).values
                cd2 = (rv < rv.rolling(720).quantile(qv)).fillna(False).values if qv < 1 else np.ones(nb, bool)
                r2, o2, x2, t2 = run(o, h, l, c, B, xa2, cd2, 0)
                m2 = ~np.isnan(r2) & (bi < sD)
                if m2.sum() < 80:
                    continue
                v = r2[m2].mean() - 7
                if best is None or v > best[0]:
                    best = (v, qv, mult, r2, ~np.isnan(r2))
        if best is None:
            continue
        _, qv, mult, r2, ok2 = best
        m3 = ok2 & (bi >= sD) & (bi < eD)
        if m3.sum() < 10:
            continue
        wf.append((y, f"변동{qv}", f"{mult}배", r2[m3].mean() - 7, float((r2[m3] > 0).mean()), int(m3.sum())))
    P(pd.DataFrame(wf, columns=["연", "선택 변동성", "선택 깊이", "그해 메이커순bp", "승률", "거래"]).round(2).to_string(index=False))
    # 비용·청산 변형
    P("\n[비용 민감도 & 청산 규칙] 테스트 구간")
    for em, en in ((0, "종가 청산"), (1, "시가 복귀 시"), (2, "목표 2x"), (3, "봉 절반")):
        r3, o3, x3, t3 = run(o, h, l, c, B, xa, cd, em)
        m3 = ~np.isnan(r3) & (bi >= TE)
        if m3.sum() < 30:
            continue
        P(f"  {en:12s} 총 {r3[m3].mean():7.2f}bp | 메이커7bp {r3[m3].mean()-7:7.2f} | 테이커12bp {r3[m3].mean()-12:7.2f} | 슬리피지+5bp {r3[m3].mean()-12:7.2f} | 승률 {float((r3[m3]>0).mean()):.3f}")
