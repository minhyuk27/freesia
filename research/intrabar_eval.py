"""후보 전략 실효성 종합 평가 — 체결 현실성·정밀 위약·월별/거래당 수익·낙폭·자본효율·스트레스·감쇠·결합.
전략: 4시간봉, 저변동(하위 q), 직전봉 범위 m배 하락 시 지정가 매수 → 종가 청산.
사용: python research/intrabar_eval.py"""
import sys, warnings
from pathlib import Path
import numba, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R))
from freesia import store
TE = pd.Timestamp("2024-01-01", tz="UTC")
rng = np.random.default_rng(197)
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
B = 240


@numba.njit(cache=True)
def run2(o, h, l, c, B, xarr, cond, buf):
    """buf: 체결 인정에 필요한 추가 하락(bp). 0이면 터치 즉시 체결 가정."""
    nb = len(c) // B
    ret = np.full(nb, np.nan); oc = np.full(nb, np.nan); xs = np.full(nb, np.nan)
    tin = np.zeros(nb, np.int32); hold = np.zeros(nb, np.int32)
    for b in range(nb):
        if not cond[b] or np.isnan(xarr[b]) or xarr[b] <= 0:
            continue
        s = b * B; e = s + B
        O = o[s]; C = c[e - 1]; x = xarr[b]
        trig = O * (1.0 - x)
        fill = trig * (1.0 - buf)
        j = -1
        for t in range(s, e):
            if l[t] <= fill:
                j = t; break
        if j < 0:
            continue
        ret[b] = (C / trig - 1.0) * 1e4
        oc[b] = (C / O - 1.0) * 1e4
        xs[b] = x * 1e4; tin[b] = j - s; hold[b] = e - 1 - j
    return ret, oc, xs, tin, hold


def load(sym):
    k = store.load("klines_1m", sym).set_index("ts")
    o, h, l, c = k.open.values, k.high.values, k.low.values, k.close.values
    idx = k.index; nb = len(c) // B
    bo = o[::B][:nb]; bc = c[B - 1::B][:nb]
    bh = np.array([h[i * B:(i + 1) * B].max() for i in range(nb)])
    bl = np.array([l[i * B:(i + 1) * B].min() for i in range(nb)])
    bi = idx[::B][:nb]
    s = pd.Series
    rp = s((bh - bl) / bo).shift(1)
    rv = s(np.log(bc)).diff().rolling(24).std()
    return o, h, l, c, bi, nb, rp, rv


CFG = {"BTCUSDT": (0.3, 1.0), "ETHUSDT": (0.3, 1.0)}
SER = {}
for sym, (qv, mult) in CFG.items():
    o, h, l, c, bi, nb, rp, rv = load(sym)
    xa = (rp * mult).values
    cd = (rv < rv.rolling(720).quantile(qv)).fillna(False).values
    P("=" * 98 + f"\n### {sym}  (저변동 하위{int(qv*100)}% · 직전범위 {mult}배)")

    # 1. 체결 현실성
    P("\n[1] 체결 가정 민감도 — 지정가보다 추가로 내려가야 체결로 인정")
    rows = []
    for buf, bn in ((0.0, "터치 즉시"), (0.0002, "2bp 더"), (0.0005, "5bp 더"), (0.0010, "10bp 더")):
        r, oc, xs, tin, hd = run2(o, h, l, c, B, xa, cd, buf)
        ok = ~np.isnan(r); te = ok & (bi >= TE)
        rows.append((bn, r[ok & (bi < TE)].mean() - 7, r[te].mean() - 7, int(te.sum()),
                     te.sum() / ((bi[-1] - TE).days / 7)))
    P(pd.DataFrame(rows, columns=["체결 조건", "학습 메이커순bp", "테스트 메이커순bp", "테스트 거래", "주당"]).round(2).to_string(index=False))

    r, oc, xs, tin, hd = run2(o, h, l, c, B, xa, cd, 0.0)
    ok = ~np.isnan(r); te = ok & (bi >= TE); trn = ok & (bi < TE)
    net = r - 7

    # 2. 정밀 위약 — 저변동 조건만 시간 회전
    P("\n[2] 정밀 위약 — 깊이·봉은 그대로, 저변동 조건만 시간 회전 300회")
    pls = []
    for _ in range(300):
        cd2 = np.roll(cd, rng.integers(200, nb - 200))
        r2, _, _, _, _ = run2(o, h, l, c, B, xa, cd2, 0.0)
        m2 = ~np.isnan(r2) & (bi >= TE)
        if m2.sum() > 50:
            pls.append(r2[m2].mean() - 7)
    pls = np.array(pls)
    P(f"  위약: 평균 {pls.mean():.2f}bp 95% {np.percentile(pls,95):.2f} 99% {np.percentile(pls,99):.2f} 최대 {pls.max():.2f} | 실제 {net[te].mean():.2f} → 백분위 {(pls < net[te].mean()).mean()*100:.1f}%")

    # 3. 거래당 수익 분포
    P("\n[3] 거래당 순수익 분포 (메이커 7bp 차감, bp)")
    for nm, m in (("학습", trn), ("테스트", te)):
        x = net[m]
        P(f"  {nm}: 평균 {x.mean():6.2f} 중앙 {np.median(x):6.2f} | 25% {np.percentile(x,25):7.2f} 75% {np.percentile(x,75):6.2f} | "
          f"최악 {x.min():8.2f} 최고 {x.max():7.2f} | 표준편차 {x.std():6.2f} | 승률 {float((x>0).mean()):.3f} | t {x.mean()/x.std()*np.sqrt(len(x)):.2f}")
    P(f"  보유시간 중앙 {np.median(hd[te]):.0f}분 / 평균 {hd[te].mean():.0f}분 | 진입까지 중앙 {np.median(tin[te]):.0f}분")

    # 4. 월별
    ser = pd.Series(net[ok], index=bi[ok])
    SER[sym] = ser
    mon = ser.resample("ME").sum() / 100        # %
    cnt = ser.resample("ME").count()
    P("\n[4] 월별 수익 (명목 1배, %)")
    for nm, m in (("학습", mon.index < TE), ("테스트", mon.index >= TE)):
        mm = mon[m]
        P(f"  {nm}: 월평균 {mm.mean():+.2f}% 중앙 {mm.median():+.2f}% | 월 승률 {float((mm>0).mean()):.3f} | "
          f"최악 {mm.min():+.2f}% 최고 {mm.max():+.2f}% | 표준편차 {mm.std():.2f}% | 월평균 거래 {cnt[m].mean():.1f}회")
    P(f"  연 환산(테스트): {mon[mon.index>=TE].mean()*12:+.2f}%")
    # 최근 12개월
    P(f"  최근 12개월: {mon.tail(12).mean():+.2f}%/월, 양수 {int((mon.tail(12)>0).sum())}/12")

    # 5. 낙폭·연속손실
    eq = (1 + ser / 1e4).cumprod()
    dd = (eq / eq.cummax() - 1)
    P(f"\n[5] 거래 시퀀스 누적: 전체 {(eq.iloc[-1]-1)*100:+.1f}% | MDD {dd.min()*100:.2f}%")
    eqte = (1 + ser[ser.index >= TE] / 1e4).cumprod()
    P(f"  테스트 구간: {(eqte.iloc[-1]-1)*100:+.1f}% | MDD {(eqte/eqte.cummax()-1).min()*100:.2f}%")
    lo = (ser < 0).astype(int)
    run_ = lo * (lo.groupby((lo != lo.shift()).cumsum()).cumcount() + 1)
    P(f"  최대 연속 손실 {int(run_.max())}회 | 월 단위 최대 연속 손실 {int((mon<0).astype(int).groupby(((mon<0)!=(mon<0).shift()).cumsum()).cumsum().max())}개월")

    # 6. 자본 효율
    P(f"\n[6] 자본 효율: 시장 노출 시간 {hd[ok].sum()/(len(bi)*B)*100:.2f}% | 동시 포지션 최대 1개(봉당 1회)")
    P(f"  노출 시간 대비 수익률(연): {mon[mon.index>=TE].mean()*12 / max(hd[te].sum()/((bi[-1]-TE).days*1440), 1e-9):.0f}% (이론 상한, 실제로는 대기 자본 필요)")

    # 7. 스트레스 구간
    P("\n[7] 스트레스 구간")
    for nm, a, b2 in (("2020-03 급락", "2020-03-01", "2020-04-01"), ("2021-05 붕괴", "2021-05-01", "2021-06-01"),
                      ("2022 하락장", "2022-01-01", "2023-01-01"), ("2024-08 엔캐리", "2024-08-01", "2024-09-01"),
                      ("2025-10 청산연쇄", "2025-10-01", "2025-11-01")):
        m = (ser.index >= pd.Timestamp(a, tz="UTC")) & (ser.index < pd.Timestamp(b2, tz="UTC"))
        if m.sum() > 0:
            P(f"  {nm:14s} 거래 {int(m.sum()):3d}회 합계 {ser[m].sum()/100:+7.2f}% 평균 {ser[m].mean():+7.2f}bp 승률 {float((ser[m]>0).mean()):.3f}")

    # 8. 감쇠
    P("\n[8] 반기별 추이 (감쇠 점검)")
    half = ser.resample("2QE").agg(["mean", "count"])
    P("  " + " | ".join(f"{i.strftime('%y-%m')} {r['mean']:.1f}bp({int(r['count'])})" for i, r in half.tail(8).iterrows()))

P("\n" + "=" * 98 + "\n[9] 두 종목 결합")
idx = SER["BTCUSDT"].index.union(SER["ETHUSDT"].index)
A = pd.DataFrame({k: v.reindex(idx) for k, v in SER.items()})
P(f"  동시 진입(같은 4시간봉) {int((A.notna().sum(axis=1) == 2).sum())}회 / BTC {int(A.BTCUSDT.notna().sum())} ETH {int(A.ETHUSDT.notna().sum())}")
P(f"  거래 수익 상관(동시 진입분): {A.dropna().corr().iloc[0,1]:.3f}")
comb = A.fillna(0).sum(axis=1) / 2          # 각 50% 명목
mon = comb.resample("ME").sum() / 100
P(f"  결합(각 50% 명목) 월평균: 학습 {mon[mon.index<TE].mean():+.2f}% / 테스트 {mon[mon.index>=TE].mean():+.2f}% | 월 승률 {float((mon[mon.index>=TE]>0).mean()):.3f}")
P(f"  결합 연 환산(테스트) {mon[mon.index>=TE].mean()*12:+.2f}% | 월 최악 {mon[mon.index>=TE].min():+.2f}%")
eq = (1 + comb / 1e4).cumprod()
P(f"  결합 누적 {(eq.iloc[-1]-1)*100:+.1f}% | MDD {(eq/eq.cummax()-1).min()*100:.2f}%")
