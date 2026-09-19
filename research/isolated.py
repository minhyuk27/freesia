"""격리마진 시뮬레이터 — 각 진입에 증거금을 격리. 청산 시 그 증거금만 손실.
명목 = 증거금 × L. 청산거리 = 1/L − 0.4%. 동시 포지션 다수 허용.
사용: python research/isolated.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from shared_cap import build, D, LEGS
TE = pd.Timestamp("2024-01-01", tz="UTC")
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
MMR = 0.004


def sim_iso(E, mfrac, L, maxpos=99):
    """mfrac: 진입당 증거금/자본. L: 격리 레버리지. 명목=증거금×L."""
    cap = 1.0
    open_ = []
    eqs = []; times = []
    nliq = 0; nent = 0
    dist = 1.0 / L - MMR
    if dist <= 0:
        return None, 0, 0
    for a, b, r, m, ts in E:
        done = [p for p in open_ if p[0] <= a]
        for p in done:
            cap += p[1]          # 정산액 반환(이미 손익 반영)
            open_.remove(p)
        if cap <= 1e-9:
            break
        if len(open_) >= maxpos:
            continue
        marg = cap * mfrac
        if marg <= 1e-12 or marg > cap:
            continue
        cap -= marg
        if m >= dist:
            ret_marg = 0.0      # 청산: 증거금 전액 손실
            nliq += 1
        else:
            ret_marg = marg * (1.0 + L * r)
            if ret_marg < 0:
                ret_marg = 0.0
        open_.append((b, ret_marg))
        nent += 1
        eqs.append(cap + sum(p[1] for p in open_)); times.append(ts)
    for p in open_:
        cap += p[1]
    eqs.append(cap); times.append(E[-1][4])
    return pd.Series(eqs, index=pd.DatetimeIndex(times)), nliq, nent


def rep(eq, nliq, nent, label):
    if eq is None or eq.iloc[-1] <= 1e-6:
        P(f"  {label:30s} 파산 (청산 {nliq}/{nent})"); return
    yrs = (eq.index[-1] - eq.index[0]).days / 365
    te = eq[eq.index >= TE]
    tstr = "-"
    if len(te) > 30:
        y2 = (te.index[-1] - te.index[0]).days / 365
        tstr = f"{((te.iloc[-1]/te.iloc[0])**(1/y2)-1)*100:+8.0f}%"
    P(f"  {label:30s} 전체 연 {(eq.iloc[-1]**(1/yrs)-1)*100:+9.0f}% MDD {float((eq/eq.cummax()-1).min())*100:6.1f}% | "
      f"테스트 연 {tstr} | 청산 {nliq:4d}/{nent} ({nliq/max(nent,1)*100:4.1f}%)")


E = build((5.0, 6.5, 8.0))
P(f"진입 이벤트 {len(E)}건 (분할 5.0/6.5/8.0, 1h+4h+8h × BTC·ETH)")
P("\n[1] 격리마진: 증거금 비율 × 레버리지")
for mfrac in (0.05, 0.10, 0.20, 0.30, 0.50):
    for L in (2, 3, 5, 8, 12, 20):
        eq, nl, ne = sim_iso(E, mfrac, L)
        rep(eq, nl, ne, f"증거금{mfrac*100:.0f}% × {L:2d}배 (명목 {mfrac*L:.1f}x)")

P("\n[2] 동시 포지션 수 제한 효과 (증거금 20% × 8배)")
for mp in (1, 2, 3, 5, 99):
    eq, nl, ne = sim_iso(E, 0.20, 8, maxpos=mp)
    rep(eq, nl, ne, f"동시 최대 {mp}개")

P("\n[3] 분할 설정별 최적 (증거금 20% × 5배)")
for sp in ((3.0, 4.5, 6.0), (4.0, 5.5, 7.0), (5.0, 6.5, 8.0), (6.0, 7.5, 9.0), (7.0, 8.5, 10.0)):
    E2 = build(sp)
    eq, nl, ne = sim_iso(E2, 0.20, 5)
    rep(eq, nl, ne, f"{str(sp)}")

P("\n[4] 최적 후보 상세")
for mfrac, L, sp in ((0.20, 5, (5.0, 6.5, 8.0)), (0.30, 5, (5.0, 6.5, 8.0)), (0.20, 8, (6.0, 7.5, 9.0))):
    E2 = build(sp)
    eq, nl, ne = sim_iso(E2, mfrac, L)
    if eq is None or eq.iloc[-1] <= 1e-6:
        continue
    yrs = (eq.index[-1] - eq.index[0]).days / 365
    yr = eq.resample("YE").last() / eq.resample("YE").first() - 1
    mon = eq.resample("ME").last() / eq.resample("ME").first() - 1
    l12 = mon.tail(12)
    P(f"\n  ■ 증거금{mfrac*100:.0f}% × {L}배, 분할{sp} (명목 {mfrac*L:.1f}x, 청산 {nl}/{ne})")
    P(f"    전체 {eq.iloc[-1]:.0f}배 연 {(eq.iloc[-1]**(1/yrs)-1)*100:+.0f}% MDD {float((eq/eq.cummax()-1).min())*100:.1f}%")
    P("    연도별: " + " | ".join(f"{i.year} {v*100:+.0f}%" for i, v in yr.items()))
    P(f"    최근 12개월 {(np.prod(1+l12)-1)*100:+.1f}% 월평균 {l12.mean()*100:+.2f}% 양수 {int((l12>0).sum())}/12")
    P(f"    월 승률 {float((mon>0).mean()):.3f} (n={len(mon)})")
