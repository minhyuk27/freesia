"""프리미엄 증폭기 검증 — 임계는 확장창 분위수(룩어헤드 없음). 민감도·워크포워드·3단 성과.
사용: python research/amp_valid.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
import exit_tier as X
from absfloor import met
from anchor import build_a
from freesia import store
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
TE = pd.Timestamp("2024-01-01", tz="UTC"); REC = pd.Timestamp("2025-01-01", tz="UTC")
SYM = ("BTCUSDT", "ETHUSDT")
Z = build_a("open", ext=0.10)

PM = {}
for s in SYM:
    pr = store.load("premium_1m", s).set_index("ts").close
    PM[s] = pr
P(f"z0.3 {len(Z)}건 | 프리미엄 데이터 로드 완료")

# 각 진입의 직전 분 프리미엄 + 그 시점까지의 확장창 분위 (룩어헤드 없음)
rec = []
for a, b, r, m, ts, key, rv, wg in Z:
    s = key[0]; ser = PM[s]
    i = ser.index.searchsorted(ts)
    if i <= 1: rec.append(np.nan); continue
    pv = float(ser.iloc[i-1])
    past = ser.iloc[max(0, i-1-525600):i-1]          # 직전 1년치로 분위 계산
    rec.append(float((past < pv).mean()) if len(past) > 10000 else np.nan)
pq = np.array(rec)
P(f"진입 시점 프리미엄 분위: 중앙 {np.nanmedian(pq):.3f} | 하위10% 이하인 진입 {int(np.nansum(pq<=0.10))}건 "
  f"/ 하위20% {int(np.nansum(pq<=0.20))}건 / 하위5% {int(np.nansum(pq<=0.05))}건")

pr_arr = np.array([x[2] for x in Z])
P("\n[1] 분위 구간별 z0.3 거래 성과 (확장창 분위, 룩어헤드 없음)")
for lo, hi in ((0, .05), (.05, .1), (.1, .2), (.2, .5), (.5, .8), (.8, 1.01)):
    sel = (pq >= lo) & (pq < hi)
    if sel.sum() < 5: continue
    P(f"  분위 {lo:.2f}~{hi:.2f}: n={int(sel.sum()):4d} 평균 {pr_arr[sel].mean()*1e4:+7.0f}bp "
      f"중앙 {np.median(pr_arr[sel])*1e4:+6.0f}bp 승률 {(pr_arr[sel]>0).mean():.3f}")


def amped(th, amp):
    E = []
    for j, (a, b, r, m, ts, key, rv, wg) in enumerate(Z):
        f = amp if (np.isfinite(pq[j]) and pq[j] <= th) else 1.0
        E.append((a, b, r, m, ts, key, rv, wg*f))
    return E


def tiers(E, lab):
    row = f"  {lab:26s} |"
    out = {}
    for w, Nm, tag in ((1.0, 1.0, "무레버"), (3.0, 5.0, "중간"), (5.0, 8.0, "공격")):
        eq, fl, liq = X.sim(E, w, Nm); m = met(eq, fl, liq)
        out[tag] = m
        row += f" {tag} {'파산' if m is None else f'{m[0]:+5.0f}%({m[2]:5.1f}%)'}"
    P(row)
    return out

P("\n[2] 임계 분위 × 증폭 배수")
for th in (0.05, 0.10, 0.20, 0.30):
    for amp in (1.0, 1.5, 2.0, 2.5):
        tiers(amped(th, amp), f"하위{th*100:.0f}% 이하 {amp:.1f}배")

P("\n[3] 선택 구성의 연도별·25년 이후")
for th, amp in ((0.10, 2.0), (0.20, 1.5), (0.10, 1.5)):
    E = amped(th, amp)
    P(f"  ── 하위{th*100:.0f}% {amp:.1f}배 ──")
    for w, Nm, tag in ((1.0, 1.0, "무레버"), (3.0, 5.0, "중간"), (5.0, 8.0, "공격")):
        eq, fl, liq = X.sim(E, w, Nm); m = met(eq, fl, liq)
        if m is None: P(f"     {tag:6s} 파산"); continue
        yr = m[5].resample("YE").last()/m[5].resample("YE").first()-1
        P(f"     {tag:6s} 연 {m[0]:+7.0f}% 최악 {m[2]:6.1f}% | 24~ {m[3]:+7.0f}% | 25~ {m[4]:+7.0f}% | "
          + " ".join(f"{i.year}{v*100:+5.0f}" for i, v in yr.items()))

P("\n[4] 워크포워드 — 증폭 규칙이 표본 밖에서도 유효한가 (하위10% 2배)")
E = amped(0.10, 2.0)
seg = lambda Ex, s, e: [x for x in Ex if (s is None or x[4] >= s) and (e is None or x[4] < e)]
for y in pd.date_range("2022-01-01", "2026-01-01", freq="YS", tz="UTC"):
    te0 = seg(Z, y, y+pd.DateOffset(years=1)); te1 = seg(E, y, y+pd.DateOffset(years=1))
    if len(te0) < 3: continue
    r0 = np.array([x[2] for x in te0])
    idx = [j for j, x in enumerate(Z) if y <= x[4] < y+pd.DateOffset(years=1)]
    namp = int(np.nansum(pq[idx] <= 0.10))
    e0, f0, l0 = X.sim(te0, 3.0, 5.0); m0 = met(e0, f0, l0)
    e1, f1, l1 = X.sim(te1, 3.0, 5.0); m1 = met(e1, f1, l1)
    P(f"  {y.year}: 거래 {len(te0)}건 중 증폭 {namp}건 | 중간구간 기본 {m0[0] if m0 else float('nan'):+7.0f}% → 증폭 {m1[0] if m1 else float('nan'):+7.0f}%")

P("\n[5] 상위 거래 제외 강건성 (하위10% 2배)")
r = np.array([x[2] for x in Z]); order = np.argsort(-r)
for drop in (0, 1, 2, 5, 10):
    keep = [j for j in range(len(Z)) if j not in set(order[:drop].tolist())]
    E2 = [(a, b, rr, m, ts, key, rv, wg*(2.0 if (np.isfinite(pq[j]) and pq[j] <= 0.10) else 1.0))
          for j, (a, b, rr, m, ts, key, rv, wg) in enumerate(Z) if j in keep]
    tiers(E2, f"상위 {drop}건 제외")
