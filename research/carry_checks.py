"""델타중립 캐리(현물 롱 + 무기한 숏) 추가 검사: 자본 배분, 표본 주기, 재조정 비용, 무위험 이자 비교, BTC+ETH 혼합, 최악 구간."""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from freesia.bt import core  # noqa: E402
import screen as S  # noqa: E402
import screen2 as S2  # noqa: E402

RF = 0.04                                                    # 무위험 이자 가정(연 4%, 스테이블코인·단기채 근사)


def carry_series(sym, tf, cap=1.2, reb_cost_yr=0.0):
    x = S2.CarryCtx(sym, tf)
    gross = x.ret * 1.2                                       # (펀딩 - Δ프리미엄), 자본 조정 전
    ppy = core.PPY[tf]
    net = gross / cap
    net = net - reb_cost_yr / ppy                             # 재조정 비용을 연율로 균등 차감
    net[0] = -0.0017 / cap                                    # 진입 비용 1회(현물 10bp+선물 5bp+슬리피지 2bp)
    return x.idx, net, ppy


def yr_table(idx, net, ppy):
    out = {}
    for Y in sorted(set(idx.year)):
        a = net[idx.year == Y]
        out[Y] = (float(np.prod(1 + a) - 1), float(a.mean() / a.std() * np.sqrt(ppy)) if a.std() > 0 else 0.0)
    return out


print("## 1) 자본 배분(증거금)에 따른 연도별 순수익 (Sharpe는 자본과 무관)")
for sym in S.SYMS:
    idx, net12, ppy = carry_series(sym, "4h", 1.2)
    print(f"{sym[:3]}:")
    for cap in (1.2, 1.5, 1.8):
        _, n, _ = carry_series(sym, "4h", cap)
        t = yr_table(idx, n, ppy)
        print(f"  자본 {cap}: " + ", ".join(f"{Y}:{r*100:+.1f}%" for Y, (r, _) in t.items()) + f" | 연평균 {np.mean([r for Y,(r,_) in t.items() if Y<2026])*100:.1f}%")
print("\n## 2) 표본 주기에 따른 연도별 Sharpe (일봉은 시가 노이즈가 평활돼 과대평가)")
for sym in S.SYMS:
    row = {}
    for tf in ("1d", "4h", "1h"):
        idx, n, ppy = carry_series(sym, tf, 1.2)
        row[tf] = yr_table(idx, n, ppy)
    for tf, t in row.items():
        print(f"  {sym[:3]} {tf}: " + ", ".join(f"{Y}:{sh:.1f}" for Y, (_, sh) in t.items()))
print("\n## 3) 재조정·운영 비용 민감도 (연 비용 %p 차감, 4시간, 자본 1.5) → 연도별 순수익 / 무위험 4% 초과?")
for sym in S.SYMS:
    for rc in (0.0, 0.01, 0.02, 0.03):
        idx, n, ppy = carry_series(sym, "4h", 1.5, rc)
        t = yr_table(idx, n, ppy)
        print(f"  {sym[:3]} 연 비용 {rc*100:.0f}%: " + ", ".join(f"{Y}:{r*100:+.1f}%" for Y, (r, _) in t.items()))
print("\n## 4) BTC+ETH 50:50 혼합 (4시간, 자본 1.5)")
a = carry_series("BTCUSDT", "4h", 1.5); b = carry_series("ETHUSDT", "4h", 1.5)
m = pd.Series(0.5 * a[1] + 0.5 * b[1], index=a[0])
t = yr_table(a[0], m.to_numpy(), a[2])
print("  " + ", ".join(f"{Y}:{r*100:+.1f}%(Sh {sh:.1f})" for Y, (r, sh) in t.items()))
full = np.prod(1 + m.to_numpy()) ** (365 / len(m) * 6 / 6) if False else None
yrs = len(m) / (6 * 365)
print(f"  전체 연환산 {((np.prod(1+m.to_numpy()))**(1/yrs)-1)*100:.1f}%, 무위험 {RF*100:.0f}% 대비 초과 {((np.prod(1+m.to_numpy()))**(1/yrs)-1-RF)*100:.1f}%p")
print("\n## 5) 펀딩·베이시스 분해 (BTC, 4시간, 자본 1.2): 연도별 펀딩 수취 vs 베이시스 손익")
x = S2.CarryCtx("BTCUSDT", "4h"); fund = x.fund / 1.2
pr = x.ret - fund
for Y in sorted(set(x.idx.year)):
    mk = x.idx.year == Y
    print(f"  {Y}: 펀딩 {fund[mk].sum()*100:+.2f}%, 베이시스(프리미엄 변동) {pr[mk].sum()*100:+.2f}%")
print("\n## 6) 최악 구간 (BTC 4시간 자본 1.2): 30일 롤링 최저, 최대 낙폭")
idx, n, ppy = carry_series("BTCUSDT", "4h", 1.2)
s = pd.Series(n, index=idx); r30 = (1 + s).rolling(180).apply(np.prod, raw=True) - 1
eq = np.cumprod(1 + n)
print(f"  30일 최저 수익 {r30.min()*100:.2f}%, 최대 낙폭 {(eq/np.maximum.accumulate(eq)-1).min()*100:.2f}%, 최악 4시간 봉 {s.min()*100:.3f}%")
