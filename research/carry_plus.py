"""캐리 심화 연구 (Binance BTC·ETH만). 사용자 제약: 다른 거래소·알트·옵션 제외.
1 펀딩 부호 구조  2 역캐리(펀딩 음수 구간)  3 분기 선물 베이시스 캐리  4 무기한+분기 결합  5 캐리 운영 최적화
사용: python research/carry_plus.py"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from freesia import store  # noqa: E402

TE = pd.Timestamp("2024-01-01", tz="UTC")
COST = 0.0006
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)


def stats(x, ppy=365):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 5 or x.std() == 0:
        return 0, 0, 0
    eq = np.cumprod(1 + x)
    return x.mean() * ppy, x.mean() / x.std() * np.sqrt(ppy), float((eq / np.maximum.accumulate(eq) - 1).min())


CAR = {}
for sym in ("BTCUSDT", "ETHUSDT"):
    f = store.load("funding", sym).set_index("ts").rate
    pr = store.load("premium_1m", sym).set_index("ts").close
    prem = pr.reindex(f.index, method="nearest", tolerance=pd.Timedelta("5min"))
    carry = (f - prem.diff()).dropna()
    CAR[sym] = (f, carry)
    P("=" * 92 + f"\n### {sym}")

    # 1. 펀딩 부호 구조
    P("\n[1] 펀딩 부호 구조 (8시간 정산 기준)")
    neg = f < 0
    runs = (neg != neg.shift()).cumsum()
    negruns = neg.groupby(runs).sum()
    negruns = negruns[negruns > 0]
    P(f"  전체 {len(f):,}회 중 음수 {neg.sum():,}회 ({neg.mean():.1%}) | 음수 연속 구간 {len(negruns)}개, 중앙 {negruns.median():.0f}회(={negruns.median()*8:.0f}시간), 최장 {negruns.max():.0f}회(={negruns.max()*8/24:.1f}일)")
    P(f"  양수일 때 평균 {f[~neg].mean()*3*365*100:.1f}%/년, 음수일 때 평균 {f[neg].mean()*3*365*100:.1f}%/년")
    rows = []
    for y in range(2020, 2027):
        m = f.index.year == y
        if m.sum() < 50:
            continue
        rows.append((y, float(neg[m].mean()), f[m].mean() * 3 * 365 * 100, f[m][neg[m]].mean() * 3 * 365 * 100 if neg[m].any() else np.nan))
    P(pd.DataFrame(rows, columns=["연", "음수비율", "평균 연%", "음수구간 평균 연%"]).round(3).to_string(index=False))

    # 2. 역캐리 (현물 숏 + 무기한 롱) — 차입 비용 반영
    P("\n[2] 역캐리 — 펀딩 음수 구간에서만 (현물 숏 + 무기한 롱). 현물 차입 비용 가정별 연 수익")
    rev = (-f + prem.diff()).dropna()       # 역캐리 8시간 수익 (부호 반대)
    sig_neg = (f < 0).reindex(rev.index).fillna(False)
    ppy = 3 * 365
    rows = []
    for borrow in (0.0, 0.05, 0.10, 0.20):     # 연 차입 이자
        bh = borrow / ppy
        net = np.where(sig_neg, rev - bh, 0.0)
        # 전환 비용
        turn = np.abs(np.diff(sig_neg.astype(float), prepend=0.0))
        net = net - turn * 2 * COST
        for nm, m in (("학습", rev.index < TE), ("테스트", rev.index >= TE)):
            a, sh, dd = stats(net[m], ppy)
            rows.append((borrow * 100, nm, a * 100, sh, dd * 100, float(sig_neg[m].mean())))
    P(pd.DataFrame(rows, columns=["차입연%", "구간", "연%", "Sharpe", "MDD%", "가동률"]).round(2).to_string(index=False))

    # 5-a. 캐리 수익 분해
    P("\n[5] 캐리 수익 분해 (연 환산 %)")
    for nm, m in (("학습", carry.index < TE), ("테스트", carry.index >= TE)):
        fm = f.reindex(carry.index)[m]
        dpm = prem.diff().reindex(carry.index)[m]
        P(f"  {nm}: 펀딩 수취 {fm.mean()*ppy*100:6.2f}% − 프리미엄 변동 {dpm.mean()*ppy*100:6.2f}% = 캐리 {carry[m].mean()*ppy*100:6.2f}%")
    # 음수 펀딩 구간이 캐리에 미치는 영향
    negm = (f.reindex(carry.index) < 0)
    P(f"  음수 펀딩 구간({negm.mean():.1%}) 제외 시 캐리: 전체 {carry.mean()*ppy*100:.2f}% → 제외 {carry[~negm].mean()*ppy*100:.2f}% (전환비용 미반영)")

# 3. 분기 선물 베이시스 캐리
P("\n" + "=" * 92 + "\n[3] 분기 선물 베이시스 캐리 (현물 롱 + 분기 선물 숏, 만기 보유)")
BAS = {}
for sym, tag in (("BTCUSDT", "btc"), ("ETHUSDT", "eth")):
    bs = pd.read_parquet(R / "data" / "ext" / f"basis_{tag}.parquet")
    d = bs.basis_ann.dropna()
    P(f"\n  {tag.upper()} 연환산 베이시스: 평균 {d.mean()*100:.2f}% 중앙 {d.median()*100:.2f}% | 음수 비율 {(d<0).mean():.1%} | 최대 {d.max()*100:.1f}%")
    rows = []
    for y in range(2021, 2027):
        m = d.index.year == y
        if m.sum() < 30:
            continue
        rows.append((y, d[m].mean() * 100, d[m].median() * 100, float((d[m] < 0).mean())))
    P(pd.DataFrame(rows, columns=["연", "평균 연%", "중앙 연%", "음수비율"]).round(2).to_string(index=False))
    # 일별 수익 근사: 베이시스는 만기까지 수렴 → 일 수익 ≈ basis_ann/365 (보유 시)
    daily = (d / 365).reindex(pd.date_range(d.index[0], d.index[-1], freq="D", tz="UTC")).ffill()
    # 롤 비용: 분기 4회 × 왕복 2다리
    roll = pd.Series(0.0, index=daily.index)
    dte = bs.dte.reindex(daily.index).ffill()
    rollday = (dte.diff() > 0)
    roll[rollday] = 4 * COST
    net = (daily - roll).fillna(0)
    for nm, m in (("학습", net.index < TE), ("테스트", net.index >= TE)):
        a, sh, dd = stats(net[m].values)
        P(f"    {nm}: 연 {a*100:.2f}% Sharpe {sh:.2f} MDD {dd*100:.2f}% (롤 비용 반영, 베이시스 변동 위험 미반영)")
    BAS[sym] = net

# 4. 결합
P("\n" + "=" * 92 + "\n[4] 무기한 캐리 + 분기 캐리 결합 (일 단위)")
D = {}
for sym in ("BTCUSDT", "ETHUSDT"):
    D[f"무기한{sym[:3]}"] = CAR[sym][1].resample("1D").sum()
    D[f"분기{sym[:3]}"] = BAS[sym]
idx = None
for v in D.values():
    idx = v.index if idx is None else idx.intersection(v.index)
M = pd.DataFrame({k: v.reindex(idx) for k, v in D.items()}).fillna(0)
P("\n일별 수익 상관")
P(M.corr().round(3).to_string())
te = idx >= TE
P("\n구성별 (증거금 배수 1.5 가정: 무기한 명목 0.67, 분기 명목 0.67)")
for nm, w in (("무기한만", {"무기한BTC": .335, "무기한ETH": .335}),
              ("분기만", {"분기BTC": .335, "분기ETH": .335}),
              ("무기한50+분기50", {"무기한BTC": .167, "무기한ETH": .167, "분기BTC": .167, "분기ETH": .167}),
              ("BTC만 무기한+분기", {"무기한BTC": .335, "분기BTC": .335})):
    x = sum(M[k] * v for k, v in w.items()).values
    a1, s1, d1 = stats(x[~te]); a2, s2, d2 = stats(x[te])
    P(f"  {nm:20s} 학습 연{a1*100:6.2f}% Sh {s1:6.2f} MDD {d1*100:6.2f}% | 테스트 연{a2*100:6.2f}% Sh {s2:6.2f} MDD {d2*100:6.2f}%")
x = sum(M[k] * v for k, v in {"무기한BTC": .167, "무기한ETH": .167, "분기BTC": .167, "분기ETH": .167}.items())
P("\n결합(무기한50+분기50) 기간별 승률")
for rule, nm in (("YE", "연"), ("QE", "분기"), ("ME", "월"), ("W", "주"), ("D", "일")):
    g = x.resample(rule).sum()
    P(f"  {nm}: {float((g>0).mean()):.3f} (n={len(g)})")
