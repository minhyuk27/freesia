"""R4-A 극단 레버리지 운용 기준 — 이론과 실측으로 상한을 못박는다.

질문: 월 +80%(연 +115,600%, 자본 1,157배/년)에 필요한 레버리지는 얼마이고,
      그 레버리지에서 자본은 어떻게 되는가.

핵심 수식:
  거래당 로그성장 g(L) = p·ln(1 + L·up − c) + q·ln(1 − L·dn − c)
  단 L·dn ≥ 1/L − mmr 이면 청산이 먼저 오므로 손실은 −1 (증거금 전액)
  연 N건이면 연 성장 = exp(N·g(L)) − 1
  월 +80% 필요조건: N·g(L) = ln(1.8)·12 = 7.054

이 파일은 전략이 아니라 '기준표'를 만든다."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R/"research"))


def P(*a, end="\n"):
    print(" ".join(str(x) for x in a), end=end, flush=True)


MMR = 0.004
TARGET = np.log(1.8)*12          # 월 +80%의 연 로그성장
P(f"월 +80% = 연 {np.exp(TARGET)-1:,.0f}배 = 연 로그성장 {TARGET:.3f}\n")


def g_trade(p, up, dn, L, cost_bp=10.0, mmr=MMR):
    """거래당 기대 로그성장. 청산이 손절보다 먼저면 손실은 증거금 전액."""
    c = cost_bp/1e4
    liq = 1.0/L - mmr
    eff_dn = min(dn, max(liq, 0.0))
    win = 1.0 + L*(up - c)
    lose = 1.0 - L*(eff_dn + c)
    if win <= 0 or lose <= 0:
        return -np.inf, eff_dn
    return p*np.log(win) + (1-p)*np.log(lose), eff_dn


def best_L(p, up, dn, cost_bp=10.0, Lmax=200.0):
    Ls = np.arange(0.1, Lmax, 0.1)
    gs = np.array([g_trade(p, up, dn, L, cost_bp)[0] for L in Ls])
    i = int(np.nanargmax(gs))
    return Ls[i], gs[i]


P("="*126); P("[1] 실측 구성별 켈리 최적 명목과 그때의 연 성장"); P("="*126)
# (이름, 승률, 익절, 손절, 연거래수, 비용bp)
CFG = [
    ("z2.4 시장가 ±(1.5/3.0)", 0.6789, 0.015, 0.030, 194, 10),
    ("z2.6 지정가 ±(3.0/3.0)", 0.5079, 0.030, 0.030, 273, 7),
    ("z2.3 대칭 ±1.5%", 0.5274, 0.015, 0.015, 187, 10),
    ("z1.4 초단기 딥(참고)", 0.634, 0.0035, 0.0035, 332, 7),
    ("z0.3 깊은 딥(참고)", 0.906, 0.0707, 0.0707, 17, 10),
    ("가상: 승률70% ±1%", 0.70, 0.01, 0.01, 1000, 7),
    ("가상: 승률80% ±0.5%", 0.80, 0.005, 0.005, 2000, 4),
    ("가상: 승률90% ±0.3%", 0.90, 0.003, 0.003, 3000, 4),
]
P(f"  {'구성':>24s} {'승률%':>6s} {'익/손%':>11s} {'연거래':>6s} {'켈리L':>7s} {'거래당g':>10s} "
  f"{'연성장%':>12s} {'월%':>8s}")
for nm, p, up, dn, N, cb in CFG:
    L, g = best_L(p, up, dn, cb)
    ann = np.exp(N*g)-1
    mon = (1+ann)**(1/12)-1 if ann > -1 else -1
    P(f"  {nm:>24s} {p*100:6.2f} {up*100:5.2f}/{dn*100:5.2f} {N:6d} {L:6.1f}x {g:10.5f} "
      f"{ann*100:12.1f} {mon*100:8.2f}")

P("\n"+"="*126); P("[2] 월 +80%에 필요한 거래당 로그성장 — 빈도별"); P("="*126)
P(f"  {'연거래':>8s} {'필요 거래당 g':>14s} {'필요 거래당 순수익%':>20s} {'해석':>40s}")
for N in (100, 250, 500, 1000, 2000, 5000, 10000, 50000):
    g = TARGET/N
    r = np.exp(g)-1
    note = ""
    if r > 0.05: note = "비현실 — 거래당 5% 초과"
    elif r > 0.01: note = "z0.3급 사건을 매번"
    elif r > 0.005: note = "매우 어려움"
    else: note = "레버리지로 도달 시도 가능"
    P(f"  {N:8d} {g:14.5f} {r*100:20.3f} {note:>40s}")

P("\n"+"="*126); P("[3] 실측 엣지(거래당 원시 bp)로 월 +80%에 필요한 명목 배수"); P("="*126)
P(f"  {'거래당 원시bp':>12s} " + " ".join(f"{'연'+str(N)+'건':>13s}" for N in (250, 500, 1000, 2000, 5000)))
for raw in (4, 8, 14, 25, 50, 100, 200):
    row = []
    for N in (250, 500, 1000, 2000, 5000):
        need = np.exp(TARGET/N)-1
        L = need/(raw/1e4)
        row.append(f"{L:11.1f}x")
    P(f"  {raw:12d} " + " ".join(f"{x:>13s}" for x in row))
P("  (단순 산술: 필요 거래당 수익 ÷ 원시 엣지. 복리·청산·켈리 무시한 하한 추정)")

P("\n"+"="*126); P("[4] 명목을 올리면 실제로 어떻게 되나 — z2.4 실측값으로"); P("="*126)
p, up, dn, N, cb = 0.6789, 0.015, 0.030, 194, 10
P(f"  승률 {p*100:.2f}%, 익절 {up*100:.1f}%, 손절 {dn*100:.1f}%, 연 {N}건, 비용 {cb}bp")
P(f"\n  {'명목':>7s} {'청산거리':>9s} {'실효손절':>9s} {'거래당g':>11s} {'연성장%':>14s} {'월%':>9s} {'상태':>14s}")
for Lx in (1, 2, 3, 4, 5, 8, 12, 20, 30, 50, 75, 100, 150):
    g, eff = g_trade(p, up, dn, Lx, cb)
    liq = 1.0/Lx - MMR
    if not np.isfinite(g):
        P(f"  {Lx:6d}x {liq*100:8.2f}% {eff*100:8.2f}% {'−∞':>11s} {'즉시파산':>14s} {'':>9s} {'1회에 전손':>14s}")
        continue
    ann = np.exp(N*g)-1
    mon = (1+ann)**(1/12)-1 if ann > -1 else -1
    st = "성장" if g > 0 else "붕괴"
    P(f"  {Lx:6d}x {liq*100:8.2f}% {eff*100:8.2f}% {g:11.5f} {ann*100:14.1f} {mon*100:9.2f} {st:>14s}")

P("\n"+"="*126); P("[5] 파산 확률 — 연속 손실에 의한 자본 소멸"); P("="*126)
P("  승률 p, 명목 L, 손절 dn일 때 자본이 x배로 줄어드는 연속손실 횟수와 그 확률")
P(f"\n  {'명목':>7s} {'1회 손실%':>10s} {'−90% 필요 연속':>15s} {'그 확률':>12s} {'연 1회 이상 발생확률':>20s}")
for Lx in (2, 3, 5, 8, 12, 20, 30, 50):
    loss = min(Lx*dn, 1.0)
    if loss >= 1.0:
        P(f"  {Lx:6d}x {loss*100:9.1f}% {'1회':>15s} {(1-p)*100:11.2f}% {'거의 확실':>20s}")
        continue
    k = int(np.ceil(np.log(0.1)/np.log(1-loss)))
    pk = (1-p)**k
    # 연 N건 중 길이 k 연속손실이 한 번이라도 날 확률(근사)
    pany = 1-(1-pk)**max(N-k, 1)
    P(f"  {Lx:6d}x {loss*100:9.1f}% {k:13d}회 {pk*100:11.4f}% {pany*100:19.2f}%")

P("\n"+"="*126); P("[6] 결론 — 각 빈도에서 도달 가능한 최대 월수익 (켈리 최적 사용)"); P("="*126)
P(f"  {'구성':>24s} {'켈리L':>7s} {'연성장%':>14s} {'월평균%':>10s} {'월 80% 대비':>12s}")
for nm, p_, up_, dn_, N_, cb_ in CFG:
    L, g = best_L(p_, up_, dn_, cb_)
    ann = np.exp(N_*g)-1
    mon = (1+ann)**(1/12)-1 if ann > -1 else -1
    ratio = (1+mon)/(1.8)
    P(f"  {nm:>24s} {L:6.1f}x {ann*100:14.1f} {mon*100:10.2f} {ratio:11.3f}배")

P("\n"+"="*126); P("[7] 월 +80%를 만드는 조합 역산 — 어떤 (승률, 빈도)면 가능한가"); P("="*126)
P("  익절=손절=1%, 비용 7bp, 켈리 최적 명목 사용 시 도달하는 월수익")
P(f"  {'승률%':>6s} " + " ".join(f"{'연'+str(N)+'건':>13s}" for N in (250, 500, 1000, 2000, 5000)))
for pw in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
    row = []
    L, g = best_L(pw, 0.01, 0.01, 7.0)
    for N in (250, 500, 1000, 2000, 5000):
        ann = np.exp(N*g)-1
        mon = (1+ann)**(1/12)-1 if ann > -1 else -1
        tag = "★" if mon >= 0.8 else " "
        row.append(f"{mon*100:11.1f}%{tag}")
    P(f"  {pw*100:6.1f} " + " ".join(f"{x:>13s}" for x in row) + f"   (켈리 {L:.1f}x)")
P("\n  ★ = 월 +80% 달성. 승률과 빈도가 동시에 얼마나 필요한지를 보여준다.")
