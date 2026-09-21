"""z4-D 최종 — 월 +80%의 구조적 경계를 확정한다.

핵심 통찰: 같은 종목의 동시 포지션은 독립 베팅이 아니다.
  켈리의 연 성장 = exp(N_독립 × g). N_독립 = 525,600분 ÷ 평균보유시간.
  거래 수를 동시 보유로 늘려도 N_독립은 늘지 않는다(전부 BTC 롱이라 완전 상관).

따라서 월수익 상한 = f(배리어). 배리어가 크면 g가 크지만 보유가 길어 N_독립이 준다.
이 절충의 최대점을 실측으로 찾는다."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R/"research"))
import z2_lib as L2
from z2_final import build_sigs, exp_z


def P(*a, end="\n"):
    print(" ".join(str(x) for x in a), end=end, flush=True)


S = "BTCUSDT"; Y = L2.YRS; MMR = L2.MMR; STEP = L2.STEP
d = L2.D[S]; idx = d["idx"]; rv = d["rv"]
O, H, Lo, C = d["o"], d["h"], d["l"], d["c"]
warm = 200000
SG = build_sigs(S)
names = list(SG.keys())
Z = {n: exp_z(np.asarray(SG[n], float)) for n in names}
mlen = min(min(len(Z[n]) for n in names), len(C)//STEP)
SIG = np.nanmean(np.vstack([Z[n][:mlen] for n in names if n.endswith("4320")]), axis=0)
MIN_Y = 365.25*1440
TARGET = np.log(1.8)*12


@numba.njit(cache=True)
def measure(o, h, l, c, sigv, th, up, dn, maxh, step, cool, start, n_end):
    nw = 0; nt = 0; sh = 0.0
    t = start
    while t < n_end - maxh - 2:
        i = t//step
        if i >= len(sigv): break
        if th > 0.0 and (not np.isfinite(sigv[i]) or sigv[i] < th): t += step; continue
        f = t+1; px = o[f]; U = px*(1.0+up); D = px*(1.0-dn)
        e = -1; win = 0
        end = min(f+maxh, n_end-1)
        for j in range(f, end+1):
            if l[j] <= D: e = j; break
            if h[j] >= U: e = j; win = 1; break
        if e < 0:
            e = end
            if c[e] > px: win = 1
        nt += 1; nw += win; sh += (e-f)
        t = e+1+cool
    return nw, nt, sh


def kelly(p, up, dn, c=0.0010):
    Lcap = 1.0/(dn+MMR)
    best = (0.0, -9e9)
    L = 0.05
    while L < Lcap:
        win = 1+L*(up-c); lose = 1-L*(dn+c)
        if lose <= 0: break
        g = p*np.log(win)+(1-p)*np.log(lose)
        if g > best[1]: best = (L, g)
        L += 0.05
    return best


P("="*130); P("[1] 배리어별 — 독립 베팅 수 × 켈리 성장 = 월수익 상한 (신호 상위10%)"); P("="*130)
TH = float(np.quantile(SIG[np.isfinite(SIG)], 1-0.10))
P(f"  {'배리어':>8s} {'승률%':>7s} {'평균보유':>9s} {'N독립/년':>9s} {'켈리L':>7s} {'거래당g':>10s} "
  f"{'연성장%':>13s} {'월%':>10s} {'목표대비':>9s}")
ROWS = []
for bar in (0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10):
    mh = min(43200, max(1440, int(bar*900000)))
    nw, nt, sh = measure(O, H, Lo, C, SIG, TH, bar, bar, mh, STEP, 0, warm, len(C))
    if nt < 30: continue
    p = nw/nt; hold = sh/nt
    Nind = MIN_Y/max(hold, 1)
    Lk, g = kelly(p, bar, bar)
    ann = np.exp(Nind*g)-1
    mon = (1+ann)**(1/12)-1 if ann > -0.999 else -1
    ROWS.append((mon, bar, p, hold, Nind, Lk, g, ann))
    P(f"  ±{bar*100:5.2f}% {p*100:7.2f} {hold:8.0f}분 {Nind:9.0f} {Lk:6.2f}x {g:10.5f} "
      f"{ann*100:13.1f} {mon*100:10.2f} {(1+mon)/1.8:8.4f}배")

P("\n"+"="*130); P("[2] 문턱별 — 승률을 올리면 (빈도는 줄어도) 상한이 오르나"); P("="*130)
P(f"  {'문턱':>10s} {'배리어':>8s} {'승률%':>7s} {'평균보유':>9s} {'N독립':>8s} {'켈리L':>7s} {'월%':>10s}")
BEST = None
for thq in (0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 1.00):
    th = float(np.quantile(SIG[np.isfinite(SIG)], 1-thq)) if thq < 1.0 else 0.0
    for bar in (0.01, 0.02, 0.03, 0.04, 0.05):
        mh = min(43200, max(1440, int(bar*900000)))
        nw, nt, sh = measure(O, H, Lo, C, SIG, th, bar, bar, mh, STEP, 0, warm, len(C))
        if nt < 30: continue
        p = nw/nt; hold = sh/nt
        Nind = MIN_Y/max(hold, 1)
        Lk, g = kelly(p, bar, bar)
        ann = np.exp(Nind*g)-1
        mon = (1+ann)**(1/12)-1 if ann > -0.999 else -1
        if BEST is None or mon > BEST[0]: BEST = (mon, thq, bar, p, hold, Nind, Lk, ann)
        if bar in (0.02, 0.04):
            P(f"  상위{thq*100:5.1f}% ±{bar*100:5.2f}% {p*100:7.2f} {hold:8.0f}분 {Nind:8.0f} "
              f"{Lk:6.2f}x {mon*100:10.2f}")
if BEST:
    mon, thq, bar, p, hold, Nind, Lk, ann = BEST
    P(f"\n  ★ 최대: 문턱 상위{thq*100:.1f}% 배리어 ±{bar*100:.2f}% → 승률 {p*100:.2f}%, "
      f"보유 {hold:.0f}분, N독립 {Nind:.0f}/년, 켈리 {Lk:.2f}배")
    P(f"     연 {ann*100:+,.1f}%  월 {mon*100:+.2f}%  (목표 +80% 대비 {(1+mon)/1.8:.4f}배)")

P("\n"+"="*130); P("[3] 비대칭 배리어에서도 (익절 좁게·손절 넓게)"); P("="*130)
P(f"  {'익절/손절':>12s} {'승률%':>7s} {'평균보유':>9s} {'N독립':>8s} {'켈리L':>7s} {'월%':>10s}")
th10 = float(np.quantile(SIG[np.isfinite(SIG)], 0.90))
BEST2 = None
for up, dn in ((0.01, 0.02), (0.015, 0.03), (0.02, 0.04), (0.03, 0.06), (0.02, 0.02),
               (0.04, 0.04), (0.02, 0.03), (0.03, 0.04), (0.04, 0.06), (0.05, 0.05)):
    mh = min(43200, max(1440, int(max(up, dn)*900000)))
    nw, nt, sh = measure(O, H, Lo, C, SIG, th10, up, dn, mh, STEP, 0, warm, len(C))
    if nt < 30: continue
    p = nw/nt; hold = sh/nt
    Nind = MIN_Y/max(hold, 1)
    Lk, g = kelly(p, up, dn)
    ann = np.exp(Nind*g)-1
    mon = (1+ann)**(1/12)-1 if ann > -0.999 else -1
    if BEST2 is None or mon > BEST2[0]: BEST2 = (mon, up, dn, p, hold, Nind, Lk, ann)
    P(f"  {up*100:5.2f}/{dn*100:5.2f}% {p*100:7.2f} {hold:8.0f}분 {Nind:8.0f} {Lk:6.2f}x {mon*100:10.2f}")
if BEST2:
    mon, up, dn, p, hold, Nind, Lk, ann = BEST2
    P(f"\n  ★ 최대: 익절 {up*100:.2f}% 손절 {dn*100:.2f}% → 승률 {p*100:.2f}%, N독립 {Nind:.0f}/년, "
      f"켈리 {Lk:.2f}배 → 월 {mon*100:+.2f}%")

P("\n"+"="*130); P("[4] 월 +80%가 요구하는 (보유시간, 승률) 조합 — 대칭 배리어 기준"); P("="*130)
P("  각 칸 = 그 조합에서 도달하는 월수익%. ★ = 월 +80% 이상")
P(f"  {'보유(분)':>9s} " + " ".join(f"{'승률'+str(int(p*100))+'%':>12s}"
                                  for p in (0.52, 0.55, 0.58, 0.60, 0.65, 0.70)))
for hold in (30, 60, 120, 240, 480, 960, 1920, 3840):
    Nind = MIN_Y/hold
    row = []
    for pw in (0.52, 0.55, 0.58, 0.60, 0.65, 0.70):
        # 보유시간에 어울리는 배리어를 rv로 추정: 대략 bar ≈ 0.0004×sqrt(hold)
        bar = max(0.002, 0.00045*np.sqrt(hold))
        Lk, g = kelly(pw, bar, bar)
        ann = np.exp(Nind*g)-1
        mon = (1+ann)**(1/12)-1 if ann > -0.999 else -1
        tag = "★" if mon >= 0.8 else " "
        row.append(f"{mon*100:10.1f}{tag}" if abs(mon) < 1e6 else f"{'거대':>10s}{tag}")
    P(f"  {hold:8d}분 " + " ".join(f"{x:>12s}" for x in row))
P("  (배리어는 보유시간에 맞춰 bar ≈ 0.045%×√분 로 추정)")

P("\n"+"="*130); P("[5] 결론 — BTC 단일 종목에서 도달 가능한 최대"); P("="*130)
if ROWS:
    ROWS.sort(key=lambda x: -x[0])
    mon, bar, p, hold, Nind, Lk, g, ann = ROWS[0]
    P(f"  대칭 배리어 최대: ±{bar*100:.2f}% → 월 {mon*100:+.2f}% (연 {ann*100:+,.1f}%)")
    P(f"    승률 {p*100:.2f}%, 평균보유 {hold:.0f}분, 독립베팅 {Nind:.0f}회/년, 켈리 명목 {Lk:.2f}배")
P(f"\n  월 +80% 달성에 필요한 것 (현재 최선 대비):")
if ROWS:
    mon, bar, p, hold, Nind, Lk, g, ann = ROWS[0]
    g_need = TARGET/Nind
    P(f"    ①빈도 유지 + 거래당 성장 {g:.5f} → {g_need:.5f} ({g_need/max(g,1e-9):.1f}배)")
    N_need = TARGET/max(g, 1e-9)
    P(f"    ②거래당 유지 + 독립베팅 {Nind:.0f} → {N_need:,.0f}회/년 "
      f"(평균보유 {MIN_Y/N_need:.1f}분으로 단축)")
    P(f"    ③승률로: ±{bar*100:.1f}%·독립 {Nind:.0f}회에서 월 +80%를 내는 승률은")
    for pw in np.arange(0.52, 0.85, 0.01):
        Lk2, g2 = kelly(pw, bar, bar)
        ann2 = np.exp(Nind*g2)-1
        mon2 = (1+ann2)**(1/12)-1 if ann2 > -0.999 else -1
        if mon2 >= 0.8:
            P(f"        승률 {pw*100:.0f}% (현재 {p*100:.2f}%, {(pw-p)*100:+.2f}%p 더 필요)")
            break
P("\n  ※ 동시 보유로 거래 수를 늘려도 이 한계는 안 바뀐다 —")
P("    같은 종목 롱 포지션들은 완전 상관이라 독립 베팅이 아니다.")
P("    독립 베팅 수를 정하는 것은 거래 수가 아니라 '평균 보유시간'이다.")
