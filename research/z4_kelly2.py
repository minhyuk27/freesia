"""z4-A(수정) 극단 레버리지 운용 기준 — 승률을 배율의 함수로 실측한다.

앞선 판의 두 오류:
  ①청산 시 손실을 1−L×(1/L−mmr)=L×mmr 로 계산 → 배율이 클수록 남는 자본이 는다는 모순.
    실제로는 청산 시 증거금 전손(거래소가 잔여 증거금을 청산수수료로 가져간다)으로 봐야 한다.
  ②승률 p를 배율과 무관한 상수로 둠. 실제로는 배율이 오르면 실효 손절이 좁아져 p가 떨어진다.

여기서는 p(L)을 데이터에서 직접 잰다."""
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
TARGET = np.log(1.8)*12
P(f"월 +80% = 연 {np.exp(TARGET)-1:,.0f}배 = 연 로그성장 {TARGET:.3f}\n")


@numba.njit(cache=True)
def measure(o, h, l, c, sigv, th, up, dn, maxh, step, cool, start, n_end):
    """주어진 익절/손절에서 승률·거래수·보유시간을 실측."""
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


TH = float(np.quantile(SIG[np.isfinite(SIG)], 1-0.25))
UP = 0.015
P("="*126); P("[1] 실측 p(L) — 배율이 오르면 실효 손절이 좁아져 승률이 떨어진다"); P("="*126)
P(f"  진입: 신호 상위25%, 익절 +{UP*100:.1f}%, 최대 보유 8640분")
P(f"\n  {'명목L':>7s} {'청산거리':>9s} {'실효손절':>9s} {'승률%':>7s} {'연거래':>7s} {'평균보유':>9s}")
PL = {}
for Lx in (1, 2, 3, 5, 8, 12, 20, 30, 40, 50, 60, 75, 100, 125, 150, 200):
    liq = 1.0/Lx - MMR
    eff = min(0.03, max(liq, 0.0005))
    nw, nt, sh = measure(O, H, Lo, C, SIG, TH, UP, eff, 8640, STEP, 60, warm, len(C))
    p = nw/max(nt, 1)
    PL[Lx] = (p, nt, eff, liq)
    P(f"  {Lx:6d}x {liq*100:8.2f}% {eff*100:8.3f}% {p*100:7.2f} {nt/Y:7.0f} {sh/max(nt,1):8.0f}분")

P("\n"+"="*126); P("[2] 수정된 기준표 — 청산 시 증거금 전손, p는 실측값"); P("="*126)
P("  거래당 로그성장 g(L) = p·ln(1 + L·(up − c)) + (1−p)·ln(max(0, 1 − L·(dn_eff + c)))")
P("  청산이 개입하면(dn_eff = 1/L − mmr) 손실 factor를 0으로 본다 → ln(0) = −∞ → 그 배율은 파산")
P(f"\n  {'명목L':>7s} {'승률%':>7s} {'연거래':>7s} {'1회손실%':>10s} {'거래당g':>11s} {'연성장%':>16s} {'월%':>10s} {'판정':>10s}")
c_ = 0.0010
BESTG = None
for Lx in sorted(PL):
    p, nt, eff, liq = PL[Lx]
    N = nt/Y
    liq_hit = liq <= 0.03 + 1e-9
    loss_frac = min(1.0, Lx*(eff + c_))
    if liq_hit and Lx > 1.0/(0.03+MMR):
        g = -np.inf
    else:
        win = 1.0 + Lx*(UP - c_)
        lose = 1.0 - loss_frac
        g = -np.inf if lose <= 0 else p*np.log(win) + (1-p)*np.log(lose)
    if not np.isfinite(g):
        P(f"  {Lx:6d}x {p*100:7.2f} {N:7.0f} {loss_frac*100:9.1f}% {'-inf':>11s} "
          f"{'파산':>16s} {'':>10s} {'전손 가능':>10s}")
        continue
    ann = np.exp(N*g)-1
    mon = (1+ann)**(1/12)-1 if ann > -0.999999 else -1
    if BESTG is None or g*N > BESTG[0]: BESTG = (g*N, Lx, p, N, ann, mon)
    P(f"  {Lx:6d}x {p*100:7.2f} {N:7.0f} {loss_frac*100:9.1f}% {g:11.5f} {ann*100:16.1f} "
      f"{mon*100:10.2f} {'성장' if g > 0 else '붕괴':>10s}")

if BESTG:
    _, Lb, pb, Nb, annb, monb = BESTG
    P(f"\n  ★ 최적 명목 {Lb}배: 승률 {pb*100:.2f}%, 연 {Nb:.0f}건 → 연 {annb*100:+.1f}% (월 {monb*100:+.2f}%)")
    P(f"     월 +80% 대비 {(1+monb)/1.8:.4f}배")

P("\n"+"="*126); P("[3] 월 +80%가 요구하는 것 — 빈도 × 거래당 순수익"); P("="*126)
P(f"  {'연거래':>8s} {'필요 거래당 순수익%':>20s} {'명목1배 원시엣지 필요bp':>24s} {'실측 대비':>12s}")
RAW = 13.8
for N in (200, 500, 1000, 2000, 5000, 10000, 20000, 50000):
    r = np.exp(TARGET/N)-1
    P(f"  {N:8d} {r*100:20.3f} {r*1e4:24.1f} {r*1e4/RAW:11.1f}배")
P(f"  (실측 거래당 원시 엣지 {RAW}bp = z2.4 기준)")

P("\n"+"="*126); P("[4] 레버리지로 메우려면 — 명목 배수별 실제 도달 월수익"); P("="*126)
P("  ※ 청산 없는 최대 명목 = 1/(손절+mmr). 손절 3%면 29.4배, 1.5%면 52.6배, 0.5%면 147배")
for dn_ in (0.005, 0.01, 0.015, 0.03):
    Lmax = 1.0/(dn_+MMR)
    nw, nt, sh = measure(O, H, Lo, C, SIG, TH, UP, dn_, 8640, STEP, 60, warm, len(C))
    p = nw/max(nt, 1); N = nt/Y
    best = None
    for Lx in np.arange(0.5, Lmax, 0.5):
        win = 1.0 + Lx*(UP - c_)
        lose = 1.0 - Lx*(dn_ + c_)
        if lose <= 0: break
        g = p*np.log(win) + (1-p)*np.log(lose)
        if best is None or g > best[1]: best = (Lx, g)
    if best is None: continue
    Lx, g = best
    ann = np.exp(N*g)-1
    mon = (1+ann)**(1/12)-1 if ann > -0.999999 else -1
    P(f"  손절 {dn_*100:4.2f}%: 승률 {p*100:5.2f}%  연 {N:4.0f}건  청산없는최대 {Lmax:5.1f}배  "
      f"켈리 {Lx:5.1f}배  → 연 {ann*100:+9.1f}%  월 {mon*100:+7.2f}%")

P("\n"+"="*126); P("[5] 역산 — 월 +80%를 만드는 (승률, 빈도) 조합. 익절=손절=1%, 비용 7bp"); P("="*126)
P("  ※ 청산 없는 최대 명목 71.4배 이내에서 켈리 최적 사용")
P(f"  {'승률%':>6s} {'켈리L':>7s} {'거래당g':>10s} " + " ".join(f"{'연'+str(N)+'건':>14s}"
                                                          for N in (500, 1000, 2000, 5000, 20000)))
up_ = 0.01; dn_ = 0.01; cc = 0.0007
Lcap = 1.0/(dn_+MMR)
for pw in (0.52, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99):
    best = None
    for Lx in np.arange(0.5, Lcap, 0.5):
        win = 1.0 + Lx*(up_-cc); lose = 1.0 - Lx*(dn_+cc)
        if lose <= 0: break
        g = pw*np.log(win)+(1-pw)*np.log(lose)
        if best is None or g > best[1]: best = (Lx, g)
    Lx, g = best
    row = []
    for N in (500, 1000, 2000, 5000, 20000):
        ann = np.exp(N*g)-1
        mon = (1+ann)**(1/12)-1 if ann > -0.999999 else -1
        tag = "★" if mon >= 0.8 else " "
        row.append(f"{mon*100:12.2f}%{tag}")
    P(f"  {pw*100:6.1f} {Lx:6.1f}x {g:10.5f} " + " ".join(f"{x:>14s}" for x in row))
P("\n  ★ = 월 +80% 달성")

P("\n"+"="*126); P("[6] 파산 확률 — 명목별 연속손실 위험 (승률 실측)"); P("="*126)
P(f"  {'명목':>7s} {'승률%':>7s} {'1회손실%':>9s} {'−90%까지':>9s} {'그 확률':>11s} {'연내 발생확률':>14s}")
for Lx in (2, 3, 5, 8, 12, 20, 29):
    p, nt, eff, liq = PL.get(Lx, (None,)*4)
    if p is None:
        nw, nt2, _ = measure(O, H, Lo, C, SIG, TH, UP, 0.03, 8640, STEP, 60, warm, len(C))
        p = nw/nt2; nt = nt2
    loss = min(Lx*0.03, 1.0)
    if loss >= 1.0:
        P(f"  {Lx:6d}x {p*100:7.2f} {loss*100:8.1f}% {'1회':>9s} {(1-p)*100:10.2f}% {'거의 확실':>14s}")
        continue
    k = int(np.ceil(np.log(0.1)/np.log(1-loss)))
    pk = (1-p)**k
    N = nt/Y
    pany = 1-(1-pk)**max(N-k, 1)
    P(f"  {Lx:6d}x {p*100:7.2f} {loss*100:8.1f}% {k:7d}회 {pk*100:10.4f}% {pany*100:13.2f}%")
