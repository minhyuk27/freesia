"""z11 얕은 딥 검증 — z8의 C 다리(연 +477%)가 진짜인지 실행 가정으로 조인다.

정체: 4시간봉 시가 대비 0.66~1.31% 아래 지정가 3층 → 같은 봉 종가 청산.
      연 877건, 승률 72.24%, 거래당 +31.5bp. (문서 z0.3과 무관 — 15.5배 얕다)

볼트에는 '얕은 딥은 무레버 +24%→+15%로 악화'로 기각돼 있으나 그것은
z0.3 프레임에 얕은 층을 '더한' 실험이었다. 단독 얕은 딥은 따로 검증된 적이 없다.

조일 것:
  ①관통 — 지정가에 닿기만 해서는 체결로 치지 않고 eps만큼 관통해야 체결
  ②체결 슬리피지 — 체결가를 불리하게
  ③플라시보 — 무작위 봉에서 같은 규칙
  ④비용 민감도 / 연도별 / 깊이 고원
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R/"research"))
import z2_lib as L2


def P(*a, end="\n"):
    print(" ".join(str(x) for x in a), end=end, flush=True)


S = "BTCUSDT"; Y = L2.YRS
d = L2.D[S]; idx = d["idx"]; rv = d["rv"]
O, H, Lo, C = d["o"], d["h"], d["l"], d["c"]
N = len(C); warm = 200000


@numba.njit(cache=True)
def dip(o, h, l, c, rvv, barmin, k0, k1, k2, w0, w1, w2,
        eps_bp, slip_bp, fee_bp, start, n_end, shift):
    """eps_bp: 이만큼 관통해야 체결. slip_bp: 체결가를 이만큼 불리하게."""
    K = 200000
    ea = np.zeros(K, np.int64); er = np.zeros(K)
    k = 0
    nb = (n_end-start)//barmin
    for b in range(nb):
        t0 = start + b*barmin + shift
        t1 = t0 + barmin - 1
        if t1 >= n_end-1 or t0 < start: continue
        if np.isnan(rvv[t0]) or rvv[t0] <= 0: continue
        px0 = o[t0]
        tot = 0.0; wsum = 0.0; ft = -1
        for li in range(3):
            kk = k0 if li == 0 else (k1 if li == 1 else k2)
            ww = w0 if li == 0 else (w1 if li == 1 else w2)
            T = px0*(1.0 - rvv[t0]*kk)
            trig = T*(1.0 - eps_bp/1e4)
            got = -1
            for j in range(t0, t1+1):
                if l[j] <= trig: got = j; break
            if got >= 0:
                fill = T*(1.0 + slip_bp/1e4)
                tot += ww*(c[t1]/fill - 1.0 - fee_bp/1e4); wsum += ww
                if ft < 0: ft = got
        if wsum > 0:
            ea[k] = ft; er[k] = tot/wsum; k += 1
            if k >= K-2: break
    return ea[:k], er[:k]


def stat(ea, er, nom=1.0):
    if len(er) < 30: return None
    m = pd.Series(nom*er, index=pd.DatetimeIndex(idx[ea])).resample("ME").sum()
    eq = (1+m).cumprod()
    if (eq <= 0.01).any(): return None
    yy = len(m)/12
    ann = (eq.iloc[-1]**(1/yy)-1)*100
    return dict(ann=ann, mon=((1+ann/100)**(1/12)-1)*100,
                mdd=(eq/eq.cummax()-1).min()*100, n=len(er), wr=np.mean(er > 0)*100,
                bp=er.mean()*1e4, mwr=np.mean(m > 0)*100)


BAR = 240; KS = (10.0, 14.0, 20.0); WS = (1.0, 3.0, 6.0)
P("="*126); P("[1] 실행 가정을 조이면 — 관통 × 슬리피지 (수수료 10bp, 명목 1배)"); P("="*126)
P(f"  {'관통':>7s} " + " ".join(f"{'슬립'+str(s)+'bp':>19s}" for s in (0, 5, 10, 20, 50)))
for eps in (0, 2, 5, 10, 20):
    row = []
    for slip in (0, 5, 10, 20, 50):
        ea, er = dip(O, H, Lo, C, rv, BAR, *KS, *WS, float(eps), float(slip), 10.0, warm, N, 0)
        r = stat(ea, er)
        row.append("        파산       " if r is None else
                   f"{r['ann']:+8.1f}%/{r['bp']:+6.1f}bp")
    P(f"  {eps:5d}bp " + " ".join(f"{x:>19s}" for x in row))
P("  (셀 = 연수익% / 거래당bp)")

P("\n"+"="*126); P("[2] 현실적 가정(관통5bp·슬립10bp·수수료10bp)에서 3단 레버리지"); P("="*126)
ea, er = dip(O, H, Lo, C, rv, BAR, *KS, *WS, 5.0, 10.0, 10.0, warm, N, 0)
P(f"  거래 {len(er):,}건 (연 {len(er)/Y:.0f}), 승률 {np.mean(er>0)*100:.2f}%, 거래당 {er.mean()*1e4:+.1f}bp")
P(f"\n  {'구간':>12s} {'명목':>6s} {'연수익%':>12s} {'월%':>9s} {'MDD%':>8s} {'월승률%':>8s}")
for lb, nom in (("무레버리지", 1), ("중간", 2), ("중간+", 3), ("공격", 5), ("최대공격", 8)):
    r = stat(ea, er, nom)
    P(f"  {lb:>12s} {nom:5d}x " + ("        파산" if r is None
                                    else f"{r['ann']:12.1f} {r['mon']:9.2f} {r['mdd']:8.1f} {r['mwr']:8.1f}"))

P("\n"+"="*126); P("[3] 플라시보 — 봉 시작을 옮겨서 (같은 규칙, 다른 위상)"); P("="*126)
base = stat(ea, er)
vals = []
for sh in range(10, BAR, 10):
    ea2, er2 = dip(O, H, Lo, C, rv, BAR, *KS, *WS, 5.0, 10.0, 10.0, warm, N, sh)
    r = stat(ea2, er2)
    if r: vals.append(r["bp"])
vals = np.array(vals)
P(f"  거래당bp 실제(위상0) {base['bp']:+7.2f} | 위상이동 {len(vals)}회 평균 {vals.mean():+6.2f} "
  f"최대 {vals.max():+6.2f} 최소 {vals.min():+6.2f}")
P(f"  → 백분위 {np.mean(vals < base['bp'])*100:.0f}%")
P("  ※ 위상을 옮겨도 같은 '봉 시가 대비 딥'을 잡으므로, 이 검정은 봉 경계 의존성만 본다.")

P("\n  진짜 대조군 — 딥 없이 봉 시가에 시장가 매수 후 종가 청산:")


@numba.njit(cache=True)
def naive(o, c, barmin, fee_bp, start, n_end):
    K = 200000
    ea = np.zeros(K, np.int64); er = np.zeros(K)
    k = 0
    nb = (n_end-start)//barmin
    for b in range(nb):
        t0 = start+b*barmin; t1 = t0+barmin-1
        if t1 >= n_end-1: break
        ea[k] = t0; er[k] = c[t1]/o[t0]-1.0-fee_bp/1e4; k += 1
        if k >= K-2: break
    return ea[:k], er[:k]


na, nr = naive(O, C, BAR, 10.0, warm, N)
rn = stat(na, nr)
if rn is None:
    P(f"    시가매수-종가청산: **파산** (거래당 {nr.mean()*1e4:+.1f}bp, 승률 {np.mean(nr>0)*100:.2f}%, n={len(nr):,})")
else:
    P(f"    시가매수-종가청산: 연 {rn['ann']:+.1f}%  거래당 {rn['bp']:+.1f}bp  승률 {rn['wr']:.2f}%  n={rn['n']:,}")
P(f"    딥 대기          : 연 {base['ann']:+.1f}%  거래당 {base['bp']:+.1f}bp  승률 {base['wr']:.2f}%  n={base['n']:,}")

P("\n"+"="*126); P("[4] 깊이 고원 — 진짜 엣지면 깊이에 단조여야 한다"); P("="*126)
P(f"  {'깊이배수':>10s} {'체결률%':>8s} {'연거래':>7s} {'승률%':>7s} {'거래당bp':>9s} {'연수익%':>10s}")
nb_tot = (N-warm)//BAR
for mult in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 15.5):
    ks = tuple(k*mult for k in KS)
    ea2, er2 = dip(O, H, Lo, C, rv, BAR, *ks, *WS, 5.0, 10.0, 10.0, warm, N, 0)
    r = stat(ea2, er2)
    if r is None:
        P(f"  rv×{ks[0]:5.1f}~ {len(er2)/nb_tot*100:7.1f} {len(er2)/Y:7.0f}       -         -       파산")
        continue
    P(f"  rv×{ks[0]:5.1f}~ {len(er2)/nb_tot*100:7.1f} {r['n']/Y:7.0f} {r['wr']:7.2f} {r['bp']:+9.1f} {r['ann']:10.1f}")
P("  (mult=15.5가 문서 z0.3의 깊이)")

P("\n"+"="*126); P("[5] 연도별 (현실 가정, 명목 1배)"); P("="*126)
m = pd.Series(er, index=pd.DatetimeIndex(idx[ea])).resample("ME").sum()
out = []
for yr in range(2020, 2027):
    sub = m[m.index.year == yr]
    if len(sub) == 0: out.append(f"{yr} n/a"); continue
    out.append(f"{yr} {((1+sub).prod()-1)*100:+7.0f}%")
P("  " + " | ".join(out))

P("\n"+"="*126); P("[6] 봉 길이별 — 15분/1시간/4시간/1일"); P("="*126)
P(f"  {'봉':>8s} {'연거래':>8s} {'승률%':>7s} {'거래당bp':>9s} {'연수익%':>10s} {'월%':>9s} {'MDD%':>8s}")
for bm in (15, 60, 240, 1440):
    ea2, er2 = dip(O, H, Lo, C, rv, bm, *KS, *WS, 5.0, 10.0, 10.0, warm, N, 0)
    r = stat(ea2, er2)
    if r is None: P(f"  {bm:6d}분        파산"); continue
    P(f"  {bm:6d}분 {r['n']/Y:8.0f} {r['wr']:7.2f} {r['bp']:+9.1f} {r['ann']:10.1f} "
      f"{r['mon']:9.2f} {r['mdd']:8.1f}")
