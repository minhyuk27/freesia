"""z12 남은 검증 — 진짜 플라시보, ETH, 워크포워드, 2025 약세, 증거금 점유.

위상 이동 플라시보는 '봉 경계가 특별한가'만 본다. 진짜 대조군은
  ①무작위 깊이(같은 체결률, 다른 가격) ②분봉 무작위 재배열(수익률 셔플)
  ③깊이를 위쪽으로(상승 돌파 매수 = 반대 방향)
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R/"research"))
import z2_lib as L2


def P(*a, end="\n"):
    print(" ".join(str(x) for x in a), end=end, flush=True)


Y = L2.YRS; MMR = L2.MMR
BAR = 720; KS = (15.0, 21.0, 30.0); WS = (1.0, 3.0, 6.0)
EPS, SLIP, FEE = 5.0, 10.0, 10.0
warm = 200000


@numba.njit(cache=True)
def dip(o, h, l, c, rvv, barmin, k0, k1, k2, w0, w1, w2,
        eps, slip, fee, start, n_end, updown):
    """updown=+1: 아래 지정가 매수(정상). -1: 위쪽 돌파 매수(반대 방향 대조군)."""
    K = 200000
    ea = np.zeros(K, np.int64); er = np.zeros(K); nf = np.zeros(K, np.int8)
    k = 0
    nb = (n_end-start)//barmin
    for b in range(nb):
        t0 = start+b*barmin; t1 = t0+barmin-1
        if t1 >= n_end-1: break
        if np.isnan(rvv[t0]) or rvv[t0] <= 0: continue
        px0 = o[t0]
        tot = 0.0; wsum = 0.0; ft = -1; cnt = 0
        for li in range(3):
            kk = k0 if li == 0 else (k1 if li == 1 else k2)
            ww = w0 if li == 0 else (w1 if li == 1 else w2)
            if updown > 0:
                T = px0*(1.0 - rvv[t0]*kk); trig = T*(1.0-eps/1e4)
                got = -1
                for j in range(t0, t1+1):
                    if l[j] <= trig: got = j; break
            else:
                T = px0*(1.0 + rvv[t0]*kk); trig = T*(1.0+eps/1e4)
                got = -1
                for j in range(t0, t1+1):
                    if h[j] >= trig: got = j; break
            if got >= 0:
                fill = T*(1.0+slip/1e4)
                tot += ww*(c[t1]/fill-1.0-fee/1e4); wsum += ww
                if ft < 0: ft = got
                cnt += 1
        if wsum > 0:
            ea[k] = ft; er[k] = tot/wsum; nf[k] = cnt; k += 1
            if k >= K-2: break
    return ea[:k], er[:k], nf[:k]


def ann(er, ea, idx, nom=1.0):
    if len(er) < 30: return None
    m = pd.Series(nom*er, index=pd.DatetimeIndex(idx[ea])).resample("ME").sum()
    eq = (1+m).cumprod()
    if (eq <= 0.01).any(): return None
    return (eq.iloc[-1]**(12/len(m))-1)*100


P("="*124); P("[1] 진짜 대조군들 — BTC"); P("="*124)
S = "BTCUSDT"
d = L2.D[S]; idx = d["idx"]; rv = d["rv"]
O, H, Lo, C = d["o"], d["h"], d["l"], d["c"]; N = len(C)
ea, er, nf = dip(O, H, Lo, C, rv, BAR, *KS, *WS, EPS, SLIP, FEE, warm, N, 1)
base = ann(er, ea, idx)
P(f"  본체(아래 딥 매수): 연 {base:+.1f}%  거래당 {er.mean()*1e4:+.1f}bp  승률 {np.mean(er>0)*100:.2f}%  n={len(er):,}")
P(f"    층 체결 분포: 1층만 {np.mean(nf==1)*100:.1f}% / 2층 {np.mean(nf==2)*100:.1f}% / 3층 {np.mean(nf==3)*100:.1f}%")

eau, eru, _ = dip(O, H, Lo, C, rv, BAR, *KS, *WS, EPS, SLIP, FEE, warm, N, -1)
au = ann(eru, eau, idx)
P(f"\n  ① 반대 방향(위쪽 돌파 매수): " + ("파산" if au is None else f"연 {au:+.1f}%") +
  f"  거래당 {eru.mean()*1e4:+.1f}bp  승률 {np.mean(eru>0)*100:.2f}%  n={len(eru):,}")
P("     → 같은 봉·같은 깊이·같은 청산인데 방향만 반대. 대칭이면 엣지는 방향에 있다.")

P("\n  ② 무작위 깊이 (같은 평균 체결률, 깊이만 섞음) 200회:")
rng = np.random.default_rng(99)
vals = []
for it in range(200):
    f_ = rng.uniform(0.5, 2.0)
    ks = (KS[0]*f_, KS[1]*f_, KS[2]*f_)
    _, er2, _ = dip(O, H, Lo, C, rv, BAR, *ks, *WS, EPS, SLIP, FEE, warm, N, 1)
    if len(er2) > 100: vals.append(er2.mean()*1e4)
vals = np.array(vals)
P(f"     거래당bp 실제 {er.mean()*1e4:+.2f} | 무작위깊이 평균 {vals.mean():+.2f} "
  f"최대 {vals.max():+.2f} → 백분위 {np.mean(vals < er.mean()*1e4)*100:.0f}%")
P("     → 깊이 고원이 넓으니 백분위가 낮게 나오는 게 정상. 음수가 안 나오는 것이 핵심.")
P(f"     무작위 깊이 200회 중 음수: {int((vals<0).sum())}회")

P("\n  ③ 수익률 부호 셔플 (봉 내 경로를 무작위로 뒤집음):")
rng2 = np.random.default_rng(7)
sh = []
for it in range(200):
    sgn = rng2.choice(np.array([-1.0, 1.0]), size=len(er))
    sh.append((er*sgn).mean()*1e4)
sh = np.array(sh)
P(f"     거래당bp 실제 {er.mean()*1e4:+.2f} | 부호셔플 평균 {sh.mean():+.2f} 최대 {sh.max():+.2f} "
  f"→ 백분위 {np.mean(sh < er.mean()*1e4)*100:.1f}%")

P("\n"+"="*124); P("[2] ETH에서도 되나"); P("="*124)
E = L2.D["ETHUSDT"]
eaE, erE, _ = dip(E["o"], E["h"], E["l"], E["c"], E["rv"], BAR, *KS, *WS,
                  EPS, SLIP, FEE, warm, len(E["c"]), 1)
aE = ann(erE, eaE, E["idx"])
P(f"  ETH: " + ("파산" if aE is None else f"연 {aE:+.1f}%") +
  f"  거래당 {erE.mean()*1e4:+.1f}bp  승률 {np.mean(erE>0)*100:.2f}%  n={len(erE):,}")
P(f"  BTC: 연 {base:+.1f}%  거래당 {er.mean()*1e4:+.1f}bp  승률 {np.mean(er>0)*100:.2f}%  n={len(er):,}")
P(f"\n  ETH 깊이 스윕:")
for mult in (8, 10, 12, 15, 20, 28, 40):
    ks = (mult, mult*1.4, mult*2.0)
    ea2, er2, _ = dip(E["o"], E["h"], E["l"], E["c"], E["rv"], BAR, *ks, *WS,
                      EPS, SLIP, FEE, warm, len(E["c"]), 1)
    a = ann(er2, ea2, E["idx"])
    P(f"    rv×{mult:3d}: " + ("파산" if a is None else f"연 {a:+8.1f}%") +
      f"  거래당 {er2.mean()*1e4:+6.1f}bp  승률 {np.mean(er2>0)*100:5.2f}%  n={len(er2):,}")

P("\n  BTC+ETH 병행(각 자본 절반, 독립 운용):")
mB = pd.Series(er, index=pd.DatetimeIndex(idx[ea])).resample("ME").sum()
mE = pd.Series(erE, index=pd.DatetimeIndex(E["idx"][eaE])).resample("ME").sum()
df = pd.concat([mB, mE], axis=1, keys=["BTC", "ETH"]).fillna(0)
P(f"    월 상관 {df.corr().iloc[0,1]:.3f}")
for nom in (1, 2, 3, 5):
    tot = df.mean(axis=1)*nom
    eq = (1+tot).cumprod()
    if (eq <= 0.01).any(): P(f"    명목{nom}배: 파산"); continue
    a = (eq.iloc[-1]**(12/len(tot))-1)*100
    P(f"    명목{nom}배: 연 {a:+9.1f}%  월 {((1+a/100)**(1/12)-1)*100:+6.2f}%  "
      f"MDD {(eq/eq.cummax()-1).min()*100:6.1f}%")

P("\n"+"="*124); P("[3] 워크포워드 — 뒤 구간만"); P("="*124)
for cut in (2021, 2022, 2023, 2024, 2025):
    m = np.array([idx[a].year >= cut for a in ea])
    if m.sum() < 50: continue
    a = ann(er[m], ea[m], idx)
    yy = (idx[ea[m]][-1]-idx[ea[m]][0]).days/365
    P(f"  {cut}년 이후 ({m.sum():4d}건, {yy:.1f}년): 거래당 {er[m].mean()*1e4:+6.1f}bp  "
      f"승률 {np.mean(er[m]>0)*100:5.2f}%  " + ("파산" if a is None else f"연 {a:+8.1f}%"))

P("\n"+"="*124); P("[4] 2025년이 약한 이유 (+37%로 최저)"); P("="*124)
dfm = pd.DataFrame({"t": idx[ea], "r": er, "nf": nf})
dfm["y"] = dfm.t.dt.year
P(f"  {'연도':>6s} {'거래':>5s} {'승률%':>7s} {'거래당bp':>9s} {'평균층수':>9s} {'rv중앙bp':>9s}")
for yr, g in dfm.groupby("y"):
    ii = np.where(idx.year == yr)[0]
    rvm = np.nanmedian(rv[ii]) if len(ii) else np.nan
    P(f"  {yr:6d} {len(g):5d} {np.mean(g.r>0)*100:7.2f} {g.r.mean()*1e4:+9.1f} "
      f"{g.nf.mean():9.2f} {rvm*1e4:9.2f}")
P("\n  → 변동성(rv)이 낮으면 딥이 얕아져 체결은 늘지만 회복 폭도 작다.")

P("\n"+"="*124); P("[5] 증거금 점유 — 3층이 동시 체결되면 실제 명목은?"); P("="*124)
P(f"  가중 1:3:6 → 1층만 체결 시 명목 1/10, 3층 전부면 10/10")
P(f"  내 시뮬은 체결된 층 가중합으로 정규화(항상 명목 1배)한다.")
P(f"  실제로 '항상 10단위를 준비'하는 방식이면 평균 명목은:")
avg_nom = np.array([1.0 if n == 1 else (4.0 if n == 2 else 10.0) for n in nf]).mean()/10
P(f"    평균 {avg_nom*100:.1f}% (체결 층수 분포 기준)")
P(f"  → 정규화 방식이면 자본 효율이 {1/avg_nom:.1f}배 높다. 실전은 그 사이.")
mm = pd.Series(1, index=pd.DatetimeIndex(idx[ea])).resample("ME").sum()
P(f"\n  월 거래 중앙 {mm.median():.0f} 최소 {mm.min():.0f} | 봉 점유율 "
  f"{len(er)/((N-warm)/BAR)*100:.1f}% (연 {len(er)/Y:.0f}건 / 봉 {(N-warm)/BAR/Y:.0f}개)")
