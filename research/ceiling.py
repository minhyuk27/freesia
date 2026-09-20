"""수익 천장 탐색 — 단계별 MAE 구조 → 단계별 사이징 → 최대 무청산 수익. 월 분포도 측정.
사용: python research/ceiling.py"""
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

PM = {s: store.load("premium_1m", s).set_index("ts").close for s in SYM}
pq = []
for a, b, r, m, ts, key, rv, wg in Z:
    ser = PM[key[0]]; i = ser.index.searchsorted(ts)
    if i <= 1: pq.append(np.nan); continue
    pv = float(ser.iloc[i-1]); past = ser.iloc[max(0, i-1-525600):i-1]
    pq.append(float((past < pv).mean()) if len(past) > 10000 else np.nan)
pq = np.array(pq)

P("[1] 단계(깊이)별 구조 — 진입가 대비 최대역행(MAE)과 수익")
df = pd.DataFrame([(x[5][2], x[2], x[3], x[5][0], x[5][1]) for x in Z],
                  columns=["k", "r", "mae", "sym", "B"])
g = df.groupby("k")
P(pd.DataFrame({
    "n": g.size(), "평균수익bp": g.r.mean()*1e4, "승률": g.r.apply(lambda s: (s > 0).mean()),
    "MAE평균%": g.mae.mean()*100, "MAE중앙%": g.mae.median()*100,
    "MAE 95%": g.mae.quantile(0.95)*100, "MAE 최대%": g.mae.max()*100,
    "수익/MAE": g.r.mean()/g.mae.mean(),
}).round(3).to_string())
P("\n  → 깊은 단계일수록 수익/MAE 비율이 높으면 그 단계에 더 큰 명목을 태울 수 있다")

P("\n[2] 단계별 사이징 — 명목을 1/MAE에 비례시키기 (단계 가중 재설계)")
MAE_M = {k: float(df[df.k == k].mae.mean()) for k in df.k.unique()}
P(f"  단계별 평균 MAE: " + " ".join(f"k={k:.0f} {v*100:.2f}%" for k, v in sorted(MAE_M.items())))


def reweight(mode, amp_th=0.10, amp=1.5):
    E = []
    for j, (a, b, r, m, ts, key, rv, wg) in enumerate(Z):
        k = key[2]
        if mode == "base": w2 = wg
        elif mode == "invmae": w2 = 1.0/MAE_M[k]
        elif mode == "invmae2": w2 = (1.0/MAE_M[k])**2
        elif mode == "edge": w2 = float(df[df.k == k].r.mean())/MAE_M[k]
        else: w2 = wg
        f = amp if (np.isfinite(pq[j]) and pq[j] <= amp_th) else 1.0
        E.append((a, b, r, m, ts, key, rv, w2*f))
    # 가중 정규화(평균 1)
    mw = np.mean([x[7] for x in E])
    return [(a, b, r, m, ts, key, rv, wg/mw) for (a, b, r, m, ts, key, rv, wg) in E]


def scan(E, lab, ws=(1, 2, 3, 5, 8, 12, 20, 30, 50), nms=(1, 2, 3, 5, 8, 12, 20, 30)):
    best = None
    for w in ws:
        for Nm in nms:
            if Nm < w: continue
            eq, fl, liq = X.sim(E, float(w), float(Nm)); m = met(eq, fl, liq)
            if m is None or liq: continue
            if best is None or m[0] > best[0]: best = (m[0], m[2], m[3], m[4], w, Nm, m[5])
    if best:
        P(f"  {lab:24s} 최대 무청산: w={best[4]:3d} Nmax={best[5]:3d} → 연 {best[0]:+8.0f}% "
          f"최악 {best[1]:6.1f}% | 24~ {best[2]:+8.0f}% | 25~ {best[3]:+8.0f}%")
    else:
        P(f"  {lab:24s} 무청산 구성 없음")
    return best


P("\n[3] 가중 방식별 최대 무청산 수익")
res = {}
for mode in ("base", "invmae", "invmae2", "edge"):
    res[mode] = scan(reweight(mode), mode)

P("\n[4] 최고 구성의 월별 분포 — 월 +80%를 넘는 달이 있나")
bm = max(res.items(), key=lambda kv: kv[1][0] if kv[1] else -1e9)
P(f"  선택: {bm[0]} w={bm[1][4]} Nmax={bm[1][5]}")
eq = bm[1][6]
mon = (eq.resample("ME").last()/eq.resample("ME").first()-1).dropna()
mon = mon[mon != 0]
P(f"  거래가 있는 달 {len(mon)}개 / 전체 81개월")
for th in (0.10, 0.30, 0.50, 0.80, 1.00, 2.00):
    n = int((mon >= th).sum())
    P(f"    월 +{th*100:.0f}% 이상: {n}개월 ({n/81*100:.1f}% of 전체, {n/max(len(mon),1)*100:.1f}% of 거래월)")
P(f"  거래월 수익: 중앙 {mon.median()*100:+.1f}% 평균 {mon.mean()*100:+.1f}% 최대 {mon.max()*100:+.1f}% 최소 {mon.min()*100:+.1f}%")
P(f"  전체 81개월 기준 월 중앙 {np.median(np.concatenate([mon.values, np.zeros(81-len(mon))]))*100:+.2f}%")
P("  상위 10개월: " + " ".join(f"{i.strftime('%y-%m')}{v*100:+.0f}%" for i, v in mon.nlargest(10).items()))

P("\n[5] 월 +80% 복리에 필요한 것 — 역산")
P("  월 +80% 복리 = 연 +115,500% (1,157배). 6.7년이면 10^38배.")
base = res["base"][0] if res["base"] else np.nan
P(f"  현재 최대 무청산 연 수익: {bm[1][0]:+.0f}%  → 필요 배수 {115500/max(bm[1][0],1):.0f}배")
P("  명목 배수로는 불가능(청산이 총명목 12x에서 발생). 남은 길은 거래당 수익 또는 거래 빈도.")
r = np.array([x[2] for x in Z])
P(f"  현재: 거래 {len(r)}건/6.7년 = {len(r)/6.7:.0f}건/년, 거래당 {r.mean()*1e4:+.0f}bp")
for need in (0.80,):
    ann = (1+need)**12 - 1
    P(f"  월 +{need*100:.0f}% 달성 조건(명목 8x 고정, 거래당 수익 유지 가정):")
    per_trade_cap = r.mean()*8
    n_need = np.log(1+ann)/np.log(1+per_trade_cap)
    P(f"    거래당 자본수익 {per_trade_cap*100:+.1f}% → 연 {n_need:.0f}건 필요 (현재 {len(r)/6.7:.0f}건, {n_need/(len(r)/6.7):.0f}배)")

P("\n[6] 참고 — 거래가 몰린 달만 보면 (패닉 달의 실제 수익)")
top = mon.nlargest(15)
P("  " + " | ".join(f"{i.strftime('%Y-%m')} {v*100:+.0f}%" for i, v in top.items()))
