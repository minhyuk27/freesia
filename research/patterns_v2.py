"""패턴 분석 단일전략 — 5계열 이론 기반 측정. 구간별 손익분기 문턱과 직접 비교.
전부 인과적(t까지의 정보만). 사용: python research/patterns_v2.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
SYM = ("BTCUSDT", "ETHUSDT")
D = {}
for s in SYM:
    k = store.load("klines_1m", s).set_index("ts")
    D[s] = dict(o=k.open.values, h=k.high.values, l=k.low.values, c=k.close.values,
                qv=k.quote_volume.values, tb=k.taker_buy_quote_volume.values, idx=k.index)
    P(f"{s}: {len(k):,}분")
HOR = (15, 30, 60, 240)
# 손익분기 문턱 (메이커 왕복 4bp / 메+테 7bp)
P("\n비용 기준: 메이커 왕복 4bp, 메+테 7bp, 테이커 10bp")


def bars(s, B):
    d = D[s]; c = d["c"]; n = len(c)//B
    st = np.arange(n)*B
    return dict(o=d["o"][st], h=np.array([d["h"][i*B:(i+1)*B].max() for i in range(n)]),
                l=np.array([d["l"][i*B:(i+1)*B].min() for i in range(n)]),
                c=d["c"][st+B-1], qv=np.array([d["qv"][i*B:(i+1)*B].sum() for i in range(n)]),
                st=st, idx=d["idx"][st], B=B)


def fwd(cl, h): return pd.Series(cl).shift(-h).values/cl - 1.0


def report(name, sel, cl, hz=(1, 2, 4), lab=""):
    n = int(sel.sum())
    if n < 100: return None
    row = f"    {name:34s} n={n:6d} "
    for h in hz:
        F = fwd(cl, h)
        ok = sel & np.isfinite(F)
        if ok.sum() < 50: continue
        row += f"| {h}봉 {F[ok].mean()*1e4:+7.2f}bp 승률{(F[ok]>0).mean():.3f} "
    P(row)
    return n


P("\n" + "="*130)
P("[A] 긴 아래꼬리 — 급락 후 즉시 흡수 (유동성 공급의 직접 증거)")
P("="*130)
for s in SYM:
    P(f"\n  ══ {s} ══")
    for B in (15, 60, 240):
        b = bars(s, B); cl = b["c"]
        rng = (b["h"]-b["l"])/b["c"]
        low_w = (np.minimum(b["c"], b["o"]) - b["l"])/b["c"]
        up_w = (b["h"] - np.maximum(b["c"], b["o"]))/b["c"]
        body = np.abs(b["c"]-b["o"])/b["c"]
        wr = low_w/np.maximum(rng, 1e-9)          # 아래꼬리 비율
        wq = pd.Series(wr).rank(pct=True).values
        rq = pd.Series(rng).rank(pct=True).values
        vq = pd.Series(b["qv"]/pd.Series(b["qv"]).rolling(100).mean().values).rank(pct=True).values
        P(f"   ── {B}분봉 ──")
        report(f"아래꼬리비율 상위5%", wq >= 0.95, cl)
        report(f"아래꼬리비율 상위1%", wq >= 0.99, cl)
        report(f"아래꼬리>60% + 범위상위20%", (wr > 0.6) & (rq >= 0.8), cl)
        report(f"아래꼬리>60% + 거래량상위20%", (wr > 0.6) & (vq >= 0.8), cl)
        report(f"망치형(아래꼬리>2×몸통, 양봉)", (low_w > 2*body) & (b["c"] > b["o"]), cl)
        report(f"망치형 + 거래량상위20%", (low_w > 2*body) & (b["c"] > b["o"]) & (vq >= 0.8), cl)
        report(f"(대조)위꼬리비율 상위5%", (pd.Series(up_w/np.maximum(rng, 1e-9)).rank(pct=True).values >= 0.95), cl)

P("\n" + "="*130)
P("[B] 스톱 헌팅 — 전저점 하향 이탈 후 회복 (청산 스톱 털기)")
P("="*130)
for s in SYM:
    P(f"\n  ══ {s} ══")
    for B in (60, 240):
        b = bars(s, B); cl = b["c"]
        for LB in (10, 20, 50):
            prev_low = pd.Series(b["l"]).rolling(LB).min().shift(1).values
            broke = b["l"] < prev_low                       # 전저점 하향 이탈
            recov = b["c"] > prev_low                        # 종가는 회복
            depth = (prev_low - b["l"])/np.maximum(prev_low, 1e-9)
            dq = pd.Series(np.where(broke, depth, np.nan)).rank(pct=True).values
            P(f"   ── {B}분봉 전저점{LB}봉 ──")
            report(f"이탈만", broke, cl)
            report(f"이탈 후 종가 회복", broke & recov, cl)
            report(f"이탈 후 회복 + 얕은이탈(하위50%)", broke & recov & (dq <= 0.5), cl)
            report(f"이탈 후 회복 + 깊은이탈(상위20%)", broke & recov & (dq >= 0.8), cl)
            report(f"이탈 후 미회복", broke & ~recov, cl)

P("\n" + "="*130)
P("[C] 라운드 넘버 — 심리적 가격대 (앵커링·주문 클러스터링)")
P("="*130)
for s in SYM:
    d = D[s]; c = d["c"]
    P(f"\n  ══ {s} ══")
    for B in (60, 240):
        b = bars(s, B); cl = b["c"]
        # 가격 자릿수에 맞는 라운드 단위
        mag = 10**np.floor(np.log10(np.maximum(cl, 1)))
        for div, lab in ((mag/10, "0.1×자릿수"), (mag/2, "0.5×자릿수"), (mag, "1×자릿수")):
            near = np.abs(cl/div - np.round(cl/div))*div/cl     # 라운드까지 상대거리
            dist_q = pd.Series(near).rank(pct=True).values
            # 라운드 아래에서 접근 후 이탈/반등
            below = (cl < np.round(cl/div)*div)
            P(f"   ── {B}분봉 {lab} ──")
            report(f"라운드 근접 하위10%", dist_q <= 0.10, cl)
            report(f"라운드 근접 & 아래", (dist_q <= 0.10) & below, cl)
            report(f"라운드 근접 & 위", (dist_q <= 0.10) & ~below, cl)
            report(f"라운드 돌파(직전 아래→현재 위)", (~below) & np.roll(below, 1), cl)
            report(f"라운드 이탈(직전 위→현재 아래)", below & ~np.roll(below, 1), cl)

P("\n" + "="*130)
P("[D] 볼륨 프로파일 — 거래량 집중 가격대(POC) 대비 위치")
P("="*130)
@numba.njit(cache=True)
def poc_dist(cl, qv, win, nb):
    """최근 win봉의 가격을 nb구간으로 나눠 거래량 최대 구간(POC) 중심까지의 상대거리"""
    n = len(cl); out = np.full(n, np.nan)
    for i in range(win, n):
        lo = 1e18; hi = -1e18
        for j in range(i-win, i):
            if cl[j] < lo: lo = cl[j]
            if cl[j] > hi: hi = cl[j]
        if hi <= lo: continue
        w = (hi-lo)/nb
        acc = np.zeros(nb)
        for j in range(i-win, i):
            k = int((cl[j]-lo)/w)
            if k >= nb: k = nb-1
            if k < 0: k = 0
            acc[k] += qv[j]
        bi = 0; bv = -1.0
        for k in range(nb):
            if acc[k] > bv: bv = acc[k]; bi = k
        poc = lo + (bi+0.5)*w
        out[i] = (cl[i]-poc)/poc
    return out

for s in SYM:
    P(f"\n  ══ {s} ══")
    for B in (60, 240):
        b = bars(s, B); cl = b["c"]
        for win in (50, 200):
            pd_ = poc_dist(cl.astype(np.float64), b["qv"].astype(np.float64), win, 30)
            q = pd.Series(pd_).rank(pct=True).values
            P(f"   ── {B}분봉 POC창{win}봉 ──")
            report("POC 대비 최하위5% (크게 아래)", q <= 0.05, cl)
            report("POC 대비 하위20%", q <= 0.20, cl)
            report("POC 근처(40~60%)", (q > 0.4) & (q <= 0.6), cl)
            report("POC 대비 상위20%", q >= 0.80, cl)
            report("POC 대비 최상위5%", q >= 0.95, cl)

P("\n" + "="*130)
P("[E] 캔들 시퀀스 — 연속 방향 패턴의 조건부 분포")
P("="*130)
for s in SYM:
    P(f"\n  ══ {s} ══")
    for B in (15, 60):
        b = bars(s, B); cl = b["c"]
        up = (b["c"] > b["o"]).astype(int)
        for L in (3, 4, 5):
            run_dn = pd.Series(1-up).rolling(L).sum().values == L
            run_up = pd.Series(up).rolling(L).sum().values == L
            P(f"   ── {B}분봉 연속{L}봉 ──")
            report(f"{L}연속 음봉", run_dn, cl)
            report(f"{L}연속 양봉", run_up, cl)
            # 연속 음봉 + 마지막 봉이 가장 큼(투매 가속)
            sz = np.abs(b["c"]-b["o"])/b["c"]
            accel = run_dn & (sz > pd.Series(sz).rolling(L).max().shift(1).values)
            report(f"{L}연속 음봉 + 가속", accel, cl)
