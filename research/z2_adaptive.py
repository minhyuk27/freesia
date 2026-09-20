"""z2 방향1·2 — 적응형 지표 + 방향 강도 지표. ±1% 우선도달 예측.
적응형: 고정 파라미터 대신 최근 데이터로 지표의 창·임계를 능동 조정.
강도: 상승/하락 초기의 '힘'을 여러 이론으로 측정.
사용: python research/z2_adaptive.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
SYM = ("BTCUSDT", "ETHUSDT")
STEP = 15; BAR = 0.01; MAXH = 4320
TH_MKT, TH_LIM = 55.0, 53.5


@numba.njit(cache=True)
def touch(h, l, c, bar, maxh, step):
    n = len(c); m = n//step
    out = np.zeros(m, np.int8); tt = np.zeros(m, np.int32)
    for k in range(m):
        i = k*step
        p = c[i]; U = p*(1.0+bar); Dn = p*(1.0-bar)
        r = 0; rt = maxh
        end = min(i+maxh, n-1)
        for j in range(i+1, end+1):
            hi = h[j] >= U; lo = l[j] <= Dn
            if lo: r = -1; rt = j-i; break
            if hi: r = 1; rt = j-i; break
        out[k] = r; tt[k] = rt
    return out, tt


D = {}
for s in SYM:
    k = store.load("klines_1m", s).set_index("ts")
    o_, t_ = touch(k.high.values, k.low.values, k.close.values, BAR, MAXH, STEP)
    D[s] = dict(k=k, o=k.open.values, h=k.high.values, l=k.low.values, c=k.close.values,
                qv=k.quote_volume.values, tb=k.taker_buy_quote_volume.values,
                nt=k.trades.values, idx=k.index, y=o_, tt=t_)
    dec = o_ != 0
    P(f"{s}: 표본 {len(o_):,} 판정 {dec.mean()*100:.1f}% 위먼저 {(o_[dec]==1).mean()*100:.2f}%")


def ev(name, sig, s, qs=(0.05, 0.10, 0.20)):
    """신호 극단 구간에서의 방향 정확도. 양수=위먼저 예측."""
    d = D[s]; y = d["y"]; n = min(len(sig), len(y))
    sig = sig[:n]; y = y[:n]
    dec = (y != 0) & np.isfinite(sig)
    if dec.sum() < 2000: return None
    q = pd.Series(np.where(dec, sig, np.nan)).rank(pct=True).values
    best = 0; row = f"    {name:30s}"
    for qq in qs:
        hi = dec & (q >= 1-qq); lo = dec & (q <= qq)
        ah = (y[hi] == 1).mean()*100; al = (y[lo] == -1).mean()*100
        both = (ah*hi.sum() + al*lo.sum())/(hi.sum()+lo.sum())
        row += f" | 상하위{qq*100:.0f}% 위{ah:5.2f}/아래{al:5.2f}/합{both:5.2f}%"
        best = max(best, both)
    P(row + (f"  ★{best:.2f}%" if best > TH_LIM else ""))
    return best


P("\n" + "="*150)
P("[방향2] 방향 강도 지표 — 상승/하락 초기의 '힘'을 여러 이론으로 측정")
P("="*150)
for s in SYM:
    d = D[s]; c = d["c"]; qv = d["qv"]; tb = d["tb"]; nt = d["nt"]
    h = d["h"]; l = d["l"]; o = d["o"]
    lr = pd.Series(np.log(c)).diff()
    P(f"\n  ══ {s} ══")
    SIG = {}
    # 1) 주문흐름 불균형 (여러 창)
    for w in (15, 60, 240, 1440):
        SIG[f"OFI{w}"] = (2*pd.Series(tb).rolling(w).sum()/pd.Series(qv).rolling(w).sum()-1).values[::STEP]
    # 2) 거래량 가중 방향 (체결 강도)
    for w in (60, 240):
        sgn = np.sign(pd.Series(c).diff().values)
        SIG[f"거래량가중방향{w}"] = (pd.Series(sgn*qv).rolling(w).sum()/pd.Series(qv).rolling(w).sum()).values[::STEP]
    # 3) 가격충격 방향 (Kyle) — 같은 거래량에 더 크게 움직인 쪽
    for w in (60, 240):
        up_imp = pd.Series(np.where(lr > 0, lr, 0)).rolling(w).sum()/pd.Series(np.where(lr > 0, qv, 0)).rolling(w).sum()
        dn_imp = pd.Series(np.where(lr < 0, -lr, 0)).rolling(w).sum()/pd.Series(np.where(lr < 0, qv, 0)).rolling(w).sum()
        SIG[f"충격비대칭{w}"] = ((up_imp-dn_imp)/(up_imp+dn_imp)).values[::STEP]
    # 4) 상방/하방 변동성 비대칭 (반변동성)
    for w in (60, 240, 1440):
        us = pd.Series(np.where(lr > 0, lr**2, 0)).rolling(w).sum()
        ds = pd.Series(np.where(lr < 0, lr**2, 0)).rolling(w).sum()
        SIG[f"반변동성비대칭{w}"] = ((us-ds)/(us+ds)).values[::STEP]
    # 5) 꼬리 비대칭 (흡수 흔적)
    for w in (60, 240):
        uw = pd.Series(h-np.maximum(c, o)).rolling(w).sum()
        dw = pd.Series(np.minimum(c, o)-l).rolling(w).sum()
        SIG[f"꼬리비대칭{w}"] = ((dw-uw)/(dw+uw)).values[::STEP]
    # 6) 체결 건수 비대칭 (소액 vs 대형)
    for w in (60, 240):
        asz = pd.Series(qv/np.maximum(nt, 1))
        SIG[f"체결크기추세{w}"] = (asz.rolling(w).mean()/asz.rolling(w*4).mean()).values[::STEP]
    # 7) 추세 강도 × 방향 (효율비 부호)
    for w in (60, 240, 1440):
        chg = pd.Series(c).diff(w)
        path = pd.Series(np.abs(np.diff(c, prepend=c[0]))).rolling(w).sum()
        SIG[f"효율비부호{w}"] = (chg/path.replace(0, np.nan)).values[::STEP]
    # 8) 모멘텀 (여러 창)
    for w in (60, 240, 1440, 4320, 10080):
        SIG[f"모멘텀{w}"] = pd.Series(c).pct_change(w).values[::STEP]
    # 9) 최근 고저 대비 위치
    for w in (1440, 10080):
        hh = pd.Series(h).rolling(w).max(); ll = pd.Series(l).rolling(w).min()
        SIG[f"위치{w}"] = ((c-ll)/(hh-ll).replace(0, np.nan)).values[::STEP]
    # 10) 프리미엄
    try:
        pr = store.load("premium_1m", s).set_index("ts").close.reindex(d["idx"]).values
        for w in (60, 240):
            SIG[f"프리미엄{w}"] = pd.Series(pr).rolling(w).mean().values[::STEP]
            SIG[f"프리미엄변화{w}"] = pd.Series(pr).diff(w).values[::STEP]
    except Exception:
        pass
    res = []
    for nm, v in SIG.items():
        b = ev(nm, np.asarray(v, float), s)
        if b: res.append((b, nm))
    res.sort(reverse=True)
    P(f"    → 최고 3개: " + ", ".join(f"{n}({b:.2f}%)" for b, n in res[:3]))

P("\n" + "="*150)
P("[방향1] 적응형 — 지표의 창 길이를 최근 성과로 능동 선택")
P("="*150)
@numba.njit(cache=True)
def adaptive_pick(sigs, y, lookback, step_idx):
    """각 시점에서 최근 lookback개 구간에 대해 가장 잘 맞춘 지표를 선택해 사용."""
    nsig, n = sigs.shape
    out = np.zeros(n)
    pick = np.zeros(n, np.int8)
    for t in range(lookback, n):
        best = -1.0; bi = 0
        for s_ in range(nsig):
            acc = 0.0; cnt = 0.0
            for j in range(t-lookback, t):
                if y[j] == 0: continue
                v = sigs[s_, j]
                if not np.isfinite(v): continue
                pred = 1 if v > 0 else -1
                if pred == y[j]: acc += 1.0
                cnt += 1.0
            if cnt > 20:
                a = acc/cnt
                if a > best: best = a; bi = s_
        out[t] = sigs[bi, t]; pick[t] = bi
    return out, pick
P("  (적응형은 호가 데이터 준비 후 통합 실행)")
