"""z2 방향3 — 호가창(bookDepth) 분석. ±1% 구간 유동성으로 우선도달 예측.
사용: python research/z2_book.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
SYM = ("BTCUSDT", "ETHUSDT")
BAR = 0.01; MAXH = 4320
TH_MKT, TH_LIM = 55.0, 53.5


@numba.njit(cache=True)
def touch_at(h, l, c, idxs, bar, maxh):
    """주어진 분 인덱스들에서 ±bar 우선도달"""
    m = len(idxs)
    out = np.zeros(m, np.int8); tt = np.zeros(m, np.int32)
    n = len(c)
    for k in range(m):
        i = idxs[k]
        if i < 0 or i >= n-1: continue
        p = c[i]; U = p*(1.0+bar); Dn = p*(1.0-bar)
        r = 0; rt = maxh
        end = min(i+maxh, n-1)
        for j in range(i+1, end+1):
            hi = h[j] >= U; lo = l[j] <= Dn
            if lo: r = -1; rt = j-i; break
            if hi: r = 1; rt = j-i; break
        out[k] = r; tt[k] = rt
    return out, tt


DATA = {}
for s in SYM:
    k = store.load("klines_1m", s).set_index("ts")
    bd = store.load("bookdepth", s)
    if bd is None or len(bd) == 0:
        P(f"{s}: 호가 데이터 없음"); continue
    bd = bd.sort_values("ts").reset_index(drop=True)
    # 분 단위로 정렬 (각 분의 마지막 스냅샷)
    ts = pd.to_datetime(bd.ts, utc=True)
    bd["m"] = ts.dt.floor("min")
    bd = bd.groupby("m").last().reset_index()
    kidx = pd.DatetimeIndex(k.index)
    pos = kidx.searchsorted(pd.DatetimeIndex(bd.m))
    ok = (pos > 0) & (pos < len(k))
    bd = bd[ok].reset_index(drop=True); pos = pos[ok] - 1     # 직전 완결 분 (룩어헤드 방지)
    y, tt = touch_at(k.high.values, k.low.values, k.close.values, pos.astype(np.int64), BAR, MAXH)
    DATA[s] = dict(bd=bd, pos=pos, y=y, tt=tt, k=k)
    dec = y != 0
    P(f"{s}: 호가 스냅샷 {len(bd):,} ({bd.m.min().date()}~{bd.m.max().date()}) "
      f"판정 {dec.mean()*100:.1f}% 위먼저 {(y[dec]==1).mean()*100:.2f}%")
    P(f"   컬럼: {[c for c in bd.columns if c.startswith(('n','d'))]}")


def evq(sig, y, lab, qs=(0.05, 0.10, 0.20), show=True):
    dec = (y != 0) & np.isfinite(sig)
    if dec.sum() < 500: return None
    q = pd.Series(np.where(dec, sig, np.nan)).rank(pct=True).values
    best = 0; row = f"    {lab:34s}"
    for qq in qs:
        hi = dec & (q >= 1-qq); lo = dec & (q <= qq)
        if hi.sum() < 50 or lo.sum() < 50: continue
        ah = (y[hi] == 1).mean()*100; al = (y[lo] == -1).mean()*100
        both = (ah*hi.sum()+al*lo.sum())/(hi.sum()+lo.sum())
        row += f" | {qq*100:.0f}% 위{ah:5.2f}/아래{al:5.2f}/합{both:5.2f}%"
        best = max(best, both)
    if show: P(row + (f"  ★{best:.2f}%" if best > TH_LIM else ""))
    return best


P("\n" + "="*140)
P("[1] 호가 불균형 — ±1% 구간 유동성")
P("="*140)
for s in DATA:
    d = DATA[s]; bd = d["bd"]; y = d["y"]
    P(f"\n  ══ {s} ══")
    SIG = {}
    # 기본 불균형: (매수잔량 - 매도잔량)/(합)  ※ n-1=아래 1%까지 매수, n+1=위 1%까지 매도
    for lv in (1, 2, 3, 5):
        b = bd[f"n-{lv}"].values; a = bd[f"n+{lv}"].values
        SIG[f"불균형±{lv}%"] = (b-a)/(b+a)
    # 누적 1~5%
    bb = sum(bd[f"n-{i}"].values for i in (1, 2, 3, 4, 5))
    aa = sum(bd[f"n+{i}"].values for i in (1, 2, 3, 4, 5))
    SIG["불균형누적1~5%"] = (bb-aa)/(bb+aa)
    # 기울기: 깊이가 멀어질수록 어떻게 변하나
    SIG["매수기울기"] = (bd["n-5"].values-bd["n-1"].values)/bd["n-1"].values
    SIG["매도기울기"] = (bd["n+5"].values-bd["n+1"].values)/bd["n+1"].values
    SIG["기울기차"] = SIG["매수기울기"]-SIG["매도기울기"]
    # 총 유동성 수준 (얇으면 움직이기 쉽다)
    tot = bb+aa
    SIG["총유동성"] = tot
    SIG["총유동성_상대"] = tot/pd.Series(tot).rolling(2880).mean().values
    # 불균형의 변화
    for w in (20, 60, 240, 1440):
        SIG[f"불균형변화{w}"] = pd.Series(SIG["불균형±1%"]).diff(w).values
        SIG[f"불균형평균{w}"] = pd.Series(SIG["불균형±1%"]).rolling(w).mean().values
    res = []
    for nm, v in SIG.items():
        b_ = evq(np.asarray(v, float), y, nm)
        if b_: res.append((b_, nm))
    res.sort(reverse=True)
    d["SIG"] = SIG
    P(f"    → 최고: " + ", ".join(f"{n}({b:.2f}%)" for b, n in res[:5]))

P("\n" + "="*140)
P("[2] 호가 + 가격 신호 결합")
P("="*140)
for s in DATA:
    d = DATA[s]; bd = d["bd"]; y = d["y"]; k = d["k"]; pos = d["pos"]
    c = k.close.values; qv = k.quote_volume.values; tb = k.taker_buy_quote_volume.values
    lr = pd.Series(np.log(c)).diff()
    PR = {}
    for w in (1440, 4320):
        us = pd.Series(np.where(lr > 0, lr**2, 0)).rolling(w).sum()
        ds = pd.Series(np.where(lr < 0, lr**2, 0)).rolling(w).sum()
        PR[f"반변동성{w}"] = ((us-ds)/(us+ds)).values[pos]
        PR[f"OFI{w}"] = (2*pd.Series(tb).rolling(w).sum()/pd.Series(qv).rolling(w).sum()-1).values[pos]
        chg = pd.Series(c).diff(w)
        path = pd.Series(np.abs(np.diff(c, prepend=c[0]))).rolling(w).sum()
        PR[f"효율비{w}"] = (chg/path.replace(0, np.nan)).values[pos]
    P(f"\n  ══ {s} ══")
    for nm, v in PR.items():
        evq(np.asarray(v, float), y, f"[가격] {nm}")
    # 결합
    def z(v):
        v = np.asarray(v, float); m = np.isfinite(v)
        o = np.full(len(v), np.nan)
        if m.sum() > 100:
            mu = v[m].mean(); sd = v[m].std()
            if sd > 0: o[m] = (v[m]-mu)/sd
        return o
    SIG = d["SIG"]
    for lab, names in (("호가3개", ["불균형±1%", "불균형누적1~5%", "기울기차"]),
                       ("가격3개", None),
                       ("호가+가격", None)):
        if lab == "호가3개":
            M = np.vstack([z(SIG[n]) for n in names])
        elif lab == "가격3개":
            M = np.vstack([z(PR[n]) for n in ("반변동성4320", "OFI1440", "효율비1440")])
        else:
            M = np.vstack([z(SIG["불균형±1%"]), z(SIG["불균형누적1~5%"]),
                           z(PR["반변동성4320"]), z(PR["OFI1440"]), z(PR["효율비1440"])])
        evq(np.nanmean(M, axis=0), y, f"결합 {lab}")
