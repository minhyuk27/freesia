"""z0.13 — '하향 이탈 매수' 단일 패턴 전략.
전저점 또는 라운드 넘버를 아래로 깨는 순간 매수(청산 스톱 유동성 공급).
임계는 전부 확장창/고정 규칙 → 룩어헤드 없음.
사용: python research/z0_13_break.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
import exit_tier as X
from absfloor import met
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
SYM = ("BTCUSDT", "ETHUSDT")
TE = pd.Timestamp("2024-01-01", tz="UTC"); REC = pd.Timestamp("2025-01-01", tz="UTC")
D = {}
for s in SYM:
    k = store.load("klines_1m", s).set_index("ts")
    D[s] = dict(o=k.open.values, h=k.high.values, l=k.low.values, c=k.close.values,
                qv=k.quote_volume.values, idx=k.index)


def bars(s, B):
    d = D[s]; c = d["c"]; n = len(c)//B
    st = np.arange(n)*B
    return dict(o=d["o"][st], h=np.array([d["h"][i*B:(i+1)*B].max() for i in range(n)]),
                l=np.array([d["l"][i*B:(i+1)*B].min() for i in range(n)]),
                c=c[st+B-1], qv=np.array([d["qv"][i*B:(i+1)*B].sum() for i in range(n)]),
                st=st, idx=d["idx"][st], B=B)


def build(s, B, mode, LB=50, H=4, rounddiv=1.0, fee=7.0, need_unrecov=True, cool=1):
    """mode: 'prevlow' 전저점 하향이탈 / 'round' 라운드 하향이탈 / 'both' 동시"""
    b = bars(s, B); d = D[s]
    cl = b["c"]; n = len(cl)
    prev_low = pd.Series(b["l"]).rolling(LB).min().shift(1).values
    broke = b["l"] < prev_low
    unrecov = b["c"] < prev_low
    mag = 10**np.floor(np.log10(np.maximum(cl, 1)))
    div = mag*rounddiv
    below = cl < np.round(cl/div)*div
    prev_below = np.roll(below, 1); prev_below[0] = below[0]
    rbreak = below & ~prev_below
    if mode == "prevlow":
        sig = broke & (unrecov if need_unrecov else ~unrecov)
    elif mode == "round":
        sig = rbreak
    else:
        sig = (broke & unrecov) | rbreak
    E = []; last = -10**9
    for i in range(max(LB, 5), n-H-1):
        if not sig[i]: continue
        if i - last < cool: continue
        last = i
        a = int(b["st"][i]) + B - 1          # 봉 종가 시점 진입(시장가)
        e_i = i + H
        bb = int(b["st"][e_i]) + B - 1
        if bb >= len(d["c"]): break
        px = cl[i]; ex = cl[e_i]
        lo_ = d["l"][a:bb+1].min()
        E.append((a, bb, float(ex/px - 1.0 - fee/1e4), float(max(1.0-lo_/px, 0)),
                  d["idx"][a], (s, mode, B), 0.01, 1.0))
    return E


def rep(E, lab, full=False):
    if len(E) < 60: P(f"  {lab:46s} 표본부족({len(E)})"); return None
    r = np.array([x[2] for x in E]); mae = np.array([x[3] for x in E])
    out = f"  {lab:46s} n={len(E):5d}({len(E)/6.7:4.0f}/년) {r.mean()*1e4:+6.1f}bp 승률{(r>0).mean():.3f} |"
    best = None
    for w, Nm, tag in ((1.0, 1.0, "무레버"), (2.0, 3.0, "저레버"), (3.0, 5.0, "중간"), (5.0, 8.0, "공격")):
        eq, fl, liq = X.sim(E, w, Nm); m = met(eq, fl, liq)
        out += f" {tag}{'파산' if m is None else f'{m[0]:+7.1f}%'}"
        if m and (best is None or m[0] > best[0]): best = (m[0], m[1], m[2], m[3], m[4], tag, m[5])
    P(out)
    if full and best:
        mon = (best[6].resample("ME").last()/best[6].resample("ME").first()-1).dropna()
        P(f"      최고 {best[5]}: MDD {best[1]:.0f}% 최악 {best[2]:.0f}% 24~ {best[3]:+.0f}% 25~ {best[4]:+.0f}% 월승률 {float((mon>0).mean()):.2f} | MAE95 {np.quantile(mae,.95)*100:.1f}%")
        df = pd.DataFrame([(x[4], x[2]) for x in E], columns=["ts", "r"]).set_index("ts")
        g = df.groupby(df.index.year).r
        P("      연도별 " + " ".join(f"{y}{v*1e4:+5.0f}×{n}" for (y, v), (_, n) in zip(g.mean().items(), g.size().items())))
    return best


P("[1] 전저점 하향 이탈 (미회복) — 봉·되돌아보기·보유 격자")
for B in (60, 240):
    for LB in (20, 50, 100):
        for H in (2, 4, 8):
            E = sorted(sum([build(s, B, "prevlow", LB, H) for s in SYM], []), key=lambda z: z[0])
            rep(E, f"{B}분봉 전저점{LB} 보유{H}봉({B*H}분)")

P("\n[2] 라운드 넘버 하향 이탈 — 단위·봉·보유")
for B in (60, 240):
    for rd in (0.1, 0.5, 1.0):
        for H in (2, 4, 8):
            E = sorted(sum([build(s, B, "round", H=H, rounddiv=rd) for s in SYM], []), key=lambda z: z[0])
            rep(E, f"{B}분봉 라운드{rd}×자릿수 보유{H}봉")

P("\n[3] 대조 — 이탈 후 '회복'한 경우 (원리 확인)")
for B in (60, 240):
    for LB in (50,):
        for H in (4,):
            E = sorted(sum([build(s, B, "prevlow", LB, H, need_unrecov=False) for s in SYM], []), key=lambda z: z[0])
            rep(E, f"{B}분봉 전저점{LB} 보유{H}봉 [회복한 경우]")

P("\n[4] 상위 후보 상세")
CAND = [(240, "prevlow", 50, 4, 1.0), (60, "prevlow", 50, 8, 1.0),
        (240, "round", 50, 4, 1.0), (60, "round", 50, 4, 1.0), (240, "both", 50, 4, 1.0)]
best_all = []
for B, md, LB, H, rd in CAND:
    E = sorted(sum([build(s, B, md, LB, H, rounddiv=rd) for s in SYM], []), key=lambda z: z[0])
    b = rep(E, f"{B}분봉 {md} LB{LB} H{H}", full=True)
    if b: best_all.append((b[0], B, md, LB, H, rd))

P("\n[5] 최선 후보 비용 민감도 · 플라시보")
@numba.njit(cache=True)
def rnd(o, c, st, hd, fee):
    out = np.zeros(len(st))
    for i in range(len(st)):
        e = st[i]+hd[i]
        if e >= len(c): e = len(c)-1
        out[i] = (c[e]/o[st[i]]-1.0) - fee/1e4
    return out
if best_all:
    best_all.sort(reverse=True)
    _, B, md, LB, H, rd = best_all[0]
    P(f"  선택: {B}분봉 {md} LB{LB} H{H}")
    for fee, lab in ((4.0, "메이커왕복4bp"), (7.0, "메+테7bp"), (10.0, "테이커10bp"), (14.0, "비관14bp")):
        E = sorted(sum([build(s, B, md, LB, H, rounddiv=rd, fee=fee) for s in SYM], []), key=lambda z: z[0])
        rep(E, f"  비용 {lab}")
    E = sorted(sum([build(s, B, md, LB, H, rounddiv=rd) for s in SYM], []), key=lambda z: z[0])
    r = np.array([x[2] for x in E]); hd = np.array([x[1]-x[0] for x in E])
    rng = np.random.default_rng(23); nul = []
    for _ in range(400):
        acc = []
        for s in SYM:
            o = D[s]["o"]; c = D[s]["c"]; N = len(c)
            cnt = max(len(E)//2, 1)
            acc.append(rnd(o, c, rng.integers(0, N-3000, cnt).astype(np.int64),
                           rng.choice(hd, cnt).astype(np.int64), 7.0))
        nul.append(np.concatenate(acc).mean()*1e4)
    nul = np.array(nul)
    P(f"  플라시보 400회: 위약 평균 {nul.mean():+.2f}bp 최대 {nul.max():+.2f} | 실제 {r.mean()*1e4:+.2f}bp → 백분위 {(nul < r.mean()*1e4).mean()*100:.1f}%")
