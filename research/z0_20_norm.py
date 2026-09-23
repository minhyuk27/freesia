"""z0.20 신호강도 배분의 공정 비교 — 실효 평균 배수를 맞춘 뒤 비교한다.
'신호가 강하면 크게 건다'의 이득이 진짜 배분 효과인지, 그냥 레버리지를 키운 것인지 분리."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R/"research"))
import z0_20_lib as L


def P(*a, end="\n"):
    print(" ".join(str(x) for x in a), end=end, flush=True)


S = "BTCUSDT"; Y = L.YRS
d = L.D[S]; idx = d["idx"]; rv = d["rv"]; sig = d["sig"]
warm = 200000; step = 5000

med = np.full(len(rv), np.nan)
for i in range(warm, len(rv), step):
    x = rv[:i]; x = x[np.isfinite(x)]
    if len(x) >= 1000: med[i:i+step] = np.median(x)

sq = np.full(len(sig), np.nan); sw = warm//L.STEP; ss = step//L.STEP
for i in range(sw, len(sig), ss):
    x = sig[:i]; x = x[np.isfinite(x)]
    if len(x) < 1000: continue
    sq[i:i+ss] = np.searchsorted(np.sort(x), sig[i:i+ss])/len(x)


def getev(q, ek, up, dn, cool):
    th = np.quantile(d["fin"], 1-q)
    ei, ex, rr, mk, _ = L._tr(d["o"], d["h"], d["l"], d["c"], sig, th, up, dn, L.MAXH,
                              L.STEP, float(ek), rv, 240, 0.0002, cool, 300000)
    k = ei >= warm
    return [(idx[a], idx[b], r, m, a) for a, b, r, m in zip(ei[k], ex[k], rr[k], mk[k])]


Q, EK, UP, DN, COOL = 0.15, 0.0, 0.015, 0.015, 60
ev = getev(Q, EK, UP, DN, COOL)
P(f"기준 상위15% 시장가 ±1.5% 쿨60 — {len(ev)}건 (연 {len(ev)/Y:.0f}건)\n")


def mults(ev, vol=True, smode="none", scap=2.0, sfl=0.5, norm=False):
    out = []
    for (f, e, res, m, ii) in ev:
        mu = 1.0
        if vol:
            r_ = rv[ii]; mm = med[ii]
            if np.isfinite(r_) and np.isfinite(mm) and r_ > 0:
                mu *= float(np.clip(mm/r_, 0.1, 4.0))
        if smode != "none":
            s_ = sq[ii//L.STEP] if ii//L.STEP < len(sq) else np.nan
            if np.isfinite(s_):
                if smode == "lin":
                    mu *= float(np.clip(sfl+(scap-sfl)*(s_-0.5)/0.5, sfl, scap))
                elif smode == "step":
                    mu *= (scap if s_ >= 0.97 else (1.0 if s_ >= 0.90 else sfl))
        out.append(mu)
    out = np.array(out)
    if norm:   # 확장창 평균으로 정규화 — 시점별 과거 평균만 사용
        cm = np.concatenate([[np.nan], np.cumsum(out)[:-1]/np.arange(1, len(out))])
        cm = np.where(np.isfinite(cm), cm, np.nanmean(out[:50]))
        out = out/np.maximum(cm, 1e-6)
    return out


def run(ev, mu, lev, dn, frac=1.0, yr=None):
    cap = 1e7; cur = []
    for (f, e, res, m, ii), u in zip(ev, mu):
        if yr and f.year != yr: continue
        lv = lev*u
        if lv <= 0: cur.append(cap); continue
        liq = 1.0/lv - L.MMR; r = res
        if r <= -dn and liq < dn: r = -1.0/lv
        fi = 2.0 if (m//2) == 1 else 5.0
        fo = 2.0 if (m % 2) == 1 else 5.0
        pnl = cap*frac*lv*(r-(fi+fo)/1e4)
        if pnl < -cap*frac: pnl = -cap*frac
        cap += pnl; cur.append(cap)
        if cap <= 1e5: break
    if not cur: return None
    eq = np.array(cur)
    if eq[-1] <= 2e5: return None
    yy = Y if yr is None else 1.0
    peak = np.maximum.accumulate(np.concatenate([[1e7], eq]))[1:]
    return dict(ann=((eq[-1]/1e7)**(1/yy)-1)*100, mdd=(eq/peak-1).min()*100, fin=eq[-1]/1e7)


def cell(r):
    return f"{r['ann']:+8.1f}%/{r['mdd']:6.1f}%" if r else "         파산   "


SCH = [("없음(고정)", dict(vol=False)), ("변동성만", dict(vol=True)),
       ("신호선형2x만", dict(vol=False, smode="lin")), ("신호계단2x만", dict(vol=False, smode="step")),
       ("변동성+선형2x", dict(vol=True, smode="lin")), ("변동성+계단2x", dict(vol=True, smode="step"))]

P("="*122); P("[1] 승수 평균 — 어떤 방식이 실제로 레버리지를 키우는가"); P("="*122)
MU = {}
for nm, kw in SCH:
    m = mults(ev, **kw); MU[nm] = m
    P(f"  {nm:>14s}  평균 {m.mean():5.3f}  중앙 {np.median(m):5.3f}  "
      f"10% {np.percentile(m,10):4.2f}  90% {np.percentile(m,90):4.2f}")

P("\n"+"="*122); P("[2] 실효 평균 배수를 맞춘 공정 비교 — 같은 위험량에서 누가 이기나"); P("="*122)
for tgt in (1.0, 2.0, 3.0, 5.0, 8.0):
    P(f"\n  ── 실효 평균 {tgt:.0f}배 ──")
    for nm, kw in SCH:
        m = MU[nm]; lev = tgt/m.mean()
        r = run(ev, m, lev, DN)
        P(f"    {nm:>14s} (기본 {lev:5.2f}x) {cell(r)}")

P("\n"+"="*122); P("[3] 확장창 정규화(평균 1 강제) — 순수 배분 효과만"); P("="*122)
P(f"  {'방식':>14s} {'실효1x':>17s} {'실효3x':>17s} {'실효5x':>17s}")
for nm, kw in SCH[1:]:
    mn = mults(ev, norm=True, **kw)
    row = [cell(run(ev, mn, tgt/mn.mean(), DN)) for tgt in (1.0, 3.0, 5.0)]
    P(f"  {nm:>14s} " + " ".join(f"{x:>17s}" for x in row))

P("\n"+"="*122); P("[4] 최종 확정 구성 3단 레버리지 + 연도별 (변동성목표만)"); P("="*122)
m = MU["변동성만"]
P(f"  {'구간':>12s} {'배수':>6s} {'실효':>6s} {'연수익%':>10s} {'MDD%':>8s} {'최종배수':>11s}")
for lb, lev in (("무레버리지", 1), ("중간", 3), ("중간+", 5), ("공격", 8), ("최대공격", 12)):
    r = run(ev, m, lev, DN)
    if not r: P(f"  {lb:>12s} {lev:5d}x        파산"); continue
    P(f"  {lb:>12s} {lev:5d}x {lev*m.mean():6.2f} {r['ann']:10.1f} {r['mdd']:8.1f} {r['fin']:11.2f}")
for lv in (1, 3, 5):
    out = []
    for yr in range(2020, 2027):
        r = run(ev, m, lv, DN, yr=yr)
        out.append(f"{yr} {(r['fin']-1)*100:+7.0f}%" if r else f"{yr}   파산")
    P(f"  연도별({lv}x): " + " | ".join(out))

rr = np.array([e[2] for e in ev])
se = np.sqrt(0.25/len(rr))*100
P(f"\n  승률 {np.mean(rr>0)*100:.2f}% (n={len(rr)}), 50% 대비 z = {(np.mean(rr>0)*100-50)/se:.2f}")
hold = np.array([(e[1]-e[0]).total_seconds()/60 for e in ev])
gap = np.array([(ev[i+1][0]-ev[i][1]).total_seconds()/60 for i in range(len(ev)-1)])
P(f"  보유 중앙 {np.median(hold):.0f}분 / 공백 중앙 {np.median(gap):.0f}분 / 노출 "
  f"{hold.sum()/((idx[-1]-idx[warm]).total_seconds()/60)*100:.1f}%")
ts = pd.DatetimeIndex([e[0] for e in ev])
mm = pd.Series(1, index=ts).resample("ME").sum()
P(f"  월 거래 중앙 {mm.median():.0f} 평균 {mm.mean():.1f} | 거래 0건 월 {int((mm==0).sum())}/{len(mm)}")
