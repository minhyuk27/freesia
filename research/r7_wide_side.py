"""R7 넓은 배리어에서 숏·ETH·양방향 재검토.

숏은 지금까지 모든 검정에서 죽었고(연 −9~−26%), ETH도 ±1.5%에서 전 구성 음수였다.
그러나 그 검정들은 전부 ±1~1.5% 배리어였다. ±4%는 수일 스윙을 잡으므로
드리프트·펀딩의 상대적 비중이 달라진다. 다시 잰다.

추가로 롱/숏을 동시에 운용하면 독립 베팅 수가 늘어나는지도 본다."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R/"research"))
import z0_20_lib as L2
from z0_20_final import build_sigs, exp_z


def P(*a, end="\n"):
    print(" ".join(str(x) for x in a), end=end, flush=True)


Y = L2.YRS; MMR = L2.MMR; STEP = L2.STEP
warm = 200000
DAT = {}
for S in ("BTCUSDT", "ETHUSDT"):
    d = L2.D[S]
    SG = build_sigs(S)
    names = list(SG.keys())
    Z = {n: exp_z(np.asarray(SG[n], float)) for n in names}
    ml = min(min(len(Z[n]) for n in names), len(d["c"])//STEP)
    sig = np.nanmean(np.vstack([Z[n][:ml] for n in names if n.endswith("4320")]), axis=0)
    DAT[S] = dict(o=d["o"], h=d["h"], l=d["l"], c=d["c"], idx=d["idx"], sig=sig)
    P(f"{S} 준비")
P("")


@numba.njit(cache=True)
def walk(o, h, l, c, sigv, th_hi, th_lo, up, dn, maxh, step, cool, nom, mmr,
         fee_bp, side_mode, start, n_end, ruin):
    """side_mode: 1=롱만, -1=숏만, 0=양방향(신호 상위면 롱, 하위면 숏)"""
    cap = 1.0; peak = 1.0; mdd = 0.0
    ntr = 0; nwin = 0; nlong = 0; sum_r = 0.0; sum_h = 0.0
    t = start
    while t < n_end - maxh - 2:
        i = t//step
        if i >= len(sigv): break
        s = sigv[i]
        if not np.isfinite(s): t += step; continue
        side = 0
        if side_mode == 1:
            if s >= th_hi: side = 1
        elif side_mode == -1:
            if s <= th_lo: side = -1
        else:
            if s >= th_hi: side = 1
            elif s <= th_lo: side = -1
        if side == 0: t += step; continue
        f = t+1; px = o[f]
        e = -1; res = 0.0
        end = min(f+maxh, n_end-1)
        if side == 1:
            U = px*(1.0+up); D = px*(1.0-dn)
            for j in range(f, end+1):
                if l[j] <= D: e = j; res = -dn; break
                if h[j] >= U: e = j; res = up; nwin += 1; break
            if e < 0: e = end; res = c[e]/px-1.0
            nlong += 1
        else:
            U = px*(1.0-up); D = px*(1.0+dn)      # 숏: 내려가면 이익
            for j in range(f, end+1):
                if h[j] >= D: e = j; res = -dn; break
                if l[j] <= U: e = j; res = up; nwin += 1; break
            if e < 0: e = end; res = px/c[e]-1.0
        pnl = cap*nom*(res - fee_bp/1e4)
        if pnl < -cap: pnl = -cap
        cap += pnl
        if cap > peak: peak = cap
        dd = cap/peak-1.0
        if dd < mdd: mdd = dd
        ntr += 1; sum_r += res; sum_h += (e-f)
        t = e+1+cool
        if cap < ruin: break
    return cap, mdd, ntr, nwin, nlong, sum_r, sum_h


def sim(S, q=0.10, up=0.04, dn=0.04, nom=1.0, maxh=43200, cool=0, fee=10.0,
        side=1, start=None, end=None):
    sg = DAT[S]["sig"]
    fin = sg[np.isfinite(sg)]
    th_hi = float(np.quantile(fin, 1-q)); th_lo = float(np.quantile(fin, q))
    s0 = warm if start is None else start
    e0 = len(DAT[S]["c"]) if end is None else end
    cap, mdd, ntr, nwin, nlong, sr, sh = walk(
        DAT[S]["o"], DAT[S]["h"], DAT[S]["l"], DAT[S]["c"], sg, th_hi, th_lo,
        up, dn, maxh, STEP, cool, float(nom), MMR, fee, side, s0, e0, 0.01)
    if ntr < 15: return None
    yy = Y if start is None else max((e0-s0)/525600, 0.05)
    ann = (max(cap, 1e-12)**(1/yy)-1)*100
    mon = ((1+ann/100)**(1/12)-1)*100 if ann > -99.99 else -100
    return dict(ann=ann, mon=mon, mdd=mdd*100, fin=cap, n=ntr, wr=nwin/ntr*100,
                nlong=nlong, bp=sr/ntr*1e4, hold=sh/ntr, ruin=cap < 0.01)


def cell(v):
    return "      파산      " if (v is None or v["ruin"]) else f"{v['ann']:+7.1f}%/{v['wr']:5.1f}%"


P("="*130); P("[1] 롱 vs 숏 — 배리어별 (BTC, 상위/하위 10%). 셀 = 연수익%/승률%"); P("="*130)
P(f"  {'배리어':>8s} " + " ".join(f"{lab:>16s}" for lab in ("롱 전용", "숏 전용", "양방향")))
for bar in (0.01, 0.02, 0.03, 0.04, 0.06, 0.08):
    mh = min(86400, max(4320, int(bar*1200000)))
    row = [cell(sim("BTCUSDT", up=bar, dn=bar, maxh=mh, side=sd)) for sd in (1, -1, 0)]
    P(f"  ±{bar*100:5.2f}% " + " ".join(f"{x:>16s}" for x in row))

P("\n"+"="*130); P("[2] 같은 표, ETH"); P("="*130)
P(f"  {'배리어':>8s} " + " ".join(f"{lab:>16s}" for lab in ("롱 전용", "숏 전용", "양방향")))
for bar in (0.01, 0.02, 0.03, 0.04, 0.06, 0.08):
    mh = min(86400, max(4320, int(bar*1200000)))
    row = [cell(sim("ETHUSDT", up=bar, dn=bar, maxh=mh, side=sd)) for sd in (1, -1, 0)]
    P(f"  ±{bar*100:5.2f}% " + " ".join(f"{x:>16s}" for x in row))

P("\n"+"="*130); P("[3] 문턱 × 종목 (배리어 ±4%, 롱 전용)"); P("="*130)
P(f"  {'문턱':>8s} " + " ".join(f"{s.replace('USDT',''):>26s}" for s in ("BTCUSDT", "ETHUSDT")))
for q in (0.02, 0.05, 0.10, 0.20, 0.35):
    row = []
    for S in ("BTCUSDT", "ETHUSDT"):
        v = sim(S, q=q, up=0.04, dn=0.04, maxh=43200)
        row.append("           파산           " if (v is None or v["ruin"])
                   else f"{v['ann']:+7.1f}%/{v['wr']:5.1f}%/{v['n']/Y:5.1f}건")
    P(f"  상위{q*100:4.1f}% " + " ".join(f"{x:>26s}" for x in row))
P("  (셀 = 연수익% / 승률% / 연거래)")

P("\n"+"="*130); P("[4] 숏이 죽는 이유 — 배리어별 상승/하락 우선도달 기저율"); P("="*130)


@numba.njit(cache=True)
def base_rate(o, h, l, c, up, dn, maxh, step, start, n_end):
    nu = 0; nd = 0
    for i in range(start//step, n_end//step):
        t = i*step
        if t+maxh >= n_end: break
        f = t+1; px = o[f]; U = px*(1.0+up); D = px*(1.0-dn)
        for j in range(f, min(f+maxh, n_end-1)+1):
            if l[j] <= D: nd += 1; break
            if h[j] >= U: nu += 1; break
    return nu, nd


P(f"  {'배리어':>8s} " + " ".join(f"{s.replace('USDT','')+' 상승우선%':>18s}" for s in ("BTCUSDT", "ETHUSDT")))
for bar in (0.01, 0.02, 0.04, 0.08):
    mh = min(86400, max(4320, int(bar*1200000)))
    row = []
    for S in ("BTCUSDT", "ETHUSDT"):
        D_ = DAT[S]
        nu, nd = base_rate(D_["o"], D_["h"], D_["l"], D_["c"], bar, bar, mh, STEP*4, warm, len(D_["c"]))
        row.append(f"{nu/max(nu+nd,1)*100:17.2f}%")
    P(f"  ±{bar*100:5.2f}% " + " ".join(f"{x:>18s}" for x in row))
P("  (50%보다 크면 롱이 구조적으로 유리 — 드리프트)")

P("\n"+"="*130); P("[5] 롱숏 병행이 독립 베팅을 늘리나 — BTC롱 + BTC숏 + ETH롱 병합"); P("="*130)


def events(S, q, bar, side):
    D_ = DAT[S]; sg = D_["sig"]
    fin = sg[np.isfinite(sg)]
    th_hi = float(np.quantile(fin, 1-q)); th_lo = float(np.quantile(fin, q))
    o, h, l, c = D_["o"], D_["h"], D_["l"], D_["c"]
    mh = min(86400, max(4320, int(bar*1200000)))
    out = []; t = warm; n_end = len(c)
    while t < n_end-mh-2:
        i = t//STEP
        if i >= len(sg) or not np.isfinite(sg[i]): t += STEP; continue
        if side == 1 and sg[i] < th_hi: t += STEP; continue
        if side == -1 and sg[i] > th_lo: t += STEP; continue
        f = t+1; px = o[f]; e = -1; res = 0.0
        end = min(f+mh, n_end-1)
        if side == 1:
            U = px*(1+bar); Dn = px*(1-bar)
            for j in range(f, end+1):
                if l[j] <= Dn: e = j; res = -bar; break
                if h[j] >= U: e = j; res = bar; break
            if e < 0: e = end; res = c[e]/px-1
        else:
            U = px*(1-bar); Dn = px*(1+bar)
            for j in range(f, end+1):
                if h[j] >= Dn: e = j; res = -bar; break
                if l[j] <= U: e = j; res = bar; break
            if e < 0: e = end; res = px/c[e]-1
        out.append((D_["idx"][f], D_["idx"][e], res))
        t = e+1
    return out


BL = events("BTCUSDT", 0.10, 0.04, 1)
BS = events("BTCUSDT", 0.10, 0.04, -1)
EL = events("ETHUSDT", 0.10, 0.04, 1)
P(f"  BTC롱 {len(BL)}건 / BTC숏 {len(BS)}건 / ETH롱 {len(EL)}건")


def merge_single(evs):
    al = sorted([e for L_ in evs for e in L_], key=lambda x: x[0])
    out = []; last = None
    for e in al:
        if last is not None and e[0] <= last: continue
        out.append(e); last = e[1]
    return out


def eq(ev, nom, fee=10.0):
    cap = 1.0; peak = 1.0; mdd = 0.0
    for (a, b, r) in ev:
        pnl = cap*nom*(r-fee/1e4)
        if pnl < -cap: pnl = -cap
        cap += pnl
        if cap > peak: peak = cap
        mdd = min(mdd, cap/peak-1)
        if cap < 0.01: return None
    return dict(ann=(cap**(1/Y)-1)*100, mdd=mdd*100, fin=cap,
                mon=((cap**(1/Y))**(1/12)-1)*100)


P(f"\n  {'구성':>20s} {'거래':>6s} {'연거래':>6s} {'승률%':>7s} " +
  " ".join(f"{'명목'+str(n)+'배':>13s}" for n in (1, 2, 3, 5)))
for lab, evs in (("BTC롱 단독", [BL]), ("BTC롱+숏", [BL, BS]),
                 ("BTC롱+ETH롱", [BL, EL]), ("셋 다", [BL, BS, EL])):
    ev = merge_single(evs)
    wr = np.mean([x[2] > 0 for x in ev])*100
    row = []
    for nom in (1, 2, 3, 5):
        r = eq(ev, nom)
        row.append("        파산" if r is None else f"{r['ann']:+12.1f}")
    P(f"  {lab:>20s} {len(ev):6d} {len(ev)/Y:6.1f} {wr:7.2f} " + " ".join(f"{x:>13s}" for x in row))
