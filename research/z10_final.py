"""z10 최종 통합 — 결합 신호(z2 + 호가기울기 + 역롱숏) × 동시 보유 구조.

z9 결과: 호가기울기는 z2와 상관 0.002인 진짜 독립 정보원.
        결합 시 승률 60.62% → 67.83%, 명목3배 +97.6% → +255.3% (2023-01~, 3.70년)

여기서: 그 신호를 z4 동시보유 구조(증거금 점검 포함)에 넣어 빈도까지 올린다.
목표는 z4의 월 +26.77%를 넘는 것.

주의: 호가 데이터가 2023-01부터라 검증 3.7년. 표본 한계를 함께 보고한다."""
import sys, warnings, glob
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R/"research"))
import z2_lib as L2
from z2_final import build_sigs, exp_z as ez


def P(*a, end="\n"):
    print(" ".join(str(x) for x in a), end=end, flush=True)


S = "BTCUSDT"; MMR = L2.MMR; STEP = L2.STEP
d = L2.D[S]; idx = d["idx"]; rv = d["rv"]
O, H, Lo, C = d["o"], d["h"], d["l"], d["c"]
N = len(C); warm = 200000; G = N//STEP


def load(kind):
    fs = sorted(glob.glob(str(R/"data"/"parquet"/kind/S/"*.parquet")))
    if not fs: return None
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.sort_values("ts").drop_duplicates("ts")


def to_grid(df, cols):
    s = df.set_index("ts")[cols]
    ti = pd.DatetimeIndex(s.index).floor("min").as_unit("ms")
    gaps = np.diff(ti.asi8)/60000.0
    lag = int(max(1, np.median(gaps))) if len(gaps) else 1
    pos = idx.searchsorted(ti)
    out = {}
    for c in cols:
        a = np.full(N, np.nan)
        v = s[c].values.astype(float)
        ok = (pos >= 0) & (pos < N)
        a[pos[ok]] = v[ok]
        out[c] = pd.Series(a).ffill().shift(lag).values[::STEP][:G].astype(np.float32)
    return out


def egz(v, wm=200000//STEP, st=5000//STEP):
    v = np.asarray(v, float); z = np.full(len(v), np.nan)
    for i in range(wm, len(v), st):
        x = v[:i]; x = x[np.isfinite(x)]
        if len(x) < 500: continue
        mu, sd = x.mean(), x.std()
        if sd > 0: z[i:i+st] = (v[i:i+st]-mu)/sd
    return z


P("신호 구축...")
bd = load("bookdepth")
bm = to_grid(bd, [c for c in bd.columns if c != "ts"])
book = bm["n-5"]/np.maximum(bm["n-1"], 1)
m5 = load("metrics_5m")
mm = to_grid(m5, [c for c in m5.columns if c != "ts"])
revls = -mm["count_long_short_ratio"]
SG = build_sigs(S)
names = list(SG.keys())
Zn = {n: ez(np.asarray(SG[n], float)) for n in names}
ml = min(min(len(Zn[n]) for n in names), G)
z2p = np.nanmean(np.vstack([Zn[n][:ml] for n in names if n.endswith("4320")]), axis=0)
Zb, Zr, Zz = egz(book[:ml]), egz(revls[:ml]), z2p
SIGS = {
    "z2단독": Zz,
    "z2+호가": np.nanmean(np.vstack([Zz, Zb]), axis=0),
    "z2+호가+역롱숏": np.nanmean(np.vstack([Zz, Zb, Zr]), axis=0),
}
st0 = max(warm, int(np.argmax(np.isfinite(Zb)))*STEP)
YRS = (N-st0)/525600
P(f"  호가 가용 시작 {idx[st0].date()}, 검증 {YRS:.2f}년\n")


@numba.njit(cache=True)
def run(o, h, l, c, rvv, med, sig, scan, nslot, th, up, dn, maxh, tot, mmr,
        fee, s_lo, s_mid, s_hi, sq, start, n_end):
    cap = 1.0; peak = 1.0; mdd = 0.0
    ei = np.full(nslot, -1, np.int64); ep = np.zeros(nslot); es = np.zeros(nslot)
    ntr = 0; nwin = 0; sr = 0.0; nopen = 0; mo = 0; nmg = 0; worst = 0.0
    for t in range(start, n_end-1):
        for s in range(nslot):
            if ei[s] < 0: continue
            px = ep[s]; U = px*(1+up); D = px*(1-dn)
            hit = 0; res = 0.0
            if l[t] <= D: hit = 1; res = -dn
            elif h[t] >= U: hit = 1; res = up; nwin += 1
            elif (t-ei[s]) >= maxh: hit = 1; res = c[t]/px-1
            if hit:
                pnl = cap*es[s]*(res-fee/1e4)
                if pnl < -cap: pnl = -cap
                cap += pnl
                if cap > peak: peak = cap
                dd = cap/peak-1
                if dd < mdd: mdd = dd
                ntr += 1; sr += res
                ei[s] = -1; nopen -= 1
                if cap < 0.01:
                    return cap, mdd, ntr, nwin, sr, mo, nmg, worst
        if nopen > 0:
            un = 0.0; nom = 0.0
            for s in range(nslot):
                if ei[s] < 0: continue
                un += es[s]*(l[t]/ep[s]-1); nom += es[s]
            if un < worst: worst = un
            eqv = cap*(1+un)
            if eqv <= cap*nom*mmr:
                nmg += 1
                cap = eqv - cap*nom*mmr
                if cap < 0: cap = 0.0
                for s in range(nslot): ei[s] = -1
                nopen = 0
                if cap < 0.01:
                    return cap, mdd, ntr, nwin, sr, mo, nmg, worst
        if nopen > mo: mo = nopen
        if t % scan != 0: continue
        if nopen >= nslot: continue
        j = t//STEP
        if j >= len(sig): continue
        sv = sig[j]
        if not np.isfinite(sv) or sv < th: continue
        if np.isnan(rvv[t]) or rvv[t] <= 0: continue
        m = 1.0
        mmd = med[t]
        if np.isfinite(mmd) and rvv[t] > 0:
            x = mmd/rvv[t]
            m *= 0.1 if x < 0.1 else (4.0 if x > 4.0 else x)
        s_ = sq[j] if j < len(sq) else np.nan
        if np.isfinite(s_): m *= s_hi if s_ >= 0.97 else (s_mid if s_ >= 0.90 else s_lo)
        sz = tot/nslot*m
        if sz <= 0: continue
        if 1.0/sz - mmr < dn: continue
        for s in range(nslot):
            if ei[s] < 0:
                ei[s] = t; ep[s] = c[t]; es[s] = sz; nopen += 1
                break
    return cap, mdd, ntr, nwin, sr, mo, nmg, worst


med = np.full(N, np.nan)
for i in range(warm, N, 5000):
    x = rv[:i]; x = x[np.isfinite(x)]
    if len(x) >= 1000: med[i:i+5000] = np.median(x)


def qser(sig):
    sq = np.full(len(sig), np.nan)
    for i in range(warm//STEP, len(sig), 5000//STEP):
        x = sig[:i]; x = x[np.isfinite(x)]
        if len(x) < 500: continue
        sq[i:i+5000//STEP] = np.searchsorted(np.sort(x), sig[i:i+5000//STEP])/len(x)
    return sq


SQ = {k: qser(v) for k, v in SIGS.items()}


def sim(key, q=0.10, nslot=64, up=0.04, dn=0.04, tot=1.0, maxh=43200, scan=1, fee=10.0,
        s_lo=0.5, s_mid=1.0, s_hi=2.0):
    sig = SIGS[key]
    fin = sig[np.isfinite(sig)]
    th = float(np.quantile(fin, 1-q))
    cap, mdd, ntr, nwin, sr, mo, nmg, worst = run(
        O, H, Lo, C, rv, med, sig, scan, nslot, th, up, dn, maxh, float(tot), MMR,
        fee, s_lo, s_mid, s_hi, SQ[key], st0, N)
    if ntr < 20: return None
    ann = (max(cap, 1e-12)**(1/YRS)-1)*100
    mon = ((1+ann/100)**(1/12)-1)*100 if ann > -99.99 else -100
    return dict(ann=ann, mon=mon, mdd=mdd*100, fin=cap, n=ntr, wr=nwin/ntr*100,
                bp=sr/ntr*1e4, maxopen=mo, nmargin=nmg, worst=worst*100, ruin=cap < 0.01)


P("="*128); P("[1] 신호 × 총명목 (동시 64슬롯, ±4%, 상위10%). 셀 = 월% / MDD%"); P("="*128)
P(f"  {'신호':>16s} " + " ".join(f"{'명목'+str(t)+'배':>17s}" for t in (1, 2, 3, 4, 6, 8)))
BEST = None
for k in SIGS:
    row = []
    for tot in (1, 2, 3, 4, 6, 8):
        v = sim(k, tot=tot)
        if v is None or v["ruin"]: row.append("       파산      "); continue
        row.append(f"{v['mon']:+7.2f}%/{v['mdd']:6.1f}%")
        if BEST is None or v["mon"] > BEST[0]: BEST = (v["mon"], k, tot, v)
    P(f"  {k:>16s} " + " ".join(f"{x:>17s}" for x in row))

P("\n"+"="*128); P("[2] 문턱 × 슬롯 (결합신호, 총명목 6배, ±4%)"); P("="*128)
K = "z2+호가+역롱숏"
P(f"  {'문턱':>8s} " + " ".join(f"{'슬롯'+str(n):>20s}" for n in (16, 32, 64, 128)))
for q in (0.02, 0.05, 0.10, 0.20, 0.35):
    row = []
    for ns in (16, 32, 64, 128):
        v = sim(K, q=q, nslot=ns, tot=6.0)
        if v is None or v["ruin"]: row.append("        파산        "); continue
        row.append(f"{v['mon']:+6.2f}%/{v['wr']:4.1f}/{v['n']/YRS:5.0f}")
        if v["mon"] > BEST[0]: BEST = (v["mon"], K, 6.0, v)
    P(f"  상위{q*100:4.1f}% " + " ".join(f"{x:>20s}" for x in row))
P("  (셀 = 월% / 승률% / 연거래)")

P("\n"+"="*128); P("[3] 배리어 (결합신호, 슬롯64, 총명목 6배, 상위10%)"); P("="*128)
P(f"  {'배리어':>8s} {'연거래':>7s} {'승률%':>7s} {'거래당bp':>9s} {'월%':>9s} {'MDD%':>8s} {'최악미실현%':>11s}")
for bar in (0.02, 0.03, 0.04, 0.05, 0.06):
    mh = min(86400, max(4320, int(bar*1200000)))
    v = sim(K, up=bar, dn=bar, tot=6.0, maxh=mh)
    if v is None: continue
    P(f"  ±{bar*100:5.2f}% {v['n']/YRS:7.0f} {v['wr']:7.2f} {v['bp']:+9.1f} "
      + ("     파산" if v["ruin"] else f"{v['mon']:+9.2f}") + f" {v['mdd']:8.1f} {v['worst']:11.1f}")

P("\n"+"="*128); P("[4] 최종 3단 레버리지 + z4 대비"); P("="*128)
if BEST:
    mon, k, tot, v0 = BEST
    P(f"  최고 신호: {k}")
    P(f"  {'구간':>12s} {'총명목':>7s} {'월%':>10s} {'연%':>14s} {'MDD%':>8s} {'승률%':>7s} {'연거래':>7s} {'계좌청산':>8s}")
    for lb, tt in (("무레버리지", 1.0), ("중간", 2.0), ("중간+", 3.0), ("공격", 4.0),
                   ("최대공격", 6.0), ("극단", 8.0), ("초극단", 12.0)):
        v = sim(k, tot=tt)
        if v is None or v["ruin"]: P(f"  {lb:>12s} {tt:6.1f}x        파산"); continue
        P(f"  {lb:>12s} {tt:6.1f}x {v['mon']:10.2f} {v['ann']:14.1f} {v['mdd']:8.1f} "
          f"{v['wr']:7.2f} {v['n']/YRS:7.0f} {v['nmargin']:8d}")
    P(f"\n  대비: z4(z2 단독, 전 구간 6.7년) 극단 8배 = 월 +26.77%")
    v8 = sim("z2단독", tot=8.0)
    if v8 and not v8["ruin"]:
        P(f"        z2 단독을 같은 3.7년 구간에서 재측정: 월 {v8['mon']:+.2f}%")

P("\n"+"="*128); P("[5] 연도별 (결합신호, 총명목 6배)"); P("="*128)


def year_run(k, tot, yr):
    ii = np.where(idx.year == yr)[0]
    if len(ii) < 50000: return None
    a0, b0 = max(ii[0], st0), ii[-1]
    if a0 >= b0-43200: return None
    sig = SIGS[k]; fin = sig[np.isfinite(sig)]
    th = float(np.quantile(fin, 0.90))
    cap, mdd, ntr, nwin, sr, mo, nmg, worst = run(
        O, H, Lo, C, rv, med, sig, 1, 64, th, 0.04, 0.04, 43200, float(tot), MMR,
        10.0, 0.5, 1.0, 2.0, SQ[k], a0, b0)
    return None if ntr < 5 else (cap-1)*100


for k in SIGS:
    out = []
    for yr in range(2023, 2027):
        r = year_run(k, 6.0, yr)
        out.append(f"{yr}  n/a" if r is None else f"{yr} {r:+7.0f}%")
    P(f"  {k:>16s}: " + " | ".join(out))

P("\n"+"="*128); P("[6] 비용 내성 (결합신호, 총명목 6배)"); P("="*128)
for fee in (4, 7, 10, 15, 20, 30):
    row = []
    for tt in (1, 3, 6):
        v = sim(K, tot=tt, fee=fee)
        row.append("     파산" if (v is None or v["ruin"]) else f"{v['mon']:+8.2f}%")
    P(f"  왕복 {fee:2d}bp: 1배{row[0]}  3배{row[1]}  6배{row[2]}")
