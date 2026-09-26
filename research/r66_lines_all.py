"""R66 — 차트에 선을 긋는 방법 전체(5계열 약 40종, 선 120개 안팎)를 한 엔진으로 전략화·검증.

계획·목록: 볼트 "freesia 차트 선 전수 조사·검증 계획 (R66)".
선 = 봉 시작 시점에 알려진 가격 시계열(없으면 NaN). 역할 자동(위에서 내려오면 지지, 아래서 올라오면 저항 → 가격 부호 반전).
사건: 봉 전체가 띠(선±δ, δ=0.25 ATR) 밖 → 띠 진입 = 접근. 결과 반등(선+1 ATR) / 돌파(종가 선−δ 밖) / 미결(48봉).
주문(1분 격자, 대기 1봉, 보유 12봉): E1 반등 지정가 @선(5bp 관통), 손절 −0.5A, 목표 +0.5A /
      E3 스윕 지정가 @선−0.5A, 손절 −1.25A, 목표 선 / E2 돌파 스톱 @선 너머 δ, 손절 선 안쪽 0.25A, 목표 +1A.
비용: 메이커 2bp, 테이커 5+2bp. 위약: 선을 날짜마다 무작위 ±0.5~1.5 ATR 이동.
통과: n≥300, 2020~22·2023~24·2025~26 모두 비용 후 흑자, 실제−위약(비용 전) > 2 표준오차.
사용: python research/r66_lines_all.py <격자 15min|1h|4h>"""
import sys, warnings, time
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R))
from freesia import store, costs
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
CM = costs.MAKER; CT = costs.TAKER + 2e-4; PEN = 5e-4
DLT, TO, K = 0.25, 48, 5
SYMS = ("BTCUSDT", "ETHUSDT")
PER = [("20~22", "2020-01-01", "2023-01-01"), ("23~24", "2023-01-01", "2025-01-01"), ("25~26", "2025-01-01", "2027-01-01")]


# ====================== 공통 엔진 ======================
@numba.njit(cache=True)
def scan_series(Ls, h, l, c, dl, B):
    """지지 역할 사건. 반환 (j, res, m, L_j)."""
    n = len(c); out = np.zeros((n // 2 + 10, 4)); e = 0; armed = False; j = 1
    while j < n:
        L = Ls[j]
        if np.isnan(L) or np.isnan(dl[j]):
            armed = False; j += 1; continue
        if np.isnan(Ls[j - 1]) or abs(L - Ls[j - 1]) > dl[j]:     # 선이 점프하면(새 피벗·기간 교체) 다시 무장해야 한다
            armed = l[j] > L + dl[j]; j += 1; continue
        if not armed:
            if l[j] > L + dl[j]: armed = True
            j += 1; continue
        if l[j] <= L + dl[j]:
            res = 0; m = j
            while m < n and m - j <= TO:
                Lm = Ls[m]
                if np.isnan(Lm): break
                if c[m] < Lm - dl[j]: res = -1; break
                if m > j and h[m] > Lm + B[j]: res = 1; break
                m += 1
            m = min(m, n - 1)
            out[e, 0] = j; out[e, 1] = res; out[e, 2] = m; out[e, 3] = L; e += 1
            armed = False; j = m + 1; continue
        j += 1
    return out[:e]


@numba.njit(cache=True)
def sim(t0, wait, hold, ori, ye, ys, yt, se, lo, lh, ll, lc):
    """방향 맞춘 로그가격 롱 엔진. 반환 (체결, 총수익, 순수익).
    스톱 진입·손절은 시가가 이미 넘어섰으면 시가 체결(불리), 지정가 진입·목표는 지정가 그대로(보수)."""
    n = len(lc); out = np.full((len(t0), 3), np.nan)
    for e in range(len(t0)):
        s = ori[e]; fill = -1; px = ye[e]
        for i in range(t0[e], min(n, t0[e] + wait[e])):
            hi_ = lh[i] if s == 1 else -ll[i]; lo_ = ll[i] if s == 1 else -lh[i]; op = lo[i] * s
            if se:
                if hi_ >= ye[e]: fill = i; px = max(ye[e], op); break
            elif lo_ <= ye[e] - PEN: fill = i; break
        if fill < 0: out[e, 0] = 0; continue
        g = np.nan; cx = CT; j = fill
        while j < min(n, fill + hold[e]):
            hi_ = lh[j] if s == 1 else -ll[j]; lo_ = ll[j] if s == 1 else -lh[j]; op = lo[j] * s
            if lo_ <= ys[e]:
                ex = ys[e] if (j == fill or op > ys[e]) else op
                g = ex - px; cx = CT; break
            if j > fill and hi_ >= yt[e]: g = yt[e] - px; cx = CM; break
            j += 1
        if np.isnan(g): g = lc[min(j, n - 1)] * s - px; cx = CT
        out[e, 0] = 1; out[e, 1] = g; out[e, 2] = g - cx - (CT if se else CM)
    return out


def run_series(name, Lp, g, base, rng, step):
    """가격 시계열 Lp(격자 정렬) 하나 → 실제·위약 × 역할 × 규칙 거래 결과 DataFrame."""
    lh, ll, lc, atr = g.lh.values, g.ll.values, g.lc.values, g.atr.values
    dl = DLT * atr; B = atr
    L = np.log(np.where(Lp > 0, Lp, np.nan))
    day = g.index.floor("D"); ud = pd.Series(rng.uniform(0.5, 1.5, len(day.unique())) * rng.choice([-1, 1], len(day.unique())),
                                             index=day.unique())
    Lpl = L + ud.reindex(day).values * atr
    blo, blh, bll, blc = np.log(base.o.values), np.log(base.h.values), np.log(base.l.values), np.log(base.c.values)
    t0_all = base.index.searchsorted(g.index)
    rows = []
    for plc, LL in ((False, L), (True, Lpl)):
        for ori in (1, -1):
            if ori == 1: ev = scan_series(LL, lh, ll, lc, dl, B)
            else: ev = scan_series(-LL, -ll, -lh, -lc, dl, B)
            if len(ev) == 0: continue
            j = ev[:, 0].astype(int); res = ev[:, 1]; yL = ev[:, 3]; A = atr[j]; d = dl[j]
            t0 = t0_all[j]; wait = np.full(len(j), step); hold = 12 * wait
            for rule in ("E1", "E3", "E2"):
                if rule == "E1": o = sim(t0, wait, hold, np.full(len(j), ori), yL, yL - 0.5 * A, yL + 0.5 * A, False, blo, blh, bll, blc)
                elif rule == "E3": o = sim(t0, wait, hold, np.full(len(j), ori), yL - 0.5 * A, yL - 1.25 * A, yL, False, blo, blh, bll, blc)
                else: o = sim(t0, wait, hold, np.full(len(j), -ori), -yL + d, -yL - 0.25 * A, -yL + A, True, blo, blh, bll, blc)
                f = o[:, 0] == 1; t = g.index[j[f]]; gg = o[f, 1]; nn = o[f, 2]
                row = {"s": name, "plc": plc, "rule": rule, "cnt": f.sum(), "sg": gg.sum(), "sg2": (gg * gg).sum(), "sn": nn.sum()}
                for i_, (_, a, z) in enumerate(PER):
                    m_ = (t >= pd.Timestamp(a, tz="UTC")) & (t < pd.Timestamp(z, tz="UTC"))
                    row[f"c{i_}"] = m_.sum(); row[f"n{i_}"] = nn[m_].sum()
                rows.append(row)
            rows.append({"s": name, "plc": plc, "rule": "EV", "cnt": len(res), "b1": (res == 1).sum(), "bd": (res != 0).sum()})
    return rows


# ====================== 데이터 ======================
def load_1m(sym):
    k = store.load("klines_1m", sym).set_index("ts")
    return pd.DataFrame({"o": k.open, "h": k.high, "l": k.low, "c": k.close, "v": k.quote_volume})


def grid(b, tf):
    g = b.resample(tf).agg({"o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"}).dropna()
    g["lh"], g["ll"], g["lc"] = np.log(g.h), np.log(g.l), np.log(g.c)
    tr = pd.concat([g.lh - g.ll, (g.lh - g.lc.shift()).abs(), (g.ll - g.lc.shift()).abs()], axis=1).max(axis=1)
    g["atr"] = tr.rolling(14).mean().shift(1)                     # 봉 시작 시점에 알려진 ATR
    ptr = pd.concat([g.h - g.l, (g.h - g.c.shift()).abs(), (g.l - g.c.shift()).abs()], axis=1).max(axis=1)
    g["patr"] = ptr.rolling(14).mean()                             # 가격 단위 ATR (지표용, 종가 시점)
    return g


def pkey(ix, kind):
    d = ix.floor("D")
    if kind == "D": return d
    if kind == "W": return d - pd.to_timedelta(ix.weekday, unit="D")
    return d - pd.to_timedelta(ix.day - 1, unit="D")


def per_period(b, kind, t_index):
    """기간(일 D·주 W·월 M) OHLC → 직전 기간 값과 이번 기간 시가를 격자 봉 시작 시점에 맞춤."""
    p = b.groupby(pkey(b.index, kind)).agg({"o": "first", "h": "max", "l": "min", "c": "last"})
    gk = pkey(t_index, kind)
    return p.shift(1).reindex(gk), p.o.reindex(gk).values


# ====================== 선 생성기 ======================
def lines_period(b, g, sym, opt):
    ix = g.index; S = {}
    D, DO = per_period(b, "D", ix); W, WO = per_period(b, "W", ix); M, MO = per_period(b, "M", ix)
    H, L_, C, O = D.h.values, D.l.values, D.c.values, D.o.values; Rg = H - L_
    S.update({"PDH": H, "PDL": L_, "PDC": C, "PWH": W.h.values, "PWL": W.l.values, "PMH": M.h.values, "PML": M.l.values,
              "DO": DO, "WO": WO, "MO": MO})
    Pp = (H + L_ + C) / 3
    S.update({"CL_P": Pp, "CL_R1": 2 * Pp - L_, "CL_S1": 2 * Pp - H, "CL_R2": Pp + Rg, "CL_S2": Pp - Rg,
              "CL_R3": H + 2 * (Pp - L_), "CL_S3": L_ - 2 * (H - Pp)})
    S.update({f"FB_{k}{i}": Pp + sg * r * Rg for i, r in ((1, .382), (2, .618), (3, 1.0)) for k, sg in (("R", 1), ("S", -1))})
    S.update({"CM_H3": C + Rg * 1.1 / 4, "CM_H4": C + Rg * 1.1 / 2, "CM_L3": C - Rg * 1.1 / 4, "CM_L4": C - Rg * 1.1 / 2})
    Pw = (H + L_ + 2 * C) / 4
    S.update({"WD_P": Pw, "WD_R1": 2 * Pw - L_, "WD_S1": 2 * Pw - H, "WD_R2": Pw + Rg, "WD_S2": Pw - Rg})
    X = np.where(C < O, H + 2 * L_ + C, np.where(C > O, 2 * H + L_ + C, H + L_ + 2 * C))
    S.update({"DM_R1": X / 2 - L_, "DM_S1": X / 2 - H})
    # 개장 범위(00:00~00:30 UTC), 세션 고저
    day = b.index.floor("D"); mins = (b.index - day).total_seconds() / 60
    orr = b[mins < 30].groupby(day[mins < 30]).agg({"h": "max", "l": "min"})
    gd = ix.floor("D"); gm = (ix - gd).total_seconds() / 60; ok = gm >= 30
    S["ORH"] = np.where(ok, orr.h.reindex(gd).values, np.nan); S["ORL"] = np.where(ok, orr.l.reindex(gd).values, np.nan)
    for nm, a, z in (("AS", 0, 480), ("LN", 420, 960), ("NY", 810, 1200)):
        m_ = (mins >= a) & (mins < z); ss = b[m_].groupby(day[m_]).agg({"h": "max", "l": "min"})
        today = gm >= z
        for col, kk in (("h", "H"), ("l", "L")):
            v_today = ss[col].reindex(gd).values; v_prev = ss[col].shift(1).reindex(gd).values
            S[f"{nm}{kk}"] = np.where(today, v_today, v_prev)
    # 일간 돈치안
    dd = b.resample("1D").agg({"h": "max", "l": "min", "c": "last"}).dropna()
    for n_ in (20, 55, 90):
        S[f"D{n_}H"] = dd.h.rolling(n_).max().shift(1).reindex(gd).values
        S[f"D{n_}L"] = dd.l.rolling(n_).min().shift(1).reindex(gd).values
    # 라운드넘버: 전일 종가 기준 아래·위
    for stp, tag in ((1000, "RN1") if sym == "BTCUSDT" else (100, "RN1"), (5000, "RN5") if sym == "BTCUSDT" else (500, "RN5")):
        S[f"{tag}_DN"] = np.floor(C / stp) * stp; S[f"{tag}_UP"] = np.ceil(C / stp) * stp
    # VWAP 일간 ±σ, 주간
    tp = (b.h + b.l + b.c) / 3; v = b.v
    for key, nm in ((day, "VW"), (pkey(b.index, "W"), "WVW")):
        cpv = (tp * v).groupby(key).cumsum(); cv = v.groupby(key).cumsum(); cp2 = (tp * tp * v).groupby(key).cumsum()
        vw = (cpv / cv).shift(1); sd = np.sqrt((cp2 / cv - (cpv / cv) ** 2).clip(lower=0)).shift(1)
        # 새 기간 첫 분에는 이전 기간 값이 섞이지 않게
        first = pd.Series(key, index=b.index) != pd.Series(key, index=b.index).shift(1)
        vw[first.values] = np.nan; sd[first.values] = np.nan
        vwg = vw.reindex(ix).values; sdg = sd.reindex(ix).values
        S[nm] = vwg
        if nm == "VW":
            for k_ in (1, 2): S[f"VW+{k_}"] = vwg + k_ * sdg; S[f"VW-{k_}"] = vwg - k_ * sdg
    # 볼륨 프로파일 (로그가격 5bp 칸): 전일 POC/VAH/VAL, 7일 합성 POC
    bins = np.floor(np.log(b.c) / 5e-4).astype(np.int64)
    vp = pd.DataFrame({"d": day, "k": bins.values, "v": v.values}).groupby(["d", "k"]).v.sum()
    poc, vah, val = {}, {}, {}
    for d_, s in vp.groupby(level=0):
        s = s.droplevel(0).sort_index(); ks = s.index.values; vs = s.values; ip = int(np.argmax(vs)); tot = vs.sum()
        lo = hi = ip; acc = vs[ip]
        while acc < 0.7 * tot and (lo > 0 or hi < len(vs) - 1):
            a1 = vs[lo - 1] if lo > 0 else -1; a2 = vs[hi + 1] if hi < len(vs) - 1 else -1
            if a2 >= a1: hi += 1; acc += vs[hi]
            else: lo -= 1; acc += vs[lo]
        poc[d_] = np.exp((ks[ip] + .5) * 5e-4); vah[d_] = np.exp((ks[hi] + 1) * 5e-4); val[d_] = np.exp(ks[lo] * 5e-4)
    for nm, dct in (("POC", poc), ("VAH", vah), ("VAL", val)):
        S[nm] = pd.Series(dct).shift(1).reindex(gd).values
    vp7 = vp.reset_index(); rep = []
    for sft in range(1, 8):
        x = vp7.copy(); x["d"] = x.d + pd.Timedelta(days=sft); rep.append(x)
    c7 = pd.concat(rep).groupby(["d", "k"]).v.sum()
    p7 = c7.groupby(level=0).apply(lambda s: np.exp((s.droplevel(0).idxmax() + .5) * 5e-4))
    S["POC7"] = p7.reindex(gd).values
    # 옵션: 전일 최다 거래 행사가(가장 가까운 만기), 최대 고통(그 만기 누적 거래량 근사)
    if opt is not None:
        S["OPT_TOP"] = opt["top"].reindex(gd).values; S["MAXPAIN"] = opt["mp"].reindex(gd).values
    return S


def options_levels(sym):
    import glob
    cur = "BTC" if sym == "BTCUSDT" else "ETH"; fs = sorted(glob.glob(str(store.PQ / "deribit_opt" / cur / "*.parquet")))
    if not fs: return None
    agg = []
    for f in fs:
        x = pd.read_parquet(f, columns=["ts", "expiry", "strike", "cp", "amount"]); x["d"] = x.ts.dt.floor("D")
        agg.append(x.groupby(["d", "expiry", "strike", "cp"]).amount.sum().reset_index())
    X = pd.concat(agg).groupby(["d", "expiry", "strike", "cp"]).amount.sum().reset_index()
    byex = dict(tuple(X.groupby("expiry"))); exps = pd.DatetimeIndex(sorted(byex))
    top, mp = {}, {}
    for d_ in sorted(X.d.unique()):
        nd = d_ + pd.Timedelta(days=1); fut = exps[exps > nd]
        if len(fut) == 0: continue
        H = byex[fut[0]]; H = H[H.d <= d_]
        if H.empty: continue
        hd = H[H.d == d_]
        if len(hd): top[nd] = hd.groupby("strike").amount.sum().idxmax()
        cv = H[H.cp == 1].groupby("strike").amount.sum(); pv = H[H.cp != 1].groupby("strike").amount.sum()
        ks = np.union1d(cv.index.values, pv.index.values)
        pay = (cv.values[None, :] * np.maximum(0, ks[:, None] - cv.index.values[None, :])).sum(1) + \
              (pv.values[None, :] * np.maximum(0, pv.index.values[None, :] - ks[:, None])).sum(1)
        mp[nd] = ks[int(np.argmin(pay))]
    return {"top": pd.Series(top), "mp": pd.Series(mp)}


def lines_indicator(g):
    c, h, l = g.c, g.h, g.l; S = {}
    for n_ in (20, 50, 100, 200): S[f"SMA{n_}"] = c.rolling(n_).mean()
    for n_ in (9, 21, 55, 200): S[f"EMA{n_}"] = c.ewm(span=n_, adjust=False).mean()
    sd = c.rolling(20).std(); S["BB_U"] = S["SMA20"] + 2 * sd; S["BB_L"] = S["SMA20"] - 2 * sd
    e20 = c.ewm(span=20, adjust=False).mean(); S["KC_U"] = e20 + 2 * g.patr; S["KC_L"] = e20 - 2 * g.patr
    S["DC_U"] = h.rolling(20).max(); S["DC_L"] = l.rolling(20).min(); S["DC_M"] = (S["DC_U"] + S["DC_L"]) / 2
    tk = (h.rolling(9).max() + l.rolling(9).min()) / 2; kj = (h.rolling(26).max() + l.rolling(26).min()) / 2
    S["ICH_TK"] = tk; S["ICH_KJ"] = kj; S["ICH_SA"] = ((tk + kj) / 2).shift(26)
    S["ICH_SB"] = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
    S["PSAR"] = pd.Series(psar(h.values, l.values), index=g.index)
    S["STREND"] = pd.Series(supertrend(h.values, l.values, c.values, g.patr.values), index=g.index)
    S["CHAN_L"] = h.rolling(22).max() - 3 * g.patr; S["CHAN_S"] = l.rolling(22).min() + 3 * g.patr
    lr, lrs = linreg(np.log(c.values), 100)
    S["LR_M"] = pd.Series(np.exp(lr), index=g.index); S["LR_U"] = pd.Series(np.exp(lr + 2 * lrs), index=g.index)
    S["LR_L"] = pd.Series(np.exp(lr - 2 * lrs), index=g.index)
    return {k: v.shift(1).values for k, v in S.items()}             # 직전 봉 종가까지의 값을 이번 봉에 사용


@numba.njit(cache=True)
def psar(h, l, af0=0.02, afm=0.2):
    n = len(h); out = np.full(n, np.nan); up = True; ep = h[0]; sar = l[0]; af = af0
    for i in range(1, n):
        sar = sar + af * (ep - sar)
        if up:
            sar = min(sar, l[i - 1], l[i - 2] if i > 1 else l[i - 1])
            if l[i] < sar: up = False; sar = ep; ep = l[i]; af = af0
            elif h[i] > ep: ep = h[i]; af = min(afm, af + af0)
        else:
            sar = max(sar, h[i - 1], h[i - 2] if i > 1 else h[i - 1])
            if h[i] > sar: up = True; sar = ep; ep = h[i]; af = af0
            elif l[i] < ep: ep = l[i]; af = min(afm, af + af0)
        out[i] = sar + af * (ep - sar)                              # 다음 봉에 쓸 SAR (i 종가 시점 확정)
    return out


@numba.njit(cache=True)
def supertrend(h, l, c, atr, m=3.0):
    n = len(c); out = np.full(n, np.nan); fu = np.nan; fl = np.nan; up = True
    for i in range(1, n):
        if np.isnan(atr[i]): continue
        hl2 = (h[i] + l[i]) / 2; bu = hl2 + m * atr[i]; bl = hl2 - m * atr[i]
        fu = bu if (np.isnan(fu) or bu < fu or c[i - 1] > fu) else fu
        fl = bl if (np.isnan(fl) or bl > fl or c[i - 1] < fl) else fl
        if up and c[i] < fl: up = False
        elif (not up) and c[i] > fu: up = True
        out[i] = fl if up else fu
    return out


@numba.njit(cache=True)
def linreg(y, n):
    N = len(y); m = np.full(N, np.nan); s = np.full(N, np.nan); x = np.arange(n) * 1.0
    xm = x.mean(); sxx = ((x - xm) ** 2).sum()
    for i in range(n - 1, N):
        w = y[i - n + 1:i + 1]; ym = w.mean(); b = ((x - xm) * (w - ym)).sum() / sxx; a = ym - b * xm
        r = w - (a + b * x); m[i] = a + b * (n - 1); s[i] = np.sqrt((r * r).mean())
    return m, s


@numba.njit(cache=True)
def pivot_flags(h, l, k):
    n = len(h); ph = np.zeros(n, np.bool_); pl = np.zeros(n, np.bool_)
    for i in range(k, n - k):
        okh = True; okl = True
        for j in range(i - k, i + k + 1):
            if h[j] > h[i]: okh = False
            if l[j] < l[i]: okl = False
        ph[i] = okh; pl[i] = okl
    return ph, pl


@numba.njit(cache=True)
def swing_state(ph, pl, lh, ll, k):
    """봉 j 시작 시점에 확정된(i+k 종가 이전) 최근·직전 스윙 고·저점 인덱스."""
    n = len(lh); SL = np.full(n, -1); SL2 = np.full(n, -1); SH = np.full(n, -1); SH2 = np.full(n, -1)
    sl = -1; sl2 = -1; sh = -1; sh2 = -1
    for j in range(n):
        i = j - k - 1                                              # j−1 종가에 확정되는 피벗
        if i >= 0:
            if pl[i]: sl2 = sl; sl = i
            if ph[i]: sh2 = sh; sh = i
        SL[j] = sl; SL2[j] = sl2; SH[j] = sh; SH2[j] = sh2
    return SL, SL2, SH, SH2


@numba.njit(cache=True)
def trendlines(pl_idx_at, ll, atr, SL, win=150, slope_max=0.3):
    """최근 확정 스윙 저점 p2와 창 안의 이전 저점 p1을 잇는, 사이를 관통하지 않는 가장 긴 선. 반환: 봉별 선 값(로그), 채널 오프셋."""
    n = len(ll); val = np.full(n, np.nan); p2c = -1; a = 0.0; s = 0.0; span = 0
    for j in range(n):
        p2 = SL[j]
        if p2 >= 0 and p2 != p2c:
            p2c = p2; best = -1; bs = 0.0
            for p1 in range(p2 - 10, max(-1, p2 - win), -1):
                if not pl_idx_at[p1]: continue
                s_ = (ll[p2] - ll[p1]) / (p2 - p1)
                if np.isnan(atr[p2]) or abs(s_) > slope_max * atr[p2]: continue
                ok = True
                for t in range(p1, p2 + 1):
                    if ll[t] - (ll[p1] + s_ * (t - p1)) < -0.1 * atr[p2]: ok = False; break
                if ok: best = p1; bs = s_
            if best >= 0: a = ll[p2]; s = bs; span = p2 - best
            else: span = 0
        if span > 0 and j - p2c <= 2 * span + 50: val[j] = a + s * (j - p2c)
    return val


@numba.njit(cache=True)
def ob_fvg(o, h, l, c, patr):
    """오더블록(변위 ≥2 ATR 직전 반대 캔들)·FVG 최근 유효 구간. 반환 8열: OBb_top OBb_bot OBs_top OBs_bot FVb_top FVb_bot FVs_top FVs_bot (다음 봉에 사용)."""
    n = len(c); out = np.full((n, 8), np.nan)
    obb = (np.nan, np.nan); obs = (np.nan, np.nan); fvb = (np.nan, np.nan); fvs = (np.nan, np.nan)
    for i in range(3, n):
        a = patr[i]
        if not np.isnan(a):
            for q in range(i - 3, i):                              # 강세 OB: q가 음봉, 이후 종가 i가 q 고가 + 2ATR 이상
                if c[q] < o[q] and c[i] - h[q] >= 2 * a: obb = (h[q], l[q])
                if c[q] > o[q] and l[q] - c[i] >= 2 * a: obs = (h[q], l[q])
        if l[i] > h[i - 2]: fvb = (l[i], h[i - 2])
        if h[i] < l[i - 2]: fvs = (l[i - 2], h[i])
        if not np.isnan(obb[1]) and c[i] < obb[1]: obb = (np.nan, np.nan)
        if not np.isnan(obs[0]) and c[i] > obs[0]: obs = (np.nan, np.nan)
        if not np.isnan(fvb[1]) and c[i] < fvb[1]: fvb = (np.nan, np.nan)
        if not np.isnan(fvs[0]) and c[i] > fvs[0]: fvs = (np.nan, np.nan)
        if i + 1 < n:
            out[i + 1, 0] = obb[0]; out[i + 1, 1] = obb[1]; out[i + 1, 2] = obs[0]; out[i + 1, 3] = obs[1]
            out[i + 1, 4] = fvb[0]; out[i + 1, 5] = fvb[1]; out[i + 1, 6] = fvs[0]; out[i + 1, 7] = fvs[1]
    return out


def lines_swing(g):
    lh, ll, lc, atr = g.lh.values, g.ll.values, g.lc.values, g.atr.values; n = len(g); S = {}
    ph, pl = pivot_flags(lh, ll, K)
    SL, SL2, SH, SH2 = swing_state(ph, pl, lh, ll, K)
    J = np.arange(n)
    get = lambda arr, idx: np.where(idx >= 0, arr[np.maximum(idx, 0)], np.nan)
    sl, sl2, sh, sh2 = get(ll, SL), get(ll, SL2), get(lh, SH), get(lh, SH2)
    S.update({"SW_L": sl, "SW_L2": sl2, "SW_H": sh, "SW_H2": sh2})
    a_sl = get(atr, SL); a_sh = get(atr, SH)
    S["EQ_L"] = np.where(np.abs(sl - sl2) < 0.25 * a_sl, np.minimum(sl, sl2), np.nan)
    S["EQ_H"] = np.where(np.abs(sh - sh2) < 0.25 * a_sh, np.maximum(sh, sh2), np.nan)
    # 추세선(지지: 저점, 저항: 부호 반전한 고점) + 채널
    tl_s = trendlines(pl, ll, atr, SL); tl_r = -trendlines(ph, -lh, atr, SH)
    S["TL_S"] = tl_s; S["TL_R"] = tl_r
    # 채널: 지지선과 평행, 최근 50봉 고가의 최대 초과분
    dev = pd.Series(lh - tl_s).rolling(50, min_periods=10).max().shift(1).values; S["CH_U"] = tl_s + dev
    dev2 = pd.Series(tl_r - ll).rolling(50, min_periods=10).max().shift(1).values; S["CH_L"] = tl_r - dev2
    # 마지막 다리 (최근 고·저 중 나중 것이 끝)
    upleg = SH > SL; lo_, hi_ = sl, sh; rng_ = hi_ - lo_
    for r in (0.382, 0.5, 0.618, 0.786):
        S[f"FIB_{r}"] = np.where(upleg, hi_ - r * rng_, lo_ + r * rng_)
    for e in (1.272, 1.618):
        S[f"FIBX_{e}"] = np.where(upleg, lo_ + e * rng_, hi_ - e * rng_)
    # 앵커드 VWAP (격자 봉 전형가·거래대금)
    tp = (g.h + g.l + g.c).values / 3; v = g.v.values; cpv = np.concatenate([[0], np.cumsum(tp * v)]); cv = np.concatenate([[0], np.cumsum(v)])
    for nm, A_ in (("AVW_L", SL), ("AVW_H", SH)):
        ok = A_ >= 0; a0 = np.maximum(A_, 0)
        num = cpv[J] - cpv[a0]; den = cv[J] - cv[a0]
        S[nm] = np.where(ok & (den > 0), np.log(np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)), np.nan)
    # 피치포크: 최근 세 교대 피벗 (A, B, C) → 중앙선 A→(B,C 중점)
    # A = 마지막 피벗과 같은 종류의 직전 피벗, B·C = 최근 저점·고점
    tA = np.where(SH > SL, SH2, SL2); yA = np.where(SH > SL, get(lh, SH2), get(ll, SL2))
    tB, yB, tC, yC = SL, sl, SH, sh
    tM = (tB + tC) / 2; yM = (yB + yC) / 2; slope = (yM - yA) / np.where(tM - tA > 0, tM - tA, np.nan)
    ok = (tA >= 0) & (tB >= 0) & (tC >= 0)
    S["PF_M"] = np.where(ok, yA + slope * (J - tA), np.nan)
    S["PF_B"] = np.where(ok, yB + slope * (J - tB), np.nan); S["PF_C"] = np.where(ok, yC + slope * (J - tC), np.nan)
    # 갠 팬 (단위 = 피벗 시점 ATR/봉)
    for k_, tag in ((1, "1x1"), (0.5, "1x2"), (2, "2x1")):
        S[f"GF_U{tag}"] = np.where(SL >= 0, sl + k_ * a_sl * (J - SL), np.nan)
        S[f"GF_D{tag}"] = np.where(SH >= 0, sh - k_ * a_sh * (J - SH), np.nan)
    # 스피드선·피보나치 팬: 상승 다리 (SL→SH, SH가 나중)일 때 SL에서 (tSH, SL + f·(SH−SL))를 지나는 선
    dt = np.where(SH - SL > 0, SH - SL, np.nan)
    for f_, tag in ((1 / 3, "SP1"), (2 / 3, "SP2"), (1 - .382, "FF382"), (0.5, "FF5"), (1 - .618, "FF618")):
        up_ = sl + f_ * rng_ * (J - SL) / dt
        dn_ = sh - f_ * rng_ * (J - SH) / np.where(SL - SH > 0, SL - SH, np.nan)
        S[tag] = np.where(upleg, up_, dn_)
    # 갠 스퀘어 오브 9 (가격 제곱근 ±1, ±2 = 180°, 360°)
    psl, psh = np.exp(sl), np.exp(sh)
    for k_, tag in ((1, "180"), (2, "360")):
        S[f"SQ9_U{tag}"] = np.log((np.sqrt(psl) + k_) ** 2); S[f"SQ9_D{tag}"] = np.log(np.clip(np.sqrt(psh) - k_, 1e-9, None) ** 2)
    # 오더블록·FVG
    of = ob_fvg(g.o.values, g.h.values, g.l.values, g.c.values, g.patr.values)
    for i_, nm in enumerate(("OBB_T", "OBB_B", "OBS_T", "OBS_B", "FVB_T", "FVB_B", "FVS_T", "FVS_B")):
        S[nm] = np.log(of[:, i_])
    return {k: np.exp(v) for k, v in S.items()}                    # 가격으로 통일


GROUP = {}
def grp(name):
    for pre, gname in (("PD", "A 전일"), ("PW", "A 전주"), ("PM", "A 전월"), ("DO", "A 시가"), ("WO", "A 시가"), ("MO", "A 시가"),
                       ("CL_", "A 피벗 클래식"), ("FB_", "A 피벗 피보나치"), ("CM_", "A 피벗 카마릴라"), ("WD_", "A 피벗 우디"), ("DM_", "A 피벗 드마크"),
                       ("OR", "A 개장범위"), ("AS", "A 세션"), ("LN", "A 세션"), ("NY", "A 세션"), ("D20", "A 일간돈치안"), ("D55", "A 일간돈치안"),
                       ("D90", "A 일간돈치안"), ("RN", "A 라운드넘버"), ("VW", "B VWAP"), ("WVW", "B VWAP"), ("POC", "B 볼륨프로파일"),
                       ("VA", "B 볼륨프로파일"), ("AVW", "B 앵커드VWAP"), ("OPT", "B 옵션"), ("MAXP", "B 옵션"),
                       ("SMA", "C 이동평균"), ("EMA", "C 이동평균"), ("BB", "C 볼린저"), ("KC", "C 켈트너"), ("DC", "C 돈치안"),
                       ("ICH", "C 이치모쿠"), ("PSAR", "C SAR·슈퍼트렌드"), ("STREND", "C SAR·슈퍼트렌드"), ("CHAN", "C 샹들리에"),
                       ("LR", "C 선형회귀"), ("SW", "D 스윙 고저"), ("EQ", "D 동일 고저"), ("TL", "D 추세선"), ("CH_", "D 채널"),
                       ("FIBX", "D 피보나치 확장"), ("FIB", "D 피보나치 되돌림"), ("PF", "D 피치포크"), ("GF", "D 갠 팬"),
                       ("SP", "D 스피드선"), ("FF", "D 피보나치 팬"), ("SQ9", "D 스퀘어오브9"), ("OB", "E 오더블록"), ("FV", "E FVG")):
        if name.startswith(pre): return gname
    return "?"


if __name__ == "__main__":
    tf = sys.argv[1] if len(sys.argv) > 1 else "1h"; step = int(pd.Timedelta(tf) / pd.Timedelta("1min"))
    rng = np.random.default_rng(0); rows = []; t_ = time.time(); days = 0
    for sym in SYMS:
        b = load_1m(sym); g = grid(b, tf); opt = options_levels(sym)
        days += (g.index[-1] - g.index[0]).days / 2
        S = {}
        S.update(lines_period(b, g, sym, opt)); S.update(lines_indicator(g)); S.update(lines_swing(g))
        P(f"{sym} {tf}: 격자 {len(g):,}봉, 선 {len(S)}개 ({time.time()-t_:.0f}s)")
        for nm, arr in S.items():
            arr = np.asarray(arr, dtype=float)
            if np.isfinite(arr).sum() < 1000: P(f"  {nm}: 값 부족, 건너뜀"); continue
            rows += run_series(nm, arr, g, b, rng, step)
        P(f"{sym} 완료 ({time.time()-t_:.0f}s)"); del b, g
    A = pd.DataFrame(rows).groupby(["s", "plc", "rule"]).sum(numeric_only=True).reset_index()
    A.to_csv(R / "logs" / f"r66_raw_{tf}.csv", index=False)
    T = A[A.rule != "EV"]; out = []
    for (s_, rule), G in T.groupby(["s", "rule"]):
        re_ = G[~G.plc].iloc[0] if (~G.plc).any() else None; pl_ = G[G.plc].iloc[0] if G.plc.any() else None
        if re_ is None or pl_ is None or re_.cnt < 50 or pl_.cnt < 50: continue
        mg, mp_ = re_.sg / re_.cnt, pl_.sg / pl_.cnt
        vr = re_.sg2 / re_.cnt - mg ** 2; vp = pl_.sg2 / pl_.cnt - mp_ ** 2
        se = np.sqrt(vr / re_.cnt + vp / pl_.cnt) * 1e4
        per = [re_[f"n{i}"] / re_[f"c{i}"] * 1e4 if re_[f"c{i}"] > 0 else np.nan for i in range(3)]
        out.append((grp(s_), s_, rule, int(re_.cnt), re_.cnt / days, mg * 1e4, mp_ * 1e4, se, re_.sn / re_.cnt * 1e4, *per))
    O = pd.DataFrame(out, columns=["계열", "선", "규칙", "n", "하루", "총", "위약총", "se", "순", "p1", "p2", "p3"])
    O["차"] = O.총 - O.위약총
    O["통과"] = (O.n >= 300) & (O.p1 > 0) & (O.p2 > 0) & (O.p3 > 0) & (O.차 > 2 * O.se)
    EV = A[A.rule == "EV"].set_index(["s", "plc"]); brt = (EV.b1 / EV.bd).unstack()
    O["반등률"] = O.선.map(brt[False]); O["위약반등률"] = O.선.map(brt[True])
    O.to_csv(R / "logs" / f"r66_{tf}.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 1000)
    P(f"\n===== {tf}: 검정 {len(O)}개 (선 {O.선.nunique()} × 규칙 3), 통과 {O.통과.sum()}개")
    P(O[O.통과].round(2).to_string(index=False) if O.통과.any() else "  통과 없음")
    P("\n  계열별 최선(비용 후 순수익 기준)")
    best = O.sort_values("순", ascending=False).groupby("계열").head(1).sort_values("순", ascending=False)
    P(best[["계열", "선", "규칙", "n", "하루", "총", "위약총", "차", "se", "순", "p1", "p2", "p3"]].round(2).to_string(index=False))
    P("\n  실제−위약(비용 전) 상위 15")
    P(O.sort_values("차", ascending=False).head(15)[["계열", "선", "규칙", "n", "총", "위약총", "차", "se", "순", "p1", "p2", "p3"]].round(2).to_string(index=False))
    P("\n  반등률 실제−위약 상·하위 8 (선이 존중되는가)")
    bb = O.drop_duplicates("선").assign(d=lambda x: x.반등률 - x.위약반등률).sort_values("d")
    P(pd.concat([bb.tail(8), bb.head(8)])[["계열", "선", "반등률", "위약반등률", "d"]].round(3).to_string(index=False))
    (R / "logs" / f"r66_{tf}.summary").write_text(f"R66 {tf}: 검정 {len(O)}, 통과 {O.통과.sum()}\n")
