"""전략군 대량 스크리닝 (1배, 비용·펀딩 반영, 연도별 워크포워드).
사용: python research/screen.py <family,...> <tf,...>  → logs/screen_<family>.json
시도 횟수(trials)를 기록해 다중비교를 나중에 보정한다."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freesia import store  # noqa: E402
from freesia.bt import core  # noqa: E402

SYMS = ["BTCUSDT", "ETHUSDT"]


def hold(el, xl, es=None, xs=None):
    """상태 유지: 진입 신호 ~ 청산 신호 사이 포지션 유지 (numpy bool 배열들)."""
    n = len(el)
    s = np.full(n, np.nan)
    if xl is not None:
        s[xl] = 0.0
    if xs is not None:
        s[xs] = 0.0
    if es is not None:
        s[es] = -1.0
    s[el] = 1.0
    return np.nan_to_num(pd.Series(s).ffill().to_numpy())


def rsi(c: pd.Series, n: int):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


class Ctx:
    def __init__(self, sym, tf):
        self.sym, self.tf = sym, tf
        self.b = core.bars(sym, tf)
        self.hb = core.TF_HOURS[tf]
        self.idx = self.b.index
        self.C = self.b.close
        self.r = self.C.pct_change().fillna(0)
        self.fund = core.funding_bars(sym, self.idx, tf)
        self.ppy = core.PPY[tf]

    def nb(self, hours):
        return max(1, int(round(hours / self.hb)))


def modes(sig, kind):
    """sig ∈ {-1,0,1}. kind: LS 롱숏, LF 롱만."""
    return sig if kind == "LS" else np.where(sig > 0, 1.0, 0.0)


# ---------------- 전략군: 각 함수는 (이름 리스트, 포지션 행렬) 반환 ----------------
def f_tsmom(x: Ctx):
    P, N = [], []
    seen = set()
    for h in [12, 24, 48, 72, 120, 168, 240, 336, 504, 720, 1080, 1440, 2160, 3000]:
        n = x.nb(h)
        if n in seen or n >= len(x.C) // 4:
            continue
        seen.add(n)
        sig = np.sign((x.C / x.C.shift(n) - 1).fillna(0).to_numpy())
        for k in ("LS", "LF"):
            P.append(modes(sig, k)); N.append(f"tsmom h={h} {k}")
    return N, P


def f_ma(x: Ctx):
    P, N = [], []
    seen = set()
    for fh in [6, 12, 24, 48, 96, 168, 336, 504]:
        for sh in [24, 48, 96, 168, 336, 720, 1440, 2400, 3600]:
            f, s = x.nb(fh), x.nb(sh)
            if f >= s or (f, s) in seen or s >= len(x.C) // 4:
                continue
            seen.add((f, s))
            sig = np.sign((x.C.rolling(f).mean() - x.C.rolling(s).mean()).fillna(0).to_numpy())
            for k in ("LS", "LF"):
                P.append(modes(sig, k)); N.append(f"ma {fh}/{sh} {k}")
    return N, P


def f_donchian(x: Ctx):
    P, N = [], []
    seen = set()
    for h in [24, 48, 96, 168, 336, 720, 1440]:
        n = x.nb(h)
        if n in seen or n >= len(x.C) // 4:
            continue
        seen.add(n)
        hh = x.C.shift(1).rolling(n).max().to_numpy(); ll = x.C.shift(1).rolling(n).min().to_numpy()
        ll2 = x.C.shift(1).rolling(max(1, n // 2)).min().to_numpy()
        hh2 = x.C.shift(1).rolling(max(1, n // 2)).max().to_numpy()
        c = x.C.to_numpy()
        el, es = c > hh, c < ll
        P.append(hold(el, None, es, None)); N.append(f"donch {h} LS")
        P.append(hold(el, c < ll2)); N.append(f"donch {h} LF")
        P.append(-hold(es, c > hh2)); N.append(f"donch {h} SF")
    return N, P


def f_voltsmom(x: Ctx):
    P, N = [], []
    vol = x.r.rolling(x.nb(168)).std().to_numpy() * np.sqrt(x.ppy)
    for h in [72, 168, 336, 720, 1440]:
        n = x.nb(h)
        sig = np.sign((x.C / x.C.shift(n) - 1).fillna(0).to_numpy())
        for tv in [0.3, 0.5, 0.8]:
            sc = np.clip(np.nan_to_num(tv / vol, nan=0.0), 0, 1)
            for k in ("LS", "LF"):
                P.append(modes(sig, k) * sc); N.append(f"voltsmom h={h} tv={tv} {k}")
    return N, P


def f_contraction(x: Ctx):
    P, N = [], []
    c = x.C.to_numpy()
    for sh in [24, 48]:
        for lh in [240, 720]:
            ratio = (x.r.rolling(x.nb(sh)).std() / x.r.rolling(x.nb(lh)).std()).to_numpy()
            for thr in [0.5, 0.7, 0.9]:
                for bh in [24, 72, 168]:
                    n = x.nb(bh)
                    hh = x.C.shift(1).rolling(n).max().to_numpy(); ll = x.C.shift(1).rolling(n).min().to_numpy()
                    for H in [24, 72, 168]:
                        Hn = x.nb(H)
                        up = pd.Series(((c > hh) & (ratio < thr)).astype(float)).rolling(Hn).max().fillna(0).to_numpy()
                        dn = pd.Series(((c < ll) & (ratio < thr)).astype(float)).rolling(Hn).max().fillna(0).to_numpy()
                        P.append(up); N.append(f"contr s={sh} l={lh} thr={thr} N={bh} H={H} LF")
                        P.append(np.clip(up - dn, -1, 1)); N.append(f"contr s={sh} l={lh} thr={thr} N={bh} H={H} LS")
    return N, P


def f_rsipull(x: Ctx):
    P, N = [], []
    c = x.C.to_numpy()
    for n in [6, 14]:
        r = rsi(x.C, n).fillna(50).to_numpy()
        for lh in [480, 1200, 2400, 3600]:
            ma = x.C.rolling(x.nb(lh)).mean().to_numpy()
            up = c > ma
            for thr in [25, 30, 35, 40, 45]:
                for ex in [55, 65]:
                    P.append(hold((r < thr) & up, (r > ex) | (~up))); N.append(f"rsipull rsi{n} ma{lh} thr={thr} ex={ex}")
    return N, P


def f_reversal(x: Ctx):
    P, N = [], []
    seen = set()
    sd = x.r.rolling(x.nb(720)).std()
    for nh in [1, 4, 12, 24, 48]:
        n = x.nb(nh)
        if n in seen:
            continue
        seen.add(n)
        z = ((x.C / x.C.shift(n) - 1) / (sd * np.sqrt(n))).fillna(0)
        for k in [1.5, 2, 2.5, 3]:
            for H in [4, 12, 24, 48, 96]:
                Hn = x.nb(H)
                lo = pd.Series((z < -k).astype(float)).rolling(Hn).max().fillna(0).to_numpy()
                hi = pd.Series((z > k).astype(float)).rolling(Hn).max().fillna(0).to_numpy()
                P.append(lo); N.append(f"rev n={nh} k={k} H={H} dipbuy")
                P.append(-hi); N.append(f"rev n={nh} k={k} H={H} ripshort")
                P.append(np.clip(lo - hi, -1, 1)); N.append(f"rev n={nh} k={k} H={H} LS")
                P.append(np.clip(hi - lo, -1, 1)); N.append(f"mom-jump n={nh} k={k} H={H} LS")
    return N, P


def f_funding(x: Ctx):
    f = store.load("funding", x.sym).set_index("ts").rate
    P, N = [], []
    for w in [1, 3, 9]:
        fs = f.rolling(w).mean()
        z = ((fs - f.rolling(270).mean()) / f.rolling(270).std())
        fsb = fs.reindex(x.idx, method="ffill").to_numpy()
        zb = z.reindex(x.idx, method="ffill").fillna(0).to_numpy()
        for k in [0.5, 1, 1.5, 2]:
            P.append(hold(zb < -k, zb > 0, zb > k, zb < 0)); N.append(f"fund-z contr w={w} k={k} LS")
            P.append(-hold(zb < -k, zb > 0, zb > k, zb < 0)); N.append(f"fund-z trend w={w} k={k} LS")
            P.append(hold(zb < -k, zb > 0)); N.append(f"fund-z contr w={w} k={k} LF")
            P.append(-hold(zb > k, zb < 0)); N.append(f"fund-z contr w={w} k={k} SF")
        for a in [0.0002, 0.0004, 0.0008]:
            P.append(-hold(fsb > a, fsb < 0.0001)); N.append(f"fund-abs w={w} a={a} SF")
        P.append(hold(fsb < 0, fsb > 0.0001)); N.append(f"fund-neg w={w} LF")
    return N, P


def f_metrics(x: Ctx):
    m = store.load("metrics_5m", x.sym).set_index("ts")
    hb = m.resample("1h").last().reindex(x.idx, method="ffill") if x.tf != "1d" else m.resample("1D").last().reindex(x.idx, method="ffill")
    if x.tf == "4h":
        hb = m.resample("4h").last().reindex(x.idx, method="ffill")
    oiv, tls, cls_, tkr = (hb[c] for c in ["sum_open_interest_value", "sum_toptrader_long_short_ratio",
                                           "count_long_short_ratio", "sum_taker_long_short_vol_ratio"])
    P, N = [], []
    c = x.C
    for wh in [72, 168, 336]:
        w = x.nb(wh)
        for nm, ser in [("taker", tkr), ("toptrader", tls), ("acct", cls_)]:
            z = ((ser - ser.rolling(w).mean()) / ser.rolling(w).std()).fillna(0).to_numpy()
            for k in [0.5, 1, 1.5, 2]:
                for H in [4, 12, 24]:
                    Hn = x.nb(H)
                    hi = pd.Series((z > k).astype(float)).rolling(Hn).max().fillna(0).to_numpy()
                    lo = pd.Series((z < -k).astype(float)).rolling(Hn).max().fillna(0).to_numpy()
                    P.append(np.clip(hi - lo, -1, 1)); N.append(f"{nm} mom w={wh} k={k} H={H}")
                    P.append(np.clip(lo - hi, -1, 1)); N.append(f"{nm} contr w={wh} k={k} H={H}")
        d = oiv.pct_change(w)
        zoi = ((d - d.rolling(w * 4).mean()) / d.rolling(w * 4).std()).fillna(0).to_numpy()
        ret = np.sign((c / c.shift(w) - 1).fillna(0).to_numpy())
        for k in [0.5, 1, 1.5]:
            for H in [12, 24, 72]:
                Hn = x.nb(H)
                on = pd.Series((zoi > k).astype(float)).rolling(Hn).max().fillna(0).to_numpy()
                P.append(ret * on); N.append(f"OI-up trend w={wh} k={k} H={H}")
                P.append(-ret * on); N.append(f"OI-up fade w={wh} k={k} H={H}")
    return N, P


FAMILIES = {"tsmom": f_tsmom, "ma": f_ma, "donchian": f_donchian, "voltsmom": f_voltsmom,
            "contraction": f_contraction, "rsipull": f_rsipull, "reversal": f_reversal,
            "funding": f_funding, "metrics": f_metrics}


def evaluate(name, x: Ctx, N, P, first_year, min_train):
    T = len(x.idx)
    R = np.zeros((T, len(P)))
    tv = np.zeros(len(P))
    for j, p in enumerate(P):
        R[:, j] = core.run(np.asarray(p, float), x.C.to_numpy(), x.fund)
        tv[j] = core.turnover_stats(np.asarray(p, float), x.tf)
    ppy = x.ppy
    oos, used, picks = core.walk_forward(R, x.idx, x.tf, first_year, min_train)
    if used.sum() < ppy:
        return None
    ro = oos[used]
    yrs = x.idx.year.to_numpy()[used]
    bh = x.r.to_numpy()[used]
    per_year = {}
    for Y in sorted(set(yrs)):
        m = yrs == Y
        per_year[int(Y)] = round(float(ro[m].mean() / ro[m].std() * np.sqrt(ppy)) if ro[m].std() > 0 else 0, 2)
    full_sh = np.where(R.std(0) > 0, R.mean(0) / np.where(R.std(0) > 0, R.std(0), 1) * np.sqrt(ppy), 0)
    oos_all = R[used]
    p_sh = np.where(oos_all.std(0) > 0, oos_all.mean(0) / np.where(oos_all.std(0) > 0, oos_all.std(0), 1) * np.sqrt(ppy), 0)
    met = core.metrics(ro, x.tf)
    bhm = core.metrics(bh, x.tf)
    best = int(np.argmax(full_sh))
    return {"family": name, "sym": x.sym, "tf": x.tf, "n_params": len(P),
            "oos_sharpe": round(met["sharpe"], 2), "oos_ann": round(met["ann"], 3), "oos_mdd": round(met["mdd"], 3),
            "bh_sharpe": round(bhm["sharpe"], 2), "bh_ann": round(bhm["ann"], 3), "bh_mdd": round(bhm["mdd"], 3),
            "per_year_sharpe": per_year, "years_pos": int(sum(v > 0 for v in per_year.values())), "years": len(per_year),
            "all_params_oos_sharpe_median": round(float(np.median(p_sh)), 2),
            "all_params_oos_pos_frac": round(float((p_sh > 0).mean()), 2),
            "full_best": {"name": N[best], "sharpe": round(float(full_sh[best]), 2), "turnover_per_yr": round(float(tv[best]), 1)},
            "picks": {int(k): [N[v[0]], round(v[1], 2)] for k, v in picks.items()},
            "oos_years": f"{yrs.min()}-{yrs.max()}"}


def main():
    fams = sys.argv[1].split(",")
    tfs = sys.argv[2].split(",")
    out = []
    t0 = time.time()
    for fam in fams:
        for tf in tfs:
            for sym in SYMS:
                try:
                    x = Ctx(sym, tf)
                    N, P = FAMILIES[fam](x)
                    fy, mt = (2023, 1.0) if fam == "metrics" else (2022, 1.5)
                    r = evaluate(fam, x, N, P, fy, mt)
                except Exception as e:  # noqa: BLE001
                    r = {"family": fam, "sym": sym, "tf": tf, "error": repr(e)}
                if r:
                    out.append(r)
                    print(fam, tf, sym, "OOS Sh", r.get("oos_sharpe"), "n", r.get("n_params"), f"{time.time()-t0:.0f}s", flush=True)
    Path("logs").mkdir(exist_ok=True)
    Path(f"logs/screen_{sys.argv[3] if len(sys.argv) > 3 else fams[0]}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
