"""2차 스크리닝: ETH/BTC 상대가치, 델타중립 캐리, 계절성, 무최적화 추세 앙상블.
사용: python research/screen2.py <family,...> <tf,...> <tag>"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freesia import store  # noqa: E402
from freesia.bt import core  # noqa: E402
import screen as S  # noqa: E402  (research/ 를 sys.path 앞에 둠)


class PairCtx(S.Ctx):
    def __init__(self, tf):
        self.sym, self.tf = "ETH/BTC", tf
        e, b = core.bars("ETHUSDT", tf), core.bars("BTCUSDT", tf)
        self.idx = e.index.intersection(b.index)
        self.hb = core.TF_HOURS[tf]; self.ppy = core.PPY[tf]
        ce, cb = e.close.reindex(self.idx), b.close.reindex(self.idx)
        self.C = np.log(ce / cb)                       # 로그 비율
        re, rb = ce.pct_change().fillna(0), cb.pct_change().fillna(0)
        self.ret = (0.5 * (re - rb)).to_numpy()        # 자본 1: ETH 롱 0.5 / BTC 숏 0.5
        self.r = pd.Series(np.zeros(len(self.idx)), index=self.idx)
        fe = core.funding_bars("ETHUSDT", self.idx, tf); fb = core.funding_bars("BTCUSDT", self.idx, tf)
        self.fund_eff = 0.5 * fe - 0.5 * fb

    def run(self, pos):
        return core.run_ret(pos, self.ret, self.fund_eff, cost=core.COST)


class CarryCtx(S.Ctx):
    """현물 롱 + 무기한 숏(델타중립). 수익 = 펀딩 수취 − 프리미엄 변동(베이시스 손익). 자본 1.2(현물 1 + 증거금 0.2)."""
    def __init__(self, sym, tf):
        super().__init__(sym, tf)
        pr = store.load("premium_1m", sym).set_index("ts").close.resample({"1h": "1h", "4h": "4h", "1d": "1D"}[tf]).last().reindex(self.idx).ffill()
        dprem = pr.diff().fillna(0).to_numpy()
        self.ret = (-dprem + self.fund) / 1.2       # 숏 퍼프: 펀딩>0이면 수취
        self.r = pd.Series(np.zeros(len(self.idx)), index=self.idx)

    def run(self, pos):
        return core.run_ret(pos, self.ret, np.zeros(len(self.idx)), cost=0.0017 / 1.2)


def f_pairs(x: PairCtx):
    P, N = [], []
    z_src = x.C
    for h in [24, 72, 168, 336, 720, 1440]:
        n = x.nb(h)
        z = ((z_src - z_src.rolling(n).mean()) / z_src.rolling(n).std()).fillna(0).to_numpy()
        for k in [1, 1.5, 2, 2.5]:
            mr = S.hold(z < -k, z > -0.25, z > k, z < 0.25)      # 비율 낮으면 ETH 롱(+1), 높으면 ETH 숏
            P.append(mr); N.append(f"pair MR w={h} k={k}")
            P.append(-mr); N.append(f"pair trend w={h} k={k}")
    for h in [72, 168, 336, 720, 1440]:
        n = x.nb(h)
        sig = np.sign((z_src - z_src.shift(n)).fillna(0).to_numpy())
        P.append(sig); N.append(f"pair mom h={h}"); P.append(-sig); N.append(f"pair rev h={h}")
    return N, P


def f_carry(x: CarryCtx):
    f = store.load("funding", x.sym).set_index("ts").rate
    P, N = [np.ones(len(x.idx))], ["carry always-on"]
    for w in [3, 9, 27]:
        fs = f.rolling(w).mean().reindex(x.idx, method="ffill").fillna(0).to_numpy()
        for a in [0.0, 0.00005, 0.0001, 0.00015, 0.0002, 0.0003]:
            P.append(S.hold(fs > a, fs < a * 0.5)); N.append(f"carry w={w} a={a}")
    return N, P


def f_season(x: S.Ctx):
    r = x.r.to_numpy()
    hr, dw, yr = x.idx.hour.to_numpy(), x.idx.dayofweek.to_numpy(), x.idx.year.to_numpy()
    P, N = [], []
    for kind, key, nk in (("hour", hr, 24), ("dow", dw, 7)):
        if kind == "hour" and x.tf == "1d":
            continue
        if kind == "dow" and x.tf != "1d":
            continue
        for k in ([2, 4, 6, 8, 12] if kind == "hour" else [1, 2, 3, 4]):
            pos = np.zeros(len(r))
            for Y in range(2021, yr.max() + 1):
                tr = yr < Y
                m = np.array([r[tr & (key == c)].mean() if (tr & (key == c)).any() else -9 for c in range(nk)])
                top = set(np.argsort(-m)[:k])
                sel = (yr == Y) & np.isin(key, list(top))
                pos[sel] = 1.0
            P.append(pos); N.append(f"season {kind} top{k}")
    return N, P


def f_ensemble(x: S.Ctx):
    """무최적화 앙상블: 여러 룩백 신호의 평균. 파라미터 선택이 없어 과최적화 위험이 가장 낮다."""
    P, N = [], []
    hs = [24, 48, 72, 120, 168, 240, 336, 504, 720, 1080, 1440, 2160]
    vol = x.r.rolling(x.nb(168)).std().to_numpy() * np.sqrt(x.ppy)
    sig_ts = np.mean([np.sign((x.C / x.C.shift(x.nb(h)) - 1).fillna(0).to_numpy()) for h in hs], axis=0)
    fs, ss = [x.nb(h) for h in (12, 24, 48, 96)], [x.nb(h) for h in (168, 336, 720, 1440)]
    sig_ma = np.mean([np.sign((x.C.rolling(f).mean() - x.C.rolling(s).mean()).fillna(0).to_numpy())
                      for f in fs for s in ss if f < s], axis=0)
    c = x.C.to_numpy(); dsig = []
    for h in [48, 168, 336, 720]:
        n = x.nb(h); hh = x.C.shift(1).rolling(n).max().to_numpy(); ll = x.C.shift(1).rolling(n).min().to_numpy()
        dsig.append(S.hold(c > hh, None, c < ll))
    sig_dc = np.mean(dsig, axis=0)
    comb = (sig_ts + sig_ma + sig_dc) / 3
    for nm, sg in (("ts", sig_ts), ("ma", sig_ma), ("donch", sig_dc), ("ALL", comb)):
        for kind in ("LS", "LF"):
            base = sg if kind == "LS" else np.clip(sg, 0, 1)
            P.append(base); N.append(f"ens-{nm} {kind}")
            for tv in (0.3, 0.5):
                P.append(base * np.clip(np.nan_to_num(tv / vol, nan=0.0), 0, 1)); N.append(f"ens-{nm} {kind} volt={tv}")
    return N, P


def main():
    fams, tfs, tag = sys.argv[1].split(","), sys.argv[2].split(","), sys.argv[3]
    out, t0 = [], time.time()
    for fam in fams:
        for tf in tfs:
            ctxs = [PairCtx(tf)] if fam == "pairs" else \
                [(CarryCtx if fam == "carry" else S.Ctx)(s, tf) for s in S.SYMS]
            for x in ctxs:
                try:
                    N, P = {"pairs": f_pairs, "carry": f_carry, "season": f_season, "ensemble": f_ensemble}[fam](x)
                    if not P:
                        continue
                    if fam == "ensemble":  # 선택 없음: 파라미터 1개짜리 각각을 고정 성과로 평가
                        fy = 2022
                    else:
                        fy = 2022
                    r = S.evaluate(fam, x, N, P, fy, 1.5)
                    if fam == "ensemble" and r:
                        R = np.array([x.run(np.asarray(p, float)) for p in P]).T
                        yrs = x.idx.year.to_numpy(); m = yrs >= 2022
                        r["fixed"] = {N[j]: {"sh": round(float(R[m, j].mean() / R[m, j].std() * np.sqrt(x.ppy)), 2),
                                             "ann": round(float(R[m, j].mean() * x.ppy), 3),
                                             "mdd": round(core.metrics(R[m, j], x.tf)["mdd"], 3),
                                             "sh_2020_2021": round(float(R[~m, j].mean() / R[~m, j].std() * np.sqrt(x.ppy)), 2) if (~m).sum() > 100 else None}
                                      for j in range(len(N))}
                except Exception as e:  # noqa: BLE001
                    import traceback; traceback.print_exc()
                    r = {"family": fam, "sym": x.sym, "tf": tf, "error": repr(e)}
                if r:
                    out.append(r)
                    print(fam, tf, x.sym, "OOS Sh", r.get("oos_sharpe"), "n", r.get("n_params"), f"{time.time()-t0:.0f}s", flush=True)
    Path(f"logs/screen_{tag}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
