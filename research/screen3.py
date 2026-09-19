"""3차 스크리닝: BTC/ETH 로테이션, 거래량 동반 돌파, 변동성 레짐 필터 추세."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from freesia.bt import core  # noqa: E402
import screen as S  # noqa: E402


class RotCtx(S.Ctx):
    def __init__(self, tf):
        self.sym, self.tf = "BTC|ETH", tf
        b, e = core.bars("BTCUSDT", tf), core.bars("ETHUSDT", tf)
        self.idx = b.index.intersection(e.index)
        self.hb = core.TF_HOURS[tf]; self.ppy = core.PPY[tf]
        self.cb, self.ce = b.close.reindex(self.idx), e.close.reindex(self.idx)
        self.C = self.cb
        self.r = pd.Series(0.5 * (self.cb.pct_change().fillna(0) + self.ce.pct_change().fillna(0)), index=self.idx)
        self.rb, self.re = self.cb.pct_change().fillna(0).to_numpy(), self.ce.pct_change().fillna(0).to_numpy()
        self.fb = core.funding_bars("BTCUSDT", self.idx, tf); self.fe = core.funding_bars("ETHUSDT", self.idx, tf)

    def run(self, pos):
        pos = np.nan_to_num(pos)
        p = np.concatenate([[0.0], pos[:-1]]); p2 = np.concatenate([[0.0], p[:-1]])
        ret = np.where(p == 1, self.rb, np.where(p == 2, self.re, 0.0))
        fund = np.where(p == 1, self.fb, np.where(p == 2, self.fe, 0.0))
        chg = np.where(p == p2, 0.0, np.where((p == 0) | (p2 == 0), 1.0, 2.0))   # 자산 교체는 양 다리 비용
        return ret - chg * core.COST - fund


def f_rotation(x: RotCtx):
    P, N = [], []
    for h in [72, 168, 336, 720, 1440]:
        n = x.nb(h)
        mb, me = (x.cb / x.cb.shift(n) - 1).fillna(-1).to_numpy(), (x.ce / x.ce.shift(n) - 1).fillna(-1).to_numpy()
        for th in [0.0, 0.05]:
            best = np.where(me > mb, 2, 1); mx = np.maximum(mb, me)
            P.append(np.where(mx > th, best, 0)); N.append(f"rot rs={h} th={th}")
        for h2 in [336, 1440, 3000]:
            n2 = x.nb(h2)
            ma = (x.cb > x.cb.rolling(n2).mean()).to_numpy() | (x.ce > x.ce.rolling(n2).mean()).to_numpy()
            P.append(np.where(ma, np.where(me > mb, 2, 1), 0)); N.append(f"rot rs={h} ma={h2}")
    return N, P


def f_volbreak(x: S.Ctx):
    P, N = [], []
    vz_src = x.b.quote_volume
    for lh in [48, 168, 336, 720]:
        n = x.nb(lh)
        hh = x.C.shift(1).rolling(n).max().to_numpy(); c = x.C.to_numpy()
        for vh in [168, 720]:
            m = x.nb(vh)
            vz = ((vz_src - vz_src.rolling(m).mean()) / vz_src.rolling(m).std()).fillna(0).to_numpy()
            for k in [0, 0.5, 1, 1.5]:
                for ex in [0.5, 1.0]:
                    ll = x.C.shift(1).rolling(max(1, int(n * ex))).min().to_numpy()
                    P.append(S.hold((c > hh) & (vz > k), c < ll)); N.append(f"volbreak N={lh} vw={vh} k={k} exit={ex}")
    return N, P


def f_volregime(x: S.Ctx):
    P, N = [], []
    vol = x.r.rolling(x.nb(336)).std()
    volq = vol.rolling(x.nb(24 * 365)).rank(pct=True).fillna(0.5).to_numpy() if x.tf == "1d" else \
        vol.rolling(min(len(vol), x.nb(24 * 365)), min_periods=x.nb(24 * 90)).rank(pct=True).fillna(0.5).to_numpy()
    for h in [72, 168, 336, 720, 1440]:
        n = x.nb(h)
        sig = np.clip(np.sign((x.C / x.C.shift(n) - 1).fillna(0).to_numpy()), 0, 1)
        for lo, hi in [(0, 1), (0, 0.7), (0, 0.5), (0.3, 1), (0.5, 1)]:
            on = (volq >= lo) & (volq <= hi)
            P.append(sig * on); N.append(f"trend h={h} volq∈[{lo},{hi}]")
    return N, P


def main():
    tfs, tag = sys.argv[1].split(","), sys.argv[2]
    out, t0 = [], time.time()
    for fam, fn in (("rotation", f_rotation), ("volbreak", f_volbreak), ("volregime", f_volregime)):
        for tf in tfs:
            ctxs = [RotCtx(tf)] if fam == "rotation" else [S.Ctx(s, tf) for s in S.SYMS]
            for x in ctxs:
                try:
                    N, P = fn(x)
                    r = S.evaluate(fam, x, N, P, 2022, 1.5)
                except Exception as e:  # noqa: BLE001
                    import traceback; traceback.print_exc()
                    r = {"family": fam, "sym": x.sym, "tf": tf, "error": repr(e)}
                if r:
                    out.append(r)
                    print(fam, tf, x.sym, "OOS Sh", r.get("oos_sharpe"), "n", r.get("n_params"), f"{time.time()-t0:.0f}s", flush=True)
    Path(f"logs/screen_{tag}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
