"""결합 구조 + 강건성 검사.
H: 현물 BTC(또는 ETH)를 항상 보유하고, 추세 신호가 꺼지면 무기한 숏으로 헤지(펀딩 수취). 헤지비율 = 1 - 추세노출.
C: 캐리(항상 헤지) 단독.
T: 추세 롱플랫(현금) 단독.
검사: 비용 배수(1x/2x/3x), 분기점(2022-01/2023-01/2024-01), 위상 무작위(플라시보) p값, 부트스트랩 신뢰구간."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from freesia import store  # noqa: E402
from freesia.bt import core  # noqa: E402
import screen as S  # noqa: E402
import screen2 as S2  # noqa: E402

HS = [24, 48, 72, 120, 168, 240, 336, 504, 720, 1080, 1440, 2160]


def trend_exposure(x: S.Ctx, kind="ALL", volt=None):
    c = x.C.to_numpy()
    ts = np.mean([np.sign((x.C / x.C.shift(x.nb(h)) - 1).fillna(0).to_numpy()) for h in HS], axis=0)
    fs, ss = [x.nb(h) for h in (12, 24, 48, 96)], [x.nb(h) for h in (168, 336, 720, 1440)]
    ma = np.mean([np.sign((x.C.rolling(f).mean() - x.C.rolling(s).mean()).fillna(0).to_numpy())
                  for f in fs for s in ss if f < s], axis=0)
    dc = []
    for h in [48, 168, 336, 720]:
        n = x.nb(h); hh = x.C.shift(1).rolling(n).max().to_numpy(); ll = x.C.shift(1).rolling(n).min().to_numpy()
        dc.append(S.hold(c > hh, None, c < ll))
    dc = np.mean(dc, axis=0)
    sg = {"ts": ts, "ma": ma, "dc": dc, "ALL": (ts + ma + dc) / 3}[kind]
    e = np.clip(sg, 0, 1)
    if volt:
        vol = x.r.rolling(x.nb(168)).std().to_numpy() * np.sqrt(x.ppy)
        e = e * np.clip(np.nan_to_num(volt / vol, nan=0.0), 0, 1)
    return e


def series(x: S2.CarryCtx, e, cost_mult=1.0):
    """세 전략의 순수익률 시계열 반환 (자본 대비)."""
    r = x.r_price
    p = np.concatenate([[0.0], e[:-1]])                  # 봉 동안 들고 있는 추세 노출
    p2 = np.concatenate([[0.0], p[:-1]])
    hedge, hedge_prev = 1 - p, 1 - p2
    basis = x.ret * 1.2                                   # = 펀딩 - Δ프리미엄 (자본 조정 전)
    # H: 현물 항상 보유. 미헤지분만 가격 노출, 헤지분은 펀딩+베이시스. 비용은 퍼프 다리(편도 6bp)만.
    H = (p * r + hedge * basis) / 1.2 - np.abs(hedge - hedge_prev) * core.COST * cost_mult / 1.2
    T = core.run(e, x.C.to_numpy(), x.fund, core.COST * cost_mult)
    C = core.run_ret(np.ones(len(e)), x.ret, np.zeros(len(e)), cost=0.0017 / 1.2 * cost_mult)
    C[0] = 0
    return H, T, C


def sh(a, ppy):
    return float(a.mean() / a.std() * np.sqrt(ppy)) if a.std() > 0 else 0.0


def stats(a, ppy, tf):
    m = core.metrics(a, tf)
    return {"sharpe": round(sh(a, ppy), 2), "ann": round(m["ann"], 3), "vol": round(m["vol"], 3), "mdd": round(m["mdd"], 3)}


def placebo_p(e, r_fn, ppy, n=400, seed=0):
    """추세 노출 시계열을 무작위로 회전시켜 (신호-수익 정렬을 파괴) Sharpe 분포와 실제를 비교."""
    rng = np.random.default_rng(seed)
    real = sh(r_fn(e), ppy)
    L = len(e); min_shift = int(L * 0.05)
    null = [sh(r_fn(np.roll(e, int(rng.integers(min_shift, L - min_shift)))), ppy) for _ in range(n)]
    return real, float((np.array(null) >= real).mean()), float(np.median(null)), float(np.percentile(null, 95))


def boot_ci(a, ppy, n=1000, block=None, seed=1):
    rng = np.random.default_rng(seed)
    b = block or max(5, int(ppy / 12))
    L = len(a); k = int(np.ceil(L / b)); out = []
    for _ in range(n):
        st = rng.integers(0, L - b, k)
        s = np.concatenate([a[i:i + b] for i in st])[:L]
        out.append(sh(s, ppy))
    return [round(float(np.percentile(out, q)), 2) for q in (5, 50, 95)]


def main():
    tf = sys.argv[1] if len(sys.argv) > 1 else "1d"
    out = {}
    for sym in S.SYMS:
        x = S2.CarryCtx(sym, tf)
        x.r_price = x.r.to_numpy() if False else x.C.pct_change().fillna(0).to_numpy()
        x.r_price = x.C.pct_change().fillna(0).to_numpy()
        ppy, yrs = x.ppy, x.idx.year.to_numpy()
        bh = x.r_price
        res = {"BH": stats(bh[yrs >= 2020], ppy, tf)}
        for kind, volt in [("ALL", None), ("ALL", 0.5), ("ts", None), ("ma", None), ("dc", None)]:
            e = trend_exposure(x, kind, volt)
            H, T, C = series(x, e)
            key = f"{kind}{'' if not volt else f' vt={volt}'}"
            m22 = yrs >= 2022
            res[key] = {
                "H_전체": stats(H, ppy, tf), "H_2022+": stats(H[m22], ppy, tf), "H_2020-21": stats(H[~m22], ppy, tf),
                "T_전체": stats(T, ppy, tf), "T_2022+": stats(T[m22], ppy, tf),
                "C_전체": stats(C, ppy, tf), "C_2022+": stats(C[m22], ppy, tf),
                "H_연도별": {int(Y): round(sh(H[yrs == Y], ppy), 2) for Y in sorted(set(yrs))},
                "T_연도별": {int(Y): round(sh(T[yrs == Y], ppy), 2) for Y in sorted(set(yrs))},
                "H_ann_연도별%": {int(Y): round(float(H[yrs == Y].sum() * 100), 1) for Y in sorted(set(yrs))},
                "BH_ann_연도별%": {int(Y): round(float(bh[yrs == Y].sum() * 100), 1) for Y in sorted(set(yrs))},
            }
            # 비용 민감도
            res[key]["H_비용배수"] = {str(m): stats(series(x, e, m)[0], ppy, tf)["sharpe"] for m in (1, 2, 4)}
            res[key]["T_비용배수"] = {str(m): stats(series(x, e, m)[1], ppy, tf)["sharpe"] for m in (1, 2, 4)}
            # 분기점
            res[key]["H_분기점"] = {f"{Y}+": stats(H[yrs >= Y], ppy, tf)["sharpe"] for Y in (2021, 2022, 2023, 2024, 2025)}
            if kind == "ALL":
                rf = lambda ee: series(x, ee)[0]  # noqa: E731
                real, p, med, p95 = placebo_p(e, rf, ppy, n=300)
                res[key]["H_플라시보"] = {"real": round(real, 2), "p": round(p, 3), "null_median": round(med, 2), "null_p95": round(p95, 2)}
                rt = lambda ee: series(x, ee)[1]  # noqa: E731
                real, p, med, p95 = placebo_p(e, rt, ppy, n=300)
                res[key]["T_플라시보"] = {"real": round(real, 2), "p": round(p, 3), "null_median": round(med, 2), "null_p95": round(p95, 2)}
                res[key]["H_부트스트랩CI(5,50,95)"] = boot_ci(H, ppy)
                res[key]["T_부트스트랩CI(5,50,95)"] = boot_ci(T, ppy)
        out[sym] = res
    Path(f"logs/combo_{tf}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    for sym, res in out.items():
        print("=====", sym, tf, "BH", res["BH"])
        for k, v in res.items():
            if k == "BH":
                continue
            print(f"[{k}] H 전체 {v['H_전체']} | 2022+ {v['H_2022+']} | T 전체 {v['T_전체']} | C 전체 {v['C_전체']}")
            print("     H연도별", v["H_연도별"], "비용배수", v["H_비용배수"], "분기점", v["H_분기점"])
            print("     T연도별", v["T_연도별"], "비용배수", v["T_비용배수"])
            if "H_플라시보" in v:
                print("     플라시보 H", v["H_플라시보"], "T", v["T_플라시보"], "부트CI H", v["H_부트스트랩CI(5,50,95)"], "T", v["T_부트스트랩CI(5,50,95)"])
            print("     H 연도별수익% ", v["H_ann_연도별%"], " BH ", v["BH_ann_연도별%"])


if __name__ == "__main__":
    main()
