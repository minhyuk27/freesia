"""사전 지정 규칙(문헌·실무 임계값, 파라미터 피팅 없음) 필터/투표 검증 — BTC·ETH 일봉.
기준 노출 e = 추세 앙상블(롱플랫). 각 필터는 e에 곱하고, 투표는 e × (통과 수 / 전체).
검증: 전체·2022~ Sharpe, 두 종목 일관성, 필터 위상 무작위(플라시보) p값."""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from freesia.bt import core  # noqa: E402
import screen as S  # noqa: E402
import edge2 as E2  # noqa: E402


def filters(F):
    f = {}
    def has(c): return c in F and F[c].notna().any()
    if has("vix"): f["VIX<25"] = (F.vix < 25)
    if has("usd_d20"): f["달러 20일 상승 아님"] = (F.usd_d20 <= 0)
    if has("y10_d20"): f["10년물 20일 상승 아님"] = (F.y10_d20 <= 0)
    if has("nasdaq_a200"): f["나스닥 200일선 위"] = (F.nasdaq_a200 > 0)
    if has("fng"): f["공포탐욕 극단 탐욕(>75) 아님"] = (F.fng <= 75)
    if has("stable_g30"): f["스테이블코인 30일 증가"] = (F.stable_g30 > 0)
    if has("cb_prem"): f["코인베이스 프리미엄 5일 평균>0"] = (F.cb_prem.rolling(5).mean() > 0)
    if has("tbr"): f["테이커 매수 우위(3일)"] = (F.tbr > 0.5)
    if has("ex_net30"): f["거래소 순유출(30일)"] = (F.ex_net30 < 0)
    if has("adx14"): f["ADX>20"] = (F.adx14 > 20)
    if has("basis_ann"): f["분기물 베이시스 0~15%"] = ((F.basis_ann > 0) & (F.basis_ann < 0.15))
    if has("fund"): f["펀딩 과열 아님(<0.05%/일)"] = (F.fund < 0.0005)
    return {k: v.fillna(True).to_numpy().astype(float) for k, v in f.items()}  # 데이터 없는 구간은 통과(중립)


def sh(a, m):
    a = a[m]
    return round(float(a.mean() / a.std() * np.sqrt(365)), 2) if a.std() > 0 else 0.0


def main():
    out = {}
    for sym in S.SYMS:
        x, F = E2.build(sym)
        C, fund, idx = x.C.to_numpy(), x.fund, x.idx
        yrs = idx.year.to_numpy(); m22 = yrs >= 2022; mall = yrs >= 2021
        e = F["ens"].to_numpy()
        Rb = core.run(e, C, fund); bh = x.r.to_numpy()
        fl = filters(F)
        res = {"기준 추세": {"2021+": sh(Rb, mall), "2022+": sh(Rb, m22), "mdd22": round(core.metrics(Rb[m22], "1d")["mdd"], 3)},
               "매수후보유": {"2021+": sh(bh, mall), "2022+": sh(bh, m22)}}
        for k, v in fl.items():
            R = core.run(e * v, C, fund)
            res[k] = {"2021+": sh(R, mall), "2022+": sh(R, m22), "mdd22": round(core.metrics(R[m22], "1d")["mdd"], 3), "통과율": round(float(v[mall].mean()), 2)}
        # DVOL 변동성 스케일
        if "dvol" in F and F.dvol.notna().any():
            dv = (F.dvol / 100).to_numpy()
            for tv in (0.4, 0.5, 0.6):
                sc = np.where(np.isnan(dv), 1.0, np.clip(tv / dv, 0, 1))
                R = core.run(e * sc, C, fund); Rbh = core.run(sc, C, fund)
                res[f"DVOL 목표변동성 {int(tv*100)}% × 추세"] = {"2021+": sh(R, mall), "2022+": sh(R, m22), "mdd22": round(core.metrics(R[m22], "1d")["mdd"], 3)}
                res[f"DVOL 목표변동성 {int(tv*100)}% × 매수후보유"] = {"2021+": sh(Rbh, mall), "2022+": sh(Rbh, m22), "mdd22": round(core.metrics(Rbh[m22], "1d")["mdd"], 3)}
        # 투표
        names = list(fl)
        V = np.mean([fl[k] for k in names], axis=0)
        for nm, w in (("투표(전체 평균)", V), ("투표(과반 통과)", (V >= 0.5).astype(float)), ("투표(2/3 이상 통과)", (V >= 2 / 3).astype(float))):
            R = core.run(e * w, C, fund)
            res[nm] = {"2021+": sh(R, mall), "2022+": sh(R, m22), "mdd22": round(core.metrics(R[m22], "1d")["mdd"], 3), "평균 노출": round(float((e * w).mean()), 2)}
        # 플라시보: 투표 시리즈를 무작위 회전
        rng = np.random.default_rng(3); real = sh(core.run(e * V, C, fund), m22); null = []
        for _ in range(300):
            k = int(rng.integers(90, len(V) - 90)); null.append(sh(core.run(e * np.roll(V, k), C, fund), m22))
        res["투표 플라시보"] = {"real2022+": real, "null중앙": round(float(np.median(null)), 2), "null95": round(float(np.percentile(null, 95)), 2), "p": round(float((np.array(null) >= real).mean()), 3)}
        out[sym] = res
        print("=====", sym)
        for k, v in res.items():
            print(f"  {k:32} {v}")
    Path("logs/edge3.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
