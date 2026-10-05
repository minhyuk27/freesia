"""R134 — 같은 목표·다른 접근 B: 첫 작은 이탈(R131 사건)의 '시작 기제'로 계속/되돌림을 가르는가 — 사전 고정 가설 검정.

가설(부호를 데이터 보기 전에 고정, 문턱은 그해 이전 연도 중앙값 — 인과적):
  H1 현물 주도: spotlead_5·s > 문턱 → 계속        H2 새 포지션: OI 60분 변화 > 문턱 → 계속(감소 = 청산·정리 → 되돌림)
  H3 시장 전체: 다른 종목 5분 수익·s > 문턱 → 계속  H4 급발생: 체결 z(1분) − 체결 z(60분) > 문턱 → 계속(외생 충격)
  H5 레버리지 추격: 프리미엄 5분 변화·s > 문턱 → 되돌림   H6 코인베이스 주도: cblead_5·s > 문턱 → 계속
  종합 점수 S = H1 + H2 + H3 + H4 + H6 − H5 (각 ±1). S ≥ 2 → 추종, S ≤ −2 → 반대.
라벨: R131과 같다(진입가 대칭 ±X 먼저 닿는 쪽, 무작위 50%). 가설 부호가 사전 고정이므로 2022~26 전체가 표본 밖이다(문턱만 이전 연도).
비용: 진입 테이커 7bp, 목표 지정가 2bp, 손절 테이커 7bp → 승 X−9bp, 패 −X−14bp.
사용: python research/r134_mechanism.py"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
R = Path(__file__).resolve().parents[1]
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
COLS = ["sym", "t", "yr", "s", "y", "X", "cont", "MS_spotlead_5_s", "MS_cblead_5_s", "MS_othret_5_s", "MS_premchg_5_s", "DRV_oi_chg_60", "MS_tradez_1", "MS_tradez_60"]
HYP = {"H1 현물 주도": ("MS_spotlead_5_s", 1), "H2 새 포지션(OI↑)": ("DRV_oi_chg_60", 1), "H3 시장 전체": ("MS_othret_5_s", 1),
       "H4 급발생": ("burst", 1), "H5 레버리지 추격": ("MS_premchg_5_s", -1), "H6 코인베이스 주도": ("MS_cblead_5_s", 1)}


def prior_median(x, yr):
    out = np.full(len(x), np.nan)
    for Y in np.unique(yr):
        h = x[(yr < Y) & np.isfinite(x)]
        if len(h) < 300: h = x[(yr == Y) & np.isfinite(x)][:300]
        out[yr == Y] = np.median(h) if len(h) else np.nan
    return out


def stat(g, X):
    n = len(g); w = g.win.mean()
    netbp = (np.where(g.win, X - 9e-4, -X - 14e-4)).mean() * 1e4
    return n, w, (w - .5) / np.sqrt(.25 / n) if n else np.nan, netbp


if __name__ == "__main__":
    D = pd.read_parquet(R / "data" / "r132_D2.parquet", columns=COLS)
    D["burst"] = D.MS_tradez_1 - D.MS_tradez_60
    yr = D.yr.values; rows = []
    sig = {}
    for h, (c, sd) in HYP.items():
        x = D[c].values.astype(float); thr = prior_median(x, yr)
        sig[h] = np.where(np.isfinite(x), np.where(x > thr, 1, -1) * sd, 0)     # +1 = 계속 예측, −1 = 되돌림 예측
    S = sum(sig.values()); sig["종합 S≥2/≤−2"] = np.where(S >= 2, 1, np.where(S <= -2, -1, 0))
    for h, v in sig.items():
        for (y, X), idx in D.groupby(["y", "X"]).groups.items():
            g = D.loc[idx]; vv = v[D.index.get_indexer(idx)]
            m = vv != 0; gg = g[m].assign(win=(g.cont.values[m] == 1) == (vv[m] == 1))
            for per, ps in (("22~24", gg.yr <= 2024), ("25~26", gg.yr >= 2025), ("전체", gg.yr > 0)):
                n, w, z, nb = stat(gg[ps], X)
                rows.append(dict(가설=h, y=round(y * 100, 1), X=round(X * 100, 1), 기간=per, n=n, 적중=w, z=z, 순bp=nb))
    O = pd.DataFrame(rows)
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 400)
    W = O.pivot_table(index=["가설", "y", "X"], columns="기간", values=["적중", "z", "순bp"]).round(3)
    P(W.to_string())
    P("\n가설별 8조합 평균(전체 기간): ")
    P(O[O.기간 == "전체"].groupby("가설")[["적중", "z", "순bp"]].mean().round(3).to_string())
    P(f"검정 수 {len(O[O.기간=='전체'])}개(가설 7 × 조합 8). |z|>2 우연 기대 약 {0.0455*len(O[O.기간=='전체']):.1f}개")
    O.to_parquet(R / "data" / "r134.parquet", index=False)
