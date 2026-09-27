"""R83 — 실전 준비: z0.31을 '실시간으로 똑같이 계산 가능한 특징'만으로 다시 학습해도 성과가 유지되는가.

bookDepth(±1~5% 누적 호가)는 바이낸스가 일별 파일로만 제공 → 실시간 재현 불가 → bslope·bimb 제거.
나머지 50개는 공개 API(1분봉·테이커 매수, OI·롱숏비 5분, 프리미엄 지수, 펀딩)로 실시간 계산 가능.
비교: R32 전체 특징(원래) vs 실시간 동등(bslope·bimb 제외), 같은 워크포워드(2022~2026 평가, 매년 과거 전체 재학습).
계좌: R81 확정 설정(문턱 25bp, V3 α1.5 상한 6, 킬 3%/12h, MMR 0.65%) × 공격 w2.5N10 / 초공격 w3N12 / 극공격 w3.5N14 (+ 무레버 w0.5N2 참고).
출력: data/r83_wf_live.parquet (실시간 동등 예측 — 실전 엔진 검증 기준선)
사용: python research/r83_live_parity.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
from sklearn.ensemble import HistGradientBoostingRegressor
import r40_regime_leverage as RL
import r80_max_return as R80
from freesia import store

# r32_filter.py와 같은 데이터·특징·모델·워크포워드 (그 파일은 import 시 전체 실행되므로 필요한 부분만 복제)
E = pd.read_parquet(R / "data" / "r32_events.parquet")
E["ts"] = pd.DatetimeIndex(E.ts); E["yr"] = E.ts.dt.year; E["isbtc"] = (E.sym == "BTCUSDT").astype(float)
E = E.replace([np.inf, -np.inf], np.nan)
ALL = [c for c in E.columns if c not in {"sym", "f", "x", "ret", "mae", "ts", "yr"}]
LIVE = [c for c in ALL if c not in ("bslope", "bimb")]


def walk(feats):
    out = []
    for Y in range(2022, 2027):
        tr = E[E.yr < Y]; te = E[E.yr == Y]
        fu = [c for c in feats if tr[c].notna().sum() > 1000]
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=400,
                                          l2_regularization=1.0, random_state=0).fit(tr[fu].values, np.clip(tr.ret.values, -0.02, 0.02))
        out.append(te.assign(pred=m.predict(te[fu].values)))
    return pd.concat(out)


def prep(W):
    W = W.copy(); W["ts"] = pd.DatetimeIndex(W.ts); W["ret"] = W.ret - 1.2e-4
    for s in ("BTCUSDT", "ETHUSDT"):
        k = store.load("klines_1m", s).set_index("ts").close.resample("D").last(); r = np.log(k).diff()
        v30 = r.rolling(30).std(); reg = (v30 / v30.rolling(365, min_periods=120).median()).shift(1)
        m = W.sym == s; W.loc[m, "vreg"] = reg.reindex(W.loc[m, "ts"].dt.normalize()).values
    W["vreg"] = W.vreg.fillna(1.0)
    return W


if __name__ == "__main__":
    RL.MMR = 0.0065
    P(f"특징: 원래 {len(ALL)}개 / 실시간 동등 {len(LIVE)}개 (제외: bslope, bimb)")
    for tag, feats in (("원래(전체)", ALL), ("실시간 동등", LIVE)):
        W = walk(feats)
        if tag == "실시간 동등": W.to_parquet(R / "data" / "r83_wf_live.parquet", index=False)
        for th in (0.0010, 0.0025):
            S = prep(W[W.pred > th]).sort_values(["f", "sym"]).reset_index(drop=True)
            by = S.groupby(S.ts.dt.year).ret.mean() * 1e4
            P(f"\n[{tag}] 예측 > {th*1e4:.0f}bp: 거래 {len(S):,} 거래당 {S.ret.mean()*1e4:+.2f}bp 승률 {(S.ret>0).mean():.3f} | " +
              " ".join(f"{y%100}:{v:+.1f}" for y, v in by.items()))
            if th != 0.0025: continue
            V2 = np.clip(S.pred.values / 0.001, 0.5, 3.0); mult = np.minimum(np.clip(S.vreg.values ** 1.5, 0.3, 4.0) * V2, 6.0)
            for w, N, nm in ((0.5, 2, "무레버급"), (2.5, 10, "공격"), (3.0, 12, "초공격"), (3.5, 14, "극공격")):
                res = RL.sim(S, w, N, mult, X=0.03, H=720)
                if res is None: P(f"   {nm} w{w}N{N}: 파산"); continue
                d = R80.daily_from_eq(res[0]); m_all = R80.metrics(d); m_te = R80.metrics(d, "2024-01-01", "2027-01-01")
                P(f"   {nm:5s} w{w}N{N}: 전 기간 월 {m_all['mo']:+.1f}% (중앙 {m_all['med']:+.1f}%) MDD {m_all['mdd']:.0f}% 월≥100% {m_all['big']}회 | "
                  f"24~26 월 {m_te['mo']:+.1f}% MDD {m_te['mdd']:.0f}% | {m_all['yr']}")
    (R / "logs" / "r83.summary").write_text("R83 실시간 동등 특징 z0.31 검증 완료. 세부 logs/r83.log\n")
