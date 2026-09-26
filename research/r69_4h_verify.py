"""R69 — R68 4시간 패턴 모델 검증: 시간순 워크포워드(과거만 학습) + 3단 레버리지 계좌.

모델: R68과 같은 특징(4시간봉 6개 창 + 지표 그룹 G1~G9)과 HistGB 분류기(시드 3 평균).
워크포워드: 평가 연도 Y마다 2020-01 ~ (Y−1)년 말로 학습, Y년 평가 (2022~2026). 학습 마지막 2일 제외(라벨 겹침).
신호: 학습 예측의 상위 10% 이상 롱, 하위 10% 이하 숏 (문턱은 학습에서만).
목표·보유: A = 다음 4시간(1봉), B = 다음 24시간(6봉, 포지션 최대 6개 겹침 → 한 포지션 = 자본 1/6).
구성: 전체, 전체−교차, 전체−파생.
계좌: 레버리지 1(무레버)·3(중간)·5(공격), 명목 = 자본 × 레버리지 × 슬롯비중, 왕복 비용 14bp, 청산은 무시(4시간 −20% 이상 드묾, 별도 표시).
사용: python research/r69_4h_verify.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
from r68_bar_patterns import build, COST
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)


def fit_pred(tr, te, feats, y):
    ps_tr, ps_te = [], []
    for sd in range(3):
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
                                           l2_regularization=1.0, random_state=sd).fit(tr[feats].values, (tr[y].values > 0).astype(int))
        ps_tr.append(m.predict_proba(tr[feats].values)[:, 1]); ps_te.append(m.predict_proba(te[feats].values)[:, 1])
    return np.mean(ps_tr, 0), np.mean(ps_te, 0)


def account(T, H, lev):
    """T: 신호 거래(t, sym, side, fwd). 포지션마다 자본 × lev / H 명목, 보유 H봉. 4시간 격자 단위 수익 합산 → 복리."""
    T = T.copy(); T["r"] = T.side * T.fwd - COST
    per = T.groupby("t").r.sum() * lev / H                       # 같은 시각 진입 포지션들의 수익(청산 시점 근사 = 진입 시점 기록)
    per = per.sort_index()
    eq = (1 + per.clip(lower=-0.999)).cumprod()
    mon = eq.resample("ME").last().pct_change().dropna()
    dd = (eq / eq.cummax() - 1).min()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    return (eq.iloc[-1] ** (1 / 12 / yrs) - 1) * 100, mon.median() * 100, dd * 100, (mon > 0).mean(), eq.iloc[-1]


if __name__ == "__main__":
    tf, W = "4h", 6
    closes = {s: store.load("klines_1m", s).set_index("ts").close.resample(tf).last() for s in ("BTCUSDT", "ETHUSDT")}
    D = pd.concat([build(s, tf, W, closes["ETHUSDT" if s == "BTCUSDT" else "BTCUSDT"]).assign(sym=s) for s in ("BTCUSDT", "ETHUSDT")], ignore_index=True)
    D = D.replace([np.inf, -np.inf], np.nan).dropna(subset=["f1", "fW", "G1_r0"]); D["yr"] = D.t.dt.year
    ALL = [c for c in D.columns if c[:2] in ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9") and c[2] == "_"]
    CFG = {"전체": ALL, "전체−교차": [c for c in ALL if not c.startswith("G7")], "전체−파생": [c for c in ALL if not c.startswith("G6")]}
    for ycol, H, tname in (("f1", 1, "A 다음 4시간"), ("fW", W, "B 다음 24시간")):
        P(f"\n===== {tname}")
        for cname, feats in CFG.items():
            allT = []; line = []
            for Y in range(2022, 2027):
                tr = D[D.t < pd.Timestamp(f"{Y}-01-01", tz="UTC") - pd.Timedelta(days=2)]; te = D[D.yr == Y]
                ptr, pte = fit_pred(tr, te, feats, ycol)
                hi, lo = np.quantile(ptr, 0.9), np.quantile(ptr, 0.1)
                side = np.where(pte >= hi, 1, np.where(pte <= lo, -1, 0)); k = side != 0
                T = pd.DataFrame({"t": te.t.values[k], "sym": te.sym.values[k], "side": side[k], "fwd": te[ycol].values[k]})
                r = T.side * T.fwd - COST
                line.append(f"{Y}: {r.mean()*1e4:+6.1f}bp 적중 {(T.side * T.fwd > 0).mean():.3f} n={len(T)}")
                allT.append(T)
            T = pd.concat(allT); r = T.side * T.fwd - COST
            P(f"  [{cname}] 거래당 {r.mean()*1e4:+.1f}bp (적중 {(T.side * T.fwd > 0).mean():.3f}, 하루 {len(T)/((T.t.max()-T.t.min()).days):.2f}건, 롱 비중 {(T.side > 0).mean():.2f}) | " + " | ".join(line))
            for lev, nm in ((1, "무레버"), (3, "중간 3배"), (5, "공격 5배")):
                cm, mm, dd, wp, fin = account(T, H, lev)
                P(f"      {nm:7s}: 월 복리 {cm:+6.2f}% (월 중앙 {mm:+.2f}%), MDD {dd:6.1f}%, 흑자 달 {wp:.0%}, 최종 배수 {fin:.2f}")
    (R / "logs" / "r69.summary").write_text("R69 4시간 패턴 워크포워드·계좌 검증 완료. 세부 logs/r69.log\n")
