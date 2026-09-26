"""R76 — 직전 24시간을 분석해 다음 24시간이 추세 강화 / 반전(반등) / 횡보 중 무엇인지 예측.

시점: 4시간봉 종가(BTC·ETH, 2020~2026). 직전 24시간 = 4시간봉 6개.
라벨(다음 h시간, h = 24 주, 12 보조): d = 직전 24시간 수익 부호, σ_h = 최근 60일 h시간 수익 표준편차(시점까지).
  추세(0): d × 다음 수익 ≥ +0.5σ_h / 반전(1): ≤ −0.5σ_h / 횡보(2): 그 사이
특징: R68 4시간 91개(G1~G9) + 24시간 요약(수익/σ, 효율비 4시간·1시간, 1시간 경로 24개 수익/σ, 범위 안 위치, 최대 되돌림·반등, 거래량 추세)
방법
  M1 마르코프: 직전 24시간 자신의 상태(같은 규칙으로 그 이전 24시간 대비 분류) → 학습 빈도
  M2 규칙: 효율비(24시간) 3분위 × |직전 수익/σ| 3분위 칸의 학습 빈도
  M3 다항 로지스틱(표준화, 결측 중앙값, C=0.1)
  M4 HistGB 다중분류(시드 3 평균)
  M5 kNN(k=100): 1시간 경로 24개(σ 정규화)의 유사 과거
검증: 연도별 확장창 워크포워드(평가 2022~2026). 지표: 정확도·로그손실·매크로 F1, 기준선(학습 빈도) 대비.
확신 구간: 각 클래스 예측확률 상위 10%에서 실제 비율.
매매 환산(참고): 추세 예측 → d 방향 24시간 보유, 반전 예측 → 반대, 횡보 → 쉼. 확신 상위 20%만. 비용 왕복 14bp.
  겹침 없는 버전 = 매일 00:00 UTC 시점만.
사용: python research/r76_regime_next24.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import log_loss, f1_score
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
from r68_bar_patterns import build, COST
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
CN = ["추세", "반전", "횡보"]


def extra(sym):
    k = store.load("klines_1m", sym).set_index("ts")
    hb = k.resample("1h").agg({"high": "max", "low": "min", "close": "last", "quote_volume": "sum"}).dropna()
    lh = np.log(hb.close); r1 = lh.diff()
    E = pd.DataFrame(index=hb.index + pd.Timedelta("1h"))            # 1시간봉 종가 시각
    E["r24"] = (lh - lh.shift(24)).values
    E["er1h"] = (np.abs(lh - lh.shift(24)) / r1.abs().rolling(24).sum()).values
    for j in range(24): E[f"H_{j}"] = r1.shift(j).values
    hi = hb.high.rolling(24).max(); lo = hb.low.rolling(24).min()
    E["pos24"] = ((hb.close - lo) / (hi - lo)).values
    E["dd24"] = np.log(hb.close / hb.close.rolling(24).max()).values; E["ru24"] = np.log(hb.close / hb.close.rolling(24).min()).values
    v = hb.quote_volume; E["vtr"] = np.log(v.rolling(6).sum() / v.shift(6).rolling(18).sum() * 3).values
    for h in (12, 24): E[f"fut{h}"] = (lh.shift(-h) - lh).values
    return E


def label(D, h):
    sg = D.sig24 * np.sqrt(h / 24)                                    # 최근 60일 24시간 수익 표준편차(시점까지), 12시간은 √ 스케일
    d = np.sign(D.r24); z = d * D[f"fut{h}"] / sg
    y = np.where(z >= 0.5, 0, np.where(z <= -0.5, 1, 2)); return np.where(np.isnan(z) | (d == 0), -1, y), sg


if __name__ == "__main__":
    closes = {s: store.load("klines_1m", s).set_index("ts").close.resample("4h").last() for s in ("BTCUSDT", "ETHUSDT")}
    parts = []
    for s in ("BTCUSDT", "ETHUSDT"):
        B = build(s, "4h", 6, closes["ETHUSDT" if s == "BTCUSDT" else "BTCUSDT"])
        E = extra(s); B = B.join(E, on="t")
        parts.append(B.assign(sym=s))
    D = pd.concat(parts, ignore_index=True).replace([np.inf, -np.inf], np.nan).sort_values(["sym", "t"]).reset_index(drop=True)
    D["yr"] = D.t.dt.year
    D["sig24"] = D.groupby("sym").r24.transform(lambda x: x.rolling(360, min_periods=120).std())
    D["z24"] = D.r24 / D.sig24; D["az24"] = D.z24.abs()
    for j in range(24): D[f"H_{j}"] = D[f"H_{j}"] / (D.sig24 / np.sqrt(24))
    prevz = D.groupby("sym").z24.shift(6)                             # 그 이전 24시간 대비 직전 24시간 상태(마르코프용)
    D["prevstate"] = np.where(np.sign(prevz) * D.z24 >= 0.5, 0, np.where(np.sign(prevz) * D.z24 <= -0.5, 1, 2))
    FE = [c for c in D.columns if (c[:2] in ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9") and c[2] == "_")] + \
         ["z24", "az24", "er1h", "pos24", "dd24", "ru24", "vtr"] + [f"H_{j}" for j in range(24)]
    HCOL = [f"H_{j}" for j in range(24)]
    for h in (24, 12):
        y, s = label(D, h); D["y"] = y
        base = D[D.y >= 0]
        P(f"\n===== 목표: 다음 {h}시간 (라벨 분포 추세/반전/횡보 = " + "/".join(f"{(base.y == c).mean():.3f}" for c in range(3)) + ")")
        allrows = []; conf = {m: [] for m in ("M3", "M4", "M5")}; trades = {m: [] for m in ("M1", "M2", "M3", "M4", "M5")}
        for Y in range(2022, 2027):
            tr = D[(D.t < pd.Timestamp(f"{Y}-01-01", tz="UTC") - pd.Timedelta(hours=h)) & (D.y >= 0)]; te = D[(D.yr == Y) & (D.y >= 0)].copy()
            prior = np.bincount(tr.y, minlength=3) / len(tr); PR = {"기준(빈도)": np.tile(prior, (len(te), 1))}
            # M1
            m1 = tr.groupby("prevstate").y.value_counts(normalize=True).unstack().reindex(columns=range(3)).fillna(1e-3)
            PR["M1 마르코프"] = m1.reindex(te.prevstate).values
            # M2
            q_er = np.nanquantile(tr.er1h, [1 / 3, 2 / 3]); q_z = np.nanquantile(tr.az24, [1 / 3, 2 / 3])
            cell = lambda d: np.digitize(d.er1h.fillna(0.5), q_er) * 3 + np.digitize(d.az24.fillna(0), q_z)
            m2 = pd.DataFrame({"c": cell(tr), "y": tr.y}).groupby("c").y.value_counts(normalize=True).unstack().reindex(columns=range(3)).fillna(1e-3)
            PR["M2 규칙(효율비×강도)"] = m2.reindex(cell(te)).fillna(pd.Series(prior)).values
            # M3
            med = tr[FE].median(); mu = tr[FE].fillna(med).mean(); sd = tr[FE].fillna(med).std().replace(0, 1)
            Z = lambda d: ((d[FE].fillna(med) - mu) / sd).clip(-6, 6).fillna(0).values
            PR["M3 로지스틱"] = LogisticRegression(C=0.1, max_iter=500).fit(Z(tr), tr.y).predict_proba(Z(te))
            # M4
            PR["M4 부스팅"] = np.mean([HistGradientBoostingClassifier(max_iter=300, learning_rate=0.04, max_leaf_nodes=31, min_samples_leaf=200,
                                                                      l2_regularization=1.0, random_state=sd_).fit(tr[FE].values, tr.y).predict_proba(te[FE].values)
                                       for sd_ in range(3)], axis=0)
            # M5
            A = np.nan_to_num(tr[HCOL].values); nn = NearestNeighbors(n_neighbors=100).fit(A)
            _, ix = nn.kneighbors(np.nan_to_num(te[HCOL].values)); yt = tr.y.values
            PR["M5 경로모양 kNN"] = np.stack([(yt[ix] == c).mean(1) for c in range(3)], 1) * 0.98 + 0.02 / 3
            for nm, p in PR.items():
                p = np.clip(p, 1e-4, 1); p = p / p.sum(1, keepdims=True)
                pred = p.argmax(1)
                allrows.append((nm, Y, (pred == te.y.values).mean(), log_loss(te.y, p, labels=[0, 1, 2]), f1_score(te.y, pred, average="macro"), len(te)))
                key = nm.split()[0]
                if key in conf:
                    for c in range(3):
                        top = p[:, c] >= np.quantile(p[:, c], 0.9)
                        conf[key].append((c, Y, (te.y.values[top] == c).mean(), (te.y.values == c).mean()))
                if key in trades:
                    # 확신: 최대 확률의 학습 대비 상위 20% (평가 연도 안 분위 — 문턱은 해당 연도 예측 분포)
                    mx = p.max(1); sel = (mx >= np.quantile(mx, 0.8)) & (pred != 2)
                    d = np.sign(te.z24.values); side = np.where(pred == 0, d, -d)
                    r = side[sel] * te[f"fut{h}"].values[sel] - COST
                    midnight = (te.t.dt.hour.values == 0)[sel]
                    trades[key].append(pd.DataFrame({"yr": Y, "r": r, "mid": midnight}))
        Rw = pd.DataFrame(allrows, columns=["방법", "yr", "acc", "ll", "f1", "n"])
        S = Rw.groupby("방법", sort=False).apply(lambda g: pd.Series({"정확도": np.average(g.acc, weights=g.n), "로그손실": np.average(g.ll, weights=g.n),
                                                                      "매크로F1": np.average(g.f1, weights=g.n),
                                                                      "연도별 정확도": " ".join(f"{int(y)%100}:{a:.3f}" for y, a in zip(g.yr, g.acc))}))
        P(S.round(4).to_string())
        P("  확신 상위 10%에서 실제 비율 (기저) — 평가 5년 평균")
        for key, rows in conf.items():
            C = pd.DataFrame(rows, columns=["c", "yr", "hit", "base"]).groupby("c")[["hit", "base"]].mean()
            P(f"    {key}: " + " | ".join(f"{CN[c]} {C.loc[c,'hit']:.3f} ({C.loc[c,'base']:.3f})" for c in range(3)))
        P("  매매 환산(확신 상위 20%, 추세→추종·반전→역행·횡보→쉼, 비용 14bp): 전체 시점 / 매일 00시만")
        for key, rows in trades.items():
            T = pd.concat(rows)
            yr = T.groupby("yr").r.mean() * 1e4; Tm = T[T.mid]
            P(f"    {key}: {T.r.mean()*1e4:+6.1f}bp (적중 {(T.r + COST > 0).mean():.3f}, n={len(T)}) | 00시만 {Tm.r.mean()*1e4:+6.1f}bp (n={len(Tm)}) | " +
              " ".join(f"{int(k)%100}:{v:+.0f}" for k, v in yr.items()))
    (R / "logs" / "r76.summary").write_text("R76 다음 24시간 추세/반전/횡보 예측 완료. 세부 logs/r76.log\n")
