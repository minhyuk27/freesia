"""R64 — 선 사건을 실제 주문으로: 선배치 지정가(반등·스윕)와 스톱(돌파). 호가·체결 모델로 거르기. 실제 선 vs 위약 선.

사건: R63 표(data/r63_events_feat.parquet, 2023~). 주문은 접근 순간 τ에 건다(τ 이전 정보만).
좌표: 방향 맞춘 로그가격 y = ori·log(가격). 모든 규칙을 'y 위쪽이 이익'인 롱 엔진 하나로 돌린다.
  지지 반등 계열 ori=+1, 저항 반등 계열 ori=−1. 돌파(E2)는 반대 방향이라 ori를 한 번 더 뒤집는다.
규칙 (yL = 선, A = 선 시간대 ATR(로그), δ = 띠)
  E1 반등: 지정가 @yL (저가가 5bp 더 관통해야 체결), 손절 yL−0.5A, 목표 yL+xA, x ∈ {0.5, 1.0}
  E3 스윕: 지정가 @yL−0.5A, 손절 yL−1.25A, 목표 yL+xA, x ∈ {0, 0.5}
  E2 돌파: 스톱 @(돌파 방향) 선+δ, 손절 선 안쪽 0.25A, 목표 선+xA (돌파 방향), x ∈ {1.0, 2.0}
  대기 = 선 시간대 1봉, 보유 제한 = 12봉. 진입 분에 손절가가 닿으면 손절(보수), 목표는 다음 분부터.
비용: 지정가 진입·목표 메이커 2bp, 스톱 진입·손절·시간청산 테이커 5+2bp.
필터: 맥락+호가·체결 모델(HistGB 시드 5)을 2023~24 학습 → 2025~26 점수. E1·E3는 반등확률 상위 30%, E2는 하위 30%(시간대별 순위).
판정: 2025~26 거래당 순수익(bp)·표준오차·승률·하루 건수·연도별. 실제 선 − 위약 선 = 선의 가치.
사용: python research/r64_line_trades.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
from sklearn.ensemble import HistGradientBoostingClassifier
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store, costs
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
from r63_line_book import FEAT, CTX, SPLIT
CM = costs.MAKER; CT = costs.TAKER + 2e-4; PEN = 5e-4
TFL = {"15min": 15, "1h": 60, "4h": 240}


@numba.njit(cache=True)
def sim(t0, wait, hold, ori, ye, ys, yt, stop_entry, lh, ll, lc, CM, CT, PEN):
    n = len(lc); out = np.full((len(t0), 2), np.nan)
    for e in range(len(t0)):
        s = ori[e]; fill = -1
        for i in range(t0[e], min(n, t0[e] + wait[e])):
            hi_ = lh[i] if s == 1 else -ll[i]; lo_ = ll[i] if s == 1 else -lh[i]
            if stop_entry[e]:
                if hi_ >= ye[e]: fill = i; break
            elif lo_ <= ye[e] - PEN: fill = i; break
        if fill < 0: out[e, 0] = 0; continue
        res = np.nan; j = fill
        while j < min(n, fill + hold[e]):
            hi_ = lh[j] if s == 1 else -ll[j]; lo_ = ll[j] if s == 1 else -lh[j]
            if lo_ <= ys[e]: res = ys[e] - ye[e] - CT; break
            if j > fill and hi_ >= yt[e]: res = yt[e] - ye[e] - CM; break
            j += 1
        if np.isnan(res):
            yc = lc[min(j, n - 1)] * s
            res = yc - ye[e] - CT
        out[e, 0] = 1; out[e, 1] = res - (CT if stop_entry[e] else CM)
    return out


def run(ev, M, rule, x):
    ix = M.index
    lh, ll, lc = np.log(M.high.values), np.log(M.low.values), np.log(M.close.values)
    t0 = ix.searchsorted(pd.DatetimeIndex(ev.tau))
    ori = np.where(ev.side.values == "지지", 1, -1)
    A = ev.atr.values; d = ev.d.values; yL = ori * ev.lvl.values
    wait = ev.tf.map(TFL).values; hold = 12 * wait
    if rule == "E1":
        ye, ys, yt, se = yL, yL - 0.5 * A, yL + x * A, False
    elif rule == "E3":
        ye, ys, yt, se = yL - 0.5 * A, yL - 1.25 * A, yL + x * A, False
    else:
        ori = -ori; zL = -yL                                      # 돌파 방향이 위가 되도록
        ye, ys, yt, se = zL + d, zL - 0.25 * A, zL + x * A, True
    return sim(t0, wait, hold, ori, ye, ys, yt, np.full(len(ev), se), lh, ll, lc, CM, CT, PEN)


if __name__ == "__main__":
    D = pd.read_parquet(R / "data" / "r63_events_feat.parquet")
    D["p"] = np.nan
    for plc in (False, True):
        S = D[D.plc == plc]; tr = S[S.t < SPLIT]; te = S[S.t >= SPLIT]
        p = np.mean([HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=300,
                                                     l2_regularization=1.0, random_state=sd).fit(tr[CTX + FEAT].values, tr.y.values)
                     .predict_proba(te[CTX + FEAT].values)[:, 1] for sd in range(5)], axis=0)
        D.loc[te.index, "p"] = p
    T = D[D.t >= SPLIT].copy()
    T["q"] = T.groupby(["plc", "tf"]).p.rank(pct=True)
    Ms = {s: store.load("klines_1m", s).set_index("ts").loc["2024-12-20":] for s in ("BTCUSDT", "ETHUSDT")}
    days = (T.t.max() - T.t.min()).days
    rows = []
    for rule, xs in (("E1", (0.5, 1.0)), ("E3", (0.0, 0.5)), ("E2", (1.0, 2.0))):
        for x in xs:
            for plc in (False, True):
                for filt in ("전체", "필터"):
                    S = T[T.plc == plc]
                    if filt == "필터": S = S[S.q >= 0.7] if rule != "E2" else S[S.q <= 0.3]
                    res = []
                    for sym in ("BTCUSDT", "ETHUSDT"):
                        ev = S[S.sym == sym]
                        o = run(ev, Ms[sym], rule, x)
                        res.append(pd.DataFrame({"tf": ev.tf.values, "yr": ev.t.dt.year.values, "fill": o[:, 0], "r": o[:, 1]}))
                    Rr = pd.concat(res); Fd = Rr[Rr.fill == 1]
                    for tf, G in [("전체", Fd)] + list(Fd.groupby("tf")):
                        rows.append((rule, x, "위약" if plc else "실제", filt, tf, len(G) / days, G.r.mean() * 1e4,
                                     G.r.std() / np.sqrt(max(len(G), 1)) * 1e4, (G.r > 0).mean(),
                                     G[G.yr == 2025].r.mean() * 1e4, G[G.yr == 2026].r.mean() * 1e4, len(G), len(Fd) / max(len(Rr), 1)))
    O = pd.DataFrame(rows, columns=["규칙", "x", "선", "필터", "tf", "하루", "bp", "se", "승률", "y25", "y26", "n", "체결률"])
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    P(O.round(2).to_string(index=False))
    O.to_csv(R / "logs" / "r64_table.csv", index=False)
    best = O[(O.선 == "실제") & (O.n >= 100)].sort_values("bp", ascending=False).head(3)
    (R / "logs" / "r64.summary").write_text("R64 선 매매 규칙 상위(실제선, 2025~26):\n" +
                                            best[["규칙", "x", "필터", "tf", "하루", "bp", "n"]].round(1).to_string(index=False) + "\n")
