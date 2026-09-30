"""R112 — 딥 비상 손절: 평소엔 닿지 않는 먼 손절(진입가 −s)로 '한 건 초대형 딥 + 급락 지속' 파산을 막고 딥 총명목 한도를 올린다.

배경: R107~R111 — 파산은 고변동기(V3 배율 최대)에 딥 1~2건이 곧바로 12배 명목이 된 상태에서 8% 추가 하락. 손절이 없어 계좌가 청산된다.
R37의 가까운 손절(0.8~3%)은 딥 엣지를 죽였다(꼬리 되돌림이 엣지). 여기서는 s ∈ {4, 5, 6, 8, 10%}의 먼 손절만.
사용자 조건(2026-09-30): 낙폭 무관, 완전 파산만 피한다 → 비상 손절로 큰 손실(예: 16배 × 5% = −80%)을 감수하는 대신 파산을 막는 구조가 조건에 맞는가.
손절 체결: 분 저가(체결가)가 T × (1 − s) 이하가 된 첫 분에 T × (1 − s) × (1 − slip)으로 청산(스톱 시장가, 급락 중 슬리피지 slip ∈ {0.5%, 1%}), 테이커 수수료.
엔진: R109 sim4(정확 시가평가, 마크 청산). 딥 w ∈ {3, 4, 5}, N ∈ {12, 16, 20, 24, 32}, 돌파 L ∈ {4, 8}. 선택 2022~23 → 평가 2024~26.
사용: python research/r112_disaster_stop.py"""
import sys, warnings, time, itertools
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store, costs
import r102_joint_account as J, r103_breakout_cost as K, r80_max_return as R80, r107_max_growth as G7, r109_dip_exact_mtm as M
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)


@numba.njit(cache=True)
def first_hit(l, f, x, lvl):
    out = np.full(len(f), -1)
    for i in range(len(f)):
        for j in range(f[i], x[i] + 1):
            if l[j] <= lvl[i]: out[i] = j; break
    return out


def with_stop(D, W, s, slip):
    D = D.copy(); hitm = np.full(len(D), -1)
    for sym in ("BTCUSDT", "ETHUSDT"):
        k = store.load("klines_1m", sym); m = (W.sym == sym).values
        f = W.f.values[m].astype(np.int64); x = W.x.values[m].astype(np.int64)
        hitm[m] = first_hit(k.low.values, f, x, D.px.values[m] * (1 - s))
    h = hitm >= 0
    D.loc[h, "ret"] = -s - slip * (1 - s) - costs.MAKER - costs.TAKER - 1.2e-4
    D.loc[h, "x"] = D.e.values[h] + np.maximum(hitm[h] - W.f.values[h], 1)          # 손절 분 시작에 청산(그 분의 더 깊은 저가는 이 포지션에 반영 안 함)
    D.loc[h, "mae"] = np.minimum(D.mae.values[h], s + slip)
    return D, h.mean()


if __name__ == "__main__":
    t_ = time.time()
    D0_, _, _, W = M.dip_with_px()
    T8 = J.add_funding(J.breakout_trades(J.TOP8)); T8["lf"] = K.last_funding(T8)
    BK = K.events(T8, False, fmax=1e-4).drop(columns="lf")
    start = int(J.mins([pd.Timestamp("2022-01-01", tz="UTC")])[0]); Gd = G7.grid(start, "mark_1m")
    rows = []
    for s, slip in [(None, 0)] + [(s, sl) for s in (0.04, 0.05, 0.06, 0.08, 0.10) for sl in (0.005, 0.01)]:
        if s is None: D, hr = D0_, 0.0
        else: D, hr = with_stop(D0_, W, s, slip)
        yr = pd.DatetimeIndex(W.ts).year
        P(f"손절 {s} 슬리피지 {slip}: 발동 {hr*100:.2f}% · 거래당 학습(22~23) {D.ret[yr < 2024].mean()*1e4:+.1f}bp 평가 {D.ret[yr >= 2024].mean()*1e4:+.1f}bp")
        E = pd.concat([D, BK[BK.e >= start]], ignore_index=True).sort_values(["e", "k"]).reset_index(drop=True)
        for w, N, L in itertools.product((3, 4, 5), (12, 16, 20, 24, 32), (4, 8)):
            d, pk = M.run(E, Gd, w, N, L)
            if d is None:
                rows.append(dict(s=s, slip=slip, w=w, N=N, L=L, ok=False, 파산=f"{J.D0 + pd.Timedelta(minutes=int(pk[0])):%Y-%m-%d}")); continue
            a = R80.metrics(d, "2022-01-01", "2024-01-01"); b = R80.metrics(d, "2024-01-01", "2026-09-23"); sr = d["2024":"2026-09-22"]
            rows.append(dict(s=s, slip=slip, w=w, N=N, L=L, ok=True, 학습월=a["mo"], 학습MDD=a["mdd"], 평가월=b["mo"], 평가중앙=b["med"], 평가100달=b["big"],
                             평가MDD=b["mdd"], 평가샤프=sr.mean() / sr.std() * np.sqrt(365), 연도=b["yr"]))
        P(f"  완료 ({time.time()-t_:.0f}s)")
    O = pd.DataFrame(rows); pd.set_option("display.width", 250); pd.set_option("display.max_rows", 120)
    S = O[O.ok]
    P(f"\n생존 {len(S)}/{len(O)} · 손절별 생존·최대 N: " + " | ".join(f"{k}: {int(g.ok.sum())}/{len(g)} N≤{g[g.ok].N.max() if g.ok.any() else '-'}" for k, g in O.groupby(O.s.astype(str) + "/" + O.slip.astype(str))))
    P("\n===== 2022~23 월 복리 상위 12 → 평가 (사전 선택)"); P(S.sort_values("학습월", ascending=False).head(12).drop(columns=["ok", "파산"], errors="ignore").round(1).to_string(index=False))
    P("\n===== 손절 없음 vs 손절별 학습 최선")
    for k, g in S.groupby(S.s.astype(str) + "/" + S.slip.astype(str)):
        b = g.sort_values("학습월", ascending=False).iloc[0]
        P(f"  {k:12s} w{b.w} N{b.N} L{b.L} 학습 {b.학습월:+.1f}% → 평가 월 {b.평가월:+.1f}% 중앙 {b.평가중앙:+.1f}% ≥100% {b.평가100달}달 MDD {b.평가MDD:.0f}% | {b.연도}")
    O.to_parquet(R / "data" / "r112_grid.parquet", index=False)
    P(f"완료 ({time.time()-t_:.0f}s)")
