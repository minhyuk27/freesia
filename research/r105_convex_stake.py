"""R105 — 볼록 구조(고정 판돈 + 매월 보충): 계좌 일부 f만 판돈으로 떼어 초고배율로 운용하고, 매월 초 판돈을 총자산 × f로 다시 맞춘다.

근거: 한 달 손실이 f로 제한되므로 계좌 전체는 파산하지 않는다. 판돈 안에서는 켈리를 넘는 배율도 허용(판돈 청산 = −f).
전략: z0.31 딥 + z0.33 TOP8(펀딩≤1bp) 공유 계좌(R102 sim_mtm, 분 시가평가 청산). 배율 λ: w = λ, N = 4λ, L33 = λ (R102 최선 비율).
매월: 그 달에 진입하는 사건만으로 자본 1에서 시작(달을 넘는 포지션은 끝까지 보유해 그 달 성과로), 청산되면 판돈 수익 −100%.
총 계좌 월 수익 = f × 판돈 월 수익. f ∈ {0.1, 0.2, 0.3, 0.5, 1.0}, λ ∈ {1, 2, 3, 4, 6, 8, 12}.
보고(2022~23 / 평가 2024~26): 월 복리, 월 중앙, 월 ≥ +100% 비율, 최악 달, MDD, 판돈 청산 달 수. 연속 운용(R102) 대비.
사용: python research/r105_convex_stake.py"""
import sys, warnings, time
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
import r102_joint_account as J, r103_breakout_cost as K
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)


def month_returns(E, lam, LO, CL, t0):
    ts = J.D0 + pd.to_timedelta(E.e, unit="min"); mon = ts.dt.to_period("M")
    out = {}
    for m, g in E.groupby(mon.values):
        g = g.sort_values(["e", "k"])
        day, pk = J.sim_mtm(g.e.values.astype(np.int64), g.x.values.astype(np.int64), g.ret.values, g.mae.values, g.k.values.astype(np.int64),
                            g.mult.values, g.s.values.astype(np.int64), g.px.values, LO, CL, t0, float(lam), float(4 * lam), float(lam), 0.03, 720, 120, J.MMR)
        if pk < 0: out[m] = -1.0; continue
        v = day[~np.isnan(day)]; out[m] = v[-1] - 1 if len(v) else 0.0
    return pd.Series(out).sort_index()


def summ(r):
    eq = (1 + r).cumprod(); n = len(r)
    return dict(mo=(eq.iloc[-1] ** (1 / n) - 1) * 100 if eq.iloc[-1] > 0 else -100, med=r.median() * 100, big=(r >= 1).mean() * 100,
                worst=r.min() * 100, mdd=(eq / eq.cummax() - 1).min() * 100, fin=eq.iloc[-1])


if __name__ == "__main__":
    t_ = time.time()
    DIP = J.dip_events()
    T = J.add_funding(J.breakout_trades(J.TOP8)); T["lf"] = K.last_funding(T)
    BK = K.events(T, False, fmax=1e-4).drop(columns="lf")
    start = J.mins([pd.Timestamp("2022-01-01", tz="UTC")])[0]
    E = pd.concat([DIP, BK[BK.e >= start]], ignore_index=True)
    E = E[E.e < J.mins([pd.Timestamp("2026-09-01", tz="UTC")])[0]].reset_index(drop=True)        # 완결 월만
    LO, CL = J.grid(int(start))
    MR = {}
    for lam in (1, 2, 3, 4, 6, 8, 12):
        MR[lam] = month_returns(E, lam, LO, CL, int(start))
        r = MR[lam]; P(f"λ{lam:>2}: 판돈 월 중앙 {r.median()*100:+6.1f}% 평균 {r.mean()*100:+6.1f}% ≥+100% {(r>=1).sum():2d}달 청산 {(r<=-1).sum():2d}달 / {len(r)} ({time.time()-t_:.0f}s)")
    rows = []
    for lam, r in MR.items():
        for f in (0.1, 0.2, 0.3, 0.5, 1.0):
            tot = (f * r).clip(lower=-1)
            for nm, a, b in (("22~23", "2022-01", "2023-12"), ("평가 24~26", "2024-01", "2026-08")):
                x = tot[(tot.index >= pd.Period(a)) & (tot.index <= pd.Period(b))]
                rows.append(dict(λ=lam, f=f, 구간=nm, **summ(x)))
    O = pd.DataFrame(rows); pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
    for nm in ("22~23", "평가 24~26"):
        P(f"\n===== {nm}: 총 계좌 (월 복리 · 월 중앙 · 월 ≥+100% 비율 · 최악 달 · MDD · 최종 배수)")
        P(O[O.구간 == nm].drop(columns="구간").round(1).to_string(index=False))
    P("\n평가 구간 월 복리 최대 (f<1, 파산 불가 구조):"); P(O[(O.구간 == "평가 24~26") & (O.f < 1)].sort_values("mo", ascending=False).head(5).round(1).to_string(index=False))
    P("평가 구간 월 ≥+100% 비율 최대 (f<1):"); P(O[(O.구간 == "평가 24~26") & (O.f < 1)].sort_values(["big", "mo"], ascending=False).head(5).round(1).to_string(index=False))
    O.to_parquet(R / "data" / "r105_convex.parquet", index=False)
    P(f"완료 ({time.time()-t_:.0f}s)")
