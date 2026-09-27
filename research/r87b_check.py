"""R87b — R87 통과 8개 점검: (1) 상승장 베타 제거(같은 보유 기간 무작위 롱 대비 초과수익), (2) z0.31 중심 덧얹기 결합.

(1) 각 거래의 기준 = 같은 종목·같은 보유 시간의 '매일 모든 시각 롱' 평균 수익(연도별). 초과 = 거래 수익 − 방향 × 기준.
(2) 결합: z0.31(w1N4 일간) + k × 바구니(통과 전략 위험 균등, z0.31과 같은 변동성으로 맞춤), k ∈ {0, 0.25, 0.5, 0.75, 1}.
    바구니 구성·비중은 2020~2023 데이터로만, 평가 2024~26 표본 밖 샤프·레버리지별 월 복리·MDD.
사용: python research/r87b_check.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
import r87_new80 as M
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
PASS = ("V03", "V04", "K04", "O01", "O05", "S01", "S03", "H05")

if __name__ == "__main__":
    D = M.Data(); S = M.strategies(D)
    IDX = pd.date_range("2020-01-01", "2026-09-25", freq="D", tz="UTC")
    base = {}
    for s in M.SYMS:
        c = D.H[s].close
        for h in (24, 72, 168):
            f = np.log(c.shift(-h) / c); base[(s, h)] = f.groupby(f.index.year).mean()
    daily = {}
    P("===== (1) 상승장 베타 제거: 같은 보유 시간 무작위 롱 대비 초과수익 (bp)")
    for name, fn in S.items():
        code = name.split()[0]
        if code not in PASS: continue
        rows = []
        for s in M.SYMS:
            E = fn(s)
            if E is None or len(E) == 0: continue
            T = M.run(D, s, E); T["t"] = pd.to_datetime(T.t, utc=True)
            hold = int(np.median(E.hold.values) // 60); hb = min((24, 72, 168), key=lambda x: abs(x - hold))
            sd = E.sort_values("i").side.values[:len(T)] if len(E) >= len(T) else np.sign(T.r)
            T["side"] = pd.Series(E.set_index("i").side).reindex(D.K[s].index.searchsorted(T.t)).values if False else np.nan
            T["bench"] = base[(s, hb)].reindex(T.t.dt.year).values
            rows.append(T.assign(hb=hb))
        T = pd.concat(rows)
        sgn = -1 if any(k in name for k in ("숏",)) and "양방향" not in name else 1
        T["ex"] = T.r - sgn * T.bench.fillna(0) if "양방향" not in name else T.r
        daily[code] = (T.groupby(T.t.dt.floor("D")).r.sum() * 0.5).reindex(IDX).fillna(0)
        yr = T.groupby(T.t.dt.year).agg(r=("r", "mean"), ex=("ex", "mean"))
        P(f"  {name[:48]:48s} 원 {T.r.mean()*1e4:+7.1f} → 초과 {T.ex.mean()*1e4:+7.1f} (t {T.ex.mean()/(T.ex.std()/np.sqrt(len(T))):.2f}) | 연도별 초과 " +
          " ".join(f"{y%100}:{v*1e4:+.0f}" for y, v in yr.ex.items()))
    import r40_regime_leverage as RL, r80_max_return as R80
    RL.MMR = 0.0065; W = R80.dip_frame(0.0025)
    mult = np.minimum(np.clip(W.vreg.values ** 1.5, 0.3, 4.0) * np.clip(W.pred.values / 0.001, 0.5, 3.0), 6.0)
    dip = R80.daily_from_eq(RL.sim(W, 1, 4, mult, X=0.03, H=720)[0]).reindex(IDX).fillna(0)
    Mx = pd.DataFrame(daily); tr = Mx[:"2023-12-31"]; te_dip = dip["2024-01-01":]
    w = 1 / tr.std(); w = w / w.sum(); bask = (Mx * w).sum(axis=1)
    scale = dip["2022-01-01":"2023-12-31"].std() / bask[:"2023-12-31"].std(); bask = bask * scale
    P(f"\n===== (2) z0.31 + k × 바구니(8개 위험 균등, 2020~23 비중·변동성 맞춤) — 평가 2024~26 표본 밖")
    P(f"  바구니 단독 2024~26 샤프 {bask['2024-01-01':].mean()/bask['2024-01-01':].std()*np.sqrt(365):.2f}, z0.31 상관 {np.corrcoef(bask['2024-01-01':], te_dip)[0,1]:+.2f}")
    for k in (0, 0.25, 0.5, 0.75, 1.0):
        r = (dip + k * bask)["2024-01-01":]; sh = r.mean() / r.std() * np.sqrt(365)
        lv = []
        for L in (1, 2, 3):
            eq = (1 + (r * L).clip(lower=-0.99)).cumprod(); yrs = (eq.index[-1] - eq.index[0]).days / 365.25
            lv.append(f"×{L} 월 {(eq.iloc[-1]**(1/12/yrs)-1)*100:+.1f}% MDD {(eq/eq.cummax()-1).min()*100:.0f}%")
        P(f"  k={k:.2f}: 샤프 {sh:.2f} | " + " | ".join(lv))
