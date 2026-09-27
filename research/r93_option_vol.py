"""R93 — 변동성 예보(p, R90)로 옵션 스트래들 사고/팔기 (방향 중립 헤지 구조).

근거 문헌: 예측 변동성과 내재 변동성(IV)의 차이를 거래하는 변동성 스프레드 전략이 BTC·ETH 옵션에서 강건한 수익
(A Horserace of Volatility Models for Cryptocurrency, arXiv 2010.07402). 내재 변동성은 평균적으로 실현보다 높다(변동성 위험 프리미엄) → 기본은 매도 유리.
데이터: Deribit 옵션 전 체결(IV·방향). 바이낸스 옵션(같은 08:00 UTC 일일 만기)의 가격 대용. 기초는 바이낸스 선물 1분봉.
구조: 매일 08:00 UTC 만기 ATM 스트래들을 h시(0~7시 → 잔존 8−h시간, 그리고 08시 → 24시간)에 매수 또는 매도, 만기까지 보유(헤지 없음).
  가격: 진입 직전 60분, 해당 만기·행사가 ±1.5% 이내 체결의 IV 중앙값 — 매수는 '매수 주도 체결'(side=+1), 매도는 '매도 주도 체결'(−1) IV = 실제 체결 가능 가격. 3건 미만이면 건너뜀.
  프리미엄 ≈ 0.798·σ·√(T/8760) (기초 1당), 결제가 = 만기 전 30분 선물 종가 평균(데리빗 TWAP 대용).
  비용: 수수료 다리당 0.03%(프리미엄 12.5% 상한) ×2 + 결제 수수료 다리당 0.015%(상한 12.5%) ×2.
신호: pr = R90 p의 직전 90일 순위(인과), vr = IV / 직전 24시간 실현변동성.
규칙: 2022~23에서 (시각, 방향, pr 구간, vr 구간) 조합을 고르고 2024~26 검증.
사용: python research/r93_option_vol.py"""
import sys, warnings, glob, time
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)


def fee(prem, rate): return np.minimum(rate, 0.125 * prem)


def build(sym):
    cc = sym[:3]; K = store.load("klines_1m", f"{cc}USDT").set_index("ts")
    cl = K.close; lr = np.log(cl).diff()
    rv24 = (lr.pow(2).rolling(1440).sum() * 365).pow(0.5)          # 연율, t 포함 직전 24시간
    rows = []
    for f in sorted(glob.glob(str(R / "data/parquet/deribit_opt" / cc / "*.parquet"))):
        d = pd.read_parquet(f, columns=["ts", "expiry", "strike", "iv", "index_price", "side"])
        d = d[d.expiry.dt.hour == 8]
        d["hT"] = (d.expiry - d.ts).dt.total_seconds() / 3600
        d = d[(d.hT <= 25) & ((d.strike / d.index_price - 1).abs() < 0.015)]
        d["ent"] = d.ts.dt.ceil("h")                                  # 진입 시각(정시), 직전 60분 체결
        d["T"] = (d.expiry - d.ent).dt.total_seconds() / 3600
        d = d[d["T"].isin([1, 2, 3, 4, 5, 6, 7, 8, 24])]
        g = d.groupby(["ent", "T"])
        a = pd.DataFrame({"ivb": d[d.side > 0].groupby(["ent", "T"]).iv.median(), "nb": d[d.side > 0].groupby(["ent", "T"]).size(),
                          "ivs": d[d.side < 0].groupby(["ent", "T"]).iv.median(), "ns": d[d.side < 0].groupby(["ent", "T"]).size(),
                          "iv": g.iv.median()}).reset_index()
        rows.append(a)
    A = pd.concat(rows); A = A[(A.nb.fillna(0) >= 3) & (A.ns.fillna(0) >= 3)]
    A["exp"] = A.ent + pd.to_timedelta(A["T"], unit="h")
    s0 = cl.reindex(A.ent - pd.Timedelta(minutes=1)).values                 # 진입 직전 종가(= 진입가)
    twap = cl.rolling(30).mean(); s1 = twap.reindex(A.exp - pd.Timedelta(minutes=1)).values
    A["absr"] = np.abs(s1 / s0 - 1); A["rv24"] = rv24.reindex(A.ent - pd.Timedelta(minutes=1)).values
    A["vr"] = A.iv / 100 / A.rv24
    for side, col in ((1, "ivb"), (-1, "ivs")):
        prem = 0.798 * A[col] / 100 * np.sqrt(A["T"] / 8760)
        cost = 2 * fee(prem, 3e-4) + 2 * fee(prem, 1.5e-4)
        A["L" if side > 0 else "S"] = (A.absr - prem) * side - cost
    A["sym"] = f"{cc}USDT"
    return A.dropna(subset=["absr", "L", "S"])


if __name__ == "__main__":
    t_ = time.time()
    import r92_vol_timing as V
    Wp = V.load_p()
    out = []
    for s in ("BTCUSDT", "ETHUSDT"):
        A = build(s); w = Wp[Wp.sym == s][["ts", "pr", "p"]].sort_values("ts")
        A = A.sort_values("ent"); A["ent_naive"] = A.ent
        A = pd.merge_asof(A, w, left_on="ent", right_on="ts", direction="backward", tolerance=pd.Timedelta("15min"))
        out.append(A); P(f"  {s}: {len(A):,}건 ({time.time()-t_:.0f}s)")
    A = pd.concat(out); A["yr"] = A.ent.dt.year
    P("\n===== 무조건 (단위 bp of 기초, 거래당): 매수 L / 매도 S, 연도별 매도")
    for T, g in A.groupby("T"):
        P(f"  잔존 {T:2.0f}h: n={len(g):5d} IV중앙 {g.iv.median():.0f} 매수 {g.L.mean()*1e4:+6.1f} 매도 {g.S.mean()*1e4:+6.1f} | 매도 연도별 " +
          " ".join(f"{y%100}:{v*1e4:+.0f}" for y, v in g.groupby("yr").S.mean().items()))
    B = A.dropna(subset=["pr"]).copy()
    B["pb"] = pd.cut(B.pr, [0, .2, .5, .8, .95, 1.01], labels=["Q0", "Q1", "Q2", "Q3", "Q4"], right=False)
    B["vb"] = pd.qcut(B.vr, 3, labels=["v저", "v중", "v고"])
    P("\n===== p 구간별 (2022~26): 매수·매도 거래당 bp, |수익| vs 매수 프리미엄")
    for T in (4, 8, 24):
        g = B[B["T"] == T]
        P(f"  잔존 {T}h:")
        for pb, h in g.groupby("pb"):
            P(f"    {pb}: n={len(h):4d} 매수 {h.L.mean()*1e4:+6.1f} 매도 {h.S.mean()*1e4:+6.1f} | 학습 매수 {h[h.yr<2024].L.mean()*1e4:+6.1f} 평가 매수 {h[h.yr>=2024].L.mean()*1e4:+6.1f} 평가 매도 {h[h.yr>=2024].S.mean()*1e4:+6.1f}")
    # 규칙 선택: (T, 방향, pb, vb) 칸, 2022~23 평균 > 0 이고 n ≥ 40
    rows = []
    for (T, pb, vb), h in B.groupby(["T", "pb", "vb"]):
        tr = h[h.yr < 2024]; te = h[h.yr >= 2024]
        for side in ("L", "S"):
            if len(tr) >= 40 and len(te) >= 20:
                rows.append(dict(T=T, pb=pb, vb=vb, side=side, ntr=len(tr), trbp=tr[side].mean() * 1e4, ttr=tr[side].mean() / tr[side].std() * np.sqrt(len(tr)),
                                 nte=len(te), tebp=te[side].mean() * 1e4, tte=te[side].mean() / te[side].std() * np.sqrt(len(te))))
    O = pd.DataFrame(rows); pd.set_option("display.width", 250)
    P("\n===== 칸별 학습 t ≥ 2 인 규칙 → 평가"); sel = O[O.ttr >= 2].sort_values("ttr", ascending=False); P(sel.round(2).to_string())
    P(f"  선택 {len(sel)}칸 중 평가 양수 {(sel.tebp > 0).sum()}, 평가 평균 t {sel.tte.mean():.2f}")
    # 바구니: 선택 칸 전부, 일별 합(기초 명목 1 당), 3단 레버리지
    B["key"] = list(zip(B["T"], B.pb, B.vb))
    parts = []
    for _, r in sel.iterrows():
        h = B[(B["T"] == r["T"]) & (B.pb == r.pb) & (B.vb == r.vb)]; parts.append(pd.DataFrame({"t": h.ent, "r": h[r.side]}))
    if parts:
        T_ = pd.concat(parts); d = T_.groupby(T_.t.dt.floor("D")).r.sum()
        d = d.reindex(pd.date_range("2022-01-01", "2026-09-18", freq="D", tz="UTC")).fillna(0)
        for a, b, lab in (("2022-01-01", "2023-12-31", "학습"), ("2024-01-01", "2026-09-18", "평가")):
            x = d[a:b]; P(f"  바구니 {lab}: 샤프 {x.mean()/x.std()*np.sqrt(365):.2f}, 하루 {len(T_[(T_.t>=a)&(T_.t<=b)])/len(x):.2f}건")
            for L in (1, 3, 10):
                eq = (1 + (x * L).clip(lower=-0.99)).cumprod(); yrs = len(x) / 365.25
                P(f"    ×{L:2d}: 월 {(eq.iloc[-1]**(1/12/yrs)-1)*100:+.2f}% MDD {(eq/eq.cummax()-1).min()*100:.0f}%")
    A.to_parquet(R / "data" / "r93_straddle.parquet", index=False)
    (R / "logs" / "r93.summary").write_text("done\n")
