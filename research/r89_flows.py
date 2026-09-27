"""R89 — '자금 유입' 계열 심화: 스테이블코인·온체인 거래소 흐름으로 연속 포지션/이벤트, 2020~23 선택 → 2024~26 검증 → 덧얹기.

신호(일간, 값 z = 직전 365일 대비, 날짜 D 값은 D+lag일 00:00 UTC부터 사용, lag ∈ {1, 2})
  ST7 스테이블 전체 7일 증가 · USDT7 · USDC7 · ST30 스테이블 30일 증가
  NFB BTC 거래소 순유출(−(유입−유출) 7일 합 / 거래소 보유) · NFE ETH 동일 · XS30 거래소 보유 30일 감소(−)
  MVRV 저평가(−MVRV) · ADR 활성 주소 30일 증가 · CMB = ST7 + NF(종목별) 평균 · DIV = ST7 − 30일 가격 모멘텀 z
포지션: LO 롱 전용 p = clip(z,0,2)/2 · LS 롱숏 p = clip(z,−2,2)/2 · EV 상위 20% 진입 후 7일 보유(p=1)
손익: 일간 p_t × r_(t+1) − |Δp| × 7bp. 종목 BTC·ETH 각각(ETH 흐름은 ETH, BTC 흐름은 BTC에, 스테이블은 둘 다).
판정(2020~23만): LS 샤프 ≥ 0.7 이고 LO 샤프가 같은 기간 노출 맞춘 보유(평균 p × 보유)보다 높음 → 2024~26 확인.
결합: z0.31(w1N4) + R88 바구니 + 선택 흐름 바구니, 평가 2024~26.
사용: python research/r89_flows.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
EXT = R / "data" / "ext"; CT = 7e-4
IDX = pd.date_range("2020-01-01", "2026-09-18", freq="D", tz="UTC")


def z365(x): return (x - x.rolling(365, min_periods=180).mean()) / x.rolling(365, min_periods=180).std()


def tz(df):
    df.index = pd.DatetimeIndex(df.index); df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
    return df


def signals():
    st = tz(pd.read_parquet(EXT / "stablecoins.parquet")); cb = tz(pd.read_parquet(EXT / "cm_btc.parquet")); ce = tz(pd.read_parquet(EXT / "cm_eth.parquet"))
    S = {}
    S["ST7"] = z365(np.log(st.stable_total).diff(7)); S["USDT7"] = z365(np.log(st.usdt).diff(7)); S["USDC7"] = z365(np.log(st.usdc).diff(7))
    S["ST30"] = z365(np.log(st.stable_total).diff(30))
    for nm, c in (("B", cb), ("E", ce)):
        nf = -(c.FlowInExUSD - c.FlowOutExUSD).rolling(7).sum() / c.SplyExUSD
        S[f"NF{nm}"] = z365(nf); S[f"XS30{nm}"] = z365(-np.log(c.SplyExUSD).diff(30))
        S[f"MVRV{nm}"] = z365(-c.CapMVRVCur); S[f"ADR{nm}"] = z365(np.log(c.AdrActCnt.rolling(7).mean()).diff(30))
    return {k: v.reindex(pd.date_range("2018-01-01", "2026-09-18", freq="D", tz="UTC")) for k, v in S.items()}


def main():
    S = signals()
    px = {s: store.load("klines_1m", s).set_index("ts").close.resample("1D").last() for s in ("BTCUSDT", "ETHUSDT")}
    ret = {s: np.log(px[s]).diff().shift(-1).reindex(IDX) for s in px}          # 날짜 t 포지션 → t~t+1 수익
    mom = {s: z365(np.log(px[s]).diff(30)).reindex(S["ST7"].index) for s in px}
    rows = []; sleeves = {}
    for lag in (1, 2):
        for s in ("BTCUSDT", "ETHUSDT"):
            tag = "B" if s == "BTCUSDT" else "E"
            sigs = {k: S[k] for k in ("ST7", "USDT7", "USDC7", "ST30")}
            sigs.update({"NF": S[f"NF{tag}"], "XS30": S[f"XS30{tag}"], "MVRV": S[f"MVRV{tag}"], "ADR": S[f"ADR{tag}"]})
            sigs["CMB"] = (S["ST7"] + S[f"NF{tag}"]) / 2; sigs["DIV"] = S["ST7"] - mom[s]
            for nm, z in sigs.items():
                zz = z.shift(lag).reindex(IDX)
                for mode in ("LO", "LS", "EV"):
                    if mode == "LO": p = zz.clip(0, 2) / 2
                    elif mode == "LS": p = zz.clip(-2, 2) / 2
                    else:
                        rk = z.rolling(365, min_periods=180).rank(pct=True).shift(lag).reindex(IDX); on = (rk >= 0.8) & ~(rk.shift(1) >= 0.8)
                        p = pd.Series(0.0, index=IDX); cnt = 0
                        for i, v in enumerate(on.fillna(False).values):
                            if v: cnt = 7
                            if cnt > 0: p.iloc[i] = 1.0; cnt -= 1
                    p = p.fillna(0); pnl = p * ret[s] - p.diff().abs().fillna(0) * CT
                    key = (lag, s, nm, mode); sleeves[key] = pnl.fillna(0)
                    for per, a, b in (("학습", "2020-01-01", "2024-01-01"), ("평가", "2024-01-01", "2027-01-01")):
                        x = pnl[a:b].dropna(); pp = p[a:b]; bh = (pp.mean() * ret[s][a:b]).dropna()
                        sh = x.mean() / x.std() * np.sqrt(365) if x.std() > 0 else np.nan
                        shb = bh.mean() / bh.std() * np.sqrt(365) if bh.std() > 0 else np.nan
                        rows.append((lag, s, nm, mode, per, sh, shb, x.mean() * 365 * 100, pp.mean()))
    O = pd.DataFrame(rows, columns=["지연", "종목", "신호", "방식", "구간", "샤프", "노출맞춘보유샤프", "연수익%", "평균노출"])
    T = O.pivot_table(index=["지연", "종목", "신호", "방식"], columns="구간", values=["샤프", "노출맞춘보유샤프", "연수익%"]).round(2)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
    P("===== 흐름 신호 전부 (구간별 샤프 / 노출 맞춘 보유 샤프 / 연수익%)"); P(T.to_string())
    # 선택: lag 1, 2020~23 기준 LS 샤프 ≥ 0.7 인 신호의 LO(초과) 또는 EV
    tr = O[(O.구간 == "학습") & (O.지연 == 1)].set_index(["종목", "신호", "방식"])
    good = [(s, n) for (s, n, m), r in tr.iterrows() if m == "LS" and r.샤프 >= 0.7]
    P(f"\n===== 학습(2020~23) LS 샤프 ≥ 0.7 신호: {good}")
    chosen = []
    for s, n in good:
        lo = tr.loc[(s, n, "LO")]
        if lo.샤프 > lo.노출맞춘보유샤프: chosen.append((1, s, n, "LO"))
        chosen.append((1, s, n, "LS"))
    te = O[(O.구간 == "평가")].set_index(["지연", "종목", "신호", "방식"])
    for k in chosen: P(f"  선택 {k}: 평가(2024~26) 샤프 {te.loc[k].샤프:.2f} (노출 맞춘 보유 {te.loc[k].노출맞춘보유샤프:.2f}), 지연 2일 {te.loc[(2,)+k[1:]].샤프:.2f}")
    if not chosen: return
    import r40_regime_leverage as RL, r80_max_return as R80
    RL.MMR = 0.0065; W = R80.dip_frame(0.0025)
    mult = np.minimum(np.clip(W.vreg.values ** 1.5, 0.3, 4.0) * np.clip(W.pred.values / 0.001, 0.5, 3.0), 6.0)
    dip = R80.daily_from_eq(RL.sim(W, 1, 4, mult, X=0.03, H=720)[0]); dip.index = dip.index.normalize(); dip = dip.reindex(IDX).fillna(0)
    Mx = pd.DataFrame({str(k): sleeves[k] for k in chosen if k[3] == "LS"})
    trm = Mx[:"2023-12-31"]; w = 1 / trm.std(); w = w / w.sum(); fb = (Mx * w).sum(axis=1)
    fb = fb * dip["2022-01-01":"2023-12-31"].std() / fb["2022-01-01":"2023-12-31"].std()
    dt, ft = dip["2024-01-01":], fb["2024-01-01":]
    P(f"\n===== 덧얹기(평가 2024~26): 흐름 LS 바구니 단독 샤프 {ft.mean()/ft.std()*np.sqrt(365):.2f}, z0.31 상관 {np.corrcoef(dt, ft)[0,1]:+.2f}")
    for k in (0, 0.25, 0.5, 0.75, 1.0):
        r = dt + k * ft; sh = r.mean() / r.std() * np.sqrt(365); lv = []
        for L in (1, 3):
            eq = (1 + (r * L).clip(lower=-0.99)).cumprod(); yrs = (eq.index[-1] - eq.index[0]).days / 365.25
            lv.append(f"×{L} 월 {(eq.iloc[-1]**(1/12/yrs)-1)*100:+.1f}% MDD {(eq/eq.cummax()-1).min()*100:.0f}%")
        P(f"  z0.31 + {k:.2f}×흐름: 샤프 {sh:.2f} | " + " | ".join(lv))
    (R / "logs" / "r89.summary").write_text("R89 자금 흐름 심화 완료. 세부 logs/r89.log\n")


if __name__ == "__main__":
    main()
