"""R54 — 방향 D 특징: Deribit 옵션(스큐·IV·풋콜 흐름) + 코인베이스 프리미엄 → R48 15분 격자에 인과적으로 붙임.

옵션(통화별, 체결 단위 → 15분 버킷):
  체결마다 블랙숄즈 델타 = N(d1)(콜), N(d1)−1(풋),  d1 = (ln(S/K) + σ²T/2)/(σ√T), σ = iv/100, T = 만기까지 연
  대상: 만기 1~45일
  rr25  = 25델타 콜 IV − 25델타 풋 IV  (콜 델타 0.15~0.35, 풋 −0.35~−0.15, 직전 4시간 체결량 가중)
  atm   = |델타| 0.4~0.6 체결 IV (직전 1시간)
  pcv   = 풋 명목 / (풋+콜 명목) (직전 1·4시간)
  pbuy  = 풋 공격 매수 명목 − 풋 공격 매도 명목, 콜 동일 → 순 풋 매수 압력 (직전 4시간, 전체 명목으로 나눔)
  변화: rr25 4·24시간 변화, atm 1·4시간 변화
코인베이스(1분):
  cbp   = (코인베이스 현물 종가 − 바이낸스 선물 종가)/바이낸스, 15·60분 평균, 60분 변화, 7일 z점수
인과성: 격자 시점 t의 값은 t 이전(t−1 분 끝까지) 체결·봉만 사용.
출력: data/r54_dfeat.parquet (sym, t, 특징들)
사용: python research/r54_d_features.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.special import ndtr
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R))
from freesia import store
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
MAP = {"BTCUSDT": ("BTC", "BTCUSD"), "ETHUSDT": ("ETH", "ETHUSD")}


def option_buckets(cur):
    files = sorted((store.PQ / "deribit_opt" / cur).glob("*.parquet"))
    rows = []
    for f in files:
        x = pd.read_parquet(f)
        T = (x.expiry - x.ts).dt.total_seconds().values / (365.25 * 86400)
        ok = (T > 1 / 365.25) & (T < 45 / 365.25) & (x.iv.values > 1)
        x = x[ok]; T = T[ok]
        sig = x.iv.values / 100.0; S = x.index_price.values; K = x.strike.values
        d1 = (np.log(S / K) + 0.5 * sig ** 2 * T) / (sig * np.sqrt(T))
        delta = np.where(x.cp.values > 0, ndtr(d1), ndtr(d1) - 1)
        notional = x.amount.values * S
        b = x.ts.dt.floor("15min")
        df = pd.DataFrame({"b": b.values, "cp": x.cp.values, "iv": x.iv.values, "delta": delta, "n": notional, "side": x.side.values})
        c25 = (df.cp > 0) & df.delta.between(0.15, 0.35); p25 = (df.cp < 0) & df.delta.between(-0.35, -0.15)
        atm = df.delta.abs().between(0.4, 0.6)
        g = pd.DataFrame({
            "c25_ivn": (df.iv * df.n).where(c25, 0.0), "c25_n": df.n.where(c25, 0.0),
            "p25_ivn": (df.iv * df.n).where(p25, 0.0), "p25_n": df.n.where(p25, 0.0),
            "atm_ivn": (df.iv * df.n).where(atm, 0.0), "atm_n": df.n.where(atm, 0.0),
            "put_n": df.n.where(df.cp < 0, 0.0), "call_n": df.n.where(df.cp > 0, 0.0),
            "put_net": (df.n * df.side).where(df.cp < 0, 0.0), "call_net": (df.n * df.side).where(df.cp > 0, 0.0),
            "b": df.b})
        rows.append(g.groupby("b").sum())
        P(f"  {cur} {f.stem}: {len(x):,}건")
    B = pd.concat(rows).groupby(level=0).sum().sort_index()
    idx = pd.date_range(B.index.min(), B.index.max(), freq="15min")
    B = B.reindex(idx, fill_value=0.0)
    r = lambda col, w: B[col].rolling(w, min_periods=1).sum()
    F = pd.DataFrame(index=idx)
    F["rr25"] = r("c25_ivn", 16) / r("c25_n", 16) - r("p25_ivn", 16) / r("p25_n", 16)
    F["atm"] = r("atm_ivn", 4) / r("atm_n", 4)
    F["pcv1"] = r("put_n", 4) / (r("put_n", 4) + r("call_n", 4))
    F["pcv4"] = r("put_n", 16) / (r("put_n", 16) + r("call_n", 16))
    F["pbuy"] = (r("put_net", 16) - r("call_net", 16)) / (r("put_n", 16) + r("call_n", 16))
    F["drr4"] = F.rr25 - F.rr25.shift(16); F["drr24"] = F.rr25 - F.rr25.shift(96)
    F["datm1"] = F.atm - F.atm.shift(4); F["datm4"] = F.atm - F.atm.shift(16)
    F["rrz"] = (F.rr25 - F.rr25.rolling(96 * 30, min_periods=96 * 5).mean()) / F.rr25.rolling(96 * 30, min_periods=96 * 5).std()
    # 버킷 b 의 값은 [b, b+15분) 체결 포함 → b+15분부터 사용 가능
    F.index = F.index + pd.Timedelta(minutes=15)
    return F.replace([np.inf, -np.inf], np.nan).ffill(limit=8)


if __name__ == "__main__":
    L = pd.read_parquet(R / "data" / "r48_labels.parquet", columns=["sym", "t", "ts"])
    out = []
    for s, (cur, cb) in MAP.items():
        g = L[L.sym == s].copy()
        F = option_buckets(cur)
        # 격자 t(분 시작)에서 쓸 수 있는 최신 값: 인덱스 ≤ t 인 버킷 (버킷 끝 시각 기준이므로 인과적)
        g = pd.merge_asof(g.sort_values("ts"), F.rename_axis("ts").reset_index(), on="ts", direction="backward")
        cbk = store.load("coinbase_1m", cb)
        if cbk is not None:
            bn = store.load("klines_1m", s).set_index("ts").close
            cbc = cbk.set_index("ts").close.reindex(bn.index).ffill(limit=5)
            prem = (cbc - bn) / bn
            X = pd.DataFrame({"cbp": prem, "cbp15": prem.rolling(15).mean(), "cbp60": prem.rolling(60).mean()})
            X["dcbp60"] = X.cbp60 - X.cbp60.shift(60)
            X["cbpz"] = (X.cbp60 - prem.rolling(10080).mean()) / prem.rolling(10080).std()
            X.index = X.index + pd.Timedelta(minutes=1)       # 분 봉 끝난 뒤 사용
            g = pd.merge_asof(g.sort_values("ts"), X.rename_axis("ts").reset_index(), on="ts", direction="backward")
        out.append(g)
        P(f"{s} D 특징 완료: 옵션 커버 {g.rr25.notna().mean():.2f}" + (f", 코인베이스 {g.cbp.notna().mean():.2f}" if "cbp" in g else ""))
    O = pd.concat(out, ignore_index=True).drop(columns=["ts"])
    for c in O.columns:
        if O[c].dtype == np.float64: O[c] = O[c].astype(np.float32)
    O.to_parquet(R / "data" / "r54_dfeat.parquet", index=False)
    P(f"저장 data/r54_dfeat.parquet {O.shape}")
