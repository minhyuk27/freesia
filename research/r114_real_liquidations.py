"""R114 — 실제 강제청산(바이낸스 코인마진 BTCUSD_PERP·ETHUSD_PERP 청산 스냅샷, 2023-06-25 ~ 2024-10-14)으로 딥·돌파를 가르는가.

R24~25는 OI 변화 등 '추정' 청산 지문이었고 진입 신호로 기각됐다. 여기서는 실제 청산 기록을 **이미 거래하는 딥·돌파의 조건**으로만 쓴다.
주의: 코인마진 청산은 USDⓈ-M 청산의 일부(표본 지표)다. 바이낸스 스냅샷은 1초에 최대 1건만 담는다(하한값).
특징(진입 분 직전까지, 종목별, 청산된 쪽 기준): 롱 청산 대금 liqL_k, 숏 청산 대금 liqS_k (k 5·15·60분), 각자 30일 중앙값 대비 배수.
가설 C(딥): 직전 롱 청산이 많은 딥(강제 매도 주도) → 더 잘 되돌린다. 가설 B(돌파): 직전 숏 청산이 많은 돌파(스퀴즈) → 더 좋다(또는 과열로 나쁘다).
평가: 기간이 짧아 학습 2023-06~2023-12 / 평가 2024-01~2024-10. 5분위별 거래당 bp·단조성. 청산 0인 시점이 많아 '0 vs 양수·상위' 3구간도 본다.
사용: python research/r114_real_liquidations.py"""
import sys, warnings, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
import r102_joint_account as J, r103_breakout_cost as K, r109_dip_exact_mtm as M
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
A0, A1, CUT = pd.Timestamp("2023-06-26", tz="UTC"), pd.Timestamp("2024-10-14", tz="UTC"), pd.Timestamp("2024-01-01", tz="UTC")


def liq_feats(cm):
    q = store.load("liq_snap", cm); q["ts"] = pd.DatetimeIndex(q.ts)
    P(f"  {cm}: {len(q):,}건 {q.ts.min()} ~ {q.ts.max()}, side 값 {q.side.value_counts().to_dict()}")
    idx = pd.date_range(A0, A1, freq="1min")
    q["m"] = q.ts.dt.floor("min")
    L = q[q.side.astype(str).str.upper().str.startswith("S")].groupby("m").notional.sum().reindex(idx, fill_value=0)   # SELL 주문 = 롱 청산
    S = q[q.side.astype(str).str.upper().str.startswith("B")].groupby("m").notional.sum().reindex(idx, fill_value=0)   # BUY 주문 = 숏 청산
    F = {}
    for k in (5, 15, 60):
        for nm, v in (("liqL", L), ("liqS", S)):
            r = v.rolling(k).sum(); base = r.rolling(43200, min_periods=1440).median()
            F[f"{nm}{k}"] = r; F[f"{nm}{k}x"] = r / (base + 1e-9)
    return pd.DataFrame(F).astype("float32")


def table(nm, X, cols):
    tr = X.ts < CUT
    P(f"\n[{nm}] 거래 {len(X):,} (학습 {tr.sum():,} / 평가 {(~tr).sum():,}) · 거래당 bp")
    for c in cols:
        v = X[c]
        z = v <= 0; pos = v > 0; hi = v >= v[tr & pos].quantile(0.7) if (tr & pos).sum() > 20 else pos
        g = lambda m_: f"{X.ret[m_ & tr].mean()*1e4:+6.1f}({(m_ & tr).sum()})/{X.ret[m_ & ~tr].mean()*1e4:+6.1f}({(m_ & ~tr).sum()})"
        P(f"  {c:9s} 0: {g(z)} | 양수: {g(pos & ~hi)} | 상위30%: {g(hi)}   (학습/평가)")


if __name__ == "__main__":
    t_ = time.time()
    LF = {"BTCUSDT": liq_feats("BTCUSD_PERP"), "ETHUSDT": liq_feats("ETHUSD_PERP")}
    T8 = J.add_funding(J.breakout_trades(J.TOP8)); T8["lf"] = K.last_funding(T8); BK = K.events(T8, False, fmax=1e-4).drop(columns="lf")
    D, _, _, W = M.dip_with_px()
    for nm, df, cols in (("C 딥 × 롱 청산", D, ["liqL5", "liqL15", "liqL60", "liqL15x", "liqS15"]), ("B 돌파 × 숏 청산", BK, ["liqS5", "liqS15", "liqS60", "liqS15x", "liqL15"])):
        X = df[["e", "ret", "s"]].copy(); X["ts"] = J.D0 + pd.to_timedelta(X.e, unit="min")
        X = X[(X.ts > A0 + pd.Timedelta(days=2)) & (X.ts < A1)].reset_index(drop=True)
        for s_, sym in ((0, "BTCUSDT"), (1, "ETHUSDT")):
            mk = (X.s == s_).values; ts = pd.DatetimeIndex(X.loc[mk, "ts"] - pd.Timedelta(minutes=1))
            for c in LF[sym].columns: X.loc[mk, c] = LF[sym][c].reindex(ts).values
        table(nm, X, cols)
    P(f"완료 ({time.time()-t_:.0f}s)")
