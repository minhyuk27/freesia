"""R106 — 여러 날 추세(선행 조사 D4): 48h·72h·7일 세션 돌파 앙상블 + 시계열 모멘텀(TSMOM)이 z0.31·z0.33과 다른 엣지인가.

근거: Liu·Tsyvinski 암호화폐 TSMOM 1~4주. R100은 1h~24h만 봤고 짧을수록 나빴다 → 더 긴 쪽이 비어 있다.
A. 세션 돌파(롱, R101 방식): 세션 Ph ∈ {48, 72, 168}시간, 시작 시각 6h(48·72)·12h(168) 간격 앙상블.
   수준 DT N{2,4} K{0.5,0.7} · LW R3 K{1.0,1.2}, 추세 필터 MA M{0(없음),5,20}, 청산 X0 세션 종가 / X4 다음 세션 종가. 롱만(숏은 전 계열 음수였음).
B. TSMOM: 매일 00 UTC, 과거 L ∈ {7, 14, 28, 56}일 수익 부호 → 롱(양수)/현금 또는 롱/숏, 변동성 조절(목표 2%/일 ÷ 20일 변동성, 상한 3), 비용 왕복 14bp × 회전.
선택: 학습(2020~23) 일간 샤프 → 평가(2024~26). 학습 상위 20 평가 평균·양수 비율, 이웃 고원, z0.33 TOP8·z0.31 일간 상관.
최선 1개는 R102 공유 계좌(z0.31 + z0.33 + 이것)로 3단 확인.
사용: python research/r106_multiday_trend.py"""
import sys, warnings, time
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store, costs
import r100_lw_deep as B, r102_joint_account as J, r103_breakout_cost as K, r80_max_return as R80
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
D0 = J.D0; NDAY = (pd.Timestamp("2026-09-22", tz="UTC") - D0).days; TRD = 1461
IDX = pd.date_range(D0, periods=NDAY, freq="D")
sh = lambda d: d.mean() / d.std() * np.sqrt(365) if d.std() > 0 else np.nan
SES = {48: range(0, 48, 6), 72: range(0, 72, 6), 168: range(0, 168, 12)}
LEV = [("DT", 2, 0.5), ("DT", 2, 0.7), ("DT", 4, 0.5), ("DT", 4, 0.7), ("LW", 3, 1.0), ("LW", 3, 1.2)]


def daily(e, ret, w):
    d = (e // 1440).astype(int); m = (d >= 0) & (d < NDAY)
    return np.bincount(d[m], weights=(ret * w)[m], minlength=NDAY)


if __name__ == "__main__":
    t_ = time.time()
    KL = {}
    for sym in ("BTCUSDT", "ETHUSDT"):
        k = store.load("klines_1m", sym); k["ts"] = pd.DatetimeIndex(k.ts); KL[sym] = k
    ACC = {}; TR = {}
    for Ph, anchors in SES.items():
        na = len(anchors)
        for sym, k in KL.items():
            mi = J.mins(k.ts); o, h, l, c = k.open.values, k.high.values, k.low.values, k.close.values
            for a in anchors:
                s0, s1, O, H, L, C, full, sts = B.sessions(k, Ph, a)
                for (fam, N, Kk) in LEV:
                    d = J.level(fam, N, Kk, O, H, L, C)
                    for M in (0, 5, 20):
                        g = np.ones(len(O), bool) if M == 0 else (O > pd.Series(C).rolling(M).mean().shift(1).values)
                        lv = np.where(full & g, O + d, np.nan)
                        for xm in (0, 4):
                            J0, J1, PX, EX, LO_ = J.run_long(s0.astype(np.int64), s1.astype(np.int64), o, h, l, c, lv.astype(float), xm)
                            m = J0 >= 0
                            px = PX[m] * (1 + 2e-4); ret = EX[m] * (1 - 2e-4) / px - 1 - 2 * costs.TAKER
                            key = (Ph, fam, N, Kk, M, xm)
                            dd = daily(mi[J0[m]], ret, 1.0 / (2 * na))
                            if key not in ACC: ACC[key] = np.zeros(NDAY); TR[key] = []
                            ACC[key] += dd
                            TR[key].append(pd.DataFrame(dict(sym=sym, e=mi[J0[m]], x=mi[J1[m]] + 1, px=PX[m], ex=EX[m], lo=LO_[m], sz=1.0, wt=48.0 / (2 * na), cfg=str(key))))
        P(f"  세션 {Ph}h 완료 ({time.time()-t_:.0f}s)")
    rows = [dict(kind="돌파", Ph=k[0], fam=k[1], N=k[2], K=k[3], M=k[4], x=k[5], sh_tr=sh(v[:TRD]), sh_te=sh(v[TRD:]), ann_te=v[TRD:].mean() * 365 * 100) for k, v in ACC.items()]
    # B. TSMOM
    for sym, k in KL.items():
        dc = k.set_index("ts").close.resample("D").last().reindex(IDX).ffill()
        r = dc.pct_change().fillna(0); vol = r.rolling(20).std().shift(1)
        for Lb in (7, 14, 28, 56):
            sg = np.sign(dc.pct_change(Lb)).shift(1)
            for mode in ("롱/현금", "롱/숏"):
                pos = sg.clip(lower=0) if mode == "롱/현금" else sg
                pos = (pos * np.clip(0.02 / vol, 0, 3)).fillna(0)
                pnl = pos * r - pos.diff().abs().fillna(0) * 7e-4
                key = ("TSMOM", Lb, mode)
                ACC[key] = ACC.get(key, np.zeros(NDAY)) + pnl.values / 2
    for k, v in ACC.items():
        if k[0] == "TSMOM": rows.append(dict(kind="TSMOM", Ph=0, fam=k[2], N=k[1], K=0, M=0, x=0, sh_tr=sh(v[:TRD]), sh_te=sh(v[TRD:]), ann_te=v[TRD:].mean() * 365 * 100))
    O = pd.DataFrame(rows); pd.set_option("display.width", 220); pd.set_option("display.max_rows", 100)
    P("\n세션·종류별 평균 샤프(학습/평가):"); P(O.groupby(["kind", "Ph"])[["sh_tr", "sh_te"]].agg(["mean", "max"]).round(2).to_string())
    P("\nTSMOM:"); P(O[O.kind == "TSMOM"].round(2).to_string(index=False))
    top = O[O.kind == "돌파"].sort_values("sh_tr", ascending=False)
    P(f"\n돌파 학습 상위 20 평가 샤프 평균 {top.head(20).sh_te.mean():.2f}, 양수 {(top.head(20).sh_te > 0).sum()}/20"); P(top.head(15).round(2).to_string(index=False))
    # 상관: z0.33 TOP8, z0.31
    T8 = J.add_funding(J.breakout_trades(J.TOP8)); T8["lf"] = K.last_funding(T8); BK = K.events(T8, False, fmax=1e-4)
    z33 = daily(BK.e.values, BK.ret.values, BK.mult.values / 48)
    E0 = J.dip_events(); z31 = daily(E0.e.values, E0.ret.values, E0.mult.values)
    b = top.iloc[0]; bk = (b.Ph, b.fam, b.N, b.K, b.M, b.x); v = ACC[bk]; s = slice(731, NDAY)
    P(f"\n최선 {bk}: 학습 {sh(v[:TRD]):.2f} 평가 {sh(v[TRD:]):.2f} | 일간 상관(2022~) z0.33 {np.corrcoef(v[s], z33[s])[0,1]:+.2f} · z0.31 {np.corrcoef(v[s], z31[s])[0,1]:+.2f}"
      f" | 연도별 " + " ".join(f"{y%100}:{sh(v[IDX.year == y]):+.1f}" for y in range(2020, 2027)))
    tb = O[O.kind == "TSMOM"].sort_values("sh_tr", ascending=False).iloc[0]; tv = ACC[("TSMOM", tb.N, tb.fam)]
    P(f"TSMOM 최선 L{tb.N} {tb.fam}: 학습 {sh(tv[:TRD]):.2f} 평가 {sh(tv[TRD:]):.2f} | 상관 z0.33 {np.corrcoef(tv[s], z33[s])[0,1]:+.2f} · z0.31 {np.corrcoef(tv[s], z31[s])[0,1]:+.2f}")
    # 공유 계좌: z0.31 + z0.33 + 최선 여러날 돌파 (L 비중)
    Tn = J.add_funding(pd.concat(TR[bk], ignore_index=True)); Tn["lf"] = K.last_funding(Tn); En = K.events(Tn, False).drop(columns="lf")
    start = J.mins([pd.Timestamp("2022-01-01", tz="UTC")])[0]
    base = pd.concat([E0, BK.drop(columns="lf")[BK.e >= start]], ignore_index=True)
    for Ln in (0, 0.5, 1, 2):
        En2 = En[En.e >= start].copy(); En2["mult"] = En2.mult * Ln
        EE = pd.concat([base, En2], ignore_index=True).sort_values(["e", "k"]).reset_index(drop=True)
        for tier, lam in (("무레버", 1), ("중간", 2), ("공격", 3)):
            J.rep(f"+여러날 L{Ln} {tier}", EE, lam, 4 * lam, lam)
    O.to_parquet(R / "data" / "r106_grid.parquet", index=False)
    P(f"완료 ({time.time()-t_:.0f}s)")
