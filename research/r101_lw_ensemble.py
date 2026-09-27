"""R101 — R100 생존 계열(24h 세션 · 큰 K 돌파 · 추세 필터 · 롱) 심화: 시작 시각 24개 앙상블, 정밀 격자, 고원, z0.31 결합.

수준: DT N ∈ {2..6} × K ∈ {0.4, 0.5, 0.6, 0.7, 0.8, 1.0} / LW R ∈ {3, 5, 10}(평균 레인지) × K ∈ {0.6, 0.8, 1.0, 1.2}
추세 필터: 세션 시가 > 직전 M ∈ {3, 5, 10, 20} 세션 종가 평균 → 롱 / 아래 → 숏(대칭 숏도 따로 평가)
청산: X0 세션 종가 / X3 다음 세션 종가(손절 O−0.5R) / X4 다음 세션 종가(손절 없음)
앙상블: 시작 시각 0~23시 24개 각각 자본 1/24 (서로 다른 시각에 뜨는 돌파를 모두 받음). 종목 BTC·ETH 합산(각 1/2).
선택: 학습(2020~23) 앙상블 샤프 → 평가(2024~26). 고원: 이웃(K·N/R·M 한 칸) 평가 샤프 양수 비율.
사이징 변형: 변동성 조절(명목 = 목표 2% / 직전 레인지%, 상한 3).
결합: z0.31(w1N4) + k × 최선 돌파(2022~23 변동성 맞춤), 3단 레버리지.
사용: python research/r101_lw_ensemble.py"""
import sys, warnings, time
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
import r100_lw_deep as B
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
COST = 14e-4
D0 = pd.Timestamp("2020-01-01", tz="UTC"); NDAY = (pd.Timestamp("2026-09-20", tz="UTC") - D0).days; TRD = 1461
sh = lambda d: d.mean() / d.std() * np.sqrt(365) if d.std() > 0 else np.nan


@numba.njit(cache=True)
def run(s0, s1, o, h, l, c, lv, stp, side, xm):
    n = len(s0); r_ = np.full(n, np.nan)
    for k in range(n - 1):
        if np.isnan(lv[k]): continue
        a = s0[k]; b = s1[k]; endk = s1[k + 1] if xm >= 3 else b
        j0 = -1
        for j in range(a, b):
            if (side == 1 and h[j] >= lv[k]) or (side == -1 and l[j] <= lv[k]): j0 = j; break
        if j0 < 0: continue
        px = max(lv[k], o[j0]) if side == 1 else min(lv[k], o[j0])
        st = stp[k] if xm == 3 else np.nan
        if not np.isnan(st) and ((side == 1 and st >= px) or (side == -1 and st <= px)): st = np.nan
        r = np.nan
        for j in range(j0, endk):
            if not np.isnan(st) and ((side == 1 and l[j] <= st) or (side == -1 and h[j] >= st)):
                ex = st if j == j0 else (min(st, o[j]) if side == 1 else max(st, o[j])); r = side * (ex / px - 1); break
        if np.isnan(r): r = side * (c[endk - 1] / px - 1)
        r_[k] = r - COST_
    return r_


COST_ = COST


def lv_sets(O, H, L, C):
    Rg = pd.Series(H - L); Cs = pd.Series(C); Hs = pd.Series(H); Ls = pd.Series(L); out = {}
    for N in (2, 3, 4, 5, 6):
        HH = Hs.rolling(N).max().shift(1); LL = Ls.rolling(N).min().shift(1); HC = Cs.rolling(N).max().shift(1); LC = Cs.rolling(N).min().shift(1)
        rng = np.maximum(HH - LC, HC - LL).values
        for K in (0.4, 0.5, 0.6, 0.7, 0.8, 1.0): out[("DT", N, K)] = rng * K
    for Rn in (3, 5, 10):
        rr = Rg.rolling(Rn).mean().shift(1).values
        for K in (0.6, 0.8, 1.0, 1.2): out[("LW", Rn, K)] = rr * K
    return out


if __name__ == "__main__":
    t_ = time.time()
    # 결과: key(fam, N, K, M, x, side, sizing) → 일간 손익 배열(종목 합, 앙상블 평균)
    ACC = {}
    for sym in ("BTCUSDT", "ETHUSDT"):
        k = store.load("klines_1m", sym); k["ts"] = pd.DatetimeIndex(k.ts)
        o, h, l, c = k.open.values, k.high.values, k.low.values, k.close.values
        for a in range(24):
            s0, s1, O, H, L, C, full, sts = B.sessions(k, 24, a)
            sts = pd.DatetimeIndex(sts); sts = sts.tz_localize("UTC") if sts.tz is None else sts
            dayi = ((sts - D0).days).values.astype(np.int64); ok = (dayi >= 0) & (dayi < NDAY)
            Ro = pd.Series(H - L).shift(1).values; vt = np.clip(0.02 / (Ro / O), 0, 3)
            MA = {M: pd.Series(C).rolling(M).mean().shift(1).values for M in (3, 5, 10, 20)}
            for key, d in lv_sets(O, H, L, C).items():
                for side in (1, -1):
                    lv = np.where(full, O + side * d, np.nan); stp = O - side * 0.5 * Ro
                    for xm in (0, 3, 4):
                        r = run(s0.astype(np.int64), s1.astype(np.int64), o, h, l, c, lv.astype(float), stp, side, xm)
                        for M in (0, 3, 5, 10, 20):
                            g = np.ones(len(r), bool) if M == 0 else (np.sign(O - MA[M]) == side)
                            x = np.where(g, r, np.nan)
                            for sz in ("고정", "변동성조절"):
                                y = x if sz == "고정" else x * vt
                                m = ~np.isnan(y) & ok
                                dd = np.bincount(dayi[m], weights=y[m], minlength=NDAY) / 48      # 24 시각 × 2 종목
                                kk = key + (M, xm, side, sz)
                                if kk not in ACC: ACC[kk] = [np.zeros(NDAY), 0, 0.0]
                                ACC[kk][0] += dd; ACC[kk][1] += int(m.sum()); ACC[kk][2] += float(np.nansum(y[m]))
        P(f"  {sym} 완료 ({time.time()-t_:.0f}s, 구성 {len(ACC):,})")
    rows = []
    for kk, (dd, n, s) in ACC.items():
        rows.append(dict(fam=kk[0], N=kk[1], K=kk[2], M=kk[3], x=kk[4], side="롱" if kk[5] == 1 else "숏", sz=kk[6],
                         sh_tr=sh(dd[:TRD]), sh_te=sh(dd[TRD:]), ann_te=dd[TRD:].mean() * 365 * 100, n_day=n / NDAY, bp=s / max(n, 1) * 1e4))
    O_ = pd.DataFrame(rows); pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    P(f"\n구성 {len(O_):,}. 방향·필터별 평균 앙상블 샤프 (학습/평가):")
    P(O_.groupby(["side", "M"])[["sh_tr", "sh_te"]].mean().round(2).unstack("M").to_string())
    P("\n청산별:"); P(O_.groupby(["side", "x"])[["sh_tr", "sh_te"]].mean().round(2).to_string())
    top = O_.sort_values("sh_tr", ascending=False).head(25)
    P("\n===== 학습 앙상블 샤프 상위 25 → 평가"); P(top.round(2).to_string())
    # 고원: 최선 구성의 이웃
    b = top.iloc[0]
    fam_rows = O_[(O_.fam == b.fam) & (O_.x == b.x) & (O_.side == b.side) & (O_.sz == b.sz)]
    Ns = sorted(fam_rows.N.unique()); Ks = sorted(fam_rows.K.unique()); Ms = sorted(fam_rows.M.unique())
    near = fam_rows[fam_rows.N.isin(Ns[max(0, Ns.index(b.N) - 1):Ns.index(b.N) + 2]) & fam_rows.K.isin(Ks[max(0, Ks.index(b.K) - 1):Ks.index(b.K) + 2])
                    & fam_rows.M.isin(Ms[max(0, Ms.index(b.M) - 1):Ms.index(b.M) + 2])]
    P(f"\n고원: 최선 {b.fam} N{b.N} K{b.K} M{b.M} X{b.x} {b.side} {b.sz} 이웃 {len(near)}개 평가 샤프 평균 {near.sh_te.mean():.2f}, 양수 {(near.sh_te>0).mean():.0%}")
    P(f"학습 상위 50 평가 샤프: 평균 {O_.sort_values('sh_tr', ascending=False).head(50).sh_te.mean():.2f}, 양수 {(O_.sort_values('sh_tr', ascending=False).head(50).sh_te > 0).sum()}/50")
    # 롱+숏 결합(같은 설정 대칭)
    kb = (b.fam, b.N, b.K, b.M, b.x, 1, b.sz); ks = (b.fam, b.N, b.K, b.M, b.x, -1, b.sz)
    best = ACC[kb][0]; both = ACC[kb][0] + ACC[ks][0]
    idx = pd.date_range(D0, periods=NDAY, freq="D")
    P(f"최선 롱: 학습 {sh(best[:TRD]):.2f} 평가 {sh(best[TRD:]):.2f} | 롱+대칭숏: 학습 {sh(both[:TRD]):.2f} 평가 {sh(both[TRD:]):.2f} | 연도별 롱 샤프 " +
      " ".join(f"{y%100}:{sh(best[(idx.year==y)]):+.2f}" for y in range(2020, 2027)))
    # z0.31 결합 · 3단
    import r40_regime_leverage as RL, r80_max_return as R80
    RL.MMR = 0.0065; W = R80.dip_frame(0.0025)
    mult = np.minimum(np.clip(W.vreg.values ** 1.5, 0.3, 4) * np.clip(W.pred.values / 0.001, 0.5, 3), 6)
    dip = R80.daily_from_eq(RL.sim(W, 1, 4, mult, X=0.03, H=720)[0]); dip.index = dip.index.normalize()
    lw = pd.Series(best, index=idx); I = dip.index.intersection(lw.index); dip = dip.reindex(I).fillna(0); lw = lw.reindex(I).fillna(0)
    P(f"\nz0.31 일간 상관 {np.corrcoef(dip, lw)[0,1]:+.2f} (2024~ {np.corrcoef(dip['2024':], lw['2024':])[0,1]:+.2f})")
    for nm, r in (("돌파 단독(거래당 명목 = 자본 × L/48)", lw),):
        for L in (12, 24, 48):
            x = r["2024":] * L; eq = (1 + x.clip(lower=-0.99)).cumprod(); yrs = len(x) / 365.25
            P(f"  {nm} ×{L}: 월 {(eq.iloc[-1]**(1/12/yrs)-1)*100:+.2f}% MDD {(eq/eq.cummax()-1).min()*100:.0f}% 샤프 {sh(x):.2f}")
    f = lw * dip["2022":"2023"].std() / lw["2022":"2023"].std()
    for w in (0.0, 0.25, 0.5, 1.0):
        r = dip + w * f; out = []
        for L in (1, 2, 3):
            x = r["2024":] * L; eq = (1 + x.clip(lower=-0.99)).cumprod(); yrs = len(x) / 365.25
            out.append(f"×{L} 월 {(eq.iloc[-1]**(1/12/yrs)-1)*100:+.1f}% MDD {(eq/eq.cummax()-1).min()*100:.0f}%")
        P(f"  z0.31 + {w:.2f}×돌파: 샤프 학습 {sh(r[:'2023']):.2f} 평가 {sh(r['2024':]):.2f} | " + " | ".join(out))
    O_.to_parquet(R / "data" / "r101_grid.parquet", index=False)
    (R / "logs" / "r101.summary").write_text("done\n")
