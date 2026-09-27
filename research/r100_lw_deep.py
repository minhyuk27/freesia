"""R100 — LW(래리 윌리엄스 변동성 돌파)와 고전 돌파 계열 심층 재연구.

과거 판정(2026-09-19): 일봉·UTC 0시·K 0.4/0.6/0.8·롱만 → 음수 탈락. 이번엔 구조 전체를 넓게 본다.
세션: 길이 P ∈ {24h, 12h, 8h, 4h, 1h}, 시작 시각 a(24h: 0~23시 24가지, 12h: 12가지, 8h: 8가지, 4h: 4가지, 1h: 1가지) — '하루' 정의가 임의인 24시간 시장의 강건성 확인.
진입 수준(세션 k 시작 시 이전 세션 정보만):
  LW   up/dn = O ± K·R, R ∈ {직전 레인지, 3·5세션 평균}, K ∈ {0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0}
  NK   한국식 노이즈 K: K = 직전 20세션 평균(1 − |C−O|/(H−L)), R = 직전 레인지
  DT   Dual Thrust: Range = max(HH−LC, HC−LL) over N ∈ {2, 4}, K ∈ {0.3, 0.5, 0.7}
  DC   돈치안/터틀: 직전 N ∈ {5, 10, 20} 세션 최고·최저 돌파
  ORB  시가 범위 돌파(Crabel): 세션 첫 m ∈ {15, 30, 60}분 고·저 돌파(m분 이후)
체결: 1분봉에서 수준 터치 분에 진입, 가격 = max(수준, 그 분 시가)(갭 반영). 진입 분 안의 손절 터치는 손절로 처리(최악 가정).
청산: X0 세션 종가 / X1 손절 = 세션 시가(본전선) / X2 손절 = O ∓ 0.5R(원래 LW) / X3 다음 세션 종가까지(손절 X2)
비용: 진입·청산 각 테이커 5bp + 슬리피지 2bp = 왕복 14bp.
방향: 롱 / 숏 / 양방향(먼저 닿는 쪽, 같은 분 동시면 건너뜀).
필터: 없음 / 추세(시가 > 직전 N=5세션 종가 평균이면 롱만·아래면 숏만, 양방향에 적용) / NR7(직전 세션이 7세션 중 최소 레인지) /
      수축(직전 레인지 < 20세션 평균의 0.7) / 확장(> 1.3) / 고변동 예보(R90 pr ≥ 0.8, 2022~) / 저변동 예보(pr < 0.5) / 추세+NR7
평가: 거래당 bp, 하루 거래, 일간 샤프 학습 2020~23 / 평가 2024~26, 두 종목. 선택 = 학습 샤프(시작 시각 평균·두 종목 평균) 상위 → 평가.
대조: 같은 세션·같은 방향 '세션 시가 진입' (돌파 조건 없는 무작위 대조) 대비 초과.
사용: python research/r100_lw_deep.py"""
import sys, warnings, time
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
COST = 14e-4


@numba.njit(cache=True)
def run(s0, s1, o, h, l, c, up, dn, stl, sts, allow, xm):
    """세션별 롱·숏 결과(순수익)와 진입 분. xm: 0 종가, 1 본전(세션 시가) 손절, 2 O∓0.5R 손절, 3 다음 세션 종가(손절 2)."""
    n = len(s0); rl = np.full(n, np.nan); rs = np.full(n, np.nan); el = np.full(n, -1); es = np.full(n, -1)
    for k in range(n - 1):
        a = s0[k] + allow[k]; b = s1[k]
        if np.isnan(up[k]) or a >= b: continue
        endk = s1[k + 1] if xm == 3 else b
        for side in (1, -1):
            lv = up[k] if side == 1 else dn[k]
            if np.isnan(lv): continue
            j0 = -1
            for j in range(a, b):
                if (side == 1 and h[j] >= lv) or (side == -1 and l[j] <= lv): j0 = j; break
            if j0 < 0: continue
            px = max(lv, o[j0]) if side == 1 else min(lv, o[j0])
            st = np.nan
            if xm == 1: st = o[s0[k]]
            elif xm >= 2: st = stl[k] if side == 1 else sts[k]
            r = np.nan
            if not np.isnan(st) and ((side == 1 and st >= px) or (side == -1 and st <= px)): st = np.nan
            for j in range(j0, endk):
                if not np.isnan(st) and ((side == 1 and l[j] <= st) or (side == -1 and h[j] >= st)):
                    ex = min(st, o[j]) if (side == 1 and j > j0) else (max(st, o[j]) if j > j0 else st)
                    r = side * (ex / px - 1); break
            if np.isnan(r): r = side * (c[endk - 1] / px - 1)
            if side == 1: rl[k] = r - COST_; el[k] = j0
            else: rs[k] = r - COST_; es[k] = j0
    return rl, rs, el, es


@numba.njit(cache=True)
def orb_levels(s0, s1, h, l, m):
    n = len(s0); up = np.full(n, np.nan); dn = np.full(n, np.nan)
    for k in range(n):
        if s1[k] - s0[k] <= m: continue
        hh = -1e18; ll = 1e18
        for j in range(s0[k], s0[k] + m): hh = max(hh, h[j]); ll = min(ll, l[j])
        up[k] = hh; dn[k] = ll
    return up, dn


COST_ = COST


def sessions(k, Ph, a):
    mi = ((k.ts - k.ts.iloc[0]).dt.total_seconds() // 60).astype(np.int64).values
    sid = (mi - a * 60) // (Ph * 60)
    ch = np.r_[True, sid[1:] != sid[:-1]]; s0 = np.nonzero(ch)[0]; s1 = np.r_[s0[1:], len(sid)]
    O = k.open.values[s0]; C = k.close.values[s1 - 1]
    H = np.maximum.reduceat(k.high.values, s0); L = np.minimum.reduceat(k.low.values, s0)
    full = (s1 - s0) >= Ph * 60 * 0.9
    return s0, s1, O, H, L, C, full, k.ts.values[s0]


def levels(O, H, L, C):
    Rg = pd.Series(H - L); Ro = Rg.shift(1); Cs = pd.Series(C); Hs = pd.Series(H); Ls = pd.Series(L); Os = pd.Series(O)
    out = {}
    for Rn, RR in (("R1", Ro), ("R3", Rg.rolling(3).mean().shift(1)), ("R5", Rg.rolling(5).mean().shift(1))):
        for K in (0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0):
            out[f"LW {Rn} K{K}"] = (O + K * RR.values, O - K * RR.values, RR.values)
    nk = (1 - (Cs - Os).abs() / Rg.replace(0, np.nan)).rolling(20).mean().shift(1)
    out["NK 노이즈K"] = (O + nk.values * Ro.values, O - nk.values * Ro.values, Ro.values)
    for N in (2, 4):
        HH = Hs.rolling(N).max().shift(1); LL = Ls.rolling(N).min().shift(1); HC = Cs.rolling(N).max().shift(1); LC = Cs.rolling(N).min().shift(1)
        rng = np.maximum(HH - LC, HC - LL)
        for K in (0.3, 0.5, 0.7): out[f"DT N{N} K{K}"] = (O + K * rng.values, O - K * rng.values, Ro.values)
    for N in (5, 10, 20):
        out[f"DC N{N}"] = (Hs.rolling(N).max().shift(1).values, Ls.rolling(N).min().shift(1).values, Ro.values)
    return out


def filters(O, H, L, C, pr):
    Rg = pd.Series(H - L); Ro = Rg.shift(1); ma = pd.Series(C).rolling(5).mean().shift(1).values
    nr7 = (Ro == Rg.rolling(7).min().shift(1)).values; ratio = (Ro / Rg.rolling(20).mean().shift(1)).values
    tr = np.where(O > ma, 1, np.where(O < ma, -1, 0))
    return {"없음": (None, None), "추세": (None, tr), "NR7": (nr7, None), "수축": (ratio < 0.7, None), "확장": (ratio > 1.3, None),
            "고변동예보": (pr >= 0.8, None), "저변동예보": (pr < 0.5, None), "추세+NR7": (nr7, tr)}


D0 = pd.Timestamp("2020-01-01", tz="UTC"); TR = (0, 1461); TE = (1461, (pd.Timestamp("2026-09-20", tz="UTC") - D0).days)


def daily_sh(dayi, x, span):
    lo, hi = span; m = ~np.isnan(x) & (dayi >= lo) & (dayi < hi)
    if m.sum() < 5: return np.nan, int(m.sum()), np.nan
    d = np.bincount(dayi[m] - lo, weights=x[m], minlength=hi - lo)
    return (d.mean() / d.std() * np.sqrt(365) if d.std() > 0 else np.nan), int(m.sum()), x[m].mean() * 1e4


if __name__ == "__main__":
    t_ = time.time()
    import r92_vol_timing as V
    Wp = V.load_p()
    ANCH = {24: range(24), 12: range(12), 8: range(8), 4: range(4), 1: range(1)}
    rows = []
    for sym in ("BTCUSDT", "ETHUSDT"):
        k = store.load("klines_1m", sym); k["ts"] = pd.DatetimeIndex(k.ts)
        w = Wp[Wp.sym == sym].sort_values("ts")[["ts", "pr"]]
        o, h, l, c = k.open.values, k.high.values, k.low.values, k.close.values
        for Ph, anchors in ANCH.items():
            for a in anchors:
                s0, s1, O, H, L, C, full, sts = sessions(k, Ph, a)
                sts = pd.DatetimeIndex(sts).tz_localize("UTC") if pd.DatetimeIndex(sts).tz is None else pd.DatetimeIndex(sts)
                dayi = ((sts - D0).days).values.astype(np.int64)
                pr = pd.merge_asof(pd.DataFrame({"ts": sts}), w, on="ts", direction="backward").pr.values
                LV = levels(O, H, L, C)
                for m in (15, 30, 60):
                    if m < Ph * 60: u, d_ = orb_levels(s0, s1, h, l, m); LV[f"ORB {m}분"] = (u, d_, pd.Series(H - L).shift(1).values)
                FL = filters(O, H, L, C, pr)
                # 대조: 세션 시가 진입(수준 = 세션 시가 → 첫 분 즉시 진입)
                LV["대조 시가진입"] = (O * 1.0, O * 1.0, pd.Series(H - L).shift(1).values)
                for nm, (up, dn, RR) in LV.items():
                    up = np.where(full, up, np.nan); allow = np.full(len(s0), int(nm.split()[1][:-1]) if nm.startswith("ORB") else 0, np.int64)
                    stl = O - 0.5 * RR; sts_ = O + 0.5 * RR
                    for xm in (0, 1, 2, 3):
                        if nm == "대조 시가진입" and xm == 1: continue
                        rl, rs, el, es = run(s0.astype(np.int64), s1.astype(np.int64), o, h, l, c, up.astype(float), dn.astype(float), stl, sts_, allow, xm)
                        both = np.where((el >= 0) & ((es < 0) | (el < es)), rl, np.where((es >= 0) & ((el < 0) | (es < el)), rs, np.nan))
                        for fn, (gate, tr) in FL.items():
                            if fn in ("고변동예보", "저변동예보") and nm == "대조 시가진입": pass
                            g = np.ones(len(s0), bool) if gate is None else np.nan_to_num(gate).astype(bool)
                            for sd, rr in (("롱", rl), ("숏", rs), ("양", both)):
                                x = np.where(g, rr, np.nan)
                                if tr is not None:
                                    if sd == "롱": x = np.where(tr == 1, x, np.nan)
                                    elif sd == "숏": x = np.where(tr == -1, x, np.nan)
                                    else: x = np.where(g & (tr == 1) & ~np.isnan(rl) & (el >= 0) & ((es < 0) | (el <= es)), rl, np.where(g & (tr == -1) & (es >= 0) & ((el < 0) | (es <= el)), rs, np.nan))
                                a1 = daily_sh(dayi, x, TR); a2 = daily_sh(dayi, x, TE)
                                rows.append((sym, Ph, a, nm.split()[0], nm, xm, fn, sd, a1[0], a1[1], a1[2], a2[0], a2[1], a2[2]))
            P(f"  {sym} {Ph}h 완료 ({time.time()-t_:.0f}s, 행 {len(rows):,})")
    O_ = pd.DataFrame(rows, columns=["sym", "P", "a", "fam", "lvl", "x", "flt", "side", "sh_tr", "n_tr", "bp_tr", "sh_te", "n_te", "bp_te"])
    O_.to_parquet(R / "data" / "r100_all.parquet", index=False)
    pd.set_option("display.width", 260); pd.set_option("display.max_rows", 300)
    P(f"\n총 구성 {len(O_):,} (종목×세션×시작시각×수준×청산×필터×방향)")
    P("\n===== 가족 × 세션 길이: 평균 샤프 학습/평가 (필터 없음, 모든 시작 시각·청산·방향)")
    b = O_[O_.flt == "없음"]
    P(b.pivot_table(index="fam", columns="P", values=["sh_tr", "sh_te"], aggfunc="mean").round(2).to_string())
    P("\n===== 방향별 (필터 없음)"); P(b.groupby(["fam", "side"])[["sh_tr", "sh_te", "bp_tr", "bp_te"]].mean().round(2).unstack("side").to_string())
    P("\n===== 필터별 평균 (돌파 가족만)"); bb = O_[~O_.fam.isin(["대조"])]
    P(bb.groupby("flt")[["sh_tr", "sh_te", "bp_tr", "bp_te", "n_te"]].mean().round(2).to_string())
    # 시작 시각·두 종목 평균으로 강건 선택
    G = O_.groupby(["P", "lvl", "x", "flt", "side"]).agg(sh_tr=("sh_tr", "mean"), sh_te=("sh_te", "mean"), bp_tr=("bp_tr", "mean"), bp_te=("bp_te", "mean"),
                                                        n_te=("n_te", "mean"), pos_tr=("sh_tr", lambda s: (s > 0).mean()), pos_te=("sh_te", lambda s: (s > 0).mean())).reset_index()
    P("\n===== 강건 선택: 학습 샤프(시작 시각·종목 평균) 상위 25 → 평가")
    P(G.sort_values("sh_tr", ascending=False).head(25).round(2).to_string())
    P("\n===== 2022~ 기준 약세장 포함 상위(참고): 평가 샤프 상위 15 (사후 선택, 믿지 말 것)")
    P(G.sort_values("sh_te", ascending=False).head(15).round(2).to_string())
    ctl = G[G.lvl == "대조 시가진입"]
    P(f"\n대조(세션 시가 진입) 평균 샤프 학습 {ctl.sh_tr.mean():.2f} / 평가 {ctl.sh_te.mean():.2f} · 돌파 전체 평균 {G[G.lvl != '대조 시가진입'].sh_tr.mean():.2f} / {G[G.lvl != '대조 시가진입'].sh_te.mean():.2f}")
    top = G.sort_values("sh_tr", ascending=False).head(50)
    P(f"학습 상위 50의 평가 샤프: 평균 {top.sh_te.mean():.2f}, 양수 {(top.sh_te > 0).sum()}/50, 평가 샤프 ≥ 1: {(top.sh_te >= 1).sum()}")
    (R / "logs" / "r100.summary").write_text("done\n")
