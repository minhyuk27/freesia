"""R73 — 짧은 창(최근 180봉 = 1달)에서도 신호가 충분히 나오게 지표를 소형 묶음으로 설계 → 매 4시간 재학습 전체 테스트.

R72(91개 지표 × 180표본)는 과적합. 여기서는 5~6개짜리 지표 묶음과 작은 표본용 선형 모델을 쓴다.
지표 묶음 (4시간봉, 봉 i 종가까지)
  S1 모멘텀: 1·3·6·12·24·42봉 수익 / (ATR·√k)
  S2 평균회귀: 종가의 SMA 6·24·42 괴리 z, RSI6, RSI14, 볼린저 %b
  S3 캔들: 마지막 봉 몸통·윗꼬리·아랫꼬리, 6봉 범위 안 위치, 6봉 누적수익, 6봉 중 양봉 수
  S4 거래량·체결: 마지막 봉 거래량, 6봉 평균 거래량, 마지막 봉 테이커 비율, 6봉 평균 테이커 비율, OBV 기울기, 24시간 VWAP 거리
  S5 파생: 펀딩, 펀딩 변화, 프리미엄, 프리미엄 6봉 변화, OI 1봉·6봉 변화
  S6 교차: 상대 코인 1·6봉 수익, 코인베이스 프리미엄·6봉 변화, ETH/BTC 6봉 변화
  S7 변동성·국면: ATR 백분위, 실현변동성 비율, 볼린저 폭, ADX, 마지막 봉 범위/ATR
  S8 압축: 91개 전체를 창 안에서 표준화 → 주성분 5개
  조합: S1+S2, S3+S4, S1+S6, S2+S3, 소형전체(S1~S7), 참고 ALL91
모델 (창 안 표준화, 결측은 창 중앙값)
  RG 릿지: y = 부호(미래수익) 에 대한 릿지 회귀(λ = 10)
  CW 상관가중: 각 지표의 창 안 상관 ρ_j, |ρ_j|√n ≥ 1 인 지표만 점수 = Σ ρ_j z_j
신호: 현재 점수가 창 안 점수의 상위 q 이상 롱, 하위 q 이하 숏. q = 20%(충분) / 10%(엄격).
매 4시간 봉 i마다 종목별로 학습 = 봉 [i−h−179, i−h]. 목표 h = 1(4시간), 6(24시간). 비용 왕복 14bp. 위약 = 미래수익 시간 이동.
사용: python research/r73_short_window_sets.py"""
import sys, warnings, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
N = 180; QS = (0.2, 0.1); LAM = 10.0


def small_sets(D, g):
    """D: build() 결과(G* 특징), g: 같은 봉의 OHLC. 반환 dict 묶음명 → 열 목록, D에 새 열 추가."""
    lc = np.log(g.c); A = D.G1_hl0 * 0 + np.log(1 + (pd.concat([g.h - g.l, (g.h - g.c.shift()).abs(), (g.l - g.c.shift()).abs()], axis=1)
                                                         .max(axis=1).rolling(14).mean() / g.c)).values
    A = pd.Series(np.asarray(A, float), index=D.index)
    lcv = pd.Series(lc.values, index=D.index); cv = pd.Series(g.c.values, index=D.index)
    for k in (1, 3, 6, 12, 24, 42): D[f"S1_m{k}"] = (lcv - lcv.shift(k)) / (A * np.sqrt(k))
    for k in (6, 24, 42): D[f"S2_z{k}"] = (cv - cv.rolling(k).mean()) / cv.rolling(k).std()
    d = cv.diff(); au = d.clip(lower=0).ewm(alpha=1 / 6, adjust=False).mean(); ad = (-d.clip(upper=0)).ewm(alpha=1 / 6, adjust=False).mean()
    D["S2_rsi6"] = 100 - 100 / (1 + au / ad); D["S2_rsi14"] = D.G3_rsi; D["S2_pctb"] = D.G4_pctb
    for a, b in (("S3_body", "G1_body0"), ("S3_uw", "G1_uw0"), ("S3_lw", "G1_lw0"), ("S3_pos", "G1_pos"), ("S3_cum", "G1_cum"), ("S3_nup", "G1_nup")): D[a] = D[b]
    D["S4_v0"] = D.G5_v0; D["S4_v6"] = D[[f"G5_v{k}" for k in range(6)]].mean(1); D["S4_tb0"] = D.G5_tb0
    D["S4_tb6"] = D[[f"G5_tb{k}" for k in range(6)]].mean(1); D["S4_obv"] = D.G5_obv; D["S4_vwap"] = D.G5_vwap
    for a, b in (("S5_fund", "G6_fund"), ("S5_dfund", "G6_dfund"), ("S5_prem", "G6_prem"), ("S5_dprem", "G6_dprem"), ("S5_oi1", "G6_oi1"), ("S5_oi6", "G6_oiW")): D[a] = D[b]
    for a, b in (("S6_o1", "G7_o1"), ("S6_o6", "G7_oW"), ("S6_cbp", "G7_cbp"), ("S6_dcbp", "G7_dcbp"), ("S6_eb", "G7_eb")): D[a] = D[b]
    for a, b in (("S7_atrp", "G4_atrp"), ("S7_rvr", "G4_rvr"), ("S7_bbw", "G4_bbw"), ("S7_adx", "G2_adx"), ("S7_hl", "G1_hl0")): D[a] = D[b]
    S = {f"S{k}": [c for c in D.columns if c.startswith(f"S{k}_")] for k in range(1, 8)}
    return S


def prep(Xw, xi):
    med = np.nanmedian(Xw, 0); med = np.where(np.isnan(med), 0, med)
    F = np.where(np.isnan(Xw), med, Xw); mu = F.mean(0); sd = F.std(0); ok = sd > 1e-12; sd[~ok] = 1
    Z = (F - mu) / sd; zi = (np.where(np.isnan(xi), med, xi) - mu) / sd
    return Z[:, ok], zi[ok]


def scores(Z, zi, y):
    """RG 릿지와 CW 상관가중: (창 안 점수들, 현재 점수)."""
    ys = np.sign(y); n, k = Z.shape
    w = np.linalg.solve(Z.T @ Z + LAM * np.eye(k), Z.T @ (ys - ys.mean()))
    rg_w, rg_i = Z @ w, zi @ w
    yz = (y - y.mean()) / (y.std() + 1e-12); rho = (Z * yz[:, None]).mean(0)
    m = np.abs(rho) * np.sqrt(n) >= 1.0
    cw_w, cw_i = (Z[:, m] @ rho[m], zi[m] @ rho[m]) if m.any() else (np.zeros(n), 0.0)
    return (rg_w, rg_i), (cw_w, cw_i)


def job(args):
    sym, ycol, H, plc, X, sets, y, t = args
    if plc: y = np.roll(y, int(np.random.default_rng(int(sym == "ETHUSDT") * 2 + int(ycol == "fW")).integers(500, len(y) - 500)))
    out = []
    for i in range(N + H + 200, len(y)):
        a, b = i - H - N + 1, i - H + 1; yw = y[a:b]
        if np.isnan(yw).any() or np.isnan(y[i]): continue
        for sname, cols in sets.items():
            if sname == "S8":
                Z, zi = prep(X[a:b][:, cols], X[i][cols])
                U, s, Vt = np.linalg.svd(Z, full_matrices=False); Z, zi = Z @ Vt[:5].T, zi @ Vt[:5].T
                sz = Z.std(0) + 1e-12; Z = Z / sz; zi = zi / sz
            else:
                Z, zi = prep(X[a:b][:, cols], X[i][cols])
            if Z.shape[1] == 0: continue
            for mname, (sw, si) in zip(("RG", "CW"), scores(Z, zi, yw)):
                if np.all(sw == 0): continue
                for q in QS:
                    hi, lo = np.quantile(sw, 1 - q), np.quantile(sw, q)
                    side = 1 if si >= hi else (-1 if si <= lo else 0)
                    if side: out.append((t[i], sym, sname, mname, q, side, y[i]))
    return ycol, plc, pd.DataFrame(out, columns=["t", "sym", "set", "model", "q", "side", "fwd"])


if __name__ == "__main__":
    from freesia import store
    from r68_bar_patterns import build, COST
    t_ = time.time()
    closes = {s: store.load("klines_1m", s).set_index("ts").close.resample("4h").last() for s in ("BTCUSDT", "ETHUSDT")}
    jobs = []
    for s in ("BTCUSDT", "ETHUSDT"):
        k = store.load("klines_1m", s).set_index("ts")
        g = k.resample("4h").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna().rename(columns={"open": "o", "high": "h", "low": "l", "close": "c"})
        D = build(s, "4h", 6, closes["ETHUSDT" if s == "BTCUSDT" else "BTCUSDT"])
        D = D.replace([np.inf, -np.inf], np.nan)
        g.index = g.index + pd.Timedelta("4h"); g = g.reindex(pd.DatetimeIndex(D.t))
        D = D.reset_index(drop=True); g = g.reset_index(drop=True)
        S = small_sets(D, g); D = D.replace([np.inf, -np.inf], np.nan)
        D = D.dropna(subset=["f1", "fW", "G1_r5"]).sort_values("t").reset_index(drop=True)
        ALL91 = [c for c in D.columns if c[:2] in ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9") and c[2] == "_"]
        cols = sorted(set(sum(S.values(), [])) | set(ALL91)); X = D[cols].values.astype(float); ix = {c: j for j, c in enumerate(cols)}
        SETS = {**S, "S8": ALL91, "S1+S2": S["S1"] + S["S2"], "S3+S4": S["S3"] + S["S4"], "S1+S6": S["S1"] + S["S6"],
                "S2+S3": S["S2"] + S["S3"], "소형전체": sum((S[f"S{k}"] for k in range(1, 8)), []), "ALL91": ALL91}
        SETS = {k: [ix[c] for c in v] for k, v in SETS.items()}
        for ycol, H in (("f1", 1), ("fW", 6)):
            for plc in (False, True): jobs.append((s, ycol, H, plc, X, SETS, D[ycol].values, D.t.values))
    P(f"작업 {len(jobs)}개, 묶음 {len(SETS)}개: " + " ".join(f"{k}({len(v)})" for k, v in SETS.items()) + f" ({time.time()-t_:.0f}s)")
    res = {}
    with Pool(2) as pool:
        for ycol, plc, df in pool.imap_unordered(job, jobs):
            res.setdefault((ycol, plc), []).append(df); P(f"  완료 {ycol} {'위약' if plc else '실제'} 신호 {len(df):,} ({time.time()-t_:.0f}s)")
    NAME = {"S1": "모멘텀", "S2": "평균회귀", "S3": "캔들", "S4": "거래량·체결", "S5": "파생", "S6": "교차", "S7": "변동성·국면", "S8": "압축(PCA5)"}
    rows = []
    for ycol, H, tname in (("f1", 1, "다음 4시간"), ("fW", 6, "다음 24시간")):
        Rl, Pl = pd.concat(res[(ycol, False)]), pd.concat(res[(ycol, True)])
        for d_ in (Rl, Pl): d_["t"] = pd.to_datetime(d_.t, utc=True); d_["r"] = d_.side * d_.fwd - COST
        days = (Rl.t.max() - Rl.t.min()).days
        P(f"\n===== 목표 {tname} (매 4시간 최근 180봉 재학습) — 거래당 비용 후 bp [적중, 하루] | 위약 | 연도(실제)")
        for (sn, mn, q), G in Rl.groupby(["set", "model", "q"], sort=False):
            Gp = Pl[(Pl.set == sn) & (Pl.model == mn) & (Pl.q == q)]
            yr = G.groupby(G.t.dt.year).r.mean() * 1e4
            lab = f"{sn} {NAME.get(sn, '')}"
            P(f"  {lab:18s} {mn} q{int(q*100):2d} | {G.r.mean()*1e4:+6.1f}bp [{(G.side*G.fwd>0).mean():.3f}, {len(G)/days:4.1f}] | {Gp.r.mean()*1e4:+6.1f} | " +
              " ".join(f"{k%100}:{v:+.0f}" for k, v in yr.items()))
            rows.append((tname, sn, mn, q, G.r.mean() * 1e4, (G.side * G.fwd > 0).mean(), len(G) / days, Gp.r.mean() * 1e4, (yr > 0).sum(), len(yr)))
    O = pd.DataFrame(rows, columns=["목표", "묶음", "모델", "q", "bp", "적중", "하루", "위약bp", "흑자연도", "연도수"])
    O.to_csv(R / "logs" / "r73.csv", index=False)
    P("\n  상위 10 (거래당 bp)"); P(O.sort_values("bp", ascending=False).head(10).round(3).to_string(index=False))
    (R / "logs" / "r73.summary").write_text("R73 짧은 창 소형 지표 묶음 완료. 상위:\n" + O.sort_values("bp", ascending=False).head(3).round(2).to_string(index=False) + "\n")
