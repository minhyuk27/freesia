"""R126 — 딥 전략의 출구(청산 방식)를 바꾸면(+넓은 봉·깊이 격자로) 예측 모델 재학습 후
더 일관적인가(승률↑, 손실 주간 축소)?

배경: 현재 운영 사건(data/r32_events.parquet, r32_dip_events.py)은 전부 mode 0(봉 종가 테이커 청산)
      이다. r26_hf_twosided.run()의 mode 1(지정가 TP, 메이커)은 체결 유보 리스크는 있지만
      수수료가 절반이고 역행 제한 효과가 있다. 여기서는 출구 방식·격자를 바꾼 뒤
      r32_filter.walk()와 같은 절차(HistGradientBoostingRegressor, 확장창 연 단위 2022~2026,
      목표 ret ±2% 절단, 전체 특징)로 재학습하고, r118 공유계좌 엔진으로 주간/월간 일관성을 비교한다.

사전 등록 변형(결과를 본 뒤 추가하지 않는다):
  E0 = mode 0 (현재 운영과 같음 — 재현 점검용)
  E1 = mode 1, g∈{0.3,0.5,1.0} × H∈{0,2}봉 (6개)
  G1 = mode 0, 봉 B∈{3,5,10,15,30,60,120} × 깊이 kx∈{2,2.5,3,4,5,6,8} (1개, 넓은 격자)
각 변형: BTC·ETH 사건 생성(r27.feats + r32.extra, 전부 f-1 특징) → r32_filter 절차로 재학습
→ r119_dip_model_v2.dip_events(Wv, pc) pc∈{5,10,15bp} → r118 엔진(w3N12, w2N8 × 돌파 L2) →
2022-24 / 2025-26 주간·월간 지표.

선택 규칙(사전 고정): 2022~2024 지표만으로 고른다(1순위 주간 샤프, 동률 시 양(+)주 비율).
저장: data/r126.parquet (전 조합), data/r126_best_events.parquet (선택된 변형의 딥 사건, 엔진 표준 컬럼).
사용: python research/r126_dip_exits.py
"""
import sys, warnings, time, gc
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
from r26_hf_twosided import run, EPS, FM, FT
from r27_dip_events import feats, BARS, KS
from r32_dip_events import extra
import r118_engine as G, r119_dip_model_v2 as V, r80_max_return as R80
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
SYM = ("BTCUSDT", "ETHUSDT")
META = {"sym", "f", "x", "ret", "mae", "ts", "yr"}
G1_B = (3, 5, 10, 15, 30, 60, 120)
G1_K = (2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0)

VARIANTS = {"E0": dict(mode=0, g=1.0, H=0, Bs=BARS, Ks=KS)}
for g in (0.3, 0.5, 1.0):
    for H in (0, 2):
        VARIANTS[f"E1_g{g:g}_H{H}"] = dict(mode=1, g=g, H=H, Bs=BARS, Ks=KS)
VARIANTS["G1"] = dict(mode=0, g=1.0, H=0, Bs=G1_B, Ks=G1_K)


def mdl():
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=400, l2_regularization=1.0, random_state=0)


def walk(E, feats_):
    out = []
    for Y in range(2022, 2027):
        tr = E[E.yr < Y]; te = E[E.yr == Y]
        fu = [c for c in feats_ if tr[c].notna().sum() > 1000]
        m = mdl().fit(tr[fu].values, np.clip(tr.ret.values, -0.02, 0.02))
        out.append(te.assign(pred=m.predict(te[fu].values)))
    return pd.concat(out)


def gen_events(d, F, piv, s, B, kx, mode, g, H):
    a, b, r, m = run(d["o"], d["h"], d["l"], d["c"], d["rv1"], B, 1, kx, mode, g, H, EPS, FM, FT)
    if len(a) == 0: return None
    fm1 = a - 1; s0 = (a // B) * B
    x = kx * d["rv1"][s0] * np.sqrt(B)
    T = d["o"][s0] * (1 - x)
    df = pd.DataFrame({"sym": s, "B": B, "k": kx, "f": a, "x": b, "ret": r, "mae": m,
                        "ts": d["idx"][a], "el": (a % B) / B, "pre": (d["c"][fm1] / d["o"][s0] - 1) / x,
                        "tpiv": np.log(T / piv[fm1]) / x, "qh": (a % 15).astype(float), "hr0": (a % 60).astype(float)})
    for kk, v in F.items(): df[kk] = v[fm1]
    return df


def weekly_monthly(d, a, b):
    """주간(W-SUN)·월간 지표. d: 일간 계좌 수익률(None이면 파산)."""
    if d is None:
        return dict(posw=0.0, medw=-100.0, p25w=-100.0, worstw=-100.0, shw=np.nan, posm=0.0, medm=-100.0, bankrupt=True)
    x = d[(d.index >= pd.Timestamp(a, tz="UTC")) & (d.index < pd.Timestamp(b, tz="UTC"))]
    if len(x) == 0:
        return dict(posw=np.nan, medw=np.nan, p25w=np.nan, worstw=np.nan, shw=np.nan, posm=np.nan, medm=np.nan, bankrupt=False)
    eq = (1 + x.clip(lower=-0.999)).cumprod()
    wk_eq = eq.resample("W-SUN").last(); wk = wk_eq.pct_change(); wk.iloc[0] = wk_eq.iloc[0] - 1
    mo_eq = eq.resample("ME").last(); mo = mo_eq.pct_change(); mo.iloc[0] = mo_eq.iloc[0] - 1
    return dict(posw=(wk > 0).mean() * 100, medw=wk.median() * 100, p25w=wk.quantile(0.25) * 100, worstw=wk.min() * 100,
                shw=(wk.mean() / wk.std()) if wk.std() > 0 else np.nan, posm=(mo > 0).mean() * 100, medm=mo.median() * 100, bankrupt=False)


if __name__ == "__main__":
    t_ = time.time()

    # ---- 0. 기준선 재현 점검 (딥 문턱10bp·w3N12 + 돌파 L2) ----
    Wold = pd.read_parquet(R / "data" / "r32_wf.parquet"); Wold["ts"] = pd.DatetimeIndex(Wold.ts)
    Dold = V.dip_events(Wold, 0.001)[G.COLS]
    BK = G.std_breakout()
    db, pkb = G.run_multi(pd.concat([Dold, BK], ignore_index=True), 3, 12, {1: 2 / 48})
    m1b = weekly_monthly(db, "2022-01-01", "2024-01-01"); m2b = weekly_monthly(db, "2025-01-01", "2026-09-26")
    P(f"[기준선 재현] 2022-24: 양주{m1b['posw']:.0f}% 중앙{m1b['medw']:+.1f}% 25pct{m1b['p25w']:+.1f}% 최악{m1b['worstw']:+.1f}% 샤프{m1b['shw']:.2f} "
      f"| 2025-26: 양주{m2b['posw']:.0f}% 중앙{m2b['medw']:+.1f}% 25pct{m2b['p25w']:+.1f}% 최악{m2b['worstw']:+.1f}% 샤프{m2b['shw']:.2f} 양월{m2b['posm']:.0f}% 월중앙{m2b['medm']:+.1f}% ({time.time()-t_:.0f}s)")
    P("  목표(보고된 기준선) 2022-24: 양주65% 중앙+4.9% 25pct-1.3% 최악-70% 샤프0.32 | 2025-26: 양주67% 중앙+2.3% 25pct-1.5% 최악-40% 샤프0.34 양월67% 월중앙+14%")

    # ---- 1. 변형별 사건 생성 (종목별 처리로 메모리 관리) ----
    events_parts = {name: [] for name in VARIANTS}
    raw_stats = {name: [] for name in VARIANTS}
    for s in SYM:
        d, F = feats(s); X, piv = extra(s, d); F.update(X)
        P(f"{s} 특징 {len(F)}개 준비 ({time.time()-t_:.0f}s)")
        for name, cfg in VARIANTS.items():
            rows = []
            for B in cfg["Bs"]:
                for kx in cfg["Ks"]:
                    df = gen_events(d, F, piv, s, B, kx, cfg["mode"], cfg["g"], cfg["H"])
                    if df is not None: rows.append(df)
            if not rows: continue
            Ev = pd.concat(rows, ignore_index=True)
            for c_ in Ev.columns:
                if Ev[c_].dtype == np.float64: Ev[c_] = Ev[c_].astype(np.float32)
            events_parts[name].append(Ev)
            raw_stats[name].append((len(Ev), (Ev.ret > 0).sum(), Ev.ret.sum(), Ev.ret.values))
            del rows, Ev
        del d, F, X, piv; gc.collect()
        P(f"  {s} 변형 {len(VARIANTS)}개 사건 생성 완료 ({time.time()-t_:.0f}s)")

    events_full = {}
    for name in VARIANTS:
        E = pd.concat(events_parts[name], ignore_index=True)
        E = E.replace([np.inf, -np.inf], np.nan)
        E["ts"] = pd.DatetimeIndex(E.ts); E["yr"] = E.ts.dt.year
        E["isbtc"] = (E.sym == "BTCUSDT").astype(float)
        events_full[name] = E
        n = len(E); hit = (E.ret > 0).mean() * 100; mbp = E.ret.mean() * 1e4; p5 = np.nanpercentile(E.ret.values, 5) * 1e4
        P(f"[{name}] 원 사건 n={n:,} 승률{hit:.1f}% 평균{mbp:+.2f}bp 5퍼센타일{p5:+.1f}bp")
    del events_parts; gc.collect()

    # ---- 2. 재학습 + 엔진 ----
    rows_acc = []
    for name, E in events_full.items():
        feats_ = [c for c in E.columns if c not in META]
        W = walk(E, feats_)
        if name == "E0":
            old = pd.read_parquet(R / "data" / "r32_wf.parquet", columns=["sym", "f", "B", "k", "pred"])
            mg = W.merge(old, on=["sym", "f", "B", "k"], suffixes=("", "_old"))
            corr = np.corrcoef(mg.pred, mg.pred_old)[0, 1] if len(mg) > 10 else np.nan
            P(f"[E0 재현 점검] n={len(mg):,} (원본 {len(old):,}) corr(pred,pred_old)={corr:.4f}")
        for pc_bp in (5, 10, 15):
            pc = pc_bp / 1e4
            Dv = V.dip_events(W, pc)
            Ev = pd.concat([Dv[G.COLS], BK], ignore_index=True)
            for w_, N_, lab in ((3, 12, "w3N12"), (2, 8, "w2N8")):
                d, pk = G.run_multi(Ev, w_, N_, {1: 2 / 48})
                m1 = weekly_monthly(d, "2022-01-01", "2024-01-01")
                m2 = weekly_monthly(d, "2025-01-01", "2026-09-26")
                rows_acc.append(dict(variant=name, pc=pc_bp, size=lab, n_dip=len(Dv),
                                     **{f"a_{k}": v for k, v in m1.items()}, **{f"b_{k}": v for k, v in m2.items()}))
            del Dv, Ev
        P(f"  [{name}] 재학습+엔진 완료 ({time.time()-t_:.0f}s)")
        del W; gc.collect()

    O = pd.DataFrame(rows_acc)
    O.to_parquet(R / "data" / "r126.parquet", index=False)
    pd.set_option("display.width", 300); pd.set_option("display.max_rows", 200)
    P("\n===== 전 조합 (2022-24 / 2025-26)")
    show = O[["variant", "pc", "size", "n_dip", "a_posw", "a_medw", "a_p25w", "a_worstw", "a_shw",
              "b_posw", "b_medw", "b_p25w", "b_worstw", "b_shw", "b_posm", "b_medm"]].round(2)
    P(show.to_string(index=False))

    # ---- 3. 선택 (2022-24 주간 샤프 1순위, 양주 비율 동률 처리) ----
    cand = O[~O.a_bankrupt & ~O.b_bankrupt].dropna(subset=["a_shw"])
    if len(cand):
        best = cand.sort_values(["a_shw", "a_posw"], ascending=False).iloc[0]
        P(f"\n[선택] {best.variant} pc{best.pc}bp {best['size']}: 2022-24 샤프{best.a_shw:.2f} 양주{best.a_posw:.0f}% | "
          f"2025-26 양주{best.b_posw:.0f}% 중앙{best.b_medw:+.1f}% 25pct{best.b_p25w:+.1f}% 최악{best.b_worstw:+.1f}% 샤프{best.b_shw:.2f} 양월{best.b_posm:.0f}% 월중앙{best.b_medm:+.1f}%")
        # 최선 변형 딥 사건 재생성·저장 (엔진 표준 컬럼)
        Eb = events_full[best.variant]
        feats_ = [c for c in Eb.columns if c not in META]
        Wb = walk(Eb, feats_)
        Db = V.dip_events(Wb, best.pc / 1e4)[G.COLS]
        Db.to_parquet(R / "data" / "r126_best_events.parquet", index=False)
        P(f"저장 data/r126_best_events.parquet {Db.shape}")
    else:
        P("\n[선택] 유효 조합 없음(전부 파산 또는 결측)")
    P(f"완료 ({time.time()-t_:.0f}s)")
