"""R92 — 4시간 ±2% 타이밍 확률 p(R90, 방향 중립 변동성 예보)를 쓰는 전략들.

p 국면: 종목별 직전 90일 p 분포에서의 순위 pr (인과적, 현재 값 제외 창 비교). 구간 Q0 <20% · Q1 20~50 · Q2 50~80 · Q3 80~95 · Q4 ≥95 · ALL.

A 되돌림·분할 그리드(방향 중립, 지정가): 15분 격자에서 종목이 비어 있으면 기준가 ref=c[t−1] 위아래로 n단 지정가
  (y, y+y/2, y+y ...)를 60분 유효로 깐다(각 1/n). 한쪽 첫 체결 시 반대쪽 취소, 같은 쪽 나머지 단은 유지.
  청산: 평균 진입가에서 ref 쪽으로 되돌림 목표(tg=0: ref까지, tg=0.5: 절반) 지정가 / 손절 = 최深단 + sd(테이커) / 240분 시간 청산(테이커).
  체결 규칙: 지정가는 5bp 관통해야 체결, 목표 지정가도 5bp 관통. 체결 봉에서는 목표 불인정·손절만 인정(최악 가정).
  비용: 지정가 2bp, 테이커 7bp. 거래당 수익 = 명목 1(전 단 체결 시) 기준.
  문헌: 그리드는 무조건이면 기대값 0(Dynamic Grid Trading, arXiv 2506.11921) — 변동성 예보가 '횡보 국면'을 골라 주면 양수가 될 수 있는가.
  선택: 2022~23 샤프 최상 구성 → 2024~26 검증.
B z0.31 딥 × p: 딥 사건 시점 p 국면별 거래당 수익(진단), 그리고 배율 g(pr)을 V3에 곱해 R40 계좌(w2.5N10·킬 3%/12h)로 비교.
사용: python research/r92_vol_timing.py [A|B]"""
import sys, warnings, time
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
CT, CM, PEN = 7e-4, 2e-4, 5e-4
BK = {"Q0": (0, .2), "Q1": (.2, .5), "Q2": (.5, .8), "Q3": (.8, .95), "Q4": (.95, 1.01), "LO": (0, .5), "HI": (.8, 1.01), "ALL": (0, 1.01)}


def load_p():
    W = pd.read_parquet(R / "data" / "r90_wf.parquet").sort_values(["sym", "t"]).reset_index(drop=True)
    W["ts"] = pd.DatetimeIndex(W.ts)
    out = []
    for s, g in W.groupby("sym"):
        g = g.copy(); p = g.p.values; pr = np.full(len(p), np.nan); n = 96 * 90
        for i in range(n // 4, len(p)):
            h = p[max(0, i - n):i]; pr[i] = (h < p[i]).mean()
        g["pr"] = pr; out.append(g)
    return pd.concat(out).dropna(subset=["pr"])


@numba.njit(cache=True)
def grid(t, on, o, h, l, c, y, n, tg, sd, W, H):
    N = len(c); E = np.zeros(len(t), np.int64); Rr = np.zeros(len(t)); m = 0; busy = -1
    lv = np.zeros(n)
    for q in range(len(t)):
        i = t[q]
        if not on[q] or i <= busy or i + W + H >= N: continue
        ref = c[i - 1]
        for k in range(n): lv[k] = y * (1 + 0.5 * k)
        stopd = lv[n - 1] + sd
        side = 0; f = -1
        for j in range(i, i + W):
            dn = l[j] <= ref * (1 - lv[0]) * (1 - PEN_); up = h[j] >= ref * (1 + lv[0]) * (1 + PEN_)
            if dn and up: break
            if dn: side = 1; f = j; break
            if up: side = -1; f = j; break
        if f < 0: continue
        # 롱(side=1): 매수 단들 ref(1−lv), 손절 ref(1−stopd)
        filled = np.zeros(n, np.bool_); qty = 0.0; cost = 0.0; r = np.nan; j = f; exp_ = i + W
        while j < f + H:
            for k in range(n):
                if not filled[k] and j < exp_:
                    px = ref * (1 - side * lv[k])
                    if (side > 0 and l[j] <= px * (1 - PEN_)) or (side < 0 and h[j] >= px * (1 + PEN_)):
                        filled[k] = True; qty += 1.0 / n; cost += px / n
            if qty == 0: j += 1; continue
            avg = cost / qty
            stp = ref * (1 - side * stopd); tgt = avg + (ref - avg) * (1 - tg)
            if (side > 0 and l[j] <= stp) or (side < 0 and h[j] >= stp):
                ex = min(stp, o[j]) if side > 0 else max(stp, o[j])
                r = qty * side * (ex / avg - 1) - qty * (CM_ + CT_); break
            if j > f and ((side > 0 and h[j] >= tgt * (1 + PEN_)) or (side < 0 and l[j] <= tgt * (1 - PEN_))):
                r = qty * side * (tgt / avg - 1) - qty * 2 * CM_; break
            j += 1
        if qty == 0: busy = j; continue
        if np.isnan(r): r = qty * side * (c[j - 1] / avg - 1) - qty * (CM_ + CT_)
        E[m] = f; Rr[m] = r; m += 1; busy = j
    return E[:m], Rr[:m]


CT_, CM_, PEN_ = CT, CM, PEN


def daily(T, a, b):
    d = T.groupby(T.t.dt.floor("D")).r.sum()
    return d.reindex(pd.date_range(a, b, freq="D", tz="UTC")).fillna(0)


def modeA(W, K):
    rows = []
    for y in (0.003, 0.005, 0.0075, 0.01, 0.015):
        for n in (1, 3):
            for tg in (0.0, 0.5):
                for sd in (0.005, 0.01):
                    for bk, (a, b) in BK.items():
                        parts = []
                        for s in K:
                            g = W[W.sym == s]; on = ((g.pr >= a) & (g.pr < b)).values; k = K[s]
                            E, Rr = grid(g.t.values.astype(np.int64), on, k.open.values, k.high.values, k.low.values, k.close.values, y, n, tg, sd, 60, 240)
                            parts.append(pd.DataFrame({"t": pd.to_datetime(k.ts.values[E], utc=True), "r": Rr}))
                        T = pd.concat(parts)
                        if len(T) < 50: continue
                        tr = T[T.t < "2024-01-01"]; te = T[T.t >= "2024-01-01"]
                        dtr = daily(tr, "2022-01-01", "2023-12-31"); dte = daily(te, "2024-01-01", "2026-09-19")
                        rows.append(dict(y=y, n=n, tg=tg, sd=sd, bk=bk, ntr=len(tr) / 730, bptr=tr.r.mean() * 1e4, shtr=dtr.mean() / dtr.std() * np.sqrt(365),
                                         nte=len(te) / 993, bpte=te.r.mean() * 1e4, shte=dte.mean() / dte.std() * np.sqrt(365), T=T))
        P(f"  y={y*100:.2f}% 완료")
    O = pd.DataFrame([{k: v for k, v in r.items() if k != "T"} for r in rows])
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400)
    P("\n===== 구간별 평균 (모든 y·n·tg·sd): 거래당 bp 학습(22~23) / 평가(24~26)")
    P(O.groupby("bk")[["bptr", "bpte", "shtr", "shte", "ntr"]].mean().round(2).to_string())
    P("\n===== 학습 샤프 상위 20 구성 → 평가")
    P(O.sort_values("shtr", ascending=False).head(20).round(2).to_string())
    P("\n===== 평가 샤프 상위 10 (참고, 선택 편향)")
    P(O.sort_values("shte", ascending=False).head(10).round(2).to_string())
    best = rows[int(O.shtr.idxmax())]; T = best["T"]
    P(f"\n===== 학습 최선 {dict((k, best[k]) for k in ('y','n','tg','sd','bk'))} 3단 레버리지 (명목 = 자본 × L, 종목별)")
    for a, b, lab in (("2022-01-01", "2026-09-19", "전체"), ("2024-01-01", "2026-09-19", "평가")):
        d = daily(T[T.t >= a], a, b)
        for L in (1, 5, 15):
            eq = (1 + (d * L).clip(lower=-0.99)).cumprod(); yrs = len(d) / 365.25
            P(f"  {lab} ×{L:2d}: 월 {(eq.iloc[-1]**(1/12/yrs)-1)*100:+.2f}% MDD {(eq/eq.cummax()-1).min()*100:.0f}% | 연도별 " +
              " ".join(f"{x.year%100}:{v*100:+.0f}%" for x, v in eq.resample('YE').last().pct_change().fillna(eq.resample('YE').last().iloc[0]-1).items()))
    O.to_parquet(R / "data" / "r92a.parquet", index=False)


def modeB(W):
    import r40_regime_leverage as RL, r80_max_return as R80
    RL.MMR = 0.0065; D = R80.dip_frame(0.0025)
    parts = []
    for s, g in D.groupby("sym"):
        g = g.sort_values("f"); w = W[W.sym == s].sort_values("t")[["t", "pr", "p"]]
        parts.append(pd.merge_asof(g, w, left_on="f", right_on="t", direction="backward"))
    D = pd.concat(parts).sort_values(["f", "sym"]).reset_index(drop=True); D["pr"] = D.pr.fillna(0.5)
    P("===== z0.31 딥 사건 × p 국면 (거래당 bp, 연도별)")
    for bk, (a, b) in BK.items():
        g = D[(D.pr >= a) & (D.pr < b)]
        P(f"  {bk:3s}: n={len(g):5d} 거래당 {g.ret.mean()*1e4:+6.1f}bp 승률 {(g.ret>0).mean():.3f} 최대역행 {g.mae.mean()*1e4:5.0f}bp | " +
          " ".join(f"{y%100}:{v*1e4:+.0f}" for y, v in g.groupby(g.ts.dt.year).ret.mean().items()))
    base = np.minimum(np.clip(D.vreg.values ** 1.5, 0.3, 4) * np.clip(D.pred.values / 0.001, 0.5, 3), 6)
    tr = D[D.ts < "2024-01-01"]; edges = [0, .2, .5, .8, .95, 1.01]
    D["b"] = np.digitize(D.pr, edges[1:-1]); tr = D[D.ts < "2024-01-01"]
    wt = (tr.groupby("b").ret.mean() / tr.groupby("b").mae.mean()).clip(lower=0); wt = wt / wt[D.b.value_counts().index].mul(D.b.value_counts()).sum() * len(D)
    P("학습(22~23) 구간 가중(수익/역행, 평균 1): " + " ".join(f"Q{int(k)}:{v:.2f}" for k, v in wt.items()))
    G = [("기준", lambda x: np.ones_like(x)), ("Q4×2", lambda x: np.where(x >= .95, 2.0, 1.0)), ("Q4×3", lambda x: np.where(x >= .95, 3.0, 1.0)),
         ("Q4만", lambda x: np.where(x >= .95, 1.0, 0.0)), ("Q4만×2", lambda x: np.where(x >= .95, 2.0, 0.0)), ("Q3+Q4만", lambda x: np.where(x >= .8, 1.0, 0.0)),
         ("Q4 제외", lambda x: np.where(x >= .95, 0.0, 1.0)), ("학습 가중", lambda x: wt.reindex(np.digitize(x, edges[1:-1])).fillna(0).values)]
    P("\n===== 배율 g(pr) × V3, R40 계좌 킬 3%/12h (학습 22~23 / 평가 24~26) — 3단 w1N4 · w1.5N6 · w2.5N10")
    for w_, N_ in ((1, 4), (1.5, 6), (2.5, 10)):
        for nm, gfun in G:
            mult = np.maximum(base * gfun(D.pr.values), 1e-9)
            res = RL.sim(D, w_, N_, mult, X=0.03, H=720)
            if res is None: P(f"  w{w_}N{N_} {nm:8s}: 파산"); continue
            dr = R80.daily_from_eq(res[0]); out = []
            for a_, b_ in (("2022-01-01", "2024-01-01"), ("2024-01-01", "2026-09-20")):
                x = dr[(dr.index >= a_) & (dr.index < b_)]; m = R80.metrics(dr, a_, b_)
                out.append(f"샤프 {x.mean()/x.std()*np.sqrt(365):.2f} 월 {m['mo']:+.1f}% MDD {m['mdd']:.0f}%")
            P(f"  w{w_}N{N_} {nm:8s} 학습 {out[0]} | 평가 {out[1]} | 최악여유 {res[2]*100:.0f}%")


if __name__ == "__main__":
    t_ = time.time(); W = load_p(); P(f"p 순위 준비 {len(W):,} ({time.time()-t_:.0f}s)")
    mode = sys.argv[1] if len(sys.argv) > 1 else "A"
    if mode == "A":
        modeA(W, {s: store.load("klines_1m", s) for s in ("BTCUSDT", "ETHUSDT")})
    else:
        modeB(W)
    (R / "logs" / f"r92{mode}.summary").write_text("done\n")
