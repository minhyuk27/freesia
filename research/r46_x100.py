"""R46 — 100배 챌린지 (사용자 규칙, 2026-09-24).

규칙: 진입마다 자본의 50%를 증거금으로 100배(명목 = 자본 × 50). 12시간 뒤 시장가 청산, 또는 강제청산.
  강제청산: 마크가격이 진입가 × (1 ∓ (1/100 − MMR)) 에 닿으면 (MMR 0.4% → 거리 0.6%). 손실 = 증거금 전액(자본 × 0.5).
  비용: 진입 테이커 5bp, 청산 테이커 5bp (명목 기준 → 자본의 50×). 한 번에 한 포지션.
  자본 배율 m: 청산 = 0.5 − 50×수수료진입, 생존 = 1 + 50×(방향×12시간 수익 − 왕복 수수료)
  변형 TP: 가격이 +tp 에 먼저 닿으면 지정가 익절(메이커 2bp). 같은 분에 청산선·익절선 모두 닿으면 청산으로 처리(보수).
진입 선택: 15분마다 후보(BTC·ETH × 롱·숏). 특징 = R27 차트·거래량·파생 + R32 오더플로우·호가·청산 개념 (f−1 분까지).
  모델: 연도별 확장창 부스팅 회귀, 목표 = ln(m). 평가 연도는 항상 과거로만 학습.
  진입 규칙: 모델 예측이 직전 30일 자기 예측 분포의 상위 q (주 평균 3회가 되게) 이상이고 포지션이 없을 때.
위약: 같은 빈도의 무작위 진입.
사용: python research/r46_x100.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
from sklearn.ensemble import HistGradientBoostingRegressor
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store, costs
from r27_dip_events import feats
from r32_dip_events import extra
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
LEV = 100.0; MARGIN = 0.5; MMR = costs.MMR_TIER1; H = 720; STEP = 15
FT = costs.TAKER; FM = costs.MAKER
NOTIONAL = LEV * MARGIN
TPS = (0.0, 0.005, 0.01, 0.02)          # 0 = 익절 없음(사용자 원안)


@numba.njit(cache=True)
def label(o, h, l, c, mh, ml, idx, side, liqd, tp, H):
    n = len(idx); out = np.empty(n); liq = np.zeros(n, np.bool_)
    for q in range(n):
        t = idx[q]; e = o[t]
        lv = e * (1 - side * liqd); tv = e * (1 + side * tp)
        done = False
        for j in range(t, t + H):
            hit_l = (ml[j] <= lv) if side > 0 else (mh[j] >= lv)
            if hit_l:
                out[q] = -1.0; liq[q] = True; done = True; break
            if tp > 0:
                hit_t = (h[j] >= tv) if side > 0 else (l[j] <= tv)
                if hit_t:
                    out[q] = tp; done = True; break
        if not done:
            out[q] = side * (c[t + H - 1] / e - 1)
    return out, liq


def mult(r, liq, tp_hit_maker):
    """자본 배율. 청산 = 증거금 전액 + 진입 수수료 손실. 생존 = 명목 × (수익 − 왕복 수수료)."""
    fee_in = FT
    fee_out = np.where(tp_hit_maker, FM, FT)
    m = 1 + NOTIONAL * (r - fee_in - fee_out)
    return np.where(liq, 1 - MARGIN - NOTIONAL * fee_in, np.maximum(m, 1 - MARGIN - NOTIONAL * fee_in))


def build():
    rows = []
    for s in ("BTCUSDT", "ETHUSDT"):
        d, F = feats(s)
        X, _ = extra(s, d)
        F.update(X)
        mk = store.load("mark_1m", s).set_index("ts").reindex(d["idx"]).ffill()
        mh, ml = mk.high.values, mk.low.values
        n = len(d["c"])
        idx = np.arange(1500, n - H - 1, STEP)
        liqd = 1 / LEV - MMR
        base = pd.DataFrame({"sym": s, "t": idx, "ts": d["idx"][idx]})
        for k_, v in F.items():
            base[k_] = v[idx - 1].astype(np.float32)
        for side in (1, -1):
            df = base.copy(); df["side"] = side
            for tp in TPS:
                r, lq = label(d["o"], d["h"], d["l"], d["c"], mh, ml, idx, side, liqd, tp, H)
                tph = (tp > 0) & (np.abs(r - tp) < 1e-12) & ~lq
                df[f"m{tp}"] = mult(r, lq, tph).astype(np.float32)
                df[f"liq{tp}"] = lq
                if tp == 0:
                    df["r12"] = r.astype(np.float32)
            rows.append(df)
        P(f"{s} 후보 {len(idx):,} × 2방향")
    return pd.concat(rows, ignore_index=True)


def account(sel, mcol):
    """한 번에 한 포지션. sel은 시간순, t(분 인덱스)·sym 포함. 같은 분 인덱스는 두 종목 공통 시계."""
    cap = 1.0; busy = -1; pts = []
    for _, r in sel.iterrows():
        if r.t < busy: continue
        cap *= r[mcol]; busy = r.t + H
        pts.append((r.ts, cap))
        if cap < 1e-6: break
    if not pts: return None
    return pd.Series([p[1] for p in pts], index=pd.DatetimeIndex([p[0] for p in pts]))


def summ(eq, lab, n_tr, weeks):
    if eq is None:
        P(f"  {lab:40s} 거래 없음"); return
    yrs = weeks / 52.18
    fin = eq.iloc[-1]
    cg = fin ** (1 / yrs) - 1 if fin > 0 else -1
    d = eq.resample("D").last().ffill()
    mdd = float((d / d.cummax() - 1).min()) * 100
    mon = d.resample("ME").last().pct_change().dropna() * 100
    P(f"  {lab:40s} 주 {n_tr/weeks:4.1f}회 | 최종 {fin:10.3g}배 연 {cg*100:+9.0f}% | 월중앙 {mon.median():+6.1f}% "
      f"월10/90 {mon.quantile(.1):+6.1f}/{mon.quantile(.9):+7.1f} | MDD {mdd:6.1f}%")


if __name__ == "__main__":
    E = build()
    E["yr"] = E.ts.dt.year
    E.to_parquet(R / "data" / "r46_cands.parquet", index=False)
    P(f"\n후보 {len(E):,} (15분 간격, 2종목×2방향)")
    P("\n[0] 무작위 진입 기저율 (평가 2021~)")
    for side in (1, -1):
        g = E[(E.side == side) & (E.yr >= 2021)]
        for tp in TPS:
            m = g[f"m{tp}"]
            P(f"  {'롱' if side>0 else '숏'} TP{tp*100:.1f}%: 청산 {g[f'liq{tp}'].mean():.3f} | 배율 평균 {m.mean():.3f} "
              f"ln평균 {np.log(m).mean():+.4f} 이익확률 {(m>1).mean():.3f}")
    FEAT = [c for c in E.columns if c not in {"sym", "t", "ts", "side", "r12", "yr"} and not c.startswith("m") and not c.startswith("liq")]
    FEAT = FEAT + ["side"]
    E["isbtc"] = (E.sym == "BTCUSDT").astype(float); FEAT.append("isbtc")
    E = E.replace([np.inf, -np.inf], np.nan)
    Q = 1 - 3 / (7 * 96 * 4)                     # 주 3회 ≈ 후보 상위 0.11%
    for tp in TPS:
        mc = f"m{tp}"
        P(f"\n[TP {tp*100:.1f}%] 워크포워드 (학습 과거 전체 → 연도 평가), 목표 ln(m)")
        W = []
        for Y in range(2021, 2027):
            tr = E[E.yr < Y]; te = E[E.yr == Y]
            fu = [c for c in FEAT if tr[c].notna().sum() > 1000]
            mdl = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, max_leaf_nodes=31,
                                                min_samples_leaf=2000, l2_regularization=1.0, random_state=0)
            mdl.fit(tr[fu].values, np.log(tr[mc].values.astype(float)))
            W.append(te.assign(pred=mdl.predict(te[fu].values)))
        W = pd.concat(W).sort_values(["t", "sym", "side"]).reset_index(drop=True)
        # 인과적 문턱: 직전 30일 자기 예측의 Q 분위
        W["thr"] = W.set_index("ts").pred.rolling("30D").quantile(Q).shift(1).values
        cand = W[(W.pred >= W.thr) & W.thr.notna()]
        weeks = (W.ts.max() - W.ts.min()).days / 7
        eq = account(cand, mc)
        n_tr = 0 if eq is None else len(eq)
        sel = cand.copy()
        P(f"  선택 후보 {len(cand)}개 → 실제 진입 {n_tr}회 (한 번에 한 포지션)")
        if eq is not None:
            taken = []; busy = -1
            for _, r in cand.iterrows():
                if r.t < busy: continue
                taken.append(r); busy = r.t + H
            T = pd.DataFrame(taken)
            P(f"  진입 거래: 청산 {T[f'liq{tp}'].mean():.3f} 이익 {(T[mc]>1).mean():.3f} 배율 평균 {T[mc].mean():.3f} "
              f"ln평균 {np.log(T[mc]).mean():+.4f} | 롱 {(T.side>0).mean():.2f} | 연도별 ln평균 " +
              " ".join(f"{y%100}{np.log(g[mc]).mean():+.3f}({len(g)})" for y, g in T.groupby(T.ts.dt.year)))
        summ(eq, "모델 선택 (원 규칙)", n_tr, weeks)
        rng = np.random.default_rng(0)
        for rep in range(3):
            rs = W.sample(n=max(len(cand), 1), random_state=rep).sort_values("t")
            eqr = account(rs, mc)
            summ(eqr, f"위약: 무작위 같은 빈도 #{rep}", 0 if eqr is None else len(eqr), weeks)
