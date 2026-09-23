"""R53 — 방향 × 결판 확률 → 기대값(EV)으로 진입 선택 (R50의 무결판 문제 대응).

두 모델(연도별 워크포워드, 평가 연도는 과거로만 학습):
  방향: P(위 먼저 | 결판)  — 결판 표본만 학습 (R50과 같음), 시드 3개 평균
  결판: P(12시간 안에 ±d 중 하나라도 닿음) — 전 표본 학습
기대값(자본 대비, R47 구조: 증거금 50% × 30배 = 명목 15배):
  EV_롱 = P결판·(P위·W − (1−P위)·Lo) + (1−P결판)·T,  EV_숏 대칭
  W = 15·(d − 테이커 − 메이커), Lo = 15·(d + 2·테이커), T = 시간 청산 = 왕복 수수료 손실(보수적 가정)
선택(인과적): max(EV)가 직전 30일 자기 분포 상위 q 이상 & EV > 0, 한 번에 한 포지션 → simulate_entries
사용: python research/r53_ev_select.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from r50_direction_model import D, FEAT
from r47_x30_frame import simulate_entries, stat, CT, CM, NOT
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)


def clf(seed, leaf=1000):
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.03, max_leaf_nodes=31,
                                          min_samples_leaf=leaf, l2_regularization=1.0, random_state=seed)


def walk(y):
    out = []
    for Y in range(2022, 2027):
        tr = D[D.yr < Y]; te = D[D.yr == Y].copy()
        fu = [c for c in FEAT if tr[c].notna().sum() > 5000]
        dec = tr[tr[y] != 0]
        pu = np.mean([clf(sd).fit(dec[fu].values, (dec[y].values == 1).astype(int)).predict_proba(te[fu].values)[:, 1]
                      for sd in (0, 1, 2)], axis=0)
        pr = clf(0, 2000).fit(tr[fu].values, (tr[y].values != 0).astype(int)).predict_proba(te[fu].values)[:, 1]
        te["pu"] = pu; te["pr"] = pr
        out.append(te)
        P(f"  {Y} 학습 완료 (결판 기저율 학습 {(tr[y]!=0).mean():.2f})")
    return pd.concat(out).sort_values(["t", "sym"]).reset_index(drop=True)


def choose(W, d, q, Tl, Ts, rule):
    Wn, Lo = NOT * (d - CT - CM), NOT * (d + 2 * CT)
    if rule == "ev":
        evl = W.pr * (W.pu * Wn - (1 - W.pu) * Lo) + (1 - W.pr) * Tl
        evs = W.pr * ((1 - W.pu) * Wn - W.pu * Lo) + (1 - W.pr) * Ts
    else:                                                          # 방향만 (R50 방식 비교용)
        evl = W.pu - 0.5; evs = 0.5 - W.pu
    W = W.assign(best=np.maximum(evl, evs), side=np.where(evl >= evs, 1, -1))
    thr = W.set_index("ts").best.rolling("30D").quantile(1 - q).shift(1).values
    S = W[(W.best >= thr) & ((W.best > 0) | (rule != "ev"))].sort_values("t")
    keep = []; busy = -1
    for r in S.itertuples():
        if r.t < busy: continue
        keep.append(r.Index); busy = r.t + 720
    return S.loc[keep]


for y, d in (("y15", 0.015), ("y10", 0.01)):
    P(f"\n===== ±{d*100:.1f}% / 12시간")
    W = walk(y)
    Tl = Ts = -NOT * 2 * CT                                        # 시간 청산 = 왕복 수수료만 잃는다고 가정(보수적)
    P(f"  시간 청산 가정 T = {Tl*100:.2f}% (수수료만)")
    wk = (W.ts.max() - W.ts.min()).days / 7
    for rule in ("dir", "ev"):
        for q in (0.01, 0.003, 0.001):
            S = choose(W, d, q, Tl, Ts, rule)
            if S.empty:
                P(f"  {rule} q={q*100:.1f}%: 거래 없음"); continue
            dec = S[S[y] != 0]
            acc = (dec[y] == dec.side).mean() if len(dec) else np.nan
            eq, T = simulate_entries(S[["ts", "sym", "side"]], d, 720)
            cg, mo, mmed, mdd, fin = stat(eq)
            by = dec.groupby("yr").apply(lambda g: (g[y] == g.side).mean())
            tm = T[T.kind == 0]
            P(f"  {'방향만' if rule=='dir' else '기대값'} q={q*100:.1f}%: 주 {len(S)/wk:4.1f}회 적중(결판) {acc:.3f} 무결판 {(S[y]==0).mean():.2f} "
              f"| 계좌 월 {mo*100:+6.1f}% MDD {mdd*100:4.0f}% 최종 {fin:.3g}배 | 시간청산 평균 {tm.r.mean()*1e4 if len(tm) else 0:+.0f}bp(가격) "
              f"| " + " ".join(f"{yy%100}:{v:.2f}" for yy, v in by.items()))
    W.to_parquet(R / "data" / f"r53_wf_{y}.parquet", index=False)
