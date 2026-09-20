"""강한 1차 신호에 메타 라벨링 — z0.6(유동성 충격) / z0.7-A(평활 괴리 극단).
2차 모델은 '이 베팅을 취할 것인가'만 학습. 롤링, 임계는 학습 구간 분포.
사용: python research/meta_strong.py"""
import sys, warnings, gc
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from bigfeat import build_features, SYM
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import exit_tier as X
from absfloor import met
from z06b import build as build_z6
from z07_dev import build as build_z7
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
STEP = 15
TE = pd.Timestamp("2024-01-01", tz="UTC"); REC = pd.Timestamp("2025-01-01", tz="UTC")

P("특징 준비...")
FEAT = {}
for s in SYM:
    Xf, c = build_features(s, step=STEP)
    FEAT[s] = dict(X=Xf.values.astype(np.float32), idx=Xf.index, cols=list(Xf.columns))
    P(f"  {s}: {Xf.shape}")
    del Xf; gc.collect()


def attach(E):
    """각 이벤트에 진입 직전 15분 격자의 특징을 붙인다 (룩어헤드 없음)"""
    out = []
    for x in E:
        s = x[5][0]; f = FEAT[s]
        j = f["idx"].searchsorted(x[4], side="right") - 1
        if j < 0: continue
        v = f["X"][j]
        if not np.isfinite(v).all(): continue
        out.append((x, v))
    return out


def meta(E, win_days=365, refit_days=30, ncol=15, meta_q=0.5, label="plain", start="2021-06-01", C=0.05):
    AV = attach(E)
    if len(AV) < 200: return []
    ts = pd.DatetimeIndex([a[0][4] for a in AV])
    Xm = np.array([a[1] for a in AV], np.float32)
    ret = np.array([a[0][2] for a in AV])
    y = (ret > 0).astype(int) if label == "plain" else (ret > 0.001).astype(int)
    keep = []
    ends = pd.date_range(pd.Timestamp(start, tz="UTC"), ts[-1], freq=f"{refit_days}D", tz="UTC")
    for e in ends:
        s0 = e - pd.Timedelta(days=win_days)
        tr = (ts >= s0) & (ts < e); te = (ts >= e) & (ts < e+pd.Timedelta(days=refit_days))
        if tr.sum() < 80 or te.sum() < 1: continue
        if len(np.unique(y[tr])) < 2: continue
        Xtr = Xm[tr]
        cr = np.abs(np.array([np.corrcoef(Xtr[:, j], ret[tr])[0, 1] if Xtr[:, j].std() > 0 else 0
                              for j in range(Xm.shape[1])]))
        cr[~np.isfinite(cr)] = 0
        cols = np.argsort(-cr)[:ncol]
        try:
            sc = StandardScaler().fit(Xtr[:, cols])
            mdl = LogisticRegression(max_iter=300, C=C).fit(sc.transform(Xtr[:, cols]), y[tr])
            ptr = mdl.predict_proba(sc.transform(Xtr[:, cols]))[:, 1]
            th = float(np.quantile(ptr, meta_q))
            pte = mdl.predict_proba(sc.transform(Xm[te][:, cols]))[:, 1]
        except Exception:
            continue
        gi = np.where(te)[0]
        for k, g in enumerate(gi):
            if pte[k] >= th: keep.append(AV[g][0])
    keep.sort(key=lambda z: z[0])
    return keep


def show(E, lab):
    if len(E) < 40: P(f"  {lab:44s} 표본 부족 ({len(E)})"); return
    r = np.array([x[2] for x in E])
    out = f"  {lab:44s} n={len(E):5d} {r.mean()*1e4:+7.1f}bp 승률{(r>0).mean():.3f} |"
    for w, Nm, tag in ((1.0, 1.0, "무레버"), (2.0, 3.0, "저레버"), (3.0, 5.0, "중간"), (5.0, 8.0, "공격")):
        eq, fl, liq = X.sim(E, w, Nm); m = met(eq, fl, liq)
        out += f" {tag} {'파산' if m is None else f'{m[0]:+7.1f}%'}"
    P(out)
    eq, fl, liq = X.sim(E, 2.0, 3.0); m = met(eq, fl, liq)
    if m is not None:
        df = pd.DataFrame([(x[4], x[2]) for x in E], columns=["ts", "r"]).set_index("ts")
        g = df.groupby(df.index.year).r
        P(f"      저레버 24~ {m[3]:+7.1f}% 25~ {m[4]:+7.1f}% 최악 {m[2]:6.1f}% | 연도별 " +
          " ".join(f"{y}{v*1e4:+6.0f}×{n}" for (y, v), (_, n) in zip(g.mean().items(), g.size().items())))


P("\n" + "="*150)
P("[1] z0.6 (유동성 충격, 지정가, 240분) — 1차 단독 vs 메타 필터")
P("="*150)
E6 = [x for x in build_z6(0.005, 1.0, 30, 240, 30) if x[4] >= pd.Timestamp("2021-06-01", tz="UTC")]
show(E6, "z0.6 1차 단독")
for mq in (0.3, 0.5, 0.7):
    for nc in (10, 20):
        show(meta(E6, meta_q=mq, ncol=nc), f"z0.6 + 메타 통과{(1-mq)*100:.0f}% 특징{nc}")

P("\n" + "="*150)
P("[2] z0.7-A (평활 괴리 극단, 시장가, 30분) — 1차 단독 vs 메타 필터")
P("="*150)
E7 = [x for x in build_z7(0.001, 30, 30, 30, mkt=1) if x[4] >= pd.Timestamp("2021-06-01", tz="UTC")]
show(E7, "z0.7-A 1차 단독")
for mq in (0.3, 0.5):
    for nc in (8, 15):
        show(meta(E7, meta_q=mq, ncol=nc, refit_days=60), f"z0.7-A + 메타 통과{(1-mq)*100:.0f}% 특징{nc}")

P("\n" + "="*150)
P("[3] z0.7 중간강도 (괴리 0.5%, 지정가 k=2, 240분) — 표본과 강도의 절충")
P("="*150)
E7b = [x for x in build_z7(0.005, 15, 240, 30, mkt=0, kx=2.0) if x[4] >= pd.Timestamp("2021-06-01", tz="UTC")]
show(E7b, "z0.7-B 1차 단독")
for mq in (0.3, 0.5, 0.7):
    show(meta(E7b, meta_q=mq, ncol=15), f"z0.7-B + 메타 통과{(1-mq)*100:.0f}%")

P("\n" + "="*150)
P("[4] 메타 모델 정규화 강도 (과적합 통제)")
P("="*150)
for C in (0.01, 0.05, 0.2, 1.0):
    show(meta(E6, meta_q=0.5, ncol=15, C=C), f"z0.6 + 메타 C={C}")
