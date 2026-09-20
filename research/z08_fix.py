"""z0.8 수정판 — 룩어헤드 제거.
(1) 진입 임계를 '학습 구간 예측 분포'에서만 결정 (표본 밖 전체 분위수 사용 금지)
(2) MAE를 종가가 아닌 저가 기준으로 계산
사용: python research/z08_fix.py"""
import sys, warnings, gc
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from bigfeat import build_features, SYM
from freesia import store
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import exit_tier as X
from absfloor import met
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
HOR = 60; STEP = 15; FEE = 7.0
TE = pd.Timestamp("2024-01-01", tz="UTC"); REC = pd.Timestamp("2025-01-01", tz="UTC")

DATA = {}
for s in SYM:
    Xf, c = build_features(s, step=STEP)
    k = store.load("klines_1m", s).set_index("ts")
    cl = c.values; lw = k.low.reindex(c.index).values
    ret = (pd.Series(cl).shift(-HOR).values/cl - 1.0)[::STEP][:len(Xf)]
    # 저가 기준 전향 최저 (룩어헤드 없음 — 보유 구간의 실제 최악)
    fwlow = pd.Series(lw).iloc[::-1].rolling(HOR, min_periods=1).min().iloc[::-1].shift(-1).values
    mae = (1.0 - fwlow/cl)[::STEP][:len(Xf)]
    m = np.isfinite(Xf.values).all(axis=1) & np.isfinite(ret) & np.isfinite(mae)
    DATA[s] = dict(X=Xf.values[m].astype(np.float32), r=ret[m].astype(np.float32),
                   mae=np.maximum(mae[m], 0).astype(np.float32), idx=Xf.index[m], cols=list(Xf.columns))
    P(f"{s}: 표본 {m.sum():,} | MAE(저가기준) 평균 {DATA[s]['mae'].mean()*100:.2f}% 최대 {DATA[s]['mae'].max()*100:.1f}%")
    del Xf; gc.collect()


def roll_events(s, q, win_days=365, ncol=10, refit_days=14, start="2021-01-01", cool_bars=4):
    """임계를 학습 구간 예측 분포의 q분위로 결정 → 다음 구간에 적용. 룩어헤드 없음."""
    d = DATA[s]; Xa = d["X"]; ret = d["r"]; idx = d["idx"]; mae = d["mae"]
    yb = (ret > 0).astype(int)
    ends = pd.date_range(pd.Timestamp(start, tz="UTC"), idx[-1], freq=f"{refit_days}D", tz="UTC")
    E = []; last = -10**9
    for e in ends:
        s0 = e - pd.Timedelta(days=win_days)
        tr = (idx >= s0) & (idx < e); te = (idx >= e) & (idx < e+pd.Timedelta(days=refit_days))
        if tr.sum() < 200 or te.sum() < 5: continue
        Xtr = Xa[tr]
        cr = np.abs(np.array([np.corrcoef(Xtr[:, j], ret[tr])[0, 1] if Xtr[:, j].std() > 0 else 0
                              for j in range(Xa.shape[1])]))
        cr[~np.isfinite(cr)] = 0
        cols = np.argsort(-cr)[:ncol]
        try:
            sc = StandardScaler().fit(Xtr[:, cols])
            mdl = LogisticRegression(max_iter=200, C=0.05).fit(sc.transform(Xtr[:, cols]), yb[tr])
            ptr = mdl.predict_proba(sc.transform(Xtr[:, cols]))[:, 1]
            th = float(np.quantile(ptr, q))              # ← 학습 구간에서만 임계 결정
            pte = mdl.predict_proba(sc.transform(Xa[te][:, cols]))[:, 1]
        except Exception:
            continue
        gi_all = np.where(te)[0]
        for j, gi in enumerate(gi_all):
            if pte[j] < th: continue
            if gi - last < cool_bars: continue
            last = gi
            E.append((int(gi*STEP), int(gi*STEP+HOR), float(ret[gi]-FEE/1e4),
                      float(mae[gi]), idx[gi], (s, q), 0.01, 1.0))
    return E


def show(E, lab):
    if len(E) < 50: P(f"  {lab}: 표본 부족 ({len(E)})"); return
    r = np.array([x[2] for x in E]); mae = np.array([x[3] for x in E])
    df = pd.DataFrame([(x[4], x[2]) for x in E], columns=["ts", "r"]).set_index("ts")
    g = df.groupby(df.index.year).r
    P(f"\n  ── {lab}: n={len(E)} ({len(E)/4.75:.0f}/년) 거래당 {r.mean()*1e4:+.2f}bp 승률 {(r>0).mean():.3f} "
      f"MAE평균 {mae.mean()*100:.2f}% 최대 {mae.max()*100:.1f}% ──")
    P("     연도별: " + " ".join(f"{y}{v*1e4:+6.1f}×{n}" for (y, v), (_, n) in zip(g.mean().items(), g.size().items())))
    for w, Nm, tag in ((1.0, 1.0, "① 무레버"), (2.0, 3.0, "저레버"), (3.0, 5.0, "② 중간"), (5.0, 8.0, "③ 공격")):
        eq, fl, liq = X.sim(E, w, Nm); m = met(eq, fl, liq)
        if m is None: P(f"     {tag:8s} 파산 (청산 {liq})"); continue
        mo = (1+m[0]/100)**(1/12)-1
        mon = (m[5].resample("ME").last()/m[5].resample("ME").first()-1).dropna()
        P(f"     {tag:8s} 연 {m[0]:+8.1f}% (월 {mo*100:+5.1f}%) MDD {m[1]:6.1f}% 최악 {m[2]:6.1f}% | "
          f"24~ {m[3]:+8.1f}% | 25~ {m[4]:+8.1f}% | 월승률 {float((mon>0).mean()):.2f}")


P("\n[1] 룩어헤드 수정 후 — 진입 임계는 학습 구간 분위수")
for q in (0.90, 0.95, 0.99):
    E = sorted(sum([roll_events(s, q) for s in SYM], []), key=lambda z: z[0])
    show(E, f"학습분위 상위{(1-q)*100:.0f}%")

P("\n[2] 학습창 길이 비교 (상위5%)")
for wd in (90, 365, 730):
    E = sorted(sum([roll_events(s, 0.95, win_days=wd) for s in SYM], []), key=lambda z: z[0])
    show(E, f"학습창 {wd}일 상위5%")

P("\n[3] 플라시보 (상위5%, 1년창)")
@numba.njit(cache=True)
def rnd(o, c, st, hd, fee):
    out = np.zeros(len(st))
    for i in range(len(st)):
        e = st[i]+hd[i]
        if e >= len(c): e = len(c)-1
        out[i] = (c[e]/o[st[i]]-1.0) - fee/1e4
    return out
K = {s: store.load("klines_1m", s).set_index("ts") for s in SYM}
E = sorted(sum([roll_events(s, 0.95) for s in SYM], []), key=lambda z: z[0])
r = np.array([x[2] for x in E])
rng = np.random.default_rng(101); nul = []
for _ in range(500):
    acc = []
    for s in SYM:
        o = K[s].open.values; c = K[s].close.values; N = len(c)
        cnt = max(len(E)//2, 1)
        acc.append(rnd(o, c, rng.integers(0, N-200, cnt).astype(np.int64), np.full(cnt, HOR, np.int64), FEE))
    nul.append(np.concatenate(acc).mean()*1e4)
nul = np.array(nul)
P(f"  위약 평균 {nul.mean():+.2f}bp 99분위 {np.quantile(nul,0.99):+.2f} 최대 {nul.max():+.2f} | "
  f"실제 {r.mean()*1e4:+.2f}bp → 백분위 {(nul < r.mean()*1e4).mean()*100:.1f}%")
