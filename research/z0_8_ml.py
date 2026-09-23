"""z0.8 — 롤링 학습 방향 모델 전략화. ETH/BTC 1년 학습·2주 재학습·상위10특징 로지스틱.
확신 상위 분위에서 롱 진입, 60분 보유. 3단 레버리지·연도별·플라시보.
사용: python research/z0_8_ml.py"""
import sys, warnings, gc
from pathlib import Path
import numpy as np, pandas as pd
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
    cl = c.values
    ret = (pd.Series(cl).shift(-HOR).values/cl - 1.0)[::STEP][:len(Xf)]
    lo = pd.Series(cl).rolling(HOR).min().shift(-HOR).values[::STEP][:len(Xf)]
    px = cl[::STEP][:len(Xf)]
    mae = 1.0 - lo/px
    m = np.isfinite(Xf.values).all(axis=1) & np.isfinite(ret) & np.isfinite(mae)
    DATA[s] = dict(X=Xf.values[m].astype(np.float32), r=ret[m].astype(np.float32),
                   mae=mae[m].astype(np.float32), idx=Xf.index[m], cols=list(Xf.columns))
    P(f"{s}: 표본 {m.sum():,}")
    del Xf; gc.collect()


def roll(s, win_days=365, ncol=10, refit_days=14, start="2021-01-01"):
    d = DATA[s]; Xa = d["X"]; ret = d["r"]; idx = d["idx"]
    yb = (ret > 0).astype(int)
    ends = pd.date_range(pd.Timestamp(start, tz="UTC"), idx[-1], freq=f"{refit_days}D", tz="UTC")
    pr = []; ii = []; picks = []
    for e in ends:
        s0 = e - pd.Timedelta(days=win_days)
        tr = (idx >= s0) & (idx < e); te = (idx >= e) & (idx < e+pd.Timedelta(days=refit_days))
        if tr.sum() < 200 or te.sum() < 10: continue
        Xtr = Xa[tr]
        cr = np.abs(np.array([np.corrcoef(Xtr[:, j], ret[tr])[0, 1] if Xtr[:, j].std() > 0 else 0
                              for j in range(Xa.shape[1])]))
        cr[~np.isfinite(cr)] = 0
        cols = np.argsort(-cr)[:ncol]
        picks.append(cols)
        try:
            sc = StandardScaler().fit(Xtr[:, cols])
            mdl = LogisticRegression(max_iter=200, C=0.05).fit(sc.transform(Xtr[:, cols]), yb[tr])
            p = mdl.predict_proba(sc.transform(Xa[te][:, cols]))[:, 1]
        except Exception:
            continue
        pr.append(p); ii.append(np.where(te)[0])
    if not pr: return None
    return np.concatenate(pr), np.concatenate(ii), picks


P("\n[1] 예측 생성")
PRED = {}
for s in SYM:
    out = roll(s)
    if out is None: continue
    p, ii, picks = out
    PRED[s] = (p, ii)
    d = DATA[s]
    P(f"  {s}: 예측 {len(p):,}개 ({d['idx'][ii][0].date()} ~ {d['idx'][ii][-1].date()})")
    cnt = {}
    for pk in picks:
        for j in pk: cnt[d["cols"][j]] = cnt.get(d["cols"][j], 0)+1
    top = sorted(cnt.items(), key=lambda kv: -kv[1])[:12]
    P(f"    자주 선택된 특징: " + ", ".join(f"{k}({v})" for k, v in top))


def make_events(q, cool_bars=4):
    """확신 상위 q 분위에서 롱 진입, 60분 보유. cool_bars: 15분 단위 쿨다운"""
    E = []
    for s in SYM:
        if s not in PRED: continue
        p, ii = PRED[s]; d = DATA[s]
        th = np.quantile(p, q)
        last = -10**9
        for k in range(len(p)):
            if p[k] < th: continue
            gi = ii[k]
            if gi - last < cool_bars: continue
            last = gi
            E.append((int(gi*STEP), int(gi*STEP+HOR), float(d["r"][gi]-FEE/1e4),
                      float(max(d["mae"][gi], 0.0)), d["idx"][gi], (s, q), 0.01, 1.0))
    E.sort(key=lambda z: z[0])
    return E


P("\n[2] 진입 분위별 — 거래 단위 및 3단 레버리지")
for q in (0.90, 0.95, 0.98, 0.99):
    E = make_events(q)
    if len(E) < 50: continue
    r = np.array([x[2] for x in E]); mae = np.array([x[3] for x in E])
    P(f"\n  ── 상위 {(1-q)*100:.0f}% 진입: n={len(E)} ({len(E)/5.7:.0f}/년) 거래당 {r.mean()*1e4:+.2f}bp "
      f"승률 {(r>0).mean():.3f} MAE평균 {mae.mean()*100:.2f}% 최대 {mae.max()*100:.1f}% ──")
    df = pd.DataFrame([(x[4], x[2]) for x in E], columns=["ts", "r"]).set_index("ts")
    g = df.groupby(df.index.year).r
    P("     연도별: " + " ".join(f"{y}{v*1e4:+6.1f}×{n}" for (y, v), (_, n) in zip(g.mean().items(), g.size().items())))
    for w, Nm, tag in ((1.0, 1.0, "① 무레버"), (2.0, 3.0, "저레버2x/3x"), (3.0, 5.0, "② 중간"), (5.0, 8.0, "③ 공격")):
        eq, fl, liq = X.sim(E, w, Nm); m = met(eq, fl, liq)
        if m is None: P(f"     {tag:12s} 파산 (청산 {liq})"); continue
        mo = (1+m[0]/100)**(1/12)-1
        mon = (m[5].resample("ME").last()/m[5].resample("ME").first()-1).dropna()
        P(f"     {tag:12s} 연 {m[0]:+8.1f}% (월 {mo*100:+5.1f}%) MDD {m[1]:6.1f}% 최악 {m[2]:6.1f}% | "
          f"24~ {m[3]:+8.1f}% | 25~ {m[4]:+8.1f}% | 월승률 {float((mon>0).mean()):.2f}")

P("\n[3] 2022년 제외 시 (하락장 취약성 확인)")
for q in (0.95, 0.99):
    E = [x for x in make_events(q) if x[4].year != 2022]
    if len(E) < 50: continue
    r = np.array([x[2] for x in E])
    P(f"  상위{(1-q)*100:.0f}% (2022 제외) n={len(E)} 거래당 {r.mean()*1e4:+.2f}bp")
    for w, Nm, tag in ((1.0, 1.0, "무레버"), (2.0, 3.0, "저레버"), (3.0, 5.0, "중간")):
        eq, fl, liq = X.sim(E, w, Nm); m = met(eq, fl, liq)
        P(f"     {tag:8s} " + ("파산" if m is None else f"연 {m[0]:+7.1f}% 최악 {m[2]:6.1f}% 24~ {m[3]:+7.1f}% 25~ {m[4]:+7.1f}%"))

P("\n[4] 플라시보 — 같은 횟수·같은 보유시간 무작위 진입 500회")
import numba
@numba.njit(cache=True)
def rnd(o, c, st, hd, fee):
    out = np.zeros(len(st))
    for i in range(len(st)):
        e = st[i]+hd[i]
        if e >= len(c): e = len(c)-1
        out[i] = (c[e]/o[st[i]]-1.0) - fee/1e4
    return out
K = {s: store.load("klines_1m", s).set_index("ts") for s in SYM}
E = make_events(0.95); r = np.array([x[2] for x in E])
rng = np.random.default_rng(97); nul = []
for _ in range(500):
    acc = []
    for s in SYM:
        o = K[s].open.values; c = K[s].close.values; N = len(c)
        cnt = max(len(E)//2, 1)
        acc.append(rnd(o, c, rng.integers(0, N-200, cnt).astype(np.int64),
                       np.full(cnt, HOR, np.int64), FEE))
    nul.append(np.concatenate(acc).mean()*1e4)
nul = np.array(nul)
P(f"  위약 평균 {nul.mean():+.2f}bp 99분위 {np.quantile(nul,0.99):+.2f} 최대 {nul.max():+.2f} | "
  f"실제 {r.mean()*1e4:+.2f}bp → 백분위 {(nul < r.mean()*1e4).mean()*100:.1f}%")
