"""메타 라벨링 + 비용조정 라벨 + 삼중배리어 + 표본가중 + MAE예측 사이징.
1차 신호(룰)는 방향을 정하고, 2차 모델은 '이 베팅을 할 것인가'만 학습한다 (López de Prado).
전부 롤링 학습, 임계는 학습 구간 분포에서만 결정 → 룩어헤드 없음.
사용: python research/meta_label.py"""
import sys, warnings, gc
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from bigfeat import build_features, SYM
from freesia import store
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
import exit_tier as X
from absfloor import met
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
STEP = 15; FEE = 7.0
TE = pd.Timestamp("2024-01-01", tz="UTC"); REC = pd.Timestamp("2025-01-01", tz="UTC")

P("데이터 준비...")
BASE = {}
for s in SYM:
    Xf, c = build_features(s, step=STEP)
    k = store.load("klines_1m", s).set_index("ts")
    cl = c.values; lw = k.low.reindex(c.index).values; hi = k.high.reindex(c.index).values
    idx15 = c.index[::STEP][:len(Xf)]
    px = cl[::STEP][:len(Xf)]
    ema15 = pd.Series(cl).ewm(span=15, adjust=False).mean().values
    dev = ((cl - ema15)/cl)[::STEP][:len(Xf)]
    lr = pd.Series(np.log(cl)).diff()
    rv = (lr.rolling(300).std().shift(1).values*np.sqrt(60))[::STEP][:len(Xf)]   # 60분 스케일
    OUT = {}
    for H in (60, 120):
        ret = (pd.Series(cl).shift(-H).values/cl - 1.0)[::STEP][:len(Xf)]
        fl = pd.Series(lw).iloc[::-1].rolling(H, min_periods=1).min().iloc[::-1].shift(-1).values
        fh = pd.Series(hi).iloc[::-1].rolling(H, min_periods=1).max().iloc[::-1].shift(-1).values
        OUT[H] = dict(ret=ret, mae=np.maximum((1.0-fl/cl)[::STEP][:len(Xf)], 0),
                      mfe=np.maximum((fh/cl-1.0)[::STEP][:len(Xf)], 0))
    BASE[s] = dict(X=Xf.values.astype(np.float32), cols=list(Xf.columns), idx=idx15,
                   dev=dev, rv=rv, px=px, OUT=OUT)
    P(f"  {s}: {len(Xf):,}행 × {Xf.shape[1]}특징")
    del Xf; gc.collect()


def base_events(s, q=0.01, cool=4, H=60):
    """1차 신호: 가격-EMA15 괴리가 확장창 q분위 이하 (룩어헤드 없음)"""
    d = BASE[s]; dev = d["dev"]
    n = len(dev); th = np.full(n, np.nan)
    warm = 20000
    for i in range(warm, n, 672):     # 주 단위 갱신 (672 = 7일/15분)
        th[i:i+672] = np.nanquantile(dev[:i], q)
    sel = np.isfinite(th) & np.isfinite(dev) & (dev <= th)
    ii = np.where(sel)[0]
    keep = []; last = -10**9
    for i in ii:
        if i - last < cool: continue
        keep.append(i); last = i
    return np.array(keep, np.int64)


def run_meta(H=60, q=0.01, label="cost", win_days=365, refit_days=14, ncol=20,
             use_weight=True, meta_q=0.5, size_by=None, start="2021-06-01"):
    """label: 'plain'(>0) / 'cost'(>비용) / 'tb'(삼중배리어)"""
    E = []
    for s in SYM:
        d = BASE[s]; Xa = d["X"]; O = d["OUT"][H]
        ev = base_events(s, q, H=H)
        ret = O["ret"][ev]; mae = O["mae"][ev]; mfe = O["mfe"][ev]
        rv = d["rv"][ev]; ts = d["idx"][ev]; Xe = Xa[ev]
        ok = np.isfinite(ret) & np.isfinite(mae) & np.isfinite(Xe).all(axis=1) & np.isfinite(rv) & (rv > 0)
        ev, ret, mae, mfe, rv, ts, Xe = ev[ok], ret[ok], mae[ok], mfe[ok], rv[ok], ts[ok], Xe[ok]
        if label == "plain":
            y = (ret > 0).astype(int)
        elif label == "cost":
            y = (ret > FEE/1e4).astype(int)
        else:   # 삼중 배리어: 상단 1×rv 먼저 닿으면 1, 하단 1×rv 먼저면 0
            y = ((mfe >= rv) & (mae < rv)).astype(int)
            y = np.where((mfe >= rv) & (mae >= rv), (ret > 0).astype(int), y)
        ends = pd.date_range(pd.Timestamp(start, tz="UTC"), ts[-1], freq=f"{refit_days}D", tz="UTC")
        for e in ends:
            s0 = e - pd.Timedelta(days=win_days)
            tr = (ts >= s0) & (ts < e); te = (ts >= e) & (ts < e+pd.Timedelta(days=refit_days))
            if tr.sum() < 120 or te.sum() < 1: continue
            if len(np.unique(y[tr])) < 2: continue
            Xtr = Xe[tr]
            cr = np.abs(np.array([np.corrcoef(Xtr[:, j], ret[tr])[0, 1] if Xtr[:, j].std() > 0 else 0
                                  for j in range(Xa.shape[1])]))
            cr[~np.isfinite(cr)] = 0
            cols = np.argsort(-cr)[:ncol]
            # 표본 고유성 가중: 겹치는 라벨은 가중 축소
            w = None
            if use_weight:
                span = H/STEP
                gaps = np.diff(ev[tr], prepend=ev[tr][0]-int(span))
                w = np.clip(gaps/span, 0.1, 1.0)
            try:
                sc = StandardScaler().fit(Xtr[:, cols])
                mdl = LogisticRegression(max_iter=300, C=0.05).fit(sc.transform(Xtr[:, cols]), y[tr],
                                                                   sample_weight=w)
                ptr = mdl.predict_proba(sc.transform(Xtr[:, cols]))[:, 1]
                pte = mdl.predict_proba(sc.transform(Xe[te][:, cols]))[:, 1]
                th = float(np.quantile(ptr, meta_q))
                szm = None
                if size_by == "mae":
                    rg = Ridge(alpha=50.0).fit(sc.transform(Xtr[:, cols]), mae[tr])
                    szm = rg.predict(sc.transform(Xe[te][:, cols]))
            except Exception:
                continue
            gi = np.where(te)[0]
            for j, g in enumerate(gi):
                if pte[j] < th: continue
                wgt = 1.0
                if size_by == "prob": wgt = float(np.clip(pte[j]/max(th, 1e-6), 0.5, 3.0))
                elif size_by == "mae" and szm is not None:
                    wgt = float(np.clip(mae[tr].mean()/max(szm[j], 1e-4), 0.3, 3.0))
                E.append((int(ev[g]*STEP), int(ev[g]*STEP+H), float(ret[g]-FEE/1e4),
                          float(mae[g]), ts[g], (s, q), 0.01, wgt))
    E.sort(key=lambda z: z[0])
    return E


def base_only(H=60, q=0.01):
    E = []
    for s in SYM:
        d = BASE[s]; O = d["OUT"][H]
        ev = base_events(s, q, H=H)
        ret = O["ret"][ev]; mae = O["mae"][ev]; ts = d["idx"][ev]
        ok = np.isfinite(ret) & np.isfinite(mae) & (ts >= pd.Timestamp("2021-06-01", tz="UTC"))
        for i, g in enumerate(np.where(ok)[0]):
            E.append((int(ev[g]*STEP), int(ev[g]*STEP+H), float(ret[g]-FEE/1e4),
                      float(mae[g]), ts[g], (s, q), 0.01, 1.0))
    E.sort(key=lambda z: z[0])
    return E


def show(E, lab):
    if len(E) < 50: P(f"  {lab:42s} 표본 부족 ({len(E)})"); return None
    r = np.array([x[2] for x in E])
    out = f"  {lab:42s} n={len(E):5d} {r.mean()*1e4:+6.2f}bp 승률{(r>0).mean():.3f} |"
    res = {}
    for w, Nm, tag in ((1.0, 1.0, "무레버"), (2.0, 3.0, "저레버"), (3.0, 5.0, "중간")):
        eq, fl, liq = X.sim(E, w, Nm); m = met(eq, fl, liq)
        res[tag] = m
        out += f" {tag} {'파산' if m is None else f'{m[0]:+7.1f}%(MDD{m[1]:5.0f}%)'}"
    P(out)
    return res


P("\n" + "="*140)
P("[1] 1차 신호 단독 vs 메타 라벨링 (H=60분, 괴리 하위1%)")
P("="*140)
show(base_only(60, 0.01), "1차 신호 단독 (필터 없음)")
for lab in ("plain", "cost", "tb"):
    for mq in (0.3, 0.5, 0.7):
        show(run_meta(60, 0.01, label=lab, meta_q=mq), f"메타[{lab}] 상위{(1-mq)*100:.0f}% 통과")

P("\n" + "="*140)
P("[2] 표본 가중 유무 · 특징 수")
P("="*140)
for uw in (True, False):
    for nc in (10, 20, 40):
        show(run_meta(60, 0.01, label="cost", meta_q=0.5, use_weight=uw, ncol=nc),
             f"메타[cost] 가중{'O' if uw else 'X'} 특징{nc}")

P("\n" + "="*140)
P("[3] 사이징 — 확률 비례 / 예측MAE 반비례")
P("="*140)
for sb in (None, "prob", "mae"):
    show(run_meta(60, 0.01, label="cost", meta_q=0.5, size_by=sb), f"메타[cost] 사이징={sb or '균등'}")

P("\n" + "="*140)
P("[4] 1차 신호 강도 × 보유시간")
P("="*140)
for q in (0.005, 0.01, 0.02):
    for H in (60, 120):
        show(base_only(H, q), f"1차 단독 q={q} H={H}분")
        show(run_meta(H, q, label="cost", meta_q=0.5), f"메타[cost] q={q} H={H}분")

P("\n" + "="*140)
P("[5] 최선 조합 연도별")
P("="*140)
for lab, mq, sb in (("cost", 0.5, None), ("cost", 0.7, None), ("tb", 0.5, None)):
    E = run_meta(60, 0.01, label=lab, meta_q=mq, size_by=sb)
    if len(E) < 50: continue
    df = pd.DataFrame([(x[4], x[2]) for x in E], columns=["ts", "r"]).set_index("ts")
    g = df.groupby(df.index.year).r
    P(f"  메타[{lab}] 통과{(1-mq)*100:.0f}%: " +
      " ".join(f"{y}{v*1e4:+6.1f}×{n}" for (y, v), (_, n) in zip(g.mean().items(), g.size().items())))
