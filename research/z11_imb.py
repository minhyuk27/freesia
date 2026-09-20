"""z1.1 — 불균형 바 위에서 저변동 필터 + 학습 필터. 선명한 신호 + 최소 필터.
사용: python research/z11_imb.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from infobars import imbalance_bars, D, SYM
from bigfeat import build_features
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
import exit_tier as X
from absfloor import met
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
T0 = pd.Timestamp("2021-06-01", tz="UTC")
TE = pd.Timestamp("2024-01-01", tz="UTC"); REC = pd.Timestamp("2025-01-01", tz="UTC")

P("특징 준비...")
FE = {}
for s in SYM:
    Xf, c = build_features(s, step=15)
    FE[s] = dict(X=Xf.values.astype(np.float32), idx=Xf.index, cols=list(Xf.columns))
    P(f"  {s}: {Xf.shape}")
    del Xf
IR = FE[SYM[0]]["cols"].index("rvratio15_240")


def imb_events(s, mult, q_th, H, fee=7.0, kx=0.0):
    """불균형 바 하위 q_th 다음 바에서 진입(지정가 kx>0이면 더 아래), H바 뒤 청산"""
    d = D[s]
    imb_tot = np.abs(2*d["tb"] - d["qv"]).sum()
    n_target = len(d["c"])//60
    thr = imb_tot/n_target*mult
    ii, io, ih, il, ic, iv = imbalance_bars(d["c"], d["o"], d["h"], d["l"], d["qv"], d["tb"], thr)
    if len(ic) < 1000: return []
    r = np.diff(np.log(ic))
    n = len(r)
    th = np.full(n, np.nan); warm = max(400, n//12)
    for i in range(warm, n, 150):
        th[i:i+150] = np.nanquantile(r[:i], q_th)
    lr = pd.Series(np.log(d["c"])).diff()
    rv = (lr.rolling(300).std().shift(1).values)*np.sqrt(60)
    E = []
    for i in range(warm, n-H-1):
        if not np.isfinite(th[i]) or r[i] > th[i]: continue
        a0 = int(ii[i+1])
        if a0+1 >= len(d["c"]): break
        if kx > 0:
            if not np.isfinite(rv[a0]) or rv[a0] <= 0: continue
            T = ic[i+1]*(1.0 - rv[a0]*kx)
            lim = min(int(ii[i+2]) if i+2 < len(ii) else a0+120, len(d["c"])-1)
            f = -1
            for u in range(a0+1, lim+1):
                if d["l"][u] <= T*(1-0.0002): f = u; break
            if f < 0: continue
            px = T; a = f
        else:
            px = ic[i+1]; a = a0
        b = int(ii[i+1+H])
        if b >= len(d["c"]) or b <= a: continue
        lo_ = d["l"][a:b+1].min()
        E.append((a, b, float(ic[i+1+H]/px - 1.0 - fee/1e4), float(max(1.0-lo_/px, 0)),
                  d["idx"][a], (s, mult), 0.01, 1.0))
    return E


def attach_feat(E):
    out = []
    for x in E:
        f = FE[x[5][0]]
        j = f["idx"].searchsorted(x[4], side="right")-1
        if j < 0: continue
        v = f["X"][j]
        if not np.isfinite(v).all(): continue
        out.append((x, v))
    return out


def filt(E, use_lowvol=True, use_meta=True, lv_q=0.3, ncol=8, mq=0.5,
         win_days=365, refit_days=30, size="flat"):
    AV = attach_feat(E)
    if len(AV) < 150: return []
    TS = pd.DatetimeIndex([a[0][4] for a in AV])
    XM = np.array([a[1] for a in AV], np.float32)
    RET = np.array([a[0][2] for a in AV]); MAE = np.array([a[0][3] for a in AV])
    keep = []
    for e in pd.date_range(T0, TS[-1], freq=f"{refit_days}D", tz="UTC"):
        s0 = e - pd.Timedelta(days=win_days)
        tr = (TS >= s0) & (TS < e); te = (TS >= e) & (TS < e+pd.Timedelta(days=refit_days))
        if tr.sum() < 60 or te.sum() < 1: continue
        Xtr = XM[tr]; rtr = RET[tr]
        lv_th = np.nanquantile(Xtr[:, IR], lv_q) if use_lowvol else np.inf
        cr = np.abs(np.array([np.corrcoef(Xtr[:, j], rtr)[0, 1] if Xtr[:, j].std() > 0 else 0
                              for j in range(XM.shape[1])]))
        cr[~np.isfinite(cr)] = 0
        cc = np.argsort(-cr)[:ncol]
        try:
            sc = StandardScaler().fit(Xtr[:, cc])
            Ztr = sc.transform(Xtr[:, cc]); Zte = sc.transform(XM[te][:, cc])
            okm = np.ones(te.sum(), bool)
            if use_meta:
                y = (rtr > 0).astype(int)
                if len(np.unique(y)) > 1:
                    mm = LogisticRegression(max_iter=300, C=0.05).fit(Ztr, y)
                    okm = mm.predict_proba(Zte)[:, 1] >= np.quantile(mm.predict_proba(Ztr)[:, 1], mq)
            wt = np.ones(te.sum())
            if size == "mae":
                rg = Ridge(alpha=50.0).fit(Ztr, MAE[tr])
                pm = np.clip(rg.predict(Zte), 1e-4, None)
                wt = np.clip(MAE[tr].mean()/pm, 0.3, 3.0)
        except Exception:
            continue
        for k, g in enumerate(np.where(te)[0]):
            if use_lowvol and not (XM[g, IR] <= lv_th): continue
            if not okm[k]: continue
            x = AV[g][0]
            keep.append((x[0], x[1], x[2], x[3], x[4], x[5], x[6], float(wt[k])))
    keep.sort(key=lambda z: z[0])
    return keep


def rep(E, lab):
    if len(E) < 50: P(f"  {lab:46s} 표본 부족({len(E)})"); return None
    r = np.array([x[2] for x in E])
    out = f"  {lab:46s} n={len(E):4d} {r.mean()*1e4:+7.1f}bp 승률{(r>0).mean():.3f} |"
    best = None
    for w, Nm, tag in ((1.0, 1.0, "무레버"), (2.0, 3.0, "저레버"), (3.0, 5.0, "중간"), (4.0, 6.0, "중간+")):
        eq, fl, liq = X.sim(E, w, Nm); m = met(eq, fl, liq)
        out += f" {tag} {'파산' if m is None else f'{m[0]:+7.1f}%'}"
        if m and (best is None or m[0] > best[0]): best = (m[0], m[2], m[3], m[4], m[5])
    P(out)
    if best:
        df = pd.DataFrame([(x[4], x[2]) for x in E], columns=["ts", "r"]).set_index("ts")
        g = df.groupby(df.index.year).r
        mon = (best[4].resample("ME").last()/best[4].resample("ME").first()-1).dropna()
        P(f"      최악 {best[1]:.0f}% 24~ {best[2]:+.0f}% 25~ {best[3]:+.0f}% 월승률 {float((mon>0).mean()):.2f} | " +
          " ".join(f"{y}{v*1e4:+5.0f}×{n}" for (y, v), (_, n) in zip(g.mean().items(), g.size().items())))
    return best


P("\n[1] 불균형 바 기반 — 필터 조합")
for mult in (0.25, 0.5):
    for q_th in (0.02, 0.05):
        for H in (3, 5):
            E = sorted(sum([imb_events(s, mult, q_th, H) for s in SYM], []), key=lambda z: z[0])
            E = [x for x in E if x[4] >= T0]
            if len(E) < 80: continue
            rep(E, f"배수{mult} q={q_th} H={H} 필터없음")
            rep(filt(E, True, False), f"배수{mult} q={q_th} H={H} +저변동")
            rep(filt(E, True, True), f"배수{mult} q={q_th} H={H} +저변동+메타")
            rep(filt(E, False, True), f"배수{mult} q={q_th} H={H} +메타만")
            rep(filt(E, True, False, size="mae"), f"배수{mult} q={q_th} H={H} +저변동+MAE사이징")

P("\n[2] 지정가 진입 결합 (불균형 바 + 더 아래 지정가)")
for mult in (0.25, 0.5):
    for kx in (0.5, 1.0):
        E = sorted(sum([imb_events(s, mult, 0.05, 3, kx=kx) for s in SYM], []), key=lambda z: z[0])
        E = [x for x in E if x[4] >= T0]
        if len(E) < 80: continue
        rep(E, f"배수{mult} 지정가 k={kx} 필터없음")
        rep(filt(E, True, True), f"배수{mult} 지정가 k={kx} +저변동+메타")
