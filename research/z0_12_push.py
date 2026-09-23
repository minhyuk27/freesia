"""z0.12 — 최대 수익 추구. z0.6+z0.7B 합집합 × 저변동 × MAE사이징 × 사이징 범위 확대.
공격 구간(5x/8x) 생존을 목표로.
사용: python research/z0_12_push.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from meta_strong import attach, FEAT, SYM
from z0_6b import build as build_z6
from z0_7_dev import build as build_z7
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
import exit_tier as X
from absfloor import met
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
T0 = pd.Timestamp("2021-06-01", tz="UTC")
TE = pd.Timestamp("2024-01-01", tz="UTC"); REC = pd.Timestamp("2025-01-01", tz="UTC")
IR = FEAT[SYM[0]]["cols"].index("rvratio15_240")

BASES = {
    "z0.6": [x for x in build_z6(0.005, 1.0, 30, 240, 30) if x[4] >= T0],
    "z0.7B": [x for x in build_z7(0.005, 15, 240, 30, mkt=0, kx=2.0) if x[4] >= T0],
}
UNION = sorted(BASES["z0.6"] + BASES["z0.7B"], key=lambda z: z[0])
P(f"z0.6 {len(BASES['z0.6'])}건 / z0.7B {len(BASES['z0.7B'])}건 / 합집합 {len(UNION)}건")

CA = {}
def AVC(E, key):
    if key not in CA: CA[key] = attach(E)
    return CA[key]


def pipeline(E, key, use_lowvol=True, lv_q=0.3, use_meta=False, mq=0.5,
             size="mae", clip_lo=0.3, clip_hi=3.0, ncol=8, win_days=365, refit_days=30):
    AV = AVC(E, key)
    if len(AV) < 150: return []
    TS = pd.DatetimeIndex([a[0][4] for a in AV])
    XM = np.array([a[1] for a in AV], np.float32)
    RET = np.array([a[0][2] for a in AV]); MAE = np.array([a[0][3] for a in AV])
    keep = []
    for e in pd.date_range(T0, TS[-1], freq=f"{refit_days}D", tz="UTC"):
        s0 = e - pd.Timedelta(days=win_days)
        tr = (TS >= s0) & (TS < e); te = (TS >= e) & (TS < e+pd.Timedelta(days=refit_days))
        if tr.sum() < 80 or te.sum() < 1: continue
        Xtr = XM[tr]; rtr = RET[tr]; mtr = MAE[tr]
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
                rg = Ridge(alpha=50.0).fit(Ztr, mtr)
                pm = np.clip(rg.predict(Zte), 1e-4, None)
                wt = np.clip(mtr.mean()/pm, clip_lo, clip_hi)
        except Exception:
            continue
        for k, g in enumerate(np.where(te)[0]):
            if use_lowvol and not (XM[g, IR] <= lv_th): continue
            if not okm[k]: continue
            x = AV[g][0]
            keep.append((x[0], x[1], x[2], x[3], x[4], x[5], x[6], float(wt[k])))
    keep.sort(key=lambda z: z[0])
    return keep


def rep(E, lab, full=False):
    if len(E) < 60: P(f"  {lab:48s} 표본부족({len(E)})"); return None
    r = np.array([x[2] for x in E])
    out = f"  {lab:48s} n={len(E):4d} {r.mean()*1e4:+6.1f}bp |"
    best = None
    for w, Nm, tag in ((1.0, 1.0, "무레버"), (2.0, 3.0, "저레버"), (3.0, 5.0, "중간"),
                       (4.0, 6.0, "중간+"), (5.0, 8.0, "공격")):
        eq, fl, liq = X.sim(E, w, Nm); m = met(eq, fl, liq)
        out += f" {tag}{'파산' if m is None else f'{m[0]:+7.1f}%'}"
        if m and (best is None or m[0] > best[0]): best = (m[0], m[1], m[2], m[3], m[4], tag, m[5])
    P(out)
    if full and best:
        mon = (best[6].resample("ME").last()/best[6].resample("ME").first()-1).dropna()
        P(f"      최고 {best[5]}: MDD {best[1]:.0f}% 최악 {best[2]:.0f}% 24~ {best[3]:+.0f}% 25~ {best[4]:+.0f}% 월승률 {float((mon>0).mean()):.2f}")
        df = pd.DataFrame([(x[4], x[2]) for x in E], columns=["ts", "r"]).set_index("ts")
        g = df.groupby(df.index.year).r
        P("      연도별 " + " ".join(f"{y}{v*1e4:+5.0f}×{n}" for (y, v), (_, n) in zip(g.mean().items(), g.size().items())))
    return best


P("\n[1] 기반 신호별 — 저변동 + MAE사이징")
for k, E in list(BASES.items()) + [("합집합", UNION)]:
    rep(pipeline(E, k, use_lowvol=False, size="flat"), f"{k} 원본")
    rep(pipeline(E, k, use_lowvol=True, size="flat"), f"{k} +저변동")
    rep(pipeline(E, k, use_lowvol=True, size="mae"), f"{k} +저변동+MAE사이징")
    rep(pipeline(E, k, use_lowvol=True, use_meta=True, size="mae"), f"{k} +저변동+메타+MAE사이징")

P("\n[2] 사이징 범위 확대 (합집합 + 저변동)")
for lo, hi in ((0.3, 3.0), (0.2, 5.0), (0.1, 8.0), (0.5, 2.0)):
    rep(pipeline(UNION, "합집합", use_lowvol=True, size="mae", clip_lo=lo, clip_hi=hi),
        f"사이징 clip({lo},{hi})")

P("\n[3] 저변동 임계 (합집합 + MAE사이징)")
for lq in (0.2, 0.3, 0.4, 0.5):
    rep(pipeline(UNION, "합집합", use_lowvol=True, lv_q=lq, size="mae"), f"저변동 하위{lq*100:.0f}%")

P("\n[4] 상위 후보 상세")
for lab, kw in (("합집합+저변동30%+MAE", dict(use_lowvol=True, lv_q=0.3, size="mae")),
                ("합집합+저변동30%+메타+MAE", dict(use_lowvol=True, lv_q=0.3, use_meta=True, size="mae")),
                ("합집합+저변동40%+MAE", dict(use_lowvol=True, lv_q=0.4, size="mae"))):
    rep(pipeline(UNION, "합집합", **kw), lab, full=True)
