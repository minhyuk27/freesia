"""z1.0 — 5대 방향 탐색의 종합. 저변동 국면 + 메타 라벨 + 켈리 필터 + MAE 사이징 조합 격자.
목표: 최고 수익. 전부 롤링·학습구간 임계 → 룩어헤드 없음.
사용: python research/z10_synth.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from meta_strong import attach, FEAT, SYM
from z06b import build as build_z6
from sklearn.linear_model import LogisticRegression, QuantileRegressor, Ridge
from sklearn.preprocessing import StandardScaler
import exit_tier as X
from absfloor import met
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
T0 = pd.Timestamp("2021-06-01", tz="UTC")
TE = pd.Timestamp("2024-01-01", tz="UTC"); REC = pd.Timestamp("2025-01-01", tz="UTC")

CACHE = {}
def prep(fm=2.0, ft=5.0, eps=0.0002):
    key = (fm, ft, eps)
    if key in CACHE: return CACHE[key]
    E6 = [x for x in build_z6(0.005, 1.0, 30, 240, 30, fm=fm, ft=ft, eps=eps) if x[4] >= T0]
    AV = attach(E6)
    cols = FEAT[SYM[0]]["cols"]
    ir = cols.index("rvratio15_240")
    d = dict(AV=AV, TS=pd.DatetimeIndex([a[0][4] for a in AV]),
             XM=np.array([a[1] for a in AV], np.float32),
             RET=np.array([a[0][2] for a in AV]), MAE=np.array([a[0][3] for a in AV]), ir=ir)
    CACHE[key] = d
    return d

D0 = prep()
P(f"1차 신호 {len(D0['AV'])}건")


def synth(use_lowvol=True, use_meta=True, use_kelly=False, size="flat",
          win_days=365, refit_days=30, ncol=10, mq=0.5, data=None, lv_q=0.5):
    d = data or D0
    AV, TS, XM, RET, MAE, ir = d["AV"], d["TS"], d["XM"], d["RET"], d["MAE"], d["ir"]
    keep = []
    ends = pd.date_range(T0, TS[-1], freq=f"{refit_days}D", tz="UTC")
    for e in ends:
        s0 = e - pd.Timedelta(days=win_days)
        tr = (TS >= s0) & (TS < e); te = (TS >= e) & (TS < e+pd.Timedelta(days=refit_days))
        if tr.sum() < 80 or te.sum() < 1: continue
        Xtr = XM[tr]; rtr = RET[tr]; mtr = MAE[tr]
        # 저변동 국면: 학습 구간의 rv비율 분위로 임계 결정 (룩어헤드 없음)
        lv_th = np.nanquantile(Xtr[:, ir], lv_q) if use_lowvol else np.inf
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
                    ptr = mm.predict_proba(Ztr)[:, 1]; pte = mm.predict_proba(Zte)[:, 1]
                    okm = pte >= np.quantile(ptr, mq)
            okk = np.ones(te.sum(), bool)
            if use_kelly:
                q1 = QuantileRegressor(quantile=0.25, alpha=0.01, solver="highs").fit(Ztr, rtr)
                q3 = QuantileRegressor(quantile=0.75, alpha=0.01, solver="highs").fit(Ztr, rtr)
                a1 = q1.predict(Zte); a3 = q3.predict(Zte)
                k_te = ((a1+a3)/2)/np.maximum(a3-a1, 1e-6)**2
                b1 = q1.predict(Ztr); b3 = q3.predict(Ztr)
                k_tr = ((b1+b3)/2)/np.maximum(b3-b1, 1e-6)**2
                okk = k_te >= np.quantile(k_tr, 0.5)
            wt = np.ones(te.sum())
            if size == "mae":
                rg = Ridge(alpha=50.0).fit(Ztr, mtr)
                pm = np.clip(rg.predict(Zte), 1e-4, None)
                wt = np.clip(mtr.mean()/pm, 0.3, 3.0)
        except Exception:
            continue
        gi = np.where(te)[0]
        for k, g in enumerate(gi):
            if use_lowvol and not (XM[g, ir] <= lv_th): continue
            if not (okm[k] and okk[k]): continue
            x = AV[g][0]
            keep.append((x[0], x[1], x[2], x[3], x[4], x[5], x[6], float(wt[k])))
    keep.sort(key=lambda z: z[0])
    return keep


def rep(E, lab, full=False):
    if len(E) < 60: P(f"  {lab:44s} 표본 부족 ({len(E)})"); return None
    r = np.array([x[2] for x in E])
    out = f"  {lab:44s} n={len(E):5d} {r.mean()*1e4:+7.1f}bp 승률{(r>0).mean():.3f} |"
    best = None
    for w, Nm, tag in ((1.0, 1.0, "무레버"), (2.0, 3.0, "저레버"), (3.0, 5.0, "중간"), (4.0, 6.0, "중간+")):
        eq, fl, liq = X.sim(E, w, Nm); m = met(eq, fl, liq)
        out += f" {tag} {'파산' if m is None else f'{m[0]:+7.1f}%'}"
        if m and (best is None or m[0] > best[0]): best = (m[0], m[2], m[3], m[4], tag, m[5])
    P(out)
    if full and best:
        P(f"      최고 {best[4]}: 최악 {best[1]:.1f}% | 24~ {best[2]:+.1f}% | 25~ {best[3]:+.1f}%")
        df = pd.DataFrame([(x[4], x[2]) for x in E], columns=["ts", "r"]).set_index("ts")
        g = df.groupby(df.index.year).r
        P("      연도별 " + " ".join(f"{y}{v*1e4:+6.0f}×{n}" for (y, v), (_, n) in zip(g.mean().items(), g.size().items())))
        mon = (best[5].resample("ME").last()/best[5].resample("ME").first()-1).dropna()
        P(f"      월승률 {float((mon>0).mean()):.2f} 월중앙 {mon.median()*100:+.1f}%")
    return best


P("\n" + "="*140)
P("[1] 구성 요소 조합 격자")
P("="*140)
rep(D0["AV"] and [a[0] for a in D0["AV"]], "기준 z0.6 단독")
combos = []
for lv in (False, True):
    for mt in (False, True):
        for kl in (False, True):
            for sz in ("flat", "mae"):
                if not (lv or mt or kl) and sz == "flat": continue
                lab = ("저변동+" if lv else "") + ("메타+" if mt else "") + ("켈리+" if kl else "") + \
                      ("MAE사이징" if sz == "mae" else "균등")
                b = rep(synth(lv, mt, kl, sz), lab)
                if b: combos.append((b[0], lab, lv, mt, kl, sz))

P("\n" + "="*140)
P("[2] 상위 조합 상세")
P("="*140)
combos.sort(reverse=True)
for sc_, lab, lv, mt, kl, sz in combos[:4]:
    rep(synth(lv, mt, kl, sz), lab, full=True)

P("\n" + "="*140)
P("[3] 저변동 임계 강도")
P("="*140)
for lq in (0.3, 0.4, 0.5, 0.6, 0.7):
    rep(synth(True, True, False, "flat", lv_q=lq), f"저변동 하위{lq*100:.0f}% + 메타")

P("\n" + "="*140)
P("[4] 최선 조합 비용 민감도")
P("="*140)
if combos:
    _, lab, lv, mt, kl, sz = combos[0]
    for fm, ft, eps, nm in ((2, 5, 0.0, "낙관"), (2, 5, 0.0002, "기준"), (3, 6, 0.0005, "보수"),
                            (4, 7, 0.001, "비관"), (5, 8, 0.002, "최악")):
        d = prep(fm, ft, eps)
        rep(synth(lv, mt, kl, sz, data=d), f"{nm} 메{fm}/테{ft} 관통{eps*1e4:.0f}bp")
