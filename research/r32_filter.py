"""R32-2 — 전문가 개념 특징이 R27 딥 선택을 개선하는가.

비교: A 기존(R27) 특징 / B 기존+전문가 특징 / C 전문가 특징만(+B,k,종목)
절차: R27과 동일한 확장창 워크포워드(2022~2026 평가), 라벨 셔플 플라시보, 계좌 시뮬 3단.
추가: 2025-10-10 폭락일 손익, 전문가 특징 순열 중요도.
사용: python research/r32_filter.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R / "research"))
from r28_hf_port import sim
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
E = pd.read_parquet(R / "data" / "r32_events.parquet")
E["ts"] = pd.DatetimeIndex(E.ts); E["yr"] = E.ts.dt.year
E["isbtc"] = (E.sym == "BTCUSDT").astype(float)
E = E.replace([np.inf, -np.inf], np.nan)
META = {"sym", "f", "x", "ret", "mae", "ts", "yr"}
NEW = ["cvd5", "cvd30", "cvddiv", "absorb", "piv", "swept", "tpiv", "vpin15", "vpin60", "accel", "oi15",
       "quad", "dprem5", "dprem30", "qh", "hr0", "bslope", "bimb"]
ALL = [c for c in E.columns if c not in META]
OLD = [c for c in ALL if c not in NEW]
BASE = ["B", "k", "isbtc"]
SETS = {"A 기존": OLD, "B 기존+전문가": ALL, "C 전문가만": NEW + BASE}
YRS = (E.ts.max() - pd.Timestamp("2022-01-01", tz="UTC")).days / 365.25


def mdl(seed=0):
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, max_leaf_nodes=15,
                                         min_samples_leaf=400, l2_regularization=1.0, random_state=seed)


def walk(feats, shuffle=False, seed=0):
    rng = np.random.default_rng(seed); out = []
    for Y in range(2022, 2027):
        tr = E[E.yr < Y]; te = E[E.yr == Y]
        y = np.clip(tr.ret.values, -0.02, 0.02).copy()
        if shuffle:
            for yy in tr.yr.unique():
                mk = (tr.yr == yy).values; y[mk] = rng.permutation(y[mk])
        fu = [c for c in feats if tr[c].notna().sum() > 1000]   # 호가 특징은 2023 이전 결측
        m = mdl(seed).fit(tr[fu].values, y)
        out.append(te.assign(pred=m.predict(te[fu].values)))
    return pd.concat(out)


def rep(S, lab):
    if len(S) < 50:
        P(f"  {lab:34s} 표본부족({len(S)})"); return
    epi = S.assign(h=S.ts.dt.floor("15min")).drop_duplicates(["sym", "h"])
    by = S.groupby("yr").ret.mean() * 1e4
    S2 = S.sort_values(["f", "sym"]).reset_index(drop=True)
    acc = []
    for w, Nm in ((0.5, 1.0), (1.0, 3.0), (1.0, 5.0)):
        res, liq = sim(S2, w, Nm)
        acc.append("파산" if res is None else f"{(res[0].iloc[-1]**(1/YRS)-1)*100:+5.0f}%/{res[1]*100:4.0f}")
    P(f"  {lab:34s} {len(S)/YRS:5.0f}/년 하루{len(epi)/YRS/365:4.1f}사건 {S.ret.mean()*1e4:+6.2f}bp 승{(S.ret>0).mean():.3f} "
      f"| 무레버 {acc[0]} 중간 {acc[1]} 공격 {acc[2]} (연%/MDD) | " +
      " ".join(f"{y%100}{v:+5.1f}" for y, v in by.items()))


P(f"이벤트 {len(E):,}, 기존 {len(OLD)} + 전문가 {len(NEW)} 특징")
P("\n[0] 전문가 특징 단독 5분위 (경계=2020~22, 평가=2023~26), 풀 B10/15/30 × k3·4")
pool = E[E.B.isin([10, 15, 30]) & E.k.isin([3.0, 4.0])]
tr, te = pool[pool.yr <= 2022], pool[pool.yr >= 2023]
for c in NEW:
    x = tr[c].dropna() if c not in ("bslope", "bimb") else te[c].dropna()
    if x.nunique() < 3: continue
    qs = np.unique(np.nanquantile(x, [0.2, 0.4, 0.6, 0.8]))
    src = tr if c not in ("bslope", "bimb") else te
    mt = [src.ret[np.digitize(src[c], qs) == i].mean() * 1e4 for i in range(len(qs) + 1)]
    me = [te.ret[np.digitize(te[c], qs) == i].mean() * 1e4 for i in range(len(qs) + 1)]
    P(f"  {c:8s} 학습 " + " ".join(f"{v:+6.1f}" for v in mt) + " | 평가 " + " ".join(f"{v:+6.1f}" for v in me))

WS = {}
for lab, fs in SETS.items():
    P(f"\n[{lab}] 특징 {len(fs)}개")
    W = walk(fs); WS[lab] = W
    for th in (0.0005, 0.001, 0.002):
        rep(W[W.pred > th], f"예측>{th*1e4:.0f}bp")
    Wsh = walk(fs, shuffle=True)
    rep(Wsh[Wsh.pred > 0.0005], "플라시보(셔플) 예측>5bp")

P("\n[3] 2025-10-10~11 폭락 구간 (예측>10bp 선택 거래)")
for lab, W in WS.items():
    d = W[(W.ts >= "2025-10-10 18:00") & (W.ts < "2025-10-11 06:00") & (W.pred > 0.001)]
    base = E[(E.ts >= "2025-10-10 18:00") & (E.ts < "2025-10-11 06:00")]
    P(f"  {lab:12s} 선택 {len(d):4d}건 합계 {d.ret.sum()*1e4:+8.0f}bp 평균 {d.ret.mean()*1e4 if len(d) else 0:+6.1f} "
      f"| 전체 딥 {len(base)}건 평균 {base.ret.mean()*1e4:+6.1f}bp")

P("\n[4] 순열 중요도 (B 모델, 2024~2026 평가, 예측>10bp 선택의 거래당 bp 하락)")
tr = E[E.yr < 2024]; te = E[E.yr >= 2024]
m = mdl().fit(tr[ALL].values, np.clip(tr.ret.values, -0.02, 0.02))
X = te[ALL].values.copy(); base = te.ret[m.predict(X) > 0.001].mean()
rng = np.random.default_rng(0); imp = []
for j, c in enumerate(ALL):
    Xp = X.copy(); Xp[:, j] = rng.permutation(Xp[:, j])
    imp.append(((base - te.ret[m.predict(Xp) > 0.001].mean()) * 1e4, c))
imp.sort(reverse=True)
P(f"  기준 {base*1e4:+.2f}bp | " + " ".join(f"{c}{'*' if c in NEW else ''} {d_:+.1f}" for d_, c in imp[:18]))
WS["B 기존+전문가"].to_parquet(R / "data" / "r32_wf.parquet", index=False)
