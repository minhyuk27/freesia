"""R129 — R90 예보 창(240분) 안에서 큰 움직임이 "어떻게 시작되는가"에 대한 기술적(descriptive) 해부.

사건: R90 p ≥ th30(학습 문턱, 표본 밖 상위 30%), 종목별 240분 겹침 제거(시간순 그리디).
      등급 T10(p≥th10) / T20(p≥th20) / T30(그 외, p≥th30만). 기준가 ref = c[t−1] (R90과 동일, 미래 참조 없음).
대조(CONTROL): p < th30 인 시점에서 종목·연도별 사건 수와 같은 수만큼 무작위 비겹침 창(240분) 추출(시드 고정).

용어(둘 다 "사후적 사후지식(hindsight)" — 예측 특징으로는 절대 쓰지 않는다. 묘사용 라벨일 뿐):
  h2            = ref 대비 ±2% 가 먼저 닿은 분(= R90의 y=1 판정 시점). 같은 분에 양쪽 다 닿으면 "모호"로 표시하고 방향 분석에서 뺀다.
  D             = h2 에서의 방향(+1/−1), 못 닿으면 0.
  onset_extreme = D=+1 이면 [t,h2] 구간의 최저가 분, D=−1 이면 최고가 분 — "진짜 턴"이 일어난 지점.
  onset_from_ref= h2 이전, 가격이 ref ±0.25% 안에 마지막으로 있었던 분("기준선에서 출발") — 다른 정의, 둘 다 보고.
  첫 이탈(first excursion) = t부터 ref ± X%(X=0.1/0.2/0.3/0.5%) 를 처음 벗어난 분·방향. 사건이 일어나기 "전에" 이미 방향이
      보이는지 보는 핵심 조기탐지 지표 — p(R90)는 방향을 예측하지 않으므로 무작위에서는 50%에 가까워야 한다(검증 포인트).
  false_starts  = onset_extreme 이전, D와 반대 방향으로 ref에서 0.25%/0.5% 벗어났다가 되돌아온 횟수(히스테리시스).

신호(signature) 분석: onset_extreme 전후 1/5/10분 특징(r129_feats.minute_panel, 전부 인과적) vs 같은 창 안
  무작위 분(onset ±15분 제외)의 특징 — TIMING(크기: 평균차/합동표준편차, AUC) · DIRECTION(부호: sign(특징)==D 비율, z검정).
  기간 분리 2022–24 vs 2025–26, 등급 T10/T20/T30(+대조군 CTRL 참고). agg1s·bookDepth 특징은 2023-01~만 존재.

출력: data/r129_windows.parquet(창별 통계) · data/r129_sig.parquet(신호 테이블). 특징 함수는 r129_feats.minute_panel.
사용: python research/r129_onset_anatomy.py"""
import sys, time, bisect, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
import r129_feats as RF
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)

SYMS = ("BTCUSDT", "ETHUSDT")
W = 240              # R90 예보 창 길이(분)
XS = np.array([0.001, 0.002, 0.003, 0.005])
SEED = 129


# ---------- 1) 창 구성 ----------

def dedup_time(df, gap):
    """시간순 그리디 겹침 제거(사건용) — 각 종목별로 t가 last_end 이상이면 채택."""
    keep = []; last = {}
    for s, t in zip(df.sym.values, df.t.values):
        ok = t >= last.get(s, -10**12) + gap
        keep.append(ok)
        if ok: last[s] = t
    return df[np.array(keep)]


def dedup_random(df, gap, rng):
    """무작위 순서 그리디 겹침 제거(대조군용) — 시간순이 아니라 뽑히는 순서 자체를 무작위로 섞는다."""
    starts = {}; keep_idx = []
    order = rng.permutation(len(df))
    sym_v = df.sym.values; t_v = df.t.values.astype(np.int64)
    for i in order:
        s = sym_v[i]; t = int(t_v[i])
        lst = starts.setdefault(s, [])
        pos = bisect.bisect_left(lst, t); ok = True
        if pos > 0 and lst[pos - 1] + gap > t: ok = False
        if ok and pos < len(lst) and t + gap > lst[pos]: ok = False
        if ok: bisect.insort(lst, t); keep_idx.append(i)
    return df.iloc[sorted(keep_idx)]


def build_windows():
    T = pd.read_parquet(R / "data" / "r90_wf.parquet")
    ev = T[T.p >= T.th30].sort_values(["sym", "t"]).copy()
    ev = dedup_time(ev, W)
    ev["tier"] = np.where(ev.p >= ev.th10, "T10", np.where(ev.p >= ev.th20, "T20", "T30"))
    target = ev.groupby(["sym", "yr"]).size()
    pool = T[T.p < T.th30].sort_values(["sym", "t"]).copy()
    rng = np.random.default_rng(SEED)
    ctrl_parts = []
    for (s, yr), n in target.items():
        cand = pool[(pool.sym == s) & (pool.yr == yr)]
        if len(cand) == 0: continue
        got = dedup_random(cand, W, rng)
        if len(got) > n: got = got.sample(n=n, random_state=SEED)
        ctrl_parts.append(got)
    ctrl = pd.concat(ctrl_parts); ctrl["tier"] = "CTRL"
    WIN = pd.concat([ev[["sym", "t", "ts", "yr", "p", "tier"]], ctrl[["sym", "t", "ts", "yr", "p", "tier"]]]).reset_index(drop=True)
    return WIN


# ---------- 2) 해부(anatomy): 사후지식 라벨 ----------

@numba.njit(cache=True)
def anatomy(t, h, l, c, W, xs):
    n = len(c); nQ = len(t)
    out = np.full((nQ, 15), np.nan)
    for q in range(nQ):
        i = t[q]
        if i < 1 or i + W >= n: continue
        ref = c[i - 1]
        d = 0; h2 = -1; amb2 = 0
        for j in range(i, i + W):
            up = h[j] >= ref * 1.02; dn = l[j] <= ref * 0.98
            if up and dn: amb2 = 1; h2 = j; d = 0; break
            if up: h2 = j; d = 1; break
            if dn: h2 = j; d = -1; break
        out[q, 0] = d; out[q, 1] = amb2; out[q, 2] = h2
        for xi in range(4):
            X = xs[xi]; fm = -1; fd = 0
            for j in range(i, i + W):
                u = h[j] >= ref * (1 + X); dn = l[j] <= ref * (1 - X)
                if u and dn: fm = j; fd = 0; break
                if u: fm = j; fd = 1; break
                if dn: fm = j; fd = -1; break
            out[q, 5 + xi] = fm; out[q, 9 + xi] = fd
        if d == 0 or h2 < 0: continue
        best = l[i] if d == 1 else h[i]; bm = i
        for j in range(i, h2 + 1):
            if d == 1 and l[j] < best: best = l[j]; bm = j
            if d == -1 and h[j] > best: best = h[j]; bm = j
        out[q, 3] = bm
        band = 0.0025; lastm = -1
        for j in range(i, h2 + 1):
            if h[j] >= ref * (1 - band) and l[j] <= ref * (1 + band): lastm = j
        out[q, 4] = lastm
        for yi in range(2):
            Y = 0.0025 if yi == 0 else 0.005; reset = Y / 2.0; armed = True; cnt = 0
            for j in range(i, bm):
                dev = (ref - l[j]) / ref if d == 1 else (h[j] - ref) / ref
                if dev >= Y and armed: cnt += 1; armed = False
                if dev < reset: armed = True
            out[q, 13 + yi] = cnt
    return out


ANAT_COLS = ["D", "amb2", "h2", "onset_ext", "onset_ref",
             "fe_m1", "fe_m2", "fe_m3", "fe_m5",
             "fe_d1", "fe_d2", "fe_d3", "fe_d5",
             "fs025", "fs050"]


def run_anatomy(WIN, K):
    parts = []
    for s in SYMS:
        g = WIN[WIN.sym == s]; k = K[s]
        out = anatomy(g.t.values.astype(np.int64), k.high.values, k.low.values, k.close.values, W, XS)
        parts.append(pd.DataFrame(out, columns=ANAT_COLS, index=g.index))
    A = pd.concat(parts).sort_index()
    return pd.concat([WIN, A], axis=1)


# ---------- 3) 신호(signature) 분석 ----------

def pick_control_minute(row, rng):
    t, h2, bm = int(row.t), row.h2, row.onset_ext
    if np.isnan(h2) or np.isnan(bm): return np.nan
    h2 = int(h2); bm = int(bm)
    lo, hi = t + 5, h2 - 5
    if hi <= lo: return np.nan
    cands = [m for m in range(lo, hi + 1) if abs(m - bm) > 15]
    if not cands: return np.nan
    return cands[rng.integers(0, len(cands))]


def zdir(correct, n):
    if n < 20: return np.nan, np.nan
    p = correct / n
    return p, (p - 0.5) / np.sqrt(0.25 / n)


def signature_tables(WIN2, K):
    """onset_ext 전후 1/5/10분 특징 vs 같은 창 무작위 분(±15분 제외) — 심볼별로 처리."""
    rng = np.random.default_rng(SEED + 1)
    rows_t = []; rows_d = []
    valid = WIN2[(WIN2.D != 0) & (WIN2.amb2 == 0) & WIN2.onset_ext.notna() & WIN2.h2.notna()].copy()
    valid["ctrl_m"] = valid.apply(lambda r: pick_control_minute(r, rng), axis=1)
    valid = valid.dropna(subset=["ctrl_m"])
    valid["ctrl_m"] = valid.ctrl_m.astype(int); valid["onset_ext"] = valid.onset_ext.astype(int)
    P(f"  신호 분석 대상 창: {len(valid)} (D≠0·모호아님·대조분 확보)")
    valid["period"] = np.where(valid.yr <= 2024, "22-24", "25-26")
    LBS = RF.LOOKBACKS
    for s in SYMS:
        g = valid[valid.sym == s]
        if len(g) == 0: continue
        D_ = g.D.values
        before = RF.minute_panel(s, g.onset_ext.values - 1)
        ctrlB = RF.minute_panel(s, g.ctrl_m.values)
        after = {L: RF.minute_panel(s, g.onset_ext.values + L) for L in LBS}
        ctrlA = {L: RF.minute_panel(s, g.ctrl_m.values + L) for L in LBS}
        for L in LBS:
            for feat in RF.FEAT_NAMES:
                col = f"{feat}_{L}"
                phases = {"before": (before[col].values, ctrlB[col].values),
                          "after": (after[L][col].values, ctrlA[L][col].values)}
                for phase, (gv, cv) in phases.items():
                    for per in ("22-24", "25-26", "전체"):
                        m = (g.period.values == per) if per != "전체" else np.ones(len(g), bool)
                        for tier in ("T10", "T20", "T30", "CTRL", "전체등급"):
                            mt = m & ((g.tier.values == tier) if tier != "전체등급" else np.ones(len(g), bool))
                            gm = gv[mt]; cm_ = cv[mt]
                            av = np.abs(gm[~np.isnan(gm)]); ac = np.abs(cm_[~np.isnan(cm_)])
                            if len(av) >= 30 and len(ac) >= 30:
                                sd = np.sqrt((np.nanvar(av) + np.nanvar(ac)) / 2) + 1e-12
                                es = (np.nanmean(av) - np.nanmean(ac)) / sd
                                try:
                                    auc = roc_auc_score(np.r_[np.ones(len(av)), np.zeros(len(ac))], np.r_[av, ac])
                                except ValueError:
                                    auc = np.nan
                                rows_t.append(dict(sym=s, L=L, feat=feat, phase=phase, period=per, tier=tier,
                                                    n_g=len(av), n_c=len(ac), mean_g=np.nanmean(av), mean_c=np.nanmean(ac), es=es, auc=auc))
                            if feat in RF.SIGNED:
                                sv = gm[(~np.isnan(gm)) & (gm != 0)]; dd = D_[mt][(~np.isnan(gm)) & (gm != 0)]
                                if len(sv) >= 20:
                                    correct = (np.sign(sv) == dd).sum(); p_, z_ = zdir(correct, len(sv))
                                    rows_d.append(dict(sym=s, L=L, feat=feat, phase=phase, period=per, tier=tier, n=len(sv), frac=p_, z=z_))
    TT = pd.DataFrame(rows_t); DD = pd.DataFrame(rows_d)
    # 심볼 합쳐서 가중평균(표본수 가중) — 보고용 요약
    def agg_t(g):
        w = g.n_g + g.n_c
        return pd.Series(dict(n_g=g.n_g.sum(), n_c=g.n_c.sum(),
                               es=np.average(g.es, weights=w), auc=np.average(g.auc, weights=w)))
    TTg = TT.groupby(["L", "feat", "phase", "period", "tier"]).apply(agg_t).reset_index() if len(TT) else TT

    def agg_d(g):
        n = g.n.sum(); correct = (g.frac * g.n).sum()
        p_, z_ = zdir(correct, n)
        return pd.Series(dict(n=n, frac=p_, z=z_))
    DDg = DD.groupby(["L", "feat", "phase", "period", "tier"]).apply(agg_d).reset_index() if len(DD) else DD
    return TT, DD, TTg, DDg


# ---------- main ----------

if __name__ == "__main__":
    t0 = time.time()
    K = {s: store.load("klines_1m", s) for s in SYMS}
    WIN = build_windows()
    P(f"창 구성 완료 ({time.time()-t0:.0f}s) — 사건(T10/T20/T30) {sum(WIN.tier!='CTRL')}, 대조(CTRL) {sum(WIN.tier=='CTRL')}")
    P(WIN.groupby(["tier", "yr"]).size().unstack(fill_value=0).to_string())

    WIN2 = run_anatomy(WIN, K)
    WIN2["hit2"] = ((WIN2.h2 >= 0)).astype(int)
    WIN2["onset_delay_ext"] = WIN2.onset_ext - WIN2.t
    WIN2["onset_delay_ref"] = WIN2.onset_ref - WIN2.t
    WIN2["run_ext"] = WIN2.h2 - WIN2.onset_ext
    WIN2["run_ref"] = WIN2.h2 - WIN2.onset_ref
    P(f"해부 완료 ({time.time()-t0:.0f}s)")

    P("\n===== 사건·연도별 수 및 ±2% 적중률(240분 안에 h2 도달, amb2 포함) — R90 y 기저율과 비교")
    hit = WIN2.groupby(["tier"]).hit2.agg(["size", "mean"])
    P(hit.round(3).to_string())
    P(WIN2.groupby(["tier", "yr"]).hit2.mean().unstack(fill_value=np.nan).round(3).to_string())

    P("\n===== hindsight onset 분포(분, t 기준 지연) — onset_extreme(최저/최고가) vs onset_from_ref(기준선 재출발)")
    ok = WIN2[(WIN2.D != 0) & (WIN2.amb2 == 0)]
    for tier in ("T10", "T20", "T30", "CTRL"):
        g = ok[ok.tier == tier]
        if len(g) < 10: continue
        qe = g.onset_delay_ext.quantile([.25, .5, .75]).values
        qr = g.onset_delay_ref.dropna().quantile([.25, .5, .75]).values if g.onset_delay_ref.notna().sum() > 10 else (np.nan,) * 3
        P(f"  {tier:4s} n={len(g):5d} | onset_extreme 지연 분기 {qe[0]:.0f}/{qe[1]:.0f}/{qe[2]:.0f} | "
          f"onset_from_ref 지연 분기 {qr[0]:.0f}/{qr[1]:.0f}/{qr[2]:.0f} (결측 {g.onset_delay_ref.isna().mean():.2f}) | "
          f"false_starts(0.25%) 평균 {g.fs025.mean():.2f} (0.5%) {g.fs050.mean():.2f}")

    P("\n===== 첫 이탈(first excursion) 방향 == D 비율 — 50%에 가까워야 정상(아니면 ref/룩어헤드 버그 의심)")
    for xi, X in enumerate((0.1, 0.2, 0.3, 0.5)):
        fm = ok[f"fe_m{['1','2','3','5'][xi]}"]; fd = ok[f"fe_d{['1','2','3','5'][xi]}"]
        m = (fm >= 0) & (fd != 0)
        for tier in ("T10", "T20", "T30", "CTRL"):
            mm = m & (ok.tier == tier)
            n = mm.sum()
            if n < 20: continue
            corr = (fd[mm] == ok.D[mm]).sum(); p_, z_ = zdir(corr, n)
            P(f"  X={X:.1f}% {tier:4s} n={n:5d} 첫이탈이 D방향 비율 {p_:.3f} (z={z_:+.2f})")
        for per, yrs in (("22-24", (2022, 2023, 2024)), ("25-26", (2025, 2026))):
            mm = m & ok.yr.isin(yrs)
            n = mm.sum()
            if n < 20: continue
            corr = (fd[mm] == ok.D[mm]).sum(); p_, z_ = zdir(corr, n)
            P(f"    기간 {per} n={n:5d} 비율 {p_:.3f} (z={z_:+.2f})")

    WIN2.to_parquet(R / "data" / "r129_windows.parquet", index=False)
    P(f"\n저장: data/r129_windows.parquet ({len(WIN2)}행) ({time.time()-t0:.0f}s)")

    P("\n===== 신호(signature) 분석 — onset 전후 1/5/10분 vs 같은 창 무작위 분")
    TT, DD, TTg, DDg = signature_tables(WIN2, K)
    pd.set_option("display.width", 240); pd.set_option("display.max_rows", 200); pd.set_option("display.max_colwidth", 30)
    if len(TTg):
        P("\n-- TIMING 상위 |AUC-0.5| (전체등급·전체기간, before/after 모두) --")
        sel = TTg[(TTg.tier == "전체등급") & (TTg.period == "전체")].copy()
        sel["dist"] = (sel.auc - 0.5).abs()
        P(sel.sort_values("dist", ascending=False).head(20)[["L", "feat", "phase", "n_g", "n_c", "es", "auc"]].round(3).to_string(index=False))
        P("\n-- TIMING 기간 분리(22-24 vs 25-26, 전체등급) 상위 --")
        sel2 = TTg[(TTg.tier == "전체등급") & (TTg.period != "전체")].copy()
        sel2["dist"] = (sel2.auc - 0.5).abs()
        P(sel2.sort_values("dist", ascending=False).head(20)[["period", "L", "feat", "phase", "n_g", "n_c", "es", "auc"]].round(3).to_string(index=False))
        TT.to_parquet(R / "data" / "r129_sig.parquet", index=False)
        P(f"저장: data/r129_sig.parquet ({len(TT)}행, 심볼별 비집계)")
    if len(DDg):
        P("\n-- DIRECTION |z|>2 (방향 신호 — 사전 신호면 매매적 의미, 사후면 그냥 결과 확인) --")
        sig = DDg[(DDg.z.abs() > 2) & (DDg.tier == "전체등급")]
        P(sig.sort_values("z", key=lambda s: s.abs(), ascending=False).head(30).round(3).to_string(index=False))
    (R / "logs" / "r129.summary").write_text(f"R129 완료 ({time.time()-t0:.0f}s)\n")
    P(f"\n전체 완료 ({time.time()-t0:.0f}s)")
