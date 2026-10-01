"""R118d — 미국 현물 ETF 일별 순유입(data/ext/etf_flows.parquet)이 다음날 BTC·ETH 수익을 예측하는가.

가설(Lim 2025, SSRN 6592830): ETF 일별 순유입이 다음날 BTC 수익을 예측(Newey-West t≈3.1, 2024-01~2025-04),
양방향 피드백(수익 → 유입도 유발)도 있음. 타이밍: 미국 거래일 D의 유입액은 미국 장 마감(약 21:00 UTC) 이후,
흔히 그날 저녁 늦게 공개된다 → D의 유입은 D+1 00:00 UTC부터 사용 가능하다고 보수적으로 가정(민감도로 D+1 13:30 UTC도 확인).

특징(자산별로 그 자산의 ETF 유입·수익만 사용, BTC↔BTC, ETH↔ETH):
  flow      : 그날 유입(USD)
  flow_z20  : flow / 유입의 20일 롤링 표준편차(인과적, 미래 데이터 없음)
  sum3·sum5 : 유입의 3일·5일 합(트레일링)
  resid     : flow를 같은 날 현물(여기선 선물 종가 기준) 수익에 대해 확장창(expanding, 과거만) OLS로 회귀한 잔차
              (수익→유입 피드백 제거). 최소 30일 쌓일 때까지는 NaN.
거래 규칙에 쓰는 "z"는 위 5개 특징 모두를 20일 롤링 평균·표준편차로 다시 표준화한 값(비교 가능하게 통일).

목표: 진입 D+1 00:00 UTC 시가 → D+1 24:00(=D+2 00:00 UTC) 종가(1일 보유), 및 D+3 24:00(=D+4 00:00 UTC) 종가(3일 보유).
검정(사전 등록, 결과를 본 뒤 변형 추가 금지):
  - IC(스피어만) · Newey-West t(HAC, Bartlett 커널): 2024(학습) / 2025~26(평가) / 전체, 분기별 부호 일치율.
  - 거래 규칙: z > z*면 롱, z < −z*면 숏, z* ∈ {0.5, 1.0}, 보유 1일(겹침 없음) · 3일(겹치지 않게 선택).
    비용 = 테이커 5bp/사이드 × 2 + 슬리피지 2bp/사이드 × 2 (왕복 14bp, R113 COST와 동일) + 실제 펀딩(J.add_funding).
    특징·z*·보유기간은 2024만으로 고른다(거래 ≥10건 제약) → 2025~26 결과를 보고한다.
  - 선택된 규칙으로 표준 사건(e,x,ret,mae,k=4,mult=1,s,px,side,rule) 생성 → data/r118d_events.parquet 저장.
  - 공동 계좌: core = std_dip()+std_breakout(), run_multi(core+etf, 3, 12, {1:12/48, 4:We}, start=2024-01-01)
    We ∈ {0.5,1,2,4}, core 단독도 같은 시작으로 비교. ETF 전략(채널4 단독) vs core 단독의 일간 수익 상관도 보고.
사용: python research/r118d_etf_flows.py (로그는 호출 시 logs/r118d.log 로 리다이렉트)
"""
import sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store, costs
import r102_joint_account as J
from r118_engine import std_dip, std_breakout, run_multi, summary

P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
SLIP = 2e-4; ROUNDTRIP = 2 * (costs.TAKER + SLIP)  # = 14bp, R113 COST와 동일
ZSTAR = (0.5, 1.0); HOLDS = (1, 3)
ASSETS = {"BTC": ("btc_total", "BTCUSDT", 0), "ETH": ("eth_total", "ETHUSDT", 1)}


# ---------- 가격 격자 ----------
def px_grid(sym):
    """절대 분(2020-01-01 UTC = 0) 인덱스로 가득 채운 시가·저가·고가 배열."""
    k = store.load("klines_1m", sym); mi = J.mins(k.ts); T1 = mi.max() + 1
    o = np.full(T1, np.nan); lo = np.full(T1, np.nan); hi = np.full(T1, np.nan); cl = np.full(T1, np.nan)
    o[mi] = k.open.values; lo[mi] = k.low.values; hi[mi] = k.high.values; cl[mi] = k.close.values
    o = pd.Series(o).ffill().bfill().values; lo = pd.Series(lo).ffill().bfill().values
    hi = pd.Series(hi).ffill().bfill().values; cl = pd.Series(cl).ffill().bfill().values
    return o, lo, hi, cl


# ---------- 통계: IC(스피어만) · Newey-West HAC t ----------
def nw_ols_t(y, x, lag):
    """y ~ a + b*zscore(x) 의 베타·Newey-West(Bartlett) t값. 표본 부족 시 NaN."""
    ok = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x, float)[ok]; y = np.asarray(y, float)[ok]
    if len(x) < 30 or x.std() < 1e-12: return np.nan, np.nan, int(ok.sum())
    xz = (x - x.mean()) / x.std()
    X = np.column_stack([np.ones(len(xz)), xz])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    XtXi = np.linalg.inv(X.T @ X)
    u = X * resid[:, None]
    S = u.T @ u
    for l in range(1, lag + 1):
        w = 1 - l / (lag + 1)
        G = u[l:].T @ u[:-l]
        S += w * (G + G.T)
    cov = XtXi @ S @ XtXi
    se = np.sqrt(max(cov[1, 1], 1e-30))
    return beta[1], beta[1] / se, len(xz)


def ic_report(feat, y, dates, lag):
    tr = dates.year == 2024; ev = dates.year >= 2025
    rows = {}
    for lab, m in (("학습24", tr), ("평가25~26", ev), ("전체", np.ones(len(dates), bool))):
        f = feat[m]; yy = y[m]
        ok = np.isfinite(f) & np.isfinite(yy)
        ic = spearmanr(f[ok], yy[ok])[0] if ok.sum() >= 20 else np.nan
        _, t, n = nw_ols_t(yy, f, lag)
        rows[lab] = dict(ic=ic, t=t, n=n)
    # 분기별 부호 일치
    q = dates.to_period("Q")
    sign_all = np.sign(rows["전체"]["ic"]) if np.isfinite(rows["전체"]["ic"]) else 0
    same = []
    for qq in sorted(set(q)):
        m = np.asarray(q == qq)
        f = feat[m]; yy = y[m]; ok = np.isfinite(f) & np.isfinite(yy)
        if ok.sum() < 15: continue
        icq = spearmanr(f[ok], yy[ok])[0]
        if np.isfinite(icq): same.append(np.sign(icq) == sign_all)
    rows["분기부호일치"] = np.mean(same) if same else np.nan
    rows["분기수"] = len(same)
    return rows


# ---------- 특징 ----------
def expanding_resid(y, x, min_n=30):
    """y를 x에 대해 과거 데이터만으로 확장창 OLS 회귀한 뒤 잔차(미래데이터 없음)."""
    n = len(y); out = np.full(n, np.nan)
    sx = sy = sxx = sxy = 0.0; cnt = 0
    for i in range(n):
        if cnt >= min_n:
            mx = sx / cnt; my = sy / cnt; vxx = sxx / cnt - mx * mx; vxy = sxy / cnt - mx * my
            b = vxy / vxx if vxx > 1e-12 else 0.0; a = my - b * mx
            if np.isfinite(y[i]) and np.isfinite(x[i]): out[i] = y[i] - (a + b * x[i])
        if np.isfinite(x[i]) and np.isfinite(y[i]):
            sx += x[i]; sy += y[i]; sxx += x[i] * x[i]; sxy += x[i] * y[i]; cnt += 1
    return out


def zscore(s, win=20, minp=15):
    m = s.rolling(win, min_periods=minp).mean(); v = s.rolling(win, min_periods=minp).std()
    return (s - m) / v.replace(0, np.nan)


def build(asset, F, o, lo, hi, cl, entry_lag_min=1440):
    """asset별 특징·목표를 D(유입일) 인덱스로 구성. entry_lag_min: D일 00:00 UTC로부터 진입까지 분(기본 1일=1440)."""
    col, sym, s_ = ASSETS[asset]
    f = F[["date", col]].dropna().rename(columns={col: "flow"}).reset_index(drop=True)
    dmin = J.mins(f.date)  # D 00:00 UTC
    dmin_c = np.clip(dmin, 0, len(o) - 1)
    e = dmin + entry_lag_min  # 진입 분(기본 D+1 00:00 UTC)
    ret_same = o[np.clip(dmin + 1440, 0, len(o) - 1)] / o[dmin_c] - 1  # D 00:00~D+1 00:00(그날 수익) 근사
    flow = f.flow.values.astype(float)
    std20 = pd.Series(flow).rolling(20, min_periods=15).std().values
    feats = {
        "flow": flow,
        "flow_z20": flow / np.where(std20 > 0, std20, np.nan),
        "sum3": pd.Series(flow).rolling(3).sum().values,
        "sum5": pd.Series(flow).rolling(5).sum().values,
        "resid": expanding_resid(flow, ret_same, min_n=30),
    }
    # 보유 1·3일 목표 수익(진입 시가 → 청산 종가, 비용·펀딩 전 raw); 데이터 범위를 벗어나면 NaN
    x1 = e + 1440; x3 = e + 3 * 1440
    e_c = np.clip(e, 0, len(o) - 1)
    valid1 = (e >= 0) & (x1 < len(cl)); valid3 = (e >= 0) & (x3 < len(cl))
    ret1 = np.where(valid1, cl[np.clip(x1 - 1, 0, len(cl) - 1)] / o[e_c] - 1, np.nan)
    ret3 = np.where(valid3, cl[np.clip(x3 - 1, 0, len(cl) - 1)] / o[e_c] - 1, np.nan)
    D = pd.DataFrame(dict(date=f.date, e=e, x1=x1, x3=x3, ret1=ret1, ret3=ret3))
    for k, v in feats.items(): D[k] = v
    D["z_" + "flow"] = zscore(pd.Series(flow)).values
    D["z_flow_z20"] = zscore(pd.Series(feats["flow_z20"])).values
    D["z_sum3"] = zscore(pd.Series(feats["sum3"])).values
    D["z_sum5"] = zscore(pd.Series(feats["sum5"])).values
    D["z_resid"] = zscore(pd.Series(feats["resid"])).values
    D["sym"] = sym; D["s"] = s_
    return D


# ---------- 거래 규칙 ----------
def rule_trades(D, zcol, hold, zstar, o, lo, hi, cl):
    z = D[zcol].values; dates = D.date.values
    sel = np.where(np.abs(z) > zstar)[0]
    if hold == 3:
        keep = []; last_exit = np.datetime64("1970-01-01")
        for i in sel:
            if dates[i] <= last_exit: continue
            keep.append(i); last_exit = dates[i] + np.timedelta64(hold, "D")
        sel = np.array(keep, dtype=int)
    if len(sel) == 0:
        return pd.DataFrame(columns=["e", "x", "ret", "mae", "px", "side", "date"])
    side = np.sign(z[sel]).astype(float)
    e = D.e.values[sel].astype(np.int64); x = e + hold * 1440
    ok = x < len(cl)
    sel, side, e, x = sel[ok], side[ok], e[ok], x[ok]
    entry = o[e]; exitp = cl[x - 1]
    T = pd.DataFrame(dict(sym=D.sym.iloc[0], e=e, x=x))
    T = J.add_funding(T)
    ret = side * (exitp / entry - 1) - ROUNDTRIP - side * T.fund.values
    mae = np.array([max(1 - lo[a:b].min() / px, 0) if sd > 0 else max(hi[a:b].max() / px - 1, 0)
                    for a, b, px, sd in zip(e, x, entry, side)])
    return pd.DataFrame(dict(e=e, x=x, ret=ret, mae=mae, px=entry, side=side.astype(int), date=D.date.values[sel]))


def pick_rule(D, o, lo, hi, cl, label):
    """2024만으로 (특징, z*, 보유) 격자에서 평균 순수익 최대인 조합을 고르고(거래>=10건), 2025~26 성과를 보고."""
    feats = ["flow", "flow_z20", "sum3", "sum5", "resid"]
    rows = []
    for fe in feats:
        zcol = "z_" + fe
        for hold in HOLDS:
            for zs in ZSTAR:
                Tt = rule_trades(D, zcol, hold, zs, o, lo, hi, cl)
                if len(Tt) == 0: continue
                tr = Tt[pd.DatetimeIndex(Tt.date).year == 2024]
                ev = Tt[pd.DatetimeIndex(Tt.date).year >= 2025]
                rows.append(dict(feat=fe, hold=hold, zstar=zs, n_tr=len(tr), bp_tr=tr.ret.mean() * 1e4 if len(tr) else np.nan,
                                 n_ev=len(ev), bp_ev=ev.ret.mean() * 1e4 if len(ev) else np.nan,
                                 wr_ev=(ev.ret > 0).mean() if len(ev) else np.nan))
    O = pd.DataFrame(rows)
    P(f"  [{label}] 격자 전체({len(O)}개) 학습24 bp(건수) 상위 8개:")
    P(O.sort_values("bp_tr", ascending=False).head(8).round(3).to_string(index=False))
    cand = O[O.n_tr >= 10]
    if cand.empty: cand = O[O.n_tr >= 5]
    if cand.empty:
        P(f"  [{label}] 2024 거래 5건 이상인 조합 없음 — 규칙 선택 불가"); return None, O
    best = cand.loc[cand.bp_tr.idxmax()]
    P(f"  [{label}] 선택(2024 평균bp 최대, n>=10): feat={best.feat} z*={best.zstar} hold={int(best.hold)}일 "
      f"| 학습24 {best.n_tr:.0f}건 {best.bp_tr:+.1f}bp | 평가25~26 {best.n_ev:.0f}건 {best.bp_ev:+.1f}bp 승률 {best.wr_ev*100:.0f}%")
    return best, O


if __name__ == "__main__":
    t_ = time.time()
    F = pd.read_parquet(R / "data" / "ext" / "etf_flows.parquet")
    F["date"] = pd.DatetimeIndex(F.date).tz_convert("UTC") if pd.DatetimeIndex(F.date).tz is not None else pd.DatetimeIndex(F.date).tz_localize("UTC")
    P(f"ETF 유입 로드 {F.shape}, BTC {F.btc_total.notna().sum()}일 ETH {F.eth_total.notna().sum()}일 ({time.time()-t_:.0f}s)")

    GRID = {}
    for asset, (_, sym, _) in ASSETS.items(): GRID[sym] = px_grid(sym)

    DATA = {}; SENS = {}
    for asset, (col, sym, s_) in ASSETS.items():
        o, lo, hi, cl = GRID[sym]
        D = build(asset, F, o, lo, hi, cl, entry_lag_min=1440)
        D13 = build(asset, F, o, lo, hi, cl, entry_lag_min=1440 + 13 * 60 + 30)  # D+1 13:30 UTC 진입(민감도)
        DATA[asset] = D; SENS[asset] = D13
        P(f"\n===== {asset} 특징·목표 {len(D)}행 ({time.time()-t_:.0f}s)")

        # IC / Newey-West t
        rows = []
        for fe in ("flow", "flow_z20", "sum3", "sum5", "resid"):
            for hz, hcol, lag in ((1, "ret1", 5), (3, "ret3", 8)):
                rep = ic_report(D[fe].values, D[hcol].values, pd.DatetimeIndex(D.date), lag)
                rows.append(dict(feat=fe, hold=hz, ic_tr=rep["학습24"]["ic"], t_tr=rep["학습24"]["t"], n_tr=rep["학습24"]["n"],
                                 ic_ev=rep["평가25~26"]["ic"], t_ev=rep["평가25~26"]["t"], n_ev=rep["평가25~26"]["n"],
                                 ic_all=rep["전체"]["ic"], t_all=rep["전체"]["t"],
                                 분기부호일치=rep["분기부호일치"], 분기수=rep["분기수"]))
        IC = pd.DataFrame(rows)
        P(f"[{asset}] IC·Newey-West t (학습24 / 평가25~26 / 전체):")
        P(IC.round(3).to_string(index=False))

        # 민감도: D+1 13:30 UTC부터 사용 가능하다고 가정했을 때 IC·t (1일 보유만)
        rows13 = []
        for fe in ("flow", "flow_z20", "sum3", "sum5", "resid"):
            rep = ic_report(D13[fe].values, D13["ret1"].values, pd.DatetimeIndex(D13.date), 5)
            rows13.append(dict(feat=fe, ic_tr=rep["학습24"]["ic"], t_tr=rep["학습24"]["t"],
                               ic_ev=rep["평가25~26"]["ic"], t_ev=rep["평가25~26"]["t"]))
        P(f"[{asset}] 민감도(진입 D+1 13:30 UTC, 1일 보유): " + " | ".join(
            f"{r['feat']} t학습{r['t_tr']:+.1f} t평가{r['t_ev']:+.1f}" for r in rows13))

    # 거래 규칙 선택(2024) → 2025~26 보고
    EVENTS = []; RULES = {}
    for asset, (col, sym, s_) in ASSETS.items():
        o, lo, hi, cl = GRID[sym]
        best, grid_tbl = pick_rule(DATA[asset], o, lo, hi, cl, asset)
        if best is None: continue
        RULES[asset] = best
        Tt = rule_trades(DATA[asset], "z_" + best.feat, int(best.hold), best.zstar, o, lo, hi, cl)
        Tt["s"] = s_; Tt["k"] = 4; Tt["mult"] = 1.0
        Tt["rule"] = f"{asset} feat={best.feat} z{best.zstar} hold{int(best.hold)}d"
        EVENTS.append(Tt[["e", "x", "ret", "mae", "k", "mult", "s", "px", "side", "rule"]])

    if EVENTS:
        E = pd.concat(EVENTS, ignore_index=True).sort_values("e").reset_index(drop=True)
        E.to_parquet(R / "data" / "r118d_events.parquet", index=False)
        P(f"\n표준 사건 저장: data/r118d_events.parquet {E.shape}, 규칙: " + " | ".join(E.rule.unique()))
    else:
        E = pd.DataFrame(columns=["e", "x", "ret", "mae", "k", "mult", "s", "px", "side", "rule"])
        P("\n경고: 선택된 규칙이 없어 사건을 생성하지 못함")

    # 공동 계좌
    P(f"\n===== 공동 계좌 ({time.time()-t_:.0f}s)")
    core = pd.concat([std_dip(), std_breakout()], ignore_index=True)
    d_core, pk_core = run_multi(core, 3, 12, {1: 12 / 48}, start="2024-01-01")
    if d_core is None:
        P("  core 단독(2024~): 파산")
    else:
        P(f"  core 단독(2024~): " + str({k: (round(v, 2) if isinstance(v, float) else v) for k, v in summary(d_core).items()}))

    etf_cols = ["e", "x", "ret", "mae", "k", "mult", "s", "px", "side"]
    etf_only = E[etf_cols] if len(E) else pd.DataFrame(columns=etf_cols)
    for We in (0.5, 1, 2, 4):
        if len(etf_only):
            d, pk = run_multi(pd.concat([core, etf_only], ignore_index=True), 3, 12, {1: 12 / 48, 4: We}, start="2024-01-01")
        else:
            d = None
        if d is None:
            P(f"  core+ETF We={We}: 파산 또는 사건 없음")
        else:
            P(f"  core+ETF We={We}: " + str({k: (round(v, 2) if isinstance(v, float) else v) for k, v in summary(d).items()}))

    if len(etf_only):
        d_etf, pk_etf = run_multi(etf_only, 3, 12, {4: 1.0}, start="2024-01-01")
        if d_etf is not None and d_core is not None:
            both = pd.concat([d_core.rename("core"), d_etf.rename("etf")], axis=1).dropna()
            both = both[both.index >= "2024-01-01"]  # run_multi 시작 전 구간은 두 시리즈 모두 평평한 0% 채움값이라 상관계수가 부풀려짐 → 제외
            corr = both.core.corr(both.etf)
            P(f"  ETF 전략 단독(We=1): " + str({k: (round(v, 2) if isinstance(v, float) else v) for k, v in summary(d_etf).items()}))
            P(f"  core·ETF 일간 수익 상관계수 (n={len(both)}): {corr:+.3f}")
        else:
            P("  ETF 전략 단독 또는 core가 파산해 상관계수 계산 불가")

    (R / "logs" / "r118d.summary").write_text("done\n")
    P(f"\n완료 ({time.time()-t_:.0f}s)")
