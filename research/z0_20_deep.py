"""z0.20 심화 — 반변동성비대칭 정교화 + 호가 동적 지표(소모율·흡수율).
사용: python research/z0_20_deep.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from z0_20_book import DATA, evq, TH_MKT, TH_LIM
from freesia import store
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
SYM = list(DATA.keys())

P("\n" + "="*140)
P("[1] 반변동성 비대칭 정교화 — 최고 신호의 변형 탐색")
P("="*140)
for s in SYM:
    d = DATA[s]; k = d["k"]; pos = d["pos"]; y = d["y"]
    c = k.close.values; qv = k.quote_volume.values; h = k.high.values; l = k.low.values
    lr = pd.Series(np.log(c)).diff()
    P(f"\n  ══ {s} ══")
    S = {}
    for w in (2880, 4320, 8640):
        us = pd.Series(np.where(lr > 0, lr**2, 0)).rolling(w).sum()
        ds = pd.Series(np.where(lr < 0, lr**2, 0)).rolling(w).sum()
        S[f"기본{w}"] = ((us-ds)/(us+ds)).values[pos]
        # 절대값(변동성 크기 무관)
        ua = pd.Series(np.where(lr > 0, np.abs(lr), 0)).rolling(w).sum()
        da = pd.Series(np.where(lr < 0, np.abs(lr), 0)).rolling(w).sum()
        S[f"절대값{w}"] = ((ua-da)/(ua+da)).values[pos]
        # 3제곱(꼬리 강조)
        u3 = pd.Series(np.where(lr > 0, lr**3, 0)).rolling(w).sum()
        d3 = pd.Series(np.where(lr < 0, -lr**3, 0)).rolling(w).sum()
        S[f"3제곱{w}"] = ((u3-d3)/(u3+d3)).values[pos]
        # 거래량 가중
        uv = pd.Series(np.where(lr > 0, lr**2*qv, 0)).rolling(w).sum()
        dv = pd.Series(np.where(lr < 0, lr**2*qv, 0)).rolling(w).sum()
        S[f"거래량가중{w}"] = ((uv-dv)/(uv+dv)).values[pos]
        # 극단만 (상위 10% 크기 움직임)
        thr = lr.abs().rolling(w).quantile(0.9)
        ue = pd.Series(np.where((lr > 0) & (lr.abs() > thr), lr**2, 0)).rolling(w).sum()
        de = pd.Series(np.where((lr < 0) & (lr.abs() > thr), lr**2, 0)).rolling(w).sum()
        S[f"극단만{w}"] = ((ue-de)/(ue+de)).values[pos]
        # 개수 기반 (크기 무관, 빈도)
        un = pd.Series((lr > 0).astype(float)).rolling(w).sum()
        dn = pd.Series((lr < 0).astype(float)).rolling(w).sum()
        S[f"개수{w}"] = ((un-dn)/(un+dn)).values[pos]
        # 고저 기반 (파킨슨 비대칭)
        up_rng = pd.Series(np.log(h/np.maximum(c, np.roll(c, 1)))).rolling(w).sum()
        dn_rng = pd.Series(np.log(np.minimum(c, np.roll(c, 1))/l)).rolling(w).sum()
        S[f"고저비대칭{w}"] = ((up_rng-dn_rng)/(up_rng+dn_rng)).values[pos]
    # EWMA 버전
    for hl in (720, 2160, 4320):
        ue = pd.Series(np.where(lr > 0, lr**2, 0)).ewm(halflife=hl).mean()
        de = pd.Series(np.where(lr < 0, lr**2, 0)).ewm(halflife=hl).mean()
        S[f"EWMA반감기{hl}"] = ((ue-de)/(ue+de)).values[pos]
    res = []
    for nm, v in S.items():
        b = evq(np.asarray(v, float), y, nm)
        if b: res.append((b, nm))
    res.sort(reverse=True)
    P(f"    → 최고: " + ", ".join(f"{n}({b:.2f}%)" for b, n in res[:5]))
    d["S1"] = S

P("\n" + "="*140)
P("[2] 호가 동적 지표 — 정적 잔량이 아니라 '소모·보충' 속도")
P("="*140)
for s in SYM:
    d = DATA[s]; bd = d["bd"]; y = d["y"]; k = d["k"]; pos = d["pos"]
    qv = k.quote_volume.values; tb = k.taker_buy_quote_volume.values
    P(f"\n  ══ {s} ══")
    S = {}
    nb1 = bd["n-1"].values; na1 = bd["n+1"].values
    nb5 = sum(bd[f"n-{i}"].values for i in (1, 2, 3, 4, 5))
    na5 = sum(bd[f"n+{i}"].values for i in (1, 2, 3, 4, 5))
    # 잔량 소모율: 잔량 변화 / 그 사이 체결량
    for w in (5, 20, 60):
        dbid = pd.Series(nb1).diff(w).values
        dask = pd.Series(na1).diff(w).values
        vol = pd.Series(qv).rolling(w).sum().values[pos]
        vol = np.maximum(vol, 1.0)
        S[f"매수잔량소모{w}"] = -dbid/vol
        S[f"매도잔량소모{w}"] = -dask/vol
        S[f"소모비대칭{w}"] = (-dask+dbid)/vol
    # 흡수율: 테이커 매도량 대비 매수잔량 유지
    for w in (20, 60):
        tsell = (pd.Series(qv).rolling(w).sum()-pd.Series(tb).rolling(w).sum()).values[pos]
        tbuy = pd.Series(tb).rolling(w).sum().values[pos]
        S[f"매수흡수력{w}"] = nb1/np.maximum(tsell, 1.0)
        S[f"매도흡수력{w}"] = na1/np.maximum(tbuy, 1.0)
        S[f"흡수비대칭{w}"] = (nb1/np.maximum(tsell, 1.0)-na1/np.maximum(tbuy, 1.0))
    # 잔량 대비 체결 압력
    vol60 = pd.Series(qv).rolling(60).sum().values[pos]
    S["압력비"] = vol60/np.maximum(nb5+na5, 1.0)
    # 잔량 변화율 자체
    for w in (20, 60, 240):
        S[f"매수잔량증가{w}"] = pd.Series(nb1).pct_change(w).values
        S[f"매도잔량증가{w}"] = pd.Series(na1).pct_change(w).values
        S[f"잔량증가차{w}"] = (pd.Series(nb1).pct_change(w)-pd.Series(na1).pct_change(w)).values
    res = []
    for nm, v in S.items():
        b = evq(np.asarray(v, float), y, nm)
        if b: res.append((b, nm))
    res.sort(reverse=True)
    P(f"    → 최고: " + ", ".join(f"{n}({b:.2f}%)" for b, n in res[:5]))
    d["S2"] = S

P("\n" + "="*140)
P("[3] 최종 결합 — 가격 최고 + 호가 최고")
P("="*140)
def z(v):
    v = np.asarray(v, float); m = np.isfinite(v)
    o = np.full(len(v), np.nan)
    if m.sum() > 100:
        mu = v[m].mean(); sd = v[m].std()
        if sd > 0: o[m] = (v[m]-mu)/sd
    return o
for s in SYM:
    d = DATA[s]; y = d["y"]
    S1 = d["S1"]; S2 = d["S2"]
    r1 = sorted([(evq(np.asarray(v, float), y, "", show=False) or 0, n) for n, v in S1.items()], reverse=True)
    r2 = sorted([(evq(np.asarray(v, float), y, "", show=False) or 0, n) for n, v in S2.items()], reverse=True)
    P(f"\n  ══ {s} ══ 가격상위: {[n for _,n in r1[:3]]} / 호가상위: {[n for _,n in r2[:3]]}")
    for k_, lab in ((3, "가격3"), (5, "가격5")):
        M = np.vstack([z(S1[n]) for _, n in r1[:k_]])
        evq(np.nanmean(M, axis=0), y, f"결합 {lab}")
    for k_ in (2, 3):
        M = np.vstack([z(S2[n]) for _, n in r2[:k_]])
        evq(np.nanmean(M, axis=0), y, f"결합 호가{k_}")
    M = np.vstack([z(S1[n]) for _, n in r1[:3]] + [z(S2[n]) for _, n in r2[:2]])
    evq(np.nanmean(M, axis=0), y, "결합 가격3+호가2")
    M = np.vstack([z(S1[n]) for _, n in r1[:5]] + [z(S2[n]) for _, n in r2[:3]])
    evq(np.nanmean(M, axis=0), y, "결합 가격5+호가3")
