"""숏 방향 전용 연구 — 크립토의 구조적 비대칭(펀딩 수취·롱 청산 우위·하락 속도)을 숏이 먹을 수 있는가.
S1 기저율(펀딩 포함 숏 총수익)  S2 펀딩 조건부 무헤지 숏  S3 비대칭 배리어  S4 과열 조건부 숏  S5 방향 비대칭 감사
규율: 학습 2020~2023 / 테스트 2024~, 비용 테이커 12bp, 위약 비교, 시도 기록.
사용: python research/short_study.py"""
import sys
import warnings
from pathlib import Path

import numba
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from freesia import store  # noqa: E402
from freesia.bt import core, ind  # noqa: E402

TE = pd.Timestamp("2024-01-01", tz="UTC")
rng = np.random.default_rng(53)
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)


def tstat(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return float(x.mean() / x.std() * np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else 0.0


@numba.njit(cache=True)
def barrier_a(hh, ll, cc, tp, sl, H, side):
    """side=1 롱, -1 숏. 숏은 tp=하락목표, sl=상승손절."""
    n = len(cc); r = np.zeros(n); hold = np.zeros(n, np.int32)
    for i in range(n - 1):
        e = cc[i]
        if side == 1:
            up = e * (1 + tp); dn = e * (1 - sl)
        else:
            up = e * (1 + sl); dn = e * (1 - tp)
        done = False
        for j in range(i + 1, min(i + 1 + H, n)):
            if hh[j] >= up:
                r[i] = tp if side == 1 else -sl; hold[i] = j - i; done = True; break
            if ll[j] <= dn:
                r[i] = -sl if side == 1 else tp; hold[i] = j - i; done = True; break
        if not done:
            j = min(i + H, n - 1); g = cc[j] / e - 1
            r[i] = g if side == 1 else -g; hold[i] = j - i
    return r, hold


@numba.njit(cache=True)
def pick(ev, hold):
    out = np.zeros(len(ev), np.bool_); nxt = 0
    for i in range(len(ev)):
        if ev[i] and i >= nxt:
            out[i] = True; nxt = i + hold[i] + 1
    return out


for sym in ("BTCUSDT", "ETHUSDT"):
    b = core.bars(sym, "1h")
    idx = b.index
    f = store.load("funding", sym).set_index("ts").rate
    fh = f.resample("1h").sum().reindex(idx).fillna(0.0).values     # 시간당 펀딩(숏 수취 부호 +)
    ret = np.log(b.close).diff().shift(-1).fillna(0).values
    tr = idx < TE
    P("=" * 96 + f"\n### {sym} 1시간봉 {len(b):,}개")

    # ── S1. 기저율: 무조건 숏/롱 보유 총수익(펀딩 포함), 연도별
    P("\n[S1] 무조건 보유 연 환산 % — 숏은 가격 하락 + 펀딩 수취, 롱은 상승 - 펀딩 지불")
    rows = []
    for y in range(2020, 2027):
        m = idx.year == y
        if m.sum() < 100:
            continue
        pr = ret[m].sum() * 100
        fu = fh[m].sum() * 100
        n = m.sum() / (24 * 365)
        rows.append((y, pr / n, fu / n, (pr - fu) / n, (-pr + fu) / n))
    d = pd.DataFrame(rows, columns=["연", "가격 연%", "펀딩 연%", "롱 총 연%", "숏 총 연%"])
    P(d.round(1).to_string(index=False))
    P(f"  전체: 가격 {ret.sum()*100/ (len(b)/(24*365)):.1f}%/년, 펀딩 {fh.sum()*100/(len(b)/(24*365)):.1f}%/년 → "
      f"롱 {(ret.sum()-fh.sum())*100/(len(b)/(24*365)):.1f}%, 숏 {(-ret.sum()+fh.sum())*100/(len(b)/(24*365)):.1f}%")

    # ── S2. 펀딩 조건부 무헤지 숏
    P("\n[S2] 펀딩 조건부 무헤지 숏 — 직전 3일 평균 펀딩 > 문턱일 때만 숏 (총수익 = -가격 + 펀딩)")
    fsig = pd.Series(fh, index=idx).rolling(72).mean().shift(1).values
    rows = []
    for th in (0.0, 0.00001, 0.00002, 0.00004):        # 시간당 펀딩 문턱
        pos = (fsig > th).astype(float)
        tot = pos * (-ret + fh) - np.abs(np.diff(pos, prepend=0.0)) * 0.0006
        for nm, m in (("학습", tr), ("테스트", ~tr)):
            n = m.sum() / (24 * 365)
            a = tot[m].sum() / n * 100
            sh = tot[m].mean() / tot[m].std() * np.sqrt(24 * 365) if tot[m].std() > 0 else 0
            eq = np.cumprod(1 + tot[m]); mdd = float((eq / np.maximum.accumulate(eq) - 1).min())
            rows.append((th * 24 * 365 * 100, nm, a, sh, mdd * 100, float(pos[m].mean())))
    P(pd.DataFrame(rows, columns=["문턱(연%)", "구간", "연%", "Sharpe", "MDD%", "가동률"]).round(2).to_string(index=False))

    # ── S3. 비대칭 배리어: 숏 전용 목표·손절 격자 (신호는 단순 추세역행)
    P("\n[S3] 숏 전용 비대칭 배리어 — 신호: 24시간 상승 상위 10%(과열) 후 숏, 총수익 bp(펀딩 포함)")
    up24 = pd.Series(np.log(b.close)).diff(24).values
    hot = up24 > pd.Series(up24).rolling(720).quantile(0.9).values
    rows = []
    ntry = 0
    for tp in (0.01, 0.02, 0.03):
        for sl in (0.01, 0.02, 0.03, 0.05):
            for H in (24, 72, 168):
                ntry += 1
                r, hold = barrier_a(b.high.values, b.low.values, b.close.values, tp, sl, H, -1)
                fcum = pd.Series(fh, index=idx).rolling(H).sum().shift(-H).fillna(0).values * (hold / H)
                tot = (r + fcum) * 1e4 - 12
                for nm, m in (("학습", tr), ("테스트", ~tr)):
                    ev = hot & m & ~np.isnan(hot)
                    sel = pick(ev, hold)
                    if sel.sum() < 20:
                        continue
                    rows.append((tp * 100, sl * 100, H, nm, tot[sel].mean(), tstat(tot[sel]), int(sel.sum())))
    dd = pd.DataFrame(rows, columns=["목표%", "손절%", "H", "구간", "순bp", "t", "n"])
    pv = dd.pivot_table(index=["목표%", "손절%", "H"], columns="구간", values=["순bp", "t", "n"])
    P(pv.round(1).to_string())
    te_ = dd[dd.구간 == "테스트"]; tr_ = dd[dd.구간 == "학습"]
    P(f"  시도 {ntry} | 학습 양수 {int((tr_.순bp>0).sum())}/{len(tr_)}, 테스트 양수 {int((te_.순bp>0).sum())}/{len(te_)}, 최고 테스트 t {te_.t.max():.2f}")

    # ── S4. 과열 조건 조합 숏 (펀딩·OI·가격신장)
    P("\n[S4] 과열 조건부 숏 — 조건 조합별 (목표2%/손절2%/72시간, 펀딩 포함, 비용 12bp)")
    try:
        oi = store.load("metrics_5m", sym).set_index("ts")["sum_open_interest"].resample("1h").last().reindex(idx)
        doi = oi.pct_change(24).values
    except Exception:
        doi = np.full(len(b), np.nan)
    z24 = (up24 - pd.Series(up24).rolling(720).mean().values) / pd.Series(up24).rolling(720).std().values
    fz = (pd.Series(fh, index=idx).rolling(24).mean() - pd.Series(fh, index=idx).rolling(720).mean()) / pd.Series(fh, index=idx).rolling(720).std()
    rsi = ind.rsi(b.close, 14).values
    conds = {"가격z>1.5": z24 > 1.5, "가격z>2.5": z24 > 2.5, "펀딩z>1.5": fz.values > 1.5,
             "RSI>75": rsi > 75, "OI24h>5%": doi > 0.05}
    r2, hold2 = barrier_a(b.high.values, b.low.values, b.close.values, 0.02, 0.02, 72, -1)
    fc2 = pd.Series(fh, index=idx).rolling(72).sum().shift(-72).fillna(0).values * (hold2 / 72)
    tot2 = (r2 + fc2) * 1e4 - 12
    import itertools
    rows = []
    for k in (1, 2, 3):
        for combo in itertools.combinations(conds, k):
            m0 = np.ones(len(b), bool)
            for c in combo:
                m0 &= np.nan_to_num(conds[c], nan=0).astype(bool)
            for nm, mm in (("학습", tr), ("테스트", ~tr)):
                sel = pick(m0 & mm, hold2)
                if sel.sum() < 20:
                    continue
                rows.append((" + ".join(combo), nm, tot2[sel].mean(), tstat(tot2[sel]), int(sel.sum()),
                             sel.sum() / max((idx[mm][-1] - idx[mm][0]).days / 7, 1)))
    dc = pd.DataFrame(rows, columns=["조건", "구간", "순bp", "t", "n", "주당"])
    piv = dc.pivot_table(index="조건", columns="구간", values=["순bp", "t", "n", "주당"])
    P(piv.round(2).sort_values(("순bp", "테스트"), ascending=False).to_string())
    both = piv[(piv[("순bp", "학습")] > 0) & (piv[("순bp", "테스트")] > 0)]
    P(f"  조건 조합 {len(piv)}개 중 학습·테스트 모두 양수 {len(both)}개")
    # 위약
    best = piv[("순bp", "테스트")].max()
    pls = []
    for _ in range(200):
        n_ev = int(piv[("n", "테스트")].median())
        cand = np.where(~tr)[0]; cand = cand[cand < len(b) - 200]
        ev = np.zeros(len(b), bool); ev[rng.choice(cand, min(n_ev * 3, len(cand)), replace=False)] = True
        sel = pick(ev, hold2)
        if sel.sum() > 20:
            pls.append(tot2[sel].mean())
    P(f"  무작위 진입 200회 (숏, 같은 기하): 평균 {np.mean(pls):.1f}bp, 95% {np.percentile(pls,95):.1f}, 최대 {np.max(pls):.1f} | 조건부 최고 {best:.1f}")
