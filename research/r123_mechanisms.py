"""R123-1 — 엣지 공장 2세대 1단계: R122에 없던 진입 메커니즘 계열(롱·숏)의 거래 목록.

A. 변동성 압축 양방향 스톱(R87 V 계열 일반화): 4시간마다(00·04·…UTC) 직전 24h 범위 / 그 30일 중앙값 = 압축비.
   압축비 < q(0.5·0.7)면 기준가 P0(그 분 시가) ± x·σ24(x 0.75·1.25)에 양방향 스톱(OCO, 24h 유효), 먼저 체결된 쪽 진입.
   청산: 목표 진입가 ± tgt·x·σ24(tgt 2·3, 지정가 메이커·관통 1bp) 또는 P0 기준 48h 뒤 종가. 롱 체결·숏 체결을 별도 메커니즘으로 기록(16개).
B. 다일 추세(TSMOM): 매일 00 UTC 시가 진입, 직전 L일 수익(L 7·14·28·56) > 0이면 롱 메커니즘, < 0이면 숏 메커니즘, 보유 1·3·7일(24개).
C. 정시 진입: 매시 정각 시가 시장가 진입, 보유 60·240·1440분, 롱·숏(6개) — 조건(2단계)이 방향 문맥을 고른다.
D. 펀딩 정산: 정산 60분 전 진입 → 정산 60분 뒤 청산 / 정산 시각 진입 → 240분 보유, 롱·숏(4개).
비용: 시장가·스톱 진입·청산 테이커 5bp + 슬리피지 2bp(편도), 지정가 목표 청산 메이커 2bp, 실제 펀딩(롱 지불·숏 수취).
출력: data/r123_trades.parquet (R122와 같은 열: mech, fam, side, sym, e, x, ret, mae, px)
사용: python research/r123_mechanisms.py"""
import sys, warnings, time
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store, costs
import r102_joint_account as J
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
TK = costs.TAKER + 2e-4; MK = costs.MAKER


@numba.njit(cache=True)
def vcomp(o, h, l, c, W0s, comp, sig, q, x, tgt):
    """A: 창 시작 분 W0s(분 배열 인덱스). 반환 진입 인덱스, 청산 인덱스, 방향, 진입가, 청산가, 목표 여부, 역행."""
    n = len(W0s); EI = np.full(n, -1); XI = np.full(n, -1); SD = np.zeros(n, np.int64); PX = np.zeros(n); XP = np.zeros(n); TG = np.zeros(n, np.bool_); AD = np.zeros(n)
    N = len(c)
    for i in range(n):
        w = W0s[i]
        if not (comp[i] < q) or np.isnan(sig[i]) or w + 2880 >= N: continue
        p0 = o[w]; up = p0 * (1 + x * sig[i]); dn = p0 * (1 - x * sig[i]); f = -1; s = 0
        for j in range(w, w + 1440):
            if h[j] >= up: f = j; s = 1; break
            if l[j] <= dn: f = j; s = -1; break
        if f < 0: continue
        px = max(up, o[f]) if s == 1 else min(dn, o[f])
        tp = px * (1 + s * tgt * x * sig[i]); end = w + 2880; xi = end - 1; xp = c[end - 1]; tg = False; ad = px
        for j in range(f, end):
            if s == 1 and l[j] < ad: ad = l[j]
            if s == -1 and h[j] > ad: ad = h[j]
            if j > f and ((s == 1 and h[j] > tp * (1 + 1e-4)) or (s == -1 and l[j] < tp * (1 - 1e-4))):
                xi = j; xp = tp; tg = True; break
        EI[i] = f; XI[i] = xi; SD[i] = s; PX[i] = px; XP[i] = xp; TG[i] = tg; AD[i] = abs(ad / px - 1)
    return EI, XI, SD, PX, XP, TG, AD


def rows_from(mech, fam, sym, mi, ei, xi, side, px, xp, entry_cost, exit_cost, ad):
    ret = side * (xp / px - 1) - entry_cost - exit_cost
    return pd.DataFrame(dict(mech=mech, fam=fam, side=side, sym=sym, e=mi[ei], x=mi[xi] + 1, ret=ret.astype("float32"), mae=ad.astype("float32"), px=px))


if __name__ == "__main__":
    t_ = time.time(); out = []
    for sym in ("BTCUSDT", "ETHUSDT"):
        k = store.load("klines_1m", sym); k["ts"] = pd.DatetimeIndex(k.ts); mi = J.mins(k.ts)
        o, h, l, c = k.open.values, k.high.values, k.low.values, k.close.values; N = len(c)
        ts = pd.DatetimeIndex(k.ts); minute_of_day = (ts.hour * 60 + ts.minute).values
        lr = np.r_[0, np.diff(np.log(c))]
        # A. 변동성 압축
        W0s = np.where((minute_of_day % 240 == 0))[0]; W0s = W0s[W0s >= 1440]
        rng24 = pd.Series(h).rolling(1440).max().values - pd.Series(l).rolling(1440).min().values
        r24 = (rng24 / c)[W0s - 1]; med = pd.Series(r24).rolling(180, min_periods=60).median().shift(1).values      # 30일 = 180개 4h 창
        comp = r24 / med; sig = (pd.Series(lr).rolling(1440).std().values * np.sqrt(1440))[W0s - 1]
        for q in (0.5, 0.7):
            for x in (0.75, 1.25):
                for tgt in (2, 3):
                    EI, XI, SD, PX, XP, TG, AD = vcomp(o, h, l, c, W0s.astype(np.int64), comp, sig, q, x, tgt)
                    m = EI >= 0
                    for s in (1, -1):
                        mm = m & (SD == s)
                        if not mm.any(): continue
                        exit_cost = np.where(TG[mm], MK, TK)
                        out.append(rows_from(f"VC q{q} x{x} t{tgt} {'L' if s > 0 else 'S'}", "vc", sym, mi, EI[mm], XI[mm], s, PX[mm] * (1 + s * 2e-4), XP[mm], costs.TAKER, exit_cost, AD[mm]))
        P(f"  {sym} A 변동성 압축 ({time.time()-t_:.0f}s)")
        # B. TSMOM
        D0i = np.where(minute_of_day == 0)[0]
        for L in (7, 14, 28, 56):
            for hd in (1, 3, 7):
                ok = (D0i >= L * 1440) & (D0i + hd * 1440 < N)
                ei = D0i[ok]; xi = ei + hd * 1440 - 1
                lb = c[ei - 1] / c[ei - 1 - (L * 1440 - 1)] - 1
                for s in (1, -1):
                    sel = (lb > 0) if s > 0 else (lb < 0)
                    e2, x2 = ei[sel], xi[sel]
                    ad = np.array([abs((l[a:b + 1].min() if s > 0 else h[a:b + 1].max()) / o[a] - 1) for a, b in zip(e2, x2)])
                    out.append(rows_from(f"TS L{L} H{hd}d {'L' if s > 0 else 'S'}", "ts", sym, mi, e2, x2, s, o[e2] * (1 + s * 2e-4), c[x2] * (1 - s * 2e-4), costs.TAKER, costs.TAKER, ad))
        P(f"  {sym} B 다일 추세 ({time.time()-t_:.0f}s)")
        # C. 정시 진입
        Hi = np.where(minute_of_day % 60 == 0)[0]
        for hm in (60, 240, 1440):
            ei = Hi[Hi + hm < N]; xi = ei + hm - 1
            lo_ = pd.Series(l).rolling(hm).min().shift(-(hm - 1)).values; hi_ = pd.Series(h).rolling(hm).max().shift(-(hm - 1)).values
            for s in (1, -1):
                ad = np.abs((lo_[ei] if s > 0 else hi_[ei]) / o[ei] - 1)
                out.append(rows_from(f"HR H{hm} {'L' if s > 0 else 'S'}", "hr", sym, mi, ei, xi, s, o[ei] * (1 + s * 2e-4), c[xi] * (1 - s * 2e-4), costs.TAKER, costs.TAKER, ad))
        P(f"  {sym} C 정시 진입 ({time.time()-t_:.0f}s)")
        # D. 펀딩 정산 전후
        Fi = np.where((minute_of_day % 480 == 0))[0]
        for lab, off, hm in (("pre60_post60", -60, 120), ("at_240", 0, 240)):
            ei = Fi + off; ok = (ei >= 0) & (ei + hm < N); ei = ei[ok]; xi = ei + hm - 1
            for s in (1, -1):
                ad = np.array([abs((l[a:b + 1].min() if s > 0 else h[a:b + 1].max()) / o[a] - 1) for a, b in zip(ei, xi)])
                out.append(rows_from(f"FS {lab} {'L' if s > 0 else 'S'}", "fs", sym, mi, ei, xi, s, o[ei] * (1 + s * 2e-4), c[xi] * (1 - s * 2e-4), costs.TAKER, costs.TAKER, ad))
        P(f"  {sym} D 펀딩 정산 ({time.time()-t_:.0f}s)")
    T = pd.concat(out, ignore_index=True)
    Tf = J.add_funding(T[["sym", "e", "x"]].copy())
    T["ret"] = (T.ret.values - T.side.values * Tf.fund.values).astype("float32")
    T["e"] = T.e.astype(np.int64); T["x"] = T.x.astype(np.int64)
    T.to_parquet(R / "data" / "r123_trades.parquet", index=False)
    S = T.groupby("mech").agg(n=("ret", "size"), bp=("ret", "mean")); S["bp"] *= 1e4
    P(f"\n메커니즘 {T.mech.nunique()}개, 거래 {len(T):,} ({time.time()-t_:.0f}s)")
    P(S.sort_values("bp", ascending=False).round(1).to_string())
