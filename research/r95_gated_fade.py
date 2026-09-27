"""R95 — 선물만: 변동성 예보 p로 켜고 끄는 '대칭 깊은 지정가 되돌림' (롱 딥 매수 + 숏 급등 매도).

근거: R92B에서 z0.31 딥이 p 최상위(Q4)에서 거래당 +75bp(전 연도 양수). R91에서 큰 초기 이동 뒤 지속 확률이 무작위보행보다 낮음(되돌림).
      → ML 필터 없이도 p만으로 깊은 지정가 되돌림이 작동하는가, 숏(급등 매도)에도 대칭으로 성립하는가.
규칙: 봉 B분 시가 o에 매수 지정가 o·(1 − k·rv·√B), 매도 지정가 o·(1 + k·rv·√B) 동시 게시(봉 동안 유효, 5bp 관통 시 체결).
      rv = 직전 300분 1분 로그수익 표준편차(시가 시점까지). 청산: 봉 종가(E1) 또는 다음 봉 종가(E2) 테이커. 비용 지정가 2bp + 테이커 7bp.
      한 봉에 양쪽 다 체결되면 둘 다 기록. 종목·봉별 독립(계좌 시뮬 아님, 거래당 평균).
국면: pr(직전 90일 p 순위, 봉 시작 직전 격자 값). Q0 <.2 · Q1 .2~.5 · Q2 .5~.8 · Q3 .8~.95 · Q4 ≥.95.
선택: 2022~23 거래당 bp·t로 (B, k, 방향, 국면, 청산) 칸 고르고 2024~26 검증.
사용: python research/r95_gated_fade.py"""
import sys, warnings, time
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
CT, CM, PEN = 7e-4, 2e-4, 5e-4


@numba.njit(cache=True)
def fade(o, h, l, c, rv, B, k, start):
    n = len(c); m = (n - start) // B
    out = np.full((m, 6), np.nan)          # 봉 시작, 롱 E1, 롱 E2, 숏 E1, 숏 E2, 깊이
    for q in range(m - 2):
        i = start + q * B
        if np.isnan(rv[i]): continue
        d = k * rv[i] * np.sqrt(B); op = o[i]
        lo = op * (1 - d); hi = op * (1 + d)
        fl = -1; fs = -1
        for j in range(i, i + B):
            if fl < 0 and l[j] <= lo * (1 - PEN_): fl = j
            if fs < 0 and h[j] >= hi * (1 + PEN_): fs = j
        out[q, 0] = i; out[q, 5] = d
        e1 = c[i + B - 1]; e2 = c[i + 2 * B - 1]
        if fl >= 0:
            out[q, 1] = e1 / lo - 1 - CT_ - CM_; out[q, 2] = e2 / lo - 1 - CT_ - CM_
        if fs >= 0:
            out[q, 3] = 1 - e1 / hi - CT_ - CM_; out[q, 4] = 1 - e2 / hi - CT_ - CM_
    return out


CT_, CM_, PEN_ = CT, CM, PEN

if __name__ == "__main__":
    t_ = time.time()
    import r92_vol_timing as V
    Wp = V.load_p()
    rows = []; ev = []
    for s in ("BTCUSDT", "ETHUSDT"):
        K = store.load("klines_1m", s); lr = np.log(K.close).diff()
        rv = lr.rolling(300).std().shift(1).values
        w = Wp[Wp.sym == s].sort_values("t")
        start = int(w.t.min())
        for B in (5, 15, 60):
            for k in (3, 4, 6):
                o = fade(K.open.values, K.high.values, K.low.values, K.close.values, rv, B, k, start)
                F = pd.DataFrame(o, columns=["i", "L1", "L2", "S1", "S2", "d"]).dropna(subset=["i"])
                F["i"] = F.i.astype(np.int64)
                F = pd.merge_asof(F.sort_values("i"), w[["t", "pr"]], left_on="i", right_on="t", direction="backward").dropna(subset=["pr"])
                F["yr"] = pd.DatetimeIndex(K.ts.values[F.i.values]).year
                F["ts"] = pd.DatetimeIndex(K.ts.values[F.i.values])
                F["pb"] = np.digitize(F.pr, [.2, .5, .8, .95]); F["sym"] = s; F["B"] = B; F["k"] = k
                ev.append(F[F[["L1", "S1"]].notna().any(axis=1)])
        P(f"  {s} 완료 ({time.time()-t_:.0f}s)")
    E = pd.concat(ev)
    for (B, k, pb), g in E.groupby(["B", "k", "pb"]):
        for col in ("L1", "L2", "S1", "S2"):
            x = g.dropna(subset=[col]); tr = x[x.yr < 2024][col]; te = x[x.yr >= 2024][col]
            if len(tr) < 30 or len(te) < 20: continue
            rows.append(dict(B=B, k=k, pb=f"Q{pb}", leg=col, ntr=len(tr), trbp=tr.mean() * 1e4, ttr=tr.mean() / tr.std() * np.sqrt(len(tr)),
                             nte=len(te), tebp=te.mean() * 1e4, tte=te.mean() / te.std() * np.sqrt(len(te))))
    O = pd.DataFrame(rows); pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400)
    P("\n===== 국면 × 방향 평균 (모든 B·k, E1): 거래당 bp 학습 / 평가")
    P(O[O.leg.isin(["L1", "S1"])].groupby(["leg", "pb"])[["trbp", "tebp", "ntr", "nte"]].mean().round(1).to_string())
    P("\n===== 전체 칸 (B, k, 국면, 방향·청산)")
    P(O.round(1).to_string())
    sel = O[(O.ttr >= 2.5) & (O.trbp > 5)]
    P(f"\n===== 학습 t ≥ 2.5 & > 5bp: {len(sel)}칸, 평가 양수 {(sel.tebp > 0).sum()}, 평가 거래당 평균 {sel.tebp.mean():+.1f}bp (건수 가중 {np.average(sel.tebp, weights=sel.nte) if len(sel) else np.nan:+.1f})")
    P(sel.sort_values("ttr", ascending=False).round(1).to_string())
    E.to_parquet(R / "data" / "r95_events.parquet", index=False)
    (R / "logs" / "r95.summary").write_text("done\n")
