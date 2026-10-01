"""R118 공통 — 다중 전략 공유 계좌 엔진(롱·숏, 분 단위 마크가격 시가평가) + 사건 표준 형식.

사건 표준 형식(DataFrame, 한 행 = 거래 하나, 진입 분 순 정렬):
  e    : 진입 분(2020-01-01 00:00 UTC부터 분 수, int64)        x  : 청산 분(e < x, 이 분 시작에 청산 처리)
  ret  : 비용·펀딩·슬리피지 반영 순수익률(방향 반영, 명목 대비)   mae: 보유 중 최대 역행(양수, 정보용)
  k    : 채널 번호 — 0 = z0.31 딥(w·N·킬 스위치 규칙), 1 이상 = 고정 비중 채널(명목 = 자본 × W[k] × mult)
  mult : 거래별 크기 배수                                   s  : 종목(0 BTC, 1 ETH)
  px   : 진입가(시가평가용)                                   side: +1 롱 / −1 숏
청산 판정: 자본 + 미실현(롱은 분 저가, 숏은 분 고가 — 마크가격) ≤ MMR × 총명목 → 파산.
크기는 실현 자본(cap) 기준. 일말 평가는 마크 종가.
z0.31 딥(k=0): 명목 = min(자본 × w × mult, 자본 × N − 열린 딥 명목), 킬 스위치(120분 실현 −3% → 720분 중지).
사용: from r118_engine import run_multi, grid_hl, std_dip, std_breakout"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
import r102_joint_account as J, r103_breakout_cost as K, r80_max_return as R80, r109_dip_exact_mtm as M, r113_spot_perp_flow as F13
D0 = J.D0; MMR = J.MMR
COLS = ["e", "x", "ret", "mae", "k", "mult", "s", "px", "side"]


@numba.njit(cache=True)
def sim_multi(e_, x_, r_, k_, u_, s_, px_, sd_, LO, HI, CL, t0, w, N, Wk, X, Hh, Lk, mmr):
    T = LO.shape[1]; nd = (t0 + T) // 1440 + 1; day = np.full(nd, np.nan); info = np.zeros(3)
    ne = len(e_); order_x = np.argsort(x_)
    cap = 1.0; QL = np.zeros(2); NL = np.zeros(2); QS = np.zeros(2); NS = np.zeros(2); dipN = 0.0
    isopen = np.zeros(ne, np.bool_); nn = np.zeros(ne)
    ce_t = np.zeros(ne, np.int64); ce_r = np.zeros(ne); nc = 0; lo_p = 0; rsum = 0.0; halt = -1
    pe = 0; px_i = 0; peak = 0.0
    for t in range(T):
        m = t0 + t
        while px_i < ne and x_[order_x[px_i]] <= m:
            q = order_x[px_i]; px_i += 1
            if not isopen[q]: continue
            isopen[q] = False; n = nn[q]; cap += n * r_[q]; s = s_[q]
            if sd_[q] > 0:
                QL[s] -= n / px_[q]; NL[s] -= n
            else:
                QS[s] -= n / px_[q]; NS[s] -= n
            if k_[q] == 0:
                dipN -= n; ce_t[nc] = m; ce_r[nc] = r_[q]; nc += 1; rsum += r_[q]
        while lo_p < nc and ce_t[lo_p] <= m - Lk:
            rsum -= ce_r[lo_p]; lo_p += 1
        if rsum < -X and m + Hh > halt: halt = m + Hh
        while pe < ne and e_[pe] <= m:
            q = pe; pe += 1
            if e_[q] < m or x_[q] <= m: continue
            if k_[q] == 0:
                if m < halt: continue
                n = min(cap * w * u_[q], max(cap * N - dipN, 0.0))
            else:
                n = cap * Wk[k_[q]] * u_[q]
            if n <= 1e-12: continue
            isopen[q] = True; nn[q] = n; s = s_[q]
            if sd_[q] > 0:
                QL[s] += n / px_[q]; NL[s] += n
            else:
                QS[s] += n / px_[q]; NS[s] += n
            if k_[q] == 0: dipN += n
        tot = NL[0] + NL[1] + NS[0] + NS[1]
        if tot > 0:
            eqw = cap
            for s in range(2):
                eqw += QL[s] * LO[s, t] - NL[s] + NS[s] - QS[s] * HI[s, t]
            if eqw <= mmr * tot:
                info[0] = m; info[1] = dipN / cap; info[2] = (tot - dipN) / cap
                return day, -1.0, info
            if tot / cap > peak: peak = tot / cap
        if (m + 1) % 1440 == 0:
            v = cap
            for s in range(2):
                v += QL[s] * CL[s, t] - NL[s] + NS[s] - QS[s] * CL[s, t]
            day[m // 1440] = v
    return day, peak, info


_G = {}


def grid_hl(t0):
    """마크가격 분 저가·고가·종가 격자(t0 = 격자 시작 분)."""
    if t0 in _G: return _G[t0]
    L_, H_, C_ = [], [], []
    for sym in ("BTCUSDT", "ETHUSDT"):
        k = store.load("mark_1m", sym); mi = J.mins(k.ts)
        T1 = mi.max() + 1; ok = mi >= t0; arr = {}
        for c in ("low", "high", "close"):
            a = np.full(T1 - t0, np.nan); a[mi[ok] - t0] = k[c].values[ok]; arr[c] = pd.Series(a).ffill().bfill().values
        L_.append(arr["low"]); H_.append(arr["high"]); C_.append(arr["close"])
    T = min(len(a) for a in L_)
    _G[t0] = (np.vstack([a[:T] for a in L_]), np.vstack([a[:T] for a in H_]), np.vstack([a[:T] for a in C_]), t0)
    return _G[t0]


def run_multi(E, w, N, Wk, start="2022-01-01"):
    """E: 표준 사건, Wk: dict{채널: 비중} (채널 1 = z0.33이면 L/48). 반환 (일간 수익 또는 None, 최대 명목 또는 파산 정보)."""
    t0 = int(J.mins([pd.Timestamp(start, tz="UTC")])[0])
    E = E[(E.e >= t0)].sort_values(["e", "k"]).reset_index(drop=True)
    LO, HI, CL, t0 = grid_hl(t0)
    wk = np.zeros(int(E.k.max()) + 2 if len(E) else 2)
    for k_, v in Wk.items(): wk[k_] = v
    day, pk, info = sim_multi(E.e.values.astype(np.int64), E.x.values.astype(np.int64), E.ret.values.astype(float), E.k.values.astype(np.int64),
                              E.mult.values.astype(float), E.s.values.astype(np.int64), E.px.values.astype(float), E.side.values.astype(np.int64),
                              LO, HI, CL, t0, float(w), float(N), wk, 0.03, 720, 120, MMR)
    if pk < 0: return None, info
    eq = pd.Series(day, index=D0 + pd.to_timedelta(np.arange(len(day)), unit="D")).dropna()
    return R80.daily_from_eq(eq), pk


def summary(d):
    a = R80.metrics(d, "2022-01-01", "2024-01-01"); b = R80.metrics(d, "2024-01-01", "2026-09-23")
    sh = lambda x: x.mean() / x.std() * np.sqrt(365) if x.std() > 0 else np.nan
    return dict(학습월=a["mo"], 학습MDD=a["mdd"], 학습샤프=sh(d["2022":"2023"]), 평가월=b["mo"], 평가중앙=b["med"], 평가100달=b["big"], 평가MDD=b["mdd"],
                평가샤프=sh(d["2024":"2026-09-22"]), 연도=b["yr"])


def std_dip():
    """z0.31 딥(문턱 25bp·V3) 표준 사건(k=0, 롱)."""
    D, _, _, W = M.dip_with_px(); D["side"] = 1
    return D[COLS]


def std_breakout(pnet_filter=True):
    """z0.33 TOP8(펀딩≤1bp) 표준 사건(k=1, 롱), pnet15 하위 40% 제외(R113b)."""
    T8 = J.add_funding(J.breakout_trades(J.TOP8)); T8["lf"] = K.last_funding(T8); BK = K.events(T8, False, fmax=1e-4).drop(columns="lf")
    if pnet_filter:
        v = np.full(len(BK), np.nan)
        for s_, sym in ((0, "BTCUSDT"), (1, "ETHUSDT")):
            F = F13.feats(sym)[0]["pnet15"]; mk = (BK.s == s_).values
            v[mk] = F.reindex(pd.DatetimeIndex(D0 + pd.to_timedelta(BK.e.values[mk] - 1, unit="min"))).values
            del F
        tr = ((D0 + pd.to_timedelta(BK.e, unit="min")) < F13.SPLIT).values
        edges = pd.Series(v[tr]).quantile([.2, .4, .6, .8]).values; q = np.where(np.isnan(v), 3, np.searchsorted(edges, v) + 1)
        BK = BK[q >= 3]
    BK = BK.copy(); BK["side"] = 1
    return BK[COLS].reset_index(drop=True)


if __name__ == "__main__":
    # 자기 점검: R113b 결과(w3·N12·L16 평가 월 +35.0%)를 이 엔진으로 재현해야 한다(롱 전용이라 sim4와 같아야 함).
    E = pd.concat([std_dip(), std_breakout()], ignore_index=True)
    for L in (12, 16):
        d, pk = run_multi(E, 3, 12, {1: L / 48})
        print(f"재현 w3N12 L{L}:", "파산" if d is None else {k: (round(v, 2) if isinstance(v, float) else v) for k, v in summary(d).items()}, flush=True)
