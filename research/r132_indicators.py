"""R132 1단 — "구할 수 있는 지표는 전부" 재사용 가능한 인과적(causal) 특징 라이브러리.

핵심 규칙(전부 엄수): 결정은 분 m의 "종가 시점"에서 일어난다(klines ts = 그 분의 시가 시각).
  - 상위 시간대(5m/15m/1h/4h/1d) 봉은 "m의 종가까지 완결된" 봉만 쓴다 — 봉 i(0-base, 길이 P분)가
    끝나는 시각은 분 인덱스 (i+1)*P. 이게 m+1 이하여야 쓸 수 있다 → 쓸 수 있는 마지막 봉
    i_max(m) = (m+1)//P − 1 (정수 나눔). `gather_bar`가 이 규칙으로 봉 배열을 분 m에 직접 매핑한다
    (전체 분 길이로 펼치지 않고 결정분에서만 샘플링 — 메모리 절약).
  - 일 데이터(ext/*.parquet 등)는 1일 지연(`daily_asof`): 날짜 D의 값은 D+lag일 00:00 UTC 부터 사용.
  - 15분 격자 저장소(r97_feat/r54_dfeat/r90 pr, 격자 t는 "t−1분까지"의 특징): 분 m 에서 쓸 수 있는
    마지막 격자 t_g = t_g ≤ m+1 중 최댓값(`grid_asof`).
  - 전부 numpy/pandas로 직접 구현(talib/ta/pandas_ta 미설치, 설치하지 않음 — .venv 에 numba/sklearn 만 확인).

메모리 규칙: 분류별 봉 배열(1m 시간대는 전체 길이)은 계산 후 즉시 결정분으로 샘플링하고 버린다
  (freesia 공통 규칙 — "수백 개 전체길이 분 컬럼을 동시에 들고 있지 않는다").

사용: from r132_indicators import classical_indicators, level_features, gather_bar, bars_from_1m, ...
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import numba

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
import r113_spot_perp_flow as F13, r115_bybit_divergence as F15, r116_kimchi_premium as F16
import r119_feats as F119, r92_vol_timing as F92
import r129_feats as RF

EPS = 1e-9
TIMEFRAMES = (("1m", 1), ("5m", 5), ("15m", 15), ("1h", 60), ("4h", 240), ("1d", 1440))


def utc(ix):
    ix = pd.DatetimeIndex(ix)
    return ix.tz_convert("UTC") if ix.tz is not None else ix.tz_localize("UTC")


def daily_asof(s, ts, lag_days=1):
    """일 자료 s(날짜 인덱스) → 시점 ts(1분 UTC DatetimeIndex)에서 쓸 수 있는 값(그 날짜+lag일 00:00부터, r97과 동일 관행)."""
    s = s.copy(); s.index = utc(s.index).normalize() + pd.Timedelta(days=lag_days)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.reindex(s.index.union(ts)).ffill().reindex(ts).values


# ============================================================ 공통 유틸 ============================================================

def bars_from_1m(o, h, l, c, v, P):
    """1분 OHLCV → P분 봉(완결된 봉만, 끝에 남는 미완결 조각은 버림). 반환은 float64 배열(길이 n//P)."""
    n = len(c)
    nb = n // P
    if nb == 0:
        z = np.zeros(0)
        return z, z, z, z, z
    n2 = nb * P
    oo = o[:n2].reshape(nb, P)[:, 0]
    hh = h[:n2].reshape(nb, P).max(1)
    ll = l[:n2].reshape(nb, P).min(1)
    cc = c[:n2].reshape(nb, P)[:, -1]
    vv = v[:n2].reshape(nb, P).sum(1)
    return oo, hh, ll, cc, vv


def gather_bar(bar_arr, P, minutes):
    """봉 배열(bar_arr, 봉 i 는 분 [i*P,(i+1)*P) 커버, 분 인덱스 (i+1)*P 에 완결) 을 분 인덱스 배열
    minutes 에서 '그 분 종가 시점에 이미 완결된 마지막 봉' 값으로 직접 샘플링(전체 길이로 펼치지 않음).
    i_max(m) = (m+1)//P − 1. 봉이 아직 없으면(i_max<0) NaN."""
    minutes = np.asarray(minutes, dtype=np.int64)
    idx = (minutes + 1) // P - 1
    out = np.full(len(minutes), np.nan, dtype=np.float32)
    ok = (idx >= 0) & (idx < len(bar_arr))
    out[ok] = bar_arr[idx[ok]].astype(np.float32)
    return out


def rolling_argext(x, n, mode):
    """길이 n 슬라이딩 윈도(현재 포함, 과거 n−1개 + 현재)에서 최댓값/최솟값의 '현재로부터 몇 분 전'을
    O(len(x)) 로 계산(단조 데크). mode='max' 또는 'min'. 반환: ago(분, 과거일수록 큼), val."""
    return _rolling_argext_nb(np.asarray(x, dtype=np.float64), n, 1 if mode == "max" else -1)


@numba.njit(cache=True)
def _rolling_argext_nb(x, n, sign):
    m = len(x)
    ago = np.full(m, np.nan)
    val = np.full(m, np.nan)
    dq_idx = np.empty(m, np.int64)
    dq_val = np.empty(m, np.float64)
    head = 0
    tail = 0
    for i in range(m):
        xv = x[i] * sign
        while tail > head and dq_val[tail - 1] <= xv:
            tail -= 1
        dq_idx[tail] = i
        dq_val[tail] = xv
        tail += 1
        lo = i - n + 1
        while dq_idx[head] < lo:
            head += 1
        ago[i] = i - dq_idx[head]
        val[i] = dq_val[head] * sign
    return ago, val


# ============================================================ 1. 고전 기술 지표 (봉 해상도) ============================================================
# 반환: (dict[name->np.ndarray(float64, 봉 해상도)], set(signed 컬럼명)) — 전부 그 봉까지의 과거만 사용(인과적).

TA_SIGNED = {
    "roc12", "mom10", "macd_hist", "ppo", "cci20", "willr14_c", "stoch_k_c", "stoch_d_c", "bb_pctb_c",
    "donch_pos_c", "kelt_pos", "di_diff", "aroon_osc", "ema9_dist", "ema21_dist", "ema50_dist", "ema200_dist",
    "ema9_slope", "ema21_slope", "ema50_slope", "ema200_slope", "ema9v21", "ema21v50", "ema50v200",
    "obv_slope_z", "ad_slope", "elder_bull", "elder_bear", "trix15", "dpo20", "vortex_diff",
    "zscore_sma20", "ha_streak", "consec_cnt", "body_ratio", "psar_dist", "supertrend_side", "ret1", "mfi_c",
}


def _ema(s, n):
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def _wilder(s, n):
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def _true_range(h, l, c):
    pc = c.shift(1)
    return pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(1)


@numba.njit(cache=True)
def _psar_nb(h, l, c, step, maxstep):
    n = len(c)
    sar = np.full(n, np.nan)
    side = np.zeros(n, np.int8)
    if n < 3:
        return sar, side
    up = c[1] >= c[0]
    sar[1] = l[0] if up else h[0]
    ep = h[1] if up else l[1]
    af = step
    side[1] = 1 if up else -1
    for i in range(2, n):
        prev = sar[i - 1]
        s = prev + af * (ep - prev)
        if up:
            s = min(s, l[i - 1], l[i - 2])
            if l[i] < s:
                up = False; s = ep; ep = l[i]; af = step
            else:
                if h[i] > ep: ep = h[i]; af = min(af + step, maxstep)
        else:
            s = max(s, h[i - 1], h[i - 2])
            if h[i] > s:
                up = True; s = ep; ep = h[i]; af = step
            else:
                if l[i] < ep: ep = l[i]; af = min(af + step, maxstep)
        sar[i] = s
        side[i] = 1 if up else -1
    return sar, side


@numba.njit(cache=True)
def _supertrend_nb(h, l, c, atr, mult):
    n = len(c)
    side = np.zeros(n, np.int8)
    st = np.full(n, np.nan)
    up = np.nan; dn = np.nan; s = 1; started = False
    for i in range(n):
        if np.isnan(atr[i]):
            continue
        basic_up = (h[i] + l[i]) / 2 + mult * atr[i]
        basic_dn = (h[i] + l[i]) / 2 - mult * atr[i]
        if not started:
            # ATR 웜업 구간(NaN)에서는 밴드를 전혀 못 갱신 — up/dn이 NaN으로 "동결"되면 이후
            # 모든 비교(x < nan, x > nan)가 항상 False라 side 가 영원히 안 바뀌는 버그가 있었음.
            # ATR 이 처음 유효해지는 봉에서 밴드를 새로 시작해 해결.
            up, dn = basic_up, basic_dn; started = True
        else:
            up = basic_up if (basic_up < up or c[i - 1] > up) else up
            dn = basic_dn if (basic_dn > dn or c[i - 1] < dn) else dn
            if s == 1 and c[i] < dn: s = -1
            elif s == -1 and c[i] > up: s = 1
        side[i] = s
        st[i] = dn if s == 1 else up
    return side, st


@numba.njit(cache=True)
def _heikin_ashi_nb(o, h, l, c):
    n = len(c)
    hc = np.empty(n)
    ho = np.empty(n)
    for i in range(n):
        hc[i] = (o[i] + h[i] + l[i] + c[i]) / 4.0
    ho[0] = (o[0] + c[0]) / 2.0
    for i in range(1, n):
        ho[i] = (ho[i - 1] + hc[i - 1]) / 2.0
    return ho, hc


def classical_indicators(o, h, l, c, v):
    """o/h/l/c/v: 같은 봉 해상도의 완결 봉 numpy 배열(float64). 반환 dict(float32), signed 집합(접미사 없는 짧은 이름)."""
    n = len(c)
    out = {}
    if n < 5:
        return out
    O, H, L, C, V = (pd.Series(x) for x in (o, h, l, c, v))
    ret1 = np.log(C / C.shift(1))
    out["ret1"] = ret1.values

    # RSI(14,6)
    diff = C.diff()
    gain = diff.clip(lower=0); loss = (-diff).clip(lower=0)
    for nn, tag in ((14, "rsi14"), (6, "rsi6")):
        ag = _wilder(gain, nn); al = _wilder(loss, nn)
        rs = ag / (al + EPS)
        out[tag] = (100 - 100 / (1 + rs)).values

    # Stochastic %K/%D(14,3), Williams %R(14) — 센터링판(−50)도 같이
    hh14 = H.rolling(14).max(); ll14 = L.rolling(14).min()
    k = (C - ll14) / (hh14 - ll14 + EPS) * 100
    out["stoch_k"] = k.values
    d = k.rolling(3).mean()
    out["stoch_d"] = d.values
    out["stoch_k_c"] = (k - 50).values
    out["stoch_d_c"] = (d - 50).values
    willr = (hh14 - C) / (hh14 - ll14 + EPS) * -100
    out["willr14"] = willr.values
    out["willr14_c"] = (willr + 50).values

    # CCI(20)
    tp = (H + L + C) / 3
    sma_tp = tp.rolling(20).mean()
    mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    out["cci20"] = ((tp - sma_tp) / (0.015 * mad + EPS)).values

    # MACD(12,26,9, 가격으로 정규화) / 시그널 / 히스토그램, PPO
    ema12 = _ema(C, 12); ema26 = _ema(C, 26)
    macd = (ema12 - ema26) / C
    sig = _ema(macd, 9)
    out["macd_line"] = macd.values; out["macd_signal"] = sig.values; out["macd_hist"] = (macd - sig).values
    out["ppo"] = ((ema12 - ema26) / (ema26 + EPS) * 100).values

    # ROC / 모멘텀
    out["roc12"] = (C / C.shift(12) - 1).values
    out["mom10"] = (C - C.shift(10)).values

    # 볼린저 %B / 밴드폭, 켈트너 포지션
    sma20 = C.rolling(20).mean(); std20 = C.rolling(20).std()
    up_bb = sma20 + 2 * std20; lo_bb = sma20 - 2 * std20
    out["bb_pctb"] = ((C - lo_bb) / (up_bb - lo_bb + EPS)).values
    out["bb_pctb_c"] = out["bb_pctb"] - 0.5
    out["bb_bw"] = ((up_bb - lo_bb) / (sma20 + EPS)).values
    atr20 = _wilder(_true_range(H, L, C), 20)
    kelt_mid = _ema(C, 20)
    out["kelt_pos"] = ((C - kelt_mid) / (2 * atr20 + EPS)).values

    # 돈치안(20, 직전 N봉 기준 — 현재 봉 제외) 포지션 & 돌파
    dup = H.shift(1).rolling(20).max(); dlo = L.shift(1).rolling(20).min()
    out["donch_pos"] = ((C - dlo) / (dup - dlo + EPS)).values
    out["donch_pos_c"] = out["donch_pos"] - 0.5
    out["donch_break_up"] = (C >= dup).astype(float).values
    out["donch_break_dn"] = (C <= dlo).astype(float).values

    # ATR 비율(ATR/가격), 단기/장기 비
    atr14 = _wilder(_true_range(H, L, C), 14)
    atr50 = _wilder(_true_range(H, L, C), 50)
    out["atr_ratio"] = (atr14 / C).values
    out["atr_short_long"] = (atr14 / (atr50 + EPS)).values

    # ADX/+DI/-DI(14)
    up_move = H.diff(); dn_move = -L.diff()
    plus_dm = pd.Series(np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0), index=H.index)
    minus_dm = pd.Series(np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0), index=H.index)
    tr14 = _wilder(_true_range(H, L, C), 14)
    plus_di = 100 * _wilder(plus_dm, 14) / (tr14 + EPS)
    minus_di = 100 * _wilder(minus_dm, 14) / (tr14 + EPS)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di + EPS) * 100
    out["adx14"] = _wilder(dx, 14).values
    out["plus_di"] = plus_di.values; out["minus_di"] = minus_di.values
    out["di_diff"] = (plus_di - minus_di).values

    # Aroon(25)
    up_ago, _ = rolling_argext(h, 25, "max"); dn_ago, _ = rolling_argext(l, 25, "min")
    aroon_up = (25 - up_ago) / 25 * 100; aroon_dn = (25 - dn_ago) / 25 * 100
    out["aroon_up"] = aroon_up; out["aroon_dn"] = aroon_dn; out["aroon_osc"] = aroon_up - aroon_dn

    # 이치모쿠: 전환선(9)/기준선(26), 구름(현재 쓰는 구름은 26봉 전 계산값)
    tenkan = (H.rolling(9).max() + L.rolling(9).min()) / 2
    kijun = (H.rolling(26).max() + L.rolling(26).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((H.rolling(52).max() + L.rolling(52).min()) / 2).shift(26)
    cloud_hi = pd.concat([senkou_a, senkou_b], axis=1).max(1); cloud_lo = pd.concat([senkou_a, senkou_b], axis=1).min(1)
    out["ichi_tenkan_dist"] = ((C - tenkan) / C).values
    out["ichi_kijun_dist"] = ((C - kijun) / C).values
    out["ichi_cloud_thick"] = ((cloud_hi - cloud_lo) / C).values
    above = C > cloud_hi; below = C < cloud_lo
    out["ichi_cloud_pos"] = np.where(above, 1.0, np.where(below, -1.0, 0.0))

    # Parabolic SAR / SuperTrend
    sar, sar_side = _psar_nb(h, l, c, 0.02, 0.2)
    out["psar_side"] = sar_side.astype(float)
    out["psar_dist"] = (c - sar) / c
    st_side, st_val = _supertrend_nb(h, l, c, atr14.values, 3.0)
    out["supertrend_side"] = st_side.astype(float)

    # EMA 9/21/50/200 거리·기울기·크로스 상태
    emas = {}
    for nn in (9, 21, 50, 200):
        e = _ema(C, nn); emas[nn] = e
        out[f"ema{nn}_dist"] = ((C - e) / C).values
        out[f"ema{nn}_slope"] = ((e - e.shift(5)) / (e.abs() + EPS)).values
    out["ema9v21"] = np.sign((emas[9] - emas[21]).values)
    out["ema21v50"] = np.sign((emas[21] - emas[50]).values)
    out["ema50v200"] = np.sign((emas[50] - emas[200]).values)

    # OBV 기울기 z
    obv = (np.sign(ret1.fillna(0)) * V).cumsum()
    obv_slope = obv - obv.shift(20)
    out["obv_slope_z"] = ((obv_slope - obv_slope.rolling(500, min_periods=50).mean()) / (obv_slope.rolling(500, min_periods=50).std() + EPS)).values

    # MFI(14), CMF(20)
    tp_diff = tp.diff()
    pos_mf = (tp * V).where(tp_diff > 0, 0.0); neg_mf = (tp * V).where(tp_diff < 0, 0.0)
    mfr = pos_mf.rolling(14).sum() / (neg_mf.rolling(14).sum() + EPS)
    mfi = 100 - 100 / (1 + mfr)
    out["mfi14"] = mfi.values; out["mfi_c"] = (mfi - 50).values
    mfv = ((C - L) - (H - C)) / (H - L + EPS) * V
    out["cmf20"] = (mfv.rolling(20).sum() / (V.rolling(20).sum() + EPS)).values

    # A/D 라인 기울기, 포스 인덱스, 엘더 레이
    ad = mfv.cumsum()
    ad_diff = ad - ad.shift(20)  # 차분은 누적합의 임의 시작점과 무관(= 최근 20봉 mfv 합) — 분모도 반드시 차분형 척도로 정규화할 것
    out["ad_slope"] = (ad_diff / (mfv.abs().rolling(200, min_periods=20).mean() * 20 + EPS)).values
    fi = diff * V
    out["force_idx13"] = (_ema(fi, 13) / (C * V.rolling(50, min_periods=5).mean() + EPS)).values
    ema13 = _ema(C, 13)
    out["elder_bull"] = ((H - ema13) / C).values
    out["elder_bear"] = ((L - ema13) / C).values

    # TRIX(15), KST, Ultimate osc, Chande 모멘텀, DPO(20)
    e1 = _ema(C, 15); e2 = _ema(e1, 15); e3 = _ema(e2, 15)
    out["trix15"] = (e3.pct_change() * 100).values
    def _roc(s, n): return (s / s.shift(n) - 1) * 100
    kst = (_roc(C, 10).rolling(10).mean() * 1 + _roc(C, 15).rolling(10).mean() * 2 +
           _roc(C, 20).rolling(10).mean() * 3 + _roc(C, 30).rolling(15).mean() * 4)
    out["kst"] = kst.values; out["kst_signal"] = kst.rolling(9).mean().values
    bp = C - pd.concat([L, C.shift(1)], axis=1).min(1)
    trv = _true_range(H, L, C)
    def _avg(n): return bp.rolling(n).sum() / (trv.rolling(n).sum() + EPS)
    out["ultimate_osc"] = (100 * (4 * _avg(7) + 2 * _avg(14) + _avg(28)) / 7).values
    chup = diff.clip(lower=0).rolling(9).sum(); chdn = (-diff.clip(upper=0)).rolling(9).sum()
    out["chande_mom9"] = (100 * (chup - chdn) / (chup + chdn + EPS)).values
    out["dpo20"] = (C.shift(11) - C.rolling(20).mean()).values / C.values

    # Mass Index(25), Vortex(14)
    hl = H - L
    ratio = _ema(hl, 9) / (_ema(_ema(hl, 9), 9) + EPS)
    out["mass_idx25"] = ratio.rolling(25).sum().values
    vmp = (H - L.shift(1)).abs(); vmm = (L - H.shift(1)).abs()
    vip = vmp.rolling(14).sum() / (trv.rolling(14).sum() + EPS); vim = vmm.rolling(14).sum() / (trv.rolling(14).sum() + EPS)
    out["vortex_plus"] = vip.values; out["vortex_minus"] = vim.values; out["vortex_diff"] = (vip - vim).values

    # Heikin-Ashi 색 연속(streak), 연속 상승/하락 캔들 수
    ho, hc = _heikin_ashi_nb(o, h, l, c)
    ha_up = hc >= ho
    out["ha_streak"] = _run_streak(ha_up)
    out["consec_cnt"] = _run_streak((c >= o))

    # 캔들 해부
    body = C - O; rng = (H - L).replace(0, np.nan)
    out["body_ratio"] = (body / rng).values
    out["upper_wick"] = ((H - pd.concat([O, C], axis=1).max(1)) / rng).values
    out["lower_wick"] = ((pd.concat([O, C], axis=1).min(1) - L) / rng).values
    prev_body = body.shift(1)
    out["engulfing"] = ((body.abs() > prev_body.abs()) & (np.sign(body) != np.sign(prev_body))).astype(float).values
    lower_wick_s = pd.Series(out["lower_wick"]); body_ratio_s = pd.Series(out["body_ratio"])
    out["hammer"] = ((lower_wick_s > 0.5) & (body_ratio_s.abs() < 0.3)).astype(float).values
    out["doji"] = (body_ratio_s.abs() < 0.1).astype(float).values

    # 종가 z, 효율성 비율(Kaufman ER), 분산비(변동성 비율), 실현 왜도/첨도, 수익률 엔트로피
    out["zscore_sma20"] = ((C - sma20) / (std20 + EPS)).values
    er_num = (C - C.shift(10)).abs(); er_den = diff.abs().rolling(10).sum()
    out["eff_ratio10"] = (er_num / (er_den + EPS)).values
    var1 = ret1.rolling(20).var(); varn = (np.log(C / C.shift(10))).rolling(20).var()
    out["variance_ratio10"] = (varn / (10 * var1 + EPS)).values
    out["real_skew20"] = ret1.rolling(20).skew().values
    out["real_kurt20"] = ret1.rolling(20).kurt().values
    sign_bin = (ret1 > 0).astype(float)
    p_up = sign_bin.rolling(20).mean().clip(EPS, 1 - EPS)
    out["ret_entropy20"] = (-(p_up * np.log2(p_up) + (1 - p_up) * np.log2(1 - p_up))).values

    # 거래량 z, 상대 거래량
    vmean = V.rolling(200, min_periods=20).mean(); vstd = V.rolling(200, min_periods=20).std()
    out["vol_z"] = ((V - vmean) / (vstd + EPS)).values
    out["rel_vol"] = (V / (vmean + EPS)).values

    # NR4/NR7, 인사이드바
    rngv = (H - L)
    out["nr4"] = (rngv <= rngv.rolling(4).min()).astype(float).values
    out["nr7"] = (rngv <= rngv.rolling(7).min()).astype(float).values
    out["inside_bar"] = ((H <= H.shift(1)) & (L >= L.shift(1))).astype(float).values

    for key in list(out.keys()):
        out[key] = np.asarray(out[key], dtype=np.float32)
    return out


@numba.njit(cache=True)
def _run_streak_nb(up):
    n = len(up)
    out = np.zeros(n, dtype=np.float64)
    cnt = 0
    for i in range(n):
        cnt = cnt + 1 if (i == 0 or up[i] == up[i - 1]) else 1
        out[i] = cnt if up[i] else -cnt
    return out


def _run_streak(up_bool):
    """불리언 배열(위/아래) → 같은 상태가 몇 봉째 이어지는지(양수=연속 상승류, 음수=연속 하락류)."""
    return _run_streak_nb(np.asarray(up_bool, dtype=np.bool_)).astype(np.float32)


# ============================================================ 2. 레벨 (분 해상도, 직접 인덱싱) ============================================================

LVL_SIGNED = {
    "pivot_P_dist", "pivot_R1_dist", "pivot_R2_dist", "pivot_S1_dist", "pivot_S2_dist",
    "pday_hi_dist", "pday_lo_dist", "pday_close_dist", "pweek_hi_dist", "pweek_lo_dist", "pweek_close_dist",
    "dayopen_dist", "weekopen_dist", "monthopen_dist", "fib1d_pos_c", "fib7d_pos_c",
    "hi24h_dist", "lo24h_dist", "hi7d_dist", "lo7d_dist", "hi30d_dist", "lo30d_dist",
    "vwap_roll1d_dist", "vwap_session_dist",
}


def level_features(ts, o, h, l, c, v, round_steps):
    """ts: pandas DatetimeIndex(UTC, 1분, 전체 길이). o/h/l/c/v: 1분 float64 배열.
    round_steps: 라운드 넘버 격자 목록(예 BTC [1000,500,100], ETH [100,50,10]) — 격자마다 round_dist_{step} 1열.
    전부 '그 분까지 완결'만 쓰는 전체 길이 배열(float32) 반환 — 호출부가 결정분에서 바로 인덱싱(이 함수는
    메모리상 전체 1분 길이를 쓰므로, 호출 뒤 필요한 열만 즉시 샘플링하고 del 할 것)."""
    idx = pd.DatetimeIndex(ts)
    O, H, L, C = (pd.Series(x, index=idx) for x in (o, h, l, c))
    date = idx.floor("D")
    week = idx.to_period("W").start_time
    month = idx.to_period("M").start_time
    out = {}

    day_open = O.groupby(date).transform("first")
    week_open = O.groupby(week).transform("first")
    month_open = O.groupby(month).transform("first")
    out["dayopen_dist"] = ((C - day_open) / C).values
    out["weekopen_dist"] = ((C - week_open) / C).values
    out["monthopen_dist"] = ((C - month_open) / C).values

    daily = pd.DataFrame({"h": H, "l": L, "c": C, "o": O}).groupby(date).agg(h=("h", "max"), l=("l", "min"), c=("c", "last"), o=("o", "first"))
    prior = daily.shift(1)
    pmap = prior.reindex(date).set_axis(idx)
    out["pday_hi_dist"] = ((C - pmap.h) / C).values
    out["pday_lo_dist"] = ((C - pmap.l) / C).values
    out["pday_close_dist"] = ((C - pmap.c) / C).values
    piv = (pmap.h + pmap.l + pmap.c) / 3
    r1 = 2 * piv - pmap.l; s1 = 2 * piv - pmap.h
    r2 = piv + (pmap.h - pmap.l); s2 = piv - (pmap.h - pmap.l)
    out["pivot_P_dist"] = ((C - piv) / C).values
    out["pivot_R1_dist"] = ((C - r1) / C).values
    out["pivot_S1_dist"] = ((C - s1) / C).values
    out["pivot_R2_dist"] = ((C - r2) / C).values
    out["pivot_S2_dist"] = ((C - s2) / C).values

    weekly = pd.DataFrame({"h": H, "l": L, "c": C}).groupby(week).agg(h=("h", "max"), l=("l", "min"), c=("c", "last"))
    priorw = weekly.shift(1).reindex(week).set_axis(idx)
    out["pweek_hi_dist"] = ((C - priorw.h) / C).values
    out["pweek_lo_dist"] = ((C - priorw.l) / C).values
    out["pweek_close_dist"] = ((C - priorw.c) / C).values

    for step in round_steps:
        mod = np.mod(c, step)
        out[f"round_dist_{int(step)}"] = (np.minimum(mod, step - mod) / (step / 2.0)).astype(np.float32)  # 0=라운드넘버, 1=중간

    # VWAP 거리: 롤링 1일(1440분) / UTC 일중(그 날 0시부터 누적, 당일 리셋) — 둘 다 그 분까지 완결된 데이터만
    tp1 = (H + L + C) / 3
    Vv = pd.Series(v, index=idx)
    pv = tp1 * Vv
    vwap_roll = pv.rolling(1440, min_periods=60).sum() / (Vv.rolling(1440, min_periods=60).sum() + EPS)
    out["vwap_roll1d_dist"] = ((C - vwap_roll) / C).values
    cum_pv = pv.groupby(date).cumsum(); cum_v = Vv.groupby(date).cumsum()
    vwap_sess = cum_pv / (cum_v + EPS)
    out["vwap_session_dist"] = ((C - vwap_sess) / C).values

    hi1d, lo1d = pd.Series(h).rolling(1440, min_periods=1440).max(), pd.Series(l).rolling(1440, min_periods=1440).min()
    hi7d, lo7d = pd.Series(h).rolling(10080, min_periods=10080).max(), pd.Series(l).rolling(10080, min_periods=10080).min()
    hi30d, lo30d = pd.Series(h).rolling(43200, min_periods=43200).max(), pd.Series(l).rolling(43200, min_periods=43200).min()
    out["hi24h_dist"] = ((C.values - hi1d.values) / C.values)
    out["lo24h_dist"] = ((C.values - lo1d.values) / C.values)
    out["hi7d_dist"] = ((C.values - hi7d.values) / C.values)
    out["lo7d_dist"] = ((C.values - lo7d.values) / C.values)
    out["hi30d_dist"] = ((C.values - hi30d.values) / C.values)
    out["lo30d_dist"] = ((C.values - lo30d.values) / C.values)
    out["fib1d_pos"] = ((C.values - lo1d.values) / (hi1d.values - lo1d.values + EPS))
    out["fib7d_pos"] = ((C.values - lo7d.values) / (hi7d.values - lo7d.values + EPS))
    out["fib1d_pos_c"] = out["fib1d_pos"] - 0.5
    out["fib7d_pos_c"] = out["fib7d_pos"] - 0.5

    ago_hi24, _ = rolling_argext(h, 1440, "max"); ago_lo24, _ = rolling_argext(l, 1440, "min")
    ago_hi7, _ = rolling_argext(h, 10080, "max"); ago_lo7, _ = rolling_argext(l, 10080, "min")
    ago_hi30, _ = rolling_argext(h, 43200, "max"); ago_lo30, _ = rolling_argext(l, 43200, "min")
    out["mins_since_hi24h"] = ago_hi24; out["mins_since_lo24h"] = ago_lo24
    out["mins_since_hi7d"] = ago_hi7; out["mins_since_lo7d"] = ago_lo7
    out["mins_since_hi30d"] = ago_hi30; out["mins_since_lo30d"] = ago_lo30

    for key in list(out.keys()):
        out[key] = np.asarray(out[key], dtype=np.float32)
    return out


# ============================================================ 7. 계절성 ============================================================

SEAS_SIGNED = set()  # 전부 범주/구간형(비부호)


def _reindex_ffill(s, grid_idx, limit=None):
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.reindex(s.index.union(grid_idx)).ffill(limit=limit).reindex(grid_idx)


# ============================================================ 4. 파생상품·포지셔닝 ============================================================

DRV_SIGNED = {
    "oi_chg_5", "oi_chg_60", "oi_chg_240", "oi_chg_1440", "ttlsr_level", "ttlsr_chg60", "ttlsr_chg240",
    "glsr_level", "glsr_chg60", "glsr_chg240", "tlsr_level", "tlsr_chg60", "tlsr_chg240",
    "funding_last_rate", "funding_predicted_proxy", "premium_last", "premium_z", "mark_minus_last",
    "liq_net_60", "liq_net_240",
    "bybit_lead1", "bybit_lead5", "bybit_lead15", "bybit_bas", "bybit_dbas15", "bybit_dbas60",
    "bybit_doi15", "bybit_doi60", "bybit_doi240", "bybit_dfund",
    "kimchi_dkp5", "kimchi_dkp15", "kimchi_dkp60", "kimchi_dkp240", "kimchi_kpz", "kimchi_ulead1", "kimchi_ulead5",
    "flow_snet5", "flow_snet15", "flow_snet60", "flow_snet240", "flow_pnet5", "flow_pnet15", "flow_pnet60",
    "flow_pnet240", "flow_div5", "flow_div15", "flow_div60", "flow_div240", "flow_db15", "flow_db60", "flow_db240",
    "cb_dp5", "cb_dp15", "cb_dp60", "cb_dp240", "cb_z1d", "cb_z7d",
}


def derivatives_features(sym, grid_idx):
    """family4 — 파생상품·포지셔닝(미결제·포지션 비율·펀딩·프리미엄·청산·바이비트 괴리·김치 프리미엄·현물흐름·코인베이스 프리미엄).
    grid_idx: klines_1m 의 전체 ts(UTC DatetimeIndex) — 반환값은 이 길이의 float32 배열 dict(바로 결정분 샘플링 후 버릴 것)."""
    out = {}
    n = len(grid_idx)

    # --- metrics_5m: OI·롱숏비율·테이커비율 (5분 봉 종료 +5분부터 사용 가능 — R115 관행) ---
    m5 = store.load("metrics_5m", sym)
    if m5 is not None and len(m5):
        idx5 = utc(m5.ts) + pd.Timedelta(minutes=5)
        oi = np.log(pd.Series(m5.sum_open_interest.values, index=idx5).replace(0, np.nan))
        oi_al = _reindex_ffill(oi, grid_idx)
        for k, tag in ((5, "oi_chg_5"), (60, "oi_chg_60"), (240, "oi_chg_240"), (1440, "oi_chg_1440")):
            out[tag] = (oi_al - oi_al.shift(k)).values
        for col, pfx in (("sum_toptrader_long_short_ratio", "ttlsr"), ("count_long_short_ratio", "glsr"),
                         ("sum_taker_long_short_vol_ratio", "tlsr")):
            s_ = _reindex_ffill(pd.Series(m5[col].values, index=idx5), grid_idx)
            out[f"{pfx}_level"] = s_.values
            out[f"{pfx}_chg60"] = (s_ - s_.shift(60)).values
            out[f"{pfx}_chg240"] = (s_ - s_.shift(240)).values

    # --- funding: 마지막 확정 요율, 예측(프리미엄 평균 근사), 다음 정산까지 분 ---
    fN = store.load("funding", sym)
    if fN is not None and len(fN):
        idxf = utc(fN.ts)
        rate_al = _reindex_ffill(pd.Series(fN.rate.values, index=idxf), grid_idx)
        out["funding_last_rate"] = rate_al.values
        fts_ns = pd.Series(idxf.asi8, index=idxf)
        fts_al = _reindex_ffill(fts_ns, grid_idx)
        mins_since = (grid_idx.asi8 - fts_al.values) / 6e10
        ih_al = _reindex_ffill(pd.Series(fN.interval_hours.values, index=idxf), grid_idx)
        out["funding_mins_since"] = mins_since.astype(np.float32)
        out["funding_mins_to_next"] = (ih_al.values * 60 - mins_since).astype(np.float32)

    # --- premium_1m / mark_1m ---
    pm = store.load("premium_1m", sym); mk = store.load("mark_1m", sym); kl = store.load("klines_1m", sym)
    if pm is not None and len(pm):
        p_al = _reindex_ffill(pd.Series(pm.close.values, index=utc(pm.ts)), grid_idx, limit=10)
        out["premium_last"] = p_al.values
        pm_mean = p_al.rolling(1440, min_periods=120).mean(); pm_std = p_al.rolling(1440, min_periods=120).std()
        out["premium_z"] = ((p_al - pm_mean) / (pm_std + EPS)).values
        out["funding_predicted_proxy"] = p_al.rolling(480, min_periods=60).mean().values
    if mk is not None and len(mk) and kl is not None:
        mk_al = _reindex_ffill(pd.Series(mk.close.values, index=utc(mk.ts)), grid_idx, limit=10)
        kl_al = pd.Series(kl.close.values, index=utc(kl.ts)).reindex(grid_idx)
        out["mark_minus_last"] = ((mk_al - kl_al) / (kl_al + EPS)).values

    # --- liq_snap(코인마진 청산 스냅샷, 2023-06~2024-10 만 존재) — 분당 순 청산 notional(매도청산 −, 매수청산 +) ---
    coin_sym = {"BTCUSDT": "BTCUSD_PERP", "ETHUSDT": "ETHUSD_PERP"}.get(sym)
    ls = store.load("liq_snap", coin_sym) if coin_sym else None
    try:
        if ls is not None and len(ls):
            tsl = utc(ls.ts).floor("min")
            signed = np.where(ls.side.values == "BUY", 1.0, -1.0) * ls.notional.values  # BUY=숏 청산(매수 강제체결, 상승압력) / SELL=롱 청산(하락압력)
            g = pd.Series(signed, index=tsl).groupby(level=0).sum()
            g = g.reindex(g.index.union(grid_idx)).reindex(grid_idx)
            lo, hi = utc(ls.ts).min(), utc(ls.ts).max()
            covered = (grid_idx >= lo) & (grid_idx <= hi)
            g0 = g.fillna(0.0)
            for k, tag in ((60, "liq_net_60"), (240, "liq_net_240")):
                v_ = g0.rolling(k, min_periods=1).sum().values
                out[tag] = np.where(covered, v_, np.nan).astype(np.float32)
    except Exception:
        pass

    # --- 바이비트 vs 바이낸스 괴리(R115) ---
    try:
        Fb, _, _ = F15.feats(sym)
        Fb = Fb.reindex(grid_idx)
        ren = {"lead1": "bybit_lead1", "lead5": "bybit_lead5", "lead15": "bybit_lead15", "bas": "bybit_bas",
               "dbas15": "bybit_dbas15", "dbas60": "bybit_dbas60", "doi15": "bybit_doi15", "doi60": "bybit_doi60",
               "doi240": "bybit_doi240", "vsh15": "bybit_vsh15", "vsh60": "bybit_vsh60", "dfund": "bybit_dfund",
               "oiB15": "bybit_oiB15", "oiB60": "bybit_oiB60", "oiB240": "bybit_oiB240"}
        for c0, c1 in ren.items():
            if c0 in Fb.columns: out[c1] = Fb[c0].values
    except Exception:
        pass

    # --- 업비트 김치 프리미엄(R116) ---
    try:
        Fk, _, _ = F16.feats(sym)
        Fk = Fk.reindex(grid_idx)
        ren = {"dkp5": "kimchi_dkp5", "dkp15": "kimchi_dkp15", "dkp60": "kimchi_dkp60", "dkp240": "kimchi_dkp240",
               "kpz": "kimchi_kpz", "ulead1": "kimchi_ulead1", "ulead5": "kimchi_ulead5", "ush15": "kimchi_ush15",
               "ush60": "kimchi_ush60", "usurge": "kimchi_usurge"}
        for c0, c1 in ren.items():
            if c0 in Fk.columns: out[c1] = Fk[c0].values
    except Exception:
        pass

    # --- 현물 vs 선물 테이커 흐름(R113) ---
    try:
        Ff, _, _ = F13.feats(sym)
        Ff = Ff.reindex(grid_idx)
        for c0 in Ff.columns:
            out[f"flow_{c0}"] = Ff[c0].values
    except Exception:
        pass

    # --- 코인베이스 프리미엄(R104 패턴) ---
    try:
        import r104_coinbase_lead as F04
        lb, lq, stale = F04.load(sym, sym.replace("USDT", "USD"))
        p = (lq - lb).reindex(grid_idx)
        for k, tag in ((5, "cb_dp5"), (15, "cb_dp15"), (60, "cb_dp60"), (240, "cb_dp240")):
            out[tag] = (p - p.shift(k)).values
        for W, tag in ((1440, "cb_z1d"), (10080, "cb_z7d")):
            mu = p.rolling(W, min_periods=W // 4).mean(); sd = p.rolling(W, min_periods=W // 4).std()
            out[tag] = ((p - mu) / (sd + EPS)).values
    except Exception:
        pass

    for k_ in list(out.keys()):
        out[k_] = np.asarray(out[k_], dtype=np.float32)
    return out


# ============================================================ 5. 변동성 ============================================================

VOL_SIGNED = set()  # 실현변동성·vol-of-vol 은 전부 크기(비부호)


def realized_vol_features(close_1m):
    """family5 — 다중 수평선 실현변동성(연율화) + vol-of-vol(변동성의 변동성). close_1m: 전체 1분 종가."""
    out = {}
    C = pd.Series(close_1m)
    lr = np.log(C / C.shift(1))
    rv = {}
    for h in (60, 240, 1440, 10080, 43200):
        rv[h] = lr.rolling(h, min_periods=h // 4).std() * np.sqrt(525600)
        out[f"rv_{h}"] = rv[h].values
    out["vol_of_vol"] = rv[1440].rolling(10080, min_periods=1440).std().values
    for k in list(out.keys()):
        out[k] = np.asarray(out[k], dtype=np.float32)
    return out


# ============================================================ 6·5 — 15분 격자 저장소 as-of 조인(r97_feat/r54_dfeat/r90 pr) ============================================================

def load_grid_table(path, sym, drop=()):
    """15분 격자 parquet(열: sym,t,... — t−1분까지의 특징) → (정렬된 t 배열, 열 이름, float32 2D 값배열)."""
    d = pd.read_parquet(path)
    d = d[d.sym == sym].sort_values("t").reset_index(drop=True)
    cols = [c for c in d.columns if c not in ("sym", "t", "ts", "yr") and c not in drop]
    return d.t.values.astype(np.int64), cols, d[cols].values.astype(np.float32)


def grid_asof_idx(t_sorted, minutes):
    """분 m 에서 쓸 수 있는 마지막 격자 t_g(= t_g ≤ m+1 중 최댓값)의 위치. 없으면 −1."""
    minutes = np.asarray(minutes, dtype=np.int64)
    pos = np.searchsorted(t_sorted, minutes + 1, side="right") - 1
    return pos


def gather_grid(t_sorted, values2d, minutes):
    pos = grid_asof_idx(t_sorted, minutes)
    out = np.full((len(minutes), values2d.shape[1]), np.nan, dtype=np.float32)
    ok = pos >= 0
    out[ok] = values2d[pos[ok]]
    return out


def load_r90_pr(sym):
    """r90_wf.parquet + 인과적 90일 분위 pr(이미 R92 가 과거만으로 계산) → 15분 격자 (t, [p,pr])."""
    W = F92.load_p()
    g = W[W.sym == sym].sort_values("t").reset_index(drop=True)
    return g.t.values.astype(np.int64), ["p", "pr"], g[["p", "pr"]].values.astype(np.float32)


def newfeats_grid(sym, grid_idx):
    """r119_feats.newfeats(분 해상도, 그 분 종가까지 인과적) → klines 전체 분 그리드에 맞춰 재색인(ffill 없음)."""
    F = F119.newfeats(sym)
    F = F.reindex(grid_idx)
    return list(F.columns), F.values.astype(np.float32)


def load_etf_flows():
    return pd.read_parquet(R / "data" / "ext" / "etf_flows.parquet").set_index("date")


# ============================================================ 8. 종목간(cross-asset) ============================================================

CROSS_SIGNED = {"oth_rsi14_c", "oth_macd_hist", "oth_ret5", "oth_ret15", "oth_ret60", "oth_ret240", "ethbtc_ratio_z"}


def cross_asset_features(oth_close, oth_rsi14, oth_macd_hist):
    """family8 — 다른 종목의 RSI14(센터)·MACD 히스토그램(1m)·수익률(5·15·60·240분), ETH/BTC 비율 30일 z.
    전부 '같은 분(m)'의 값(1m 시간대는 i_max(m)=m 이라 바로 인덱싱 가능, 미래 참조 없음)."""
    out = {}
    oc = pd.Series(oth_close)
    for k in (5, 15, 60, 240):
        out[f"oth_ret{k}"] = (np.log(oc / oc.shift(k))).values
    out["oth_rsi14_c"] = np.asarray(oth_rsi14, dtype=np.float32) - 50.0
    out["oth_macd_hist"] = np.asarray(oth_macd_hist, dtype=np.float32)
    for k in list(out.keys()):
        out[k] = np.asarray(out[k], dtype=np.float32)
    return out


def ethbtc_ratio_z(btc_close, eth_close, W=43200):
    ratio = np.log(pd.Series(eth_close, dtype=np.float64)) - np.log(pd.Series(btc_close, dtype=np.float64))
    mu = ratio.rolling(W, min_periods=W // 4).mean(); sd = ratio.rolling(W, min_periods=W // 4).std()
    return ((ratio - mu) / (sd + EPS)).astype(np.float32).values


def seasonality_features(ts, funding_next_minutes=None):
    idx = pd.DatetimeIndex(ts)
    out = {}
    out["hour"] = idx.hour.values.astype(np.float32)
    out["minute_of_hour"] = idx.minute.values.astype(np.float32)
    out["dow"] = idx.dayofweek.values.astype(np.float32)
    out["is_weekend"] = (idx.dayofweek >= 5).astype(np.float32)
    out["is_us_session"] = (((idx.hour >= 13) & (idx.hour < 21))).astype(np.float32)
    out["is_eu_session"] = (((idx.hour >= 7) & (idx.hour < 16))).astype(np.float32)
    out["is_asia_session"] = (((idx.hour >= 0) & (idx.hour < 9))).astype(np.float32)
    out["is_month_end"] = (idx.day >= 28).astype(np.float32)
    minute_of_day = idx.hour.values * 60 + idx.minute.values
    out["mins_to_funding"] = (((8 * 60) - (minute_of_day % (8 * 60)))).astype(np.float32)
    return out
