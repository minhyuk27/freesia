"""1배(무레버리지) 벡터화 백테스트 코어.

가정(보수적):
- 신호는 봉 마감 시점 값만 사용, 다음 봉 수익에 적용 (24시간 시장이라 다음 봉 시가 ≈ 이전 봉 종가).
- 수수료 taker 5bp + 슬리피지 1bp = 편도 6bp × 포지션 변화량.
- 펀딩: 봉 안에 정산이 있으면 포지션 × 펀딩률만큼 지불(롱) / 수취(숏).
- 결과는 자본 1 기준 순수익률. 배율은 여기서 다루지 않는다.
"""
import numpy as np
import pandas as pd

from freesia import store

COST = 0.0006  # 편도: taker 0.05% + 슬리피지 0.01%
TF_HOURS = {"1h": 1, "4h": 4, "1d": 24}
PPY = {"1h": 24 * 365, "4h": 6 * 365, "1d": 365}
CACHE = store.ROOT / "data" / "bars"


def bars(sym: str, tf: str) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"{sym}_{tf}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    k = store.load("klines_1m", sym).set_index("ts")
    rule = {"1h": "1h", "4h": "4h", "1d": "1D"}[tf]
    b = k.resample(rule).agg({"open": "first", "high": "max", "low": "min", "close": "last",
                              "volume": "sum", "quote_volume": "sum", "trades": "sum",
                              "taker_buy_quote_volume": "sum"}).dropna(subset=["close"])
    b.to_parquet(p)
    return b


def funding_bars(sym: str, index: pd.DatetimeIndex, tf: str) -> np.ndarray:
    f = store.load("funding", sym).set_index("ts").rate
    rule = {"1h": "1h", "4h": "4h", "1d": "1D"}[tf]
    return f.resample(rule).sum().reindex(index).fillna(0.0).to_numpy()


def run(pos: np.ndarray, close: np.ndarray, fund: np.ndarray, cost: float = COST):
    """pos[i] 는 i번째 봉 마감 시점에 정한 포지션(-1~1). 수익은 i+1 봉에 적용."""
    pos = np.nan_to_num(pos)
    p_prev = np.concatenate([[0.0], pos[:-1]])          # i 봉 동안 들고 있는 포지션
    p_prev2 = np.concatenate([[0.0], p_prev[:-1]])
    ret = np.concatenate([[0.0], close[1:] / close[:-1] - 1.0])
    net = p_prev * ret - np.abs(p_prev - p_prev2) * cost - p_prev * fund
    return net


def metrics(net: np.ndarray, tf: str) -> dict:
    ppy = PPY[tf]
    m, s = net.mean(), net.std()
    eq = np.cumprod(1 + net)
    dd = (eq / np.maximum.accumulate(eq) - 1).min()
    return {"ann": m * ppy, "vol": s * np.sqrt(ppy), "sharpe": m / s * np.sqrt(ppy) if s > 0 else 0.0,
            "mdd": dd}


def turnover_stats(pos: np.ndarray, tf: str):
    d = np.abs(np.diff(np.nan_to_num(pos), prepend=0.0))
    return d.sum() / (len(pos) / PPY[tf])  # 연간 포지션 회전량(편도 합)


def walk_forward(R: np.ndarray, index: pd.DatetimeIndex, tf: str, first_test_year: int,
                 min_train_years: float = 1.5, names=None):
    """연도별 워크포워드: 해당 연도 이전 전체 데이터로 Sharpe 최고 파라미터 선택 → 그 해 성과 이어붙임.
    R: (T, P) 순수익률 행렬. 반환: OOS 수익률 배열, 연도별 (선택된 파라미터 idx)"""
    yrs = index.year.to_numpy()
    ppy = PPY[tf]
    oos = np.zeros(len(index))
    used = np.zeros(len(index), dtype=bool)
    picks = {}
    for Y in range(first_test_year, yrs.max() + 1):
        tr = yrs < Y
        te = yrs == Y
        if tr.sum() < min_train_years * ppy or te.sum() == 0:
            continue
        mu, sd = R[tr].mean(0), R[tr].std(0)
        sh = np.where(sd > 0, mu / np.where(sd > 0, sd, 1), -9)
        j = int(np.argmax(sh))
        oos[te] = R[te, j]
        used |= te
        picks[Y] = (j, float(sh[j] * np.sqrt(ppy)))
    return oos, used, picks


def run_ret(pos: np.ndarray, ret: np.ndarray, fund_eff: np.ndarray, cost: float = COST):
    """수익률 시계열을 직접 받는 버전 (페어·캐리 등 합성 수익률용)."""
    pos = np.nan_to_num(pos)
    p_prev = np.concatenate([[0.0], pos[:-1]])
    p_prev2 = np.concatenate([[0.0], p_prev[:-1]])
    return p_prev * np.nan_to_num(ret) - np.abs(p_prev - p_prev2) * cost - p_prev * fund_eff
