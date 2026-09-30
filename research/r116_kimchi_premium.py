"""R116 — 업비트 1분 김치 프리미엄 동학: 한국 개인 수요의 급변이 바이낸스 BTC·ETH를 예측하는가, 돌파·딥을 가르는가.

선행 조사: 김치 프리미엄은 자본 통제 아래 한국 개인 수요의 지표. 음→양 전환 뒤 수익이 강했다는 사례 보고(학술 근거 약함). 미국 ETF 흐름이 더 중요하다는 반론.
프리미엄 kp = log(업비트 KRW 종가) − log(바이낸스 USDT 종가 × 원달러) — 원달러(FRED DEXKOUS)는 하루 늦게(다음 날부터) 사용하므로 수준에 잡음이 있다.
  → 분 단위 '변화'는 환율 영향이 거의 없다. 수준은 7일 z로만 쓴다.
특징(t분까지, 종목별): dkp_k = kp 변화(k 5·15·60·240), kpz = (kp − 7일 평균)/7일 표준편차,
  ulead_k = 업비트 k분 수익 − 바이낸스 k분 수익 (k 1·5), ush_k = 업비트 원화 대금(달러 환산) / (업비트 + 바이낸스 대금) − 7일 평균 (k 15·60),
  usurge = 업비트 15분 대금 / 7일 같은 길이 중앙값.
A/B/C는 R113과 같다(학습 2020~23 / 평가 2024~26).
사용: python research/r116_kimchi_premium.py"""
import sys, warnings, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr, norm
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
import r102_joint_account as J, r103_breakout_cost as K, r109_dip_exact_mtm as M, r113_spot_perp_flow as F13, r115_bybit_divergence as F15
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
UP = {"BTCUSDT": "KRWBTC", "ETHUSDT": "KRWETH"}


def feats(sym):
    n = store.load("klines_1m", sym).set_index("ts"); u = store.load("upbit_1m", UP[sym]).set_index("ts")
    n.index = F15.utc(n.index); u.index = F15.utc(u.index)
    idx = pd.date_range(max(n.index[0], u.index[0]), min(n.index[-1], u.index[-1]), freq="1min")
    n = n.reindex(idx); u = u.reindex(idx); stale = n.close.isna() | u.close.isna()
    fx = pd.read_parquet(R / "data" / "ext" / "usdkrw.parquet").set_index("date").krw
    fx.index = F15.utc(fx.index) + pd.Timedelta(days=1)                            # 다음 날부터 사용
    fx = fx[~fx.index.duplicated()].reindex(idx, method="ffill")
    ln, lu = np.log(n.close.ffill()), np.log(u.close.ffill())
    kp = lu - ln - np.log(fx)
    Fd = {}
    for k in (5, 15, 60, 240): Fd[f"dkp{k}"] = kp - kp.shift(k)
    Fd["kpz"] = (kp - kp.rolling(10080, min_periods=1440).mean()) / kp.rolling(10080, min_periods=1440).std()
    for k in (1, 5): Fd[f"ulead{k}"] = (lu - lu.shift(k)) - (ln - ln.shift(k))
    vu = (u.value / fx).fillna(0); vn = n.quote_volume.fillna(0)
    for k in (15, 60):
        s_ = vu.rolling(k).sum() / (vu.rolling(k).sum() + vn.rolling(k).sum()).replace(0, np.nan)
        Fd[f"ush{k}"] = s_ - s_.rolling(10080, min_periods=1440).mean()
    v15 = vu.rolling(15).sum(); Fd["usurge"] = v15 / v15.rolling(10080, min_periods=1440).median().replace(0, np.nan)
    return pd.DataFrame(Fd).astype("float32"), ln, stale.rolling(240, min_periods=1).max().astype(bool)


if __name__ == "__main__":
    F15.analyze(feats, "r116_ic.parquet")
