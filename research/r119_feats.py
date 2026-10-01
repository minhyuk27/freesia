"""R119 공통 — 새 데이터 특징 묶음(분 단위, t분 종가까지): 현물·선물 흐름(R113), 김치 프리미엄(R116), 코인베이스 프리미엄(R104), 바이비트 OI·거래 비중(R115).

newfeats(sym) → DataFrame(index = 1분 UTC, float32). 거래에 붙일 때는 진입 분 − 1분 값을 쓴다(attach).
메모리: 종목당 약 350만 분 × 25열 × 4바이트 ≈ 350MB.
사용: from r119_feats import newfeats, attach"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
import r113_spot_perp_flow as F13, r116_kimchi_premium as F16, r104_coinbase_lead as F04, r115_bybit_divergence as F15
D0 = pd.Timestamp("2020-01-01", tz="UTC")
NEWF = ["snet5", "snet15", "snet60", "pnet5", "pnet15", "pnet60", "div5", "div15", "div60", "share5", "share60",
        "dkp5", "dkp15", "dkp60", "kpz", "ulead5", "usurge", "cdp15", "cdp60", "cz1d", "oiB60", "doi60", "vsh60", "dfund"]


def newfeats(sym):
    Fs, _, _ = F13.feats(sym)
    idx = Fs.index
    out = [Fs[["snet5", "snet15", "snet60", "pnet5", "pnet15", "pnet60", "div5", "div15", "div60", "share5", "share60"]]]
    Fk = F16.feats(sym)[0][["dkp5", "dkp15", "dkp60", "kpz", "ulead5", "usurge"]]; out.append(Fk.reindex(idx))
    b, q, st = F04.load(sym, sym.replace("USDT", "USD")); p = q - b
    out.append(pd.DataFrame({"cdp15": p - p.shift(15), "cdp60": p - p.shift(60),
                             "cz1d": (p - p.rolling(1440, min_periods=720).mean()) / p.rolling(1440, min_periods=720).std()}).astype("float32").reindex(idx))
    Fb = F15.feats(sym)[0][["oiB60", "doi60", "vsh60", "dfund"]]; out.append(Fb.reindex(idx))
    return pd.concat(out, axis=1).astype("float32")


def attach(ts_entry, F):
    """진입 시각(Timestamp 배열, UTC) − 1분의 특징 값."""
    t = pd.DatetimeIndex(ts_entry)
    t = t.tz_convert("UTC") if t.tz is not None else t.tz_localize("UTC")
    return F.reindex(t - pd.Timedelta(minutes=1)).values
