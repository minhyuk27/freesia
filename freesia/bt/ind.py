"""기술적 지표 (표준 정의). 입력 pandas Series, 출력 Series/ndarray. 룩어헤드 없음(현재 봉 마감까지의 값만 사용)."""
import numpy as np
import pandas as pd


def sma(s, n): return s.rolling(n).mean()
def ema(s, n): return s.ewm(span=n, adjust=False).mean()


def rsi(c, n):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def true_range(h, l, c):
    pc = c.shift(1)
    return pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)


def atr(h, l, c, n): return true_range(h, l, c).ewm(alpha=1 / n, adjust=False).mean()


def macd(c, f, s, g):
    m = ema(c, f) - ema(c, s)
    sig = ema(m, g)
    return m, sig, m - sig


def stoch(h, l, c, n, d=3):
    hh, ll = h.rolling(n).max(), l.rolling(n).min()
    k = 100 * (c - ll) / (hh - ll).replace(0, np.nan)
    return k, k.rolling(d).mean()


def adx(h, l, c, n):
    up, dn = h.diff(), -l.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=c.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=c.index)
    a = atr(h, l, c, n)
    pdi = 100 * pdm.ewm(alpha=1 / n, adjust=False).mean() / a
    mdi = 100 * mdm.ewm(alpha=1 / n, adjust=False).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return pdi, mdi, dx.ewm(alpha=1 / n, adjust=False).mean()


def zscore_tp(h, l, c, n):
    tp = (h + l + c) / 3
    return (tp - tp.rolling(n).mean()) / tp.rolling(n).std()      # CCI 표준편차 근사


def bbands(c, n, k):
    m, s = c.rolling(n).mean(), c.rolling(n).std()
    return m, m + k * s, m - k * s


def keltner(h, l, c, n, k):
    m = ema(c, n)
    a = atr(h, l, c, n)
    return m, m + k * a, m - k * a


def aroon(h, l, n):
    up = h.rolling(n + 1).apply(np.argmax, raw=True) * 100 / n
    dn = l.rolling(n + 1).apply(np.argmin, raw=True) * 100 / n
    return up, dn


def vortex(h, l, c, n):
    vmp, vmm = (h - l.shift(1)).abs(), (l - h.shift(1)).abs()
    tr = true_range(h, l, c).rolling(n).sum()
    return vmp.rolling(n).sum() / tr, vmm.rolling(n).sum() / tr


def trix(c, n):
    e = ema(ema(ema(c, n), n), n)
    return e.pct_change() * 100


def mfi(h, l, c, v, n):
    tp = (h + l + c) / 3
    rmf = tp * v
    pos = rmf.where(tp > tp.shift(1), 0.0).rolling(n).sum()
    neg = rmf.where(tp < tp.shift(1), 0.0).rolling(n).sum()
    return 100 - 100 / (1 + pos / neg.replace(0, np.nan))


def cmf(h, l, c, v, n):
    mfm = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
    return (mfm * v).rolling(n).sum() / v.rolling(n).sum()


def wma(s, n):
    w = np.arange(1, n + 1, dtype=float)
    return s.rolling(n).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)


def hma(c, n):
    return wma(2 * wma(c, max(2, n // 2)) - wma(c, n), max(2, int(np.sqrt(n))))


def kama(c, n, fast, slow):
    x = c.to_numpy(); out = np.full(len(x), np.nan)
    ch = np.abs(pd.Series(x).diff(n).to_numpy()); vol = pd.Series(np.abs(np.diff(x, prepend=x[0]))).rolling(n).sum().to_numpy()
    fs, ss = 2 / (fast + 1), 2 / (slow + 1)
    start = n
    if len(x) <= start:
        return pd.Series(out, index=c.index)
    out[start] = x[start]
    for i in range(start + 1, len(x)):
        er = ch[i] / vol[i] if vol[i] > 0 else 0.0
        sc = (er * (fs - ss) + ss) ** 2
        out[i] = out[i - 1] + sc * (x[i] - out[i - 1])
    return pd.Series(out, index=c.index)


def supertrend(h, l, c, n, m):
    a = atr(h, l, c, n).to_numpy(); hl2 = ((h + l) / 2).to_numpy(); cc = c.to_numpy()
    ub, lb = hl2 + m * a, hl2 - m * a
    N = len(cc); fu, fl, d = ub.copy(), lb.copy(), np.zeros(N)
    for i in range(1, N):
        if np.isnan(a[i]):
            continue
        fu[i] = ub[i] if (ub[i] < fu[i - 1] or cc[i - 1] > fu[i - 1] or np.isnan(fu[i - 1])) else fu[i - 1]
        fl[i] = lb[i] if (lb[i] > fl[i - 1] or cc[i - 1] < fl[i - 1] or np.isnan(fl[i - 1])) else fl[i - 1]
        if cc[i] > fu[i - 1]:
            d[i] = 1
        elif cc[i] < fl[i - 1]:
            d[i] = -1
        else:
            d[i] = d[i - 1]
    return d


def psar(h, l, af0, afmax):
    hh, ll = h.to_numpy(), l.to_numpy(); N = len(hh)
    d = np.ones(N); sar = ll[0]; ep = hh[0]; af = af0; up = True
    for i in range(1, N):
        sar = sar + af * (ep - sar)
        if up:
            sar = min(sar, ll[i - 1], ll[i - 2] if i > 1 else ll[i - 1])
            if ll[i] < sar:
                up, sar, ep, af = False, ep, ll[i], af0
            elif hh[i] > ep:
                ep, af = hh[i], min(af + af0, afmax)
        else:
            sar = max(sar, hh[i - 1], hh[i - 2] if i > 1 else hh[i - 1])
            if hh[i] > sar:
                up, sar, ep, af = True, ep, hh[i], af0
            elif ll[i] < ep:
                ep, af = ll[i], min(af + af0, afmax)
        d[i] = 1 if up else -1
    return d


def ichimoku(h, l, c, t, k, s):
    ten = (h.rolling(t).max() + l.rolling(t).min()) / 2
    kij = (h.rolling(k).max() + l.rolling(k).min()) / 2
    a = ((ten + kij) / 2).shift(k)
    b = ((h.rolling(s).max() + l.rolling(s).min()) / 2).shift(k)
    return ten, kij, a, b


def heikin_ashi(o, h, l, c):
    hc = ((o + h + l + c) / 4).to_numpy(); ho = np.zeros(len(hc)); ho[0] = (o.iloc[0] + c.iloc[0]) / 2
    for i in range(1, len(hc)):
        ho[i] = (ho[i - 1] + hc[i - 1]) / 2
    return ho, hc


def swings(h, l, k):
    """프랙탈 스윙: 좌우 k봉보다 높은(낮은) 봉은 k봉 뒤에 확정. 반환: 확정된 마지막 스윙 고점·저점의 값과 발생 위치."""
    H, L = h.to_numpy(), l.to_numpy(); n = len(H)
    mx = h.rolling(2 * k + 1).max().to_numpy(); mn = l.rolling(2 * k + 1).min().to_numpy()
    hk, lk = h.shift(k).to_numpy(), l.shift(k).to_numpy()
    ish, isl = hk == mx, lk == mn
    idx = np.arange(n)
    shv = pd.Series(np.where(ish, hk, np.nan)).ffill().to_numpy(); shi = pd.Series(np.where(ish, idx - k, np.nan)).ffill().to_numpy()
    slv = pd.Series(np.where(isl, lk, np.nan)).ffill().to_numpy(); sli = pd.Series(np.where(isl, idx - k, np.nan)).ffill().to_numpy()
    return shv, shi, slv, sli, ish, isl
