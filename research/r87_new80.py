"""R87 — 완전히 새로운 관점 11계열 80개 전략 → 엣지 선별 → z0.31과 결합.

계열(전부 규칙 사전 고정, 파라미터 탐색 없음)
  V 변동성 국면 전환(12) · K 김치 프리미엄(10) · C CME 주말 갭(6) · X 옵션 만기(6) · O 옵션 흐름 사건(12)
  L 코인베이스→바이낸스 선행(8) · S 스테이블코인 공급(4) · M 거시 발표 양방향 스톱(4) · H 추세·변동성 국면(6)
  Q 수익 분포 모양(6) · R 분산비 국면(6)
체결: 1분봉. 진입 = 신호 다음 분 시가(양방향 스톱은 돌파 다음 분 시가), 청산 = 목표·손절(%)·최대 보유 중 먼저. 비용 7bp(목표는 2bp).
판정: n ≥ 40, 전체 t ≥ 1.5, 2020~23 과 2024~26 모두 흑자.
결합: 통과 전략 일간 손익(거래 = 명목 1, 종목 0.5) + z0.31(R81 설정 w1N4). 비중은 2022~23로 추정(위험 균등·최대 샤프), 2024~26 표본 밖 샤프 비교.
사용: python research/r87_new80.py"""
import sys, warnings, time, glob, re
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from freesia import store
from r79_longshort_book import run_trades_x
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
CT, CM = 7e-4, 2e-4
EXT = R / "data" / "ext"; SYMS = ("BTCUSDT", "ETHUSDT")
BIG = 1000.0                                                            # 목표·손절 없음(시간 청산)


@numba.njit(cache=True)
def straddle(start, px, x, wait, h, l):
    """start 분부터 wait 분 안에 px×(1±x) 돌파한 첫 분 → (진입 분 = 다음 분, 방향). 없으면 −1."""
    n = len(h); ent = np.full(len(start), -1, np.int64); sd = np.zeros(len(start), np.int64)
    for q in range(len(start)):
        for i in range(start[q], min(n - 1, start[q] + wait[q])):
            if h[i] >= px[q] * (1 + x[q]): ent[q] = i + 1; sd[q] = 1; break
            if l[i] <= px[q] * (1 - x[q]): ent[q] = i + 1; sd[q] = -1; break
    return ent, sd


class Data:
    def __init__(self):
        self.K = {s: store.load("klines_1m", s).set_index("ts") for s in SYMS}
        self.H = {}
        for s in SYMS:
            k = self.K[s]; lr = np.log(k.close).diff()
            H = k.resample("1h").agg({"open": "first", "high": "max", "low": "min", "close": "last", "quote_volume": "sum"}).dropna()
            H["rv"] = lr.resample("1h").std().reindex(H.index)
            H["t"] = H.index + pd.Timedelta("1h")                         # 결정 시각 = 봉 종가
            self.H[s] = H
        self.cb = {s: store.load("coinbase_1m", "BTCUSD" if s == "BTCUSDT" else "ETHUSD").set_index("ts").close for s in SYMS}


def entries_at(D, s, times, side, tg, st, hold_h):
    k = D.K[s]; idx = k.index.searchsorted(pd.DatetimeIndex(times)); ok = idx < len(k) - 2
    n = ok.sum(); arr = lambda v: np.broadcast_to(np.asarray(v, dtype=np.float64), (len(idx),))[ok].copy()
    return pd.DataFrame({"i": idx[ok], "side": np.broadcast_to(np.asarray(side, np.int64), (len(idx),))[ok], "tg": arr(tg), "st": arr(st),
                         "hold": (arr(hold_h) * 60).astype(np.int64)})


def run(D, s, E):
    if E is None or len(E) == 0: return pd.DataFrame(columns=["t", "r", "sym"])
    E = E.sort_values("i"); k = D.K[s]
    u, r, ex = run_trades_x(E.i.values.astype(np.int64), E.side.values.astype(np.int64), E.tg.values, E.st.values, E.hold.values,
                            k.open.values, k.high.values, k.low.values, k.close.values, CT, CM)
    return pd.DataFrame({"t": k.index[E.i.values[u]], "r": r[u], "sym": s})


def strategies(D):
    """이름 → 함수(sym) → 진입 DataFrame. 신호는 결정 시각(봉 종가) 이후 다음 분 진입."""
    S = {}
    def add(name, fn): S[name] = fn
    # ---------- 공통 시간 특징 ----------
    HF = {}
    for s in SYMS:
        H = D.H[s].copy(); c = H.close; lc = np.log(c)
        rv2 = H.rv ** 2 * 60                                            # 1시간 분산
        H["RVd"] = rv2.rolling(24).mean(); H["RVm"] = rv2.rolling(720).mean(); H["q"] = H.RVd / H.RVm
        H["sig24"] = np.sqrt(H.RVm * 24); H["sig1"] = np.sqrt(H.RVm)
        H["r1"] = lc.diff(); H["r4"] = lc.diff(4); H["r24"] = lc.diff(24); H["r168"] = lc.diff(168)
        H["hi24"] = H.high.rolling(24).max(); H["lo24"] = H.low.rolling(24).min()
        H["sma200d"] = c.rolling(4800).mean()
        r1 = H.r1
        H["skew"] = r1.rolling(168).skew(); H["kurt"] = r1.rolling(168).kurt()
        H["VR"] = lc.diff(24).rolling(720).var() / (24 * r1.rolling(720).var())
        HF[s] = H
    # ---------- V 변동성 국면 ----------
    for x_, tm, nm in ((0.5, 2, "V01"), (0.5, 3, "V02"), (1.0, 2, "V03"), (1.0, 3, "V04")):
        def f(s, x_=x_, tm=tm):
            H = HF[s]; m = (H.q < 0.5) & (H.q.shift(1) >= 0.5)            # 깊은 압축 진입 순간
            T = H[m]; k = D.K[s]; st_ = k.index.searchsorted(pd.DatetimeIndex(T.t))
            ok = st_ < len(k) - 2; T = T[ok]; st_ = st_[ok]
            xx = (x_ * T.sig24).values
            ent, sd = straddle(st_.astype(np.int64), k.close.values[np.maximum(st_ - 1, 0)], xx, np.full(len(st_), 2880, np.int64), k.high.values, k.low.values)
            g = ent >= 0
            return pd.DataFrame({"i": ent[g], "side": sd[g], "tg": xx[g] * 100 * tm, "st": xx[g] * 100, "hold": np.full(g.sum(), 48 * 60)})
        add(f"{nm} 깊은 변동성 압축 → 양방향 스톱 ±{x_}σ24, 목표 {tm}배", f)
    for th, hold, nm in ((1.5, 12, "V05"), (2.0, 12, "V06"), (2.5, 12, "V07"), (2.0, 24, "V08")):
        def f(s, th=th, hold=hold):
            H = HF[s]; m = (H.q > th) & (H.r4.abs() > 2 * H.sig1 * 2)
            T = H[m]; return entries_at(D, s, T.t, -np.sign(T.r4).astype(int), (T.sig1 * 100).values, (T.sig1 * 100).values, hold)
        add(f"{nm} 변동성 급등(q>{th}) + 4h 큰 움직임 → 역행 {hold}h", f)
    def f(s):
        H = HF[s]; m = (H.q < 0.8) & (H.r168 > 0) & (H.index.hour == 0); T = H[m]
        return entries_at(D, s, T.t, 1, BIG, BIG, 24)
    add("V09 잔잔한 상승추세(q<0.8, 7일↑) → 일일 롱", f)
    def f(s):
        H = HF[s]; m = (H.q < 0.8) & (H.r168 < 0) & (H.index.hour == 0); T = H[m]
        return entries_at(D, s, T.t, -1, BIG, BIG, 24)
    add("V10 잔잔한 하락추세 → 일일 숏", f)
    def f(s):
        H = HF[s]; m = (H.q < 0.8) & (H.r1 < -2 * H.sig1); T = H[m]
        return entries_at(D, s, T.t, 1, (T.sig1 * 150).values, (T.sig1 * 150).values, 12)
    add("V11 저변동 국면의 1h −2σ 급락 → 롱", f)
    def f(s):
        H = HF[s]; m = (H.q > 1.5) & (H.r1 < -2 * H.sig1); T = H[m]
        return entries_at(D, s, T.t, -1, (T.sig1 * 150).values, (T.sig1 * 150).values, 12)
    add("V12 고변동 국면의 1h −2σ 급락 → 추종 숏", f)
    # ---------- K 김치 프리미엄 ----------
    up = pd.read_parquet(EXT / "upbit_1h.parquet").set_index("ts"); fx = pd.read_parquet(EXT / "usdkrw.parquet").set_index("date").krw
    fx.index = fx.index + pd.Timedelta(days=1)
    for s in SYMS:
        H = HF[s]; u = up["btc" if s == "BTCUSDT" else "eth"].reindex(H.index)
        f_ = fx.reindex(fx.index.union(H.index)).ffill().reindex(H.index)
        kp = u / (H.close * f_) - 1; H["kp"] = kp
        H["kpz"] = (kp - kp.rolling(720, min_periods=240).mean()) / kp.rolling(720, min_periods=240).std(); H["dkp"] = kp.diff(24)
        H["dkpq"] = H.dkp.rolling(2160, min_periods=500).rank(pct=True)
    for sgn, th, hold, nm, desc in ((-1, 2, 24, "K01", "김프 z>2(개인 과열) → 숏 24h"), (1, -2, 24, "K02", "김프 z<−2(역프) → 롱 24h"),
                                    (-1, 2, 72, "K03", "김프 z>2 → 숏 72h"), (1, -2, 72, "K04", "김프 z<−2 → 롱 72h"),
                                    (1, 2, 24, "K08", "김프 z>2 → 추종 롱 24h(부호 확인)")):
        def f(s, sgn=sgn, th=th, hold=hold):
            H = HF[s]; m = (H.kpz > th) if th > 0 else (H.kpz < th); m = m & ~m.shift(1).fillna(False).astype(bool)
            T = H[m]; return entries_at(D, s, T.t, sgn, BIG, BIG, hold)
        add(f"{nm} {desc}", f)
    def f(s):
        H = HF[s]; m = (H.dkpq > 0.9) & ~(H.dkpq.shift(1) > 0.9); T = H[m]; return entries_at(D, s, T.t, -1, BIG, BIG, 24)
    add("K05 김프 24h 상승 상위 10% → 숏 24h", f)
    def f(s):
        H = HF[s]; m = (H.dkpq < 0.1) & ~(H.dkpq.shift(1) < 0.1); T = H[m]; return entries_at(D, s, T.t, 1, BIG, BIG, 24)
    add("K06 김프 24h 하락 하위 10% → 롱 24h", f)
    def f(s):
        H = HF[s]; m = (H.kpz > 1.5) & (H.r24 > 2 * H.sig24 / 2) & (H.dkp > 0); m = m & ~m.shift(1).fillna(False).astype(bool); T = H[m]
        return entries_at(D, s, T.t, -1, (T.sig24 * 100).values, (T.sig24 * 100).values, 48)
    add("K07 김프↑ + 가격↑(개인 FOMO) → 숏 ±1σ24", f)
    def f(s):
        H = HF[s]; m = (H.kpz < -1.5) & (H.r24 < -H.sig24); m = m & ~m.shift(1).fillna(False).astype(bool); T = H[m]
        return entries_at(D, s, T.t, 1, (T.sig24 * 100).values, (T.sig24 * 100).values, 48)
    add("K09 역프 + 가격 급락(개인 투매) → 롱 ±1σ24", f)
    def f(s):
        H = HF[s]; m = (H.kpz.abs() < 0.5) & (H.kpz.shift(24).abs() > 2); T = H[m]
        return entries_at(D, s, T.t, -np.sign(H.kpz.shift(24)[m]).astype(int), BIG, BIG, 24)
    add("K10 김프 극단 해소 직후 → 해소 전 방향의 반대 24h", f)
    # ---------- C CME 주말 갭 ----------
    for thr, mode, nm, desc in ((0.01, "fill", "C01", "|갭|>1% → 금요일 종가로 메우기"), (0.02, "fill", "C02", "|갭|>2% → 메우기"),
                                (0.01, "cont", "C03", "|갭|>1% → 갭 방향 지속"), (0.01, "fill_up", "C05", "갭 상승 >1% → 메우기(숏)"),
                                (0.01, "fill_dn", "C06", "갭 하락 >1% → 메우기(롱)"), (0.005, "fill", "C04", "|갭|>0.5% → 메우기")):
        def f(s, thr=thr, mode=mode):
            c = D.H[s].close; fr = c[(c.index.weekday == 4) & (c.index.hour == 20)]    # 금 20:00 봉 종가 = 21:00
            rows = []
            for t, pf in fr.items():
                ts = t + pd.Timedelta(hours=49)                                          # 일 22:00 봉 시작 → 종가 23:00 근사 → 22:00 사용
                if ts not in c.index: continue
                ps = c[t + pd.Timedelta(hours=48)] if (t + pd.Timedelta(hours=48)) in c.index else np.nan
                if np.isnan(ps): continue
                g = ps / pf - 1
                if abs(g) < thr: continue
                if mode == "fill_up" and g < 0: continue
                if mode == "fill_dn" and g > 0: continue
                sd = -np.sign(g) if mode != "cont" else np.sign(g)
                rows.append((t + pd.Timedelta(hours=49), int(sd), abs(g) * 100 if mode != "cont" else abs(g) * 100, abs(g) * 150))
            if not rows: return None
            T = pd.DataFrame(rows, columns=["t", "sd", "tg", "st"])
            return entries_at(D, s, T.t, T.sd.values, T.tg.values, T.st.values, 72)
        add(f"{nm} CME 갭 {desc}", f)
    # ---------- X 옵션 만기 ----------
    from r66_lines_all import options_levels
    mp = {s: options_levels(s)["mp"] for s in SYMS}
    def fridays(kind):
        d = pd.date_range("2020-01-03", "2026-09-25", freq="W-FRI", tz="UTC")
        if kind == "all": return d
        last = d.to_series().groupby([d.year, d.month]).transform("max") == d.to_series()
        d2 = d[last.values]
        return d2 if kind == "month" else d2[d2.month.isin([3, 6, 9, 12])]
    for kind, nm in (("all", "X01"), ("month", "X02"), ("quarter", "X03")):
        def f(s, kind=kind):
            rows = []; c = D.H[s].close
            for fr in fridays(kind):
                t0 = fr - pd.Timedelta(hours=24) + pd.Timedelta(hours=8); lv = mp[s].get(fr.normalize(), np.nan)
                pr = c.asof(t0 - pd.Timedelta(hours=1)) if len(c) else np.nan
                if np.isnan(lv) or np.isnan(pr): continue
                rows.append((t0, 1 if lv > pr else -1, abs(lv / pr - 1) * 100, abs(lv / pr - 1) * 100))
            T = pd.DataFrame(rows, columns=["t", "sd", "tg", "st"]); T = T[T.tg > 0.1]
            return entries_at(D, s, T.t, T.sd.values, T.tg.values, T.st.values, 24)
        add(f"{nm} 만기({kind}) 24h 전 → 최대고통 쪽으로(고정 효과)", f)
    def f(s):
        c = D.H[s].close; rows = []
        for fr in fridays("all"):
            t = fr + pd.Timedelta(hours=8); pre = c.asof(t - pd.Timedelta(hours=1)) / c.asof(t - pd.Timedelta(hours=25)) - 1
            if np.isnan(pre) or abs(pre) < 0.005: continue
            rows.append((t, int(np.sign(pre))))
        T = pd.DataFrame(rows, columns=["t", "sd"]); return entries_at(D, s, T.t, T.sd.values, BIG, BIG, 24)
    add("X04 만기 후 → 만기 전 24h 방향 지속", f)
    def f(s):
        E = S["X04 만기 후 → 만기 전 24h 방향 지속"](s); E["side"] = -E.side; return E
    add("X05 만기 후 → 만기 전 방향 반전", f)
    def f(s):
        k = D.K[s]; H = HF[s]; fr = fridays("all") + pd.Timedelta(hours=8); st_ = k.index.searchsorted(fr); ok = st_ < len(k) - 2; st_ = st_[ok]
        sig = H.sig24.reindex(fr[ok], method="ffill").values * 0.5
        ent, sd = straddle(st_.astype(np.int64), k.close.values[st_ - 1], sig, np.full(len(st_), 1440, np.int64), k.high.values, k.low.values)
        g = ent >= 0
        return pd.DataFrame({"i": ent[g], "side": sd[g], "tg": sig[g] * 200, "st": sig[g] * 100, "hold": np.full(g.sum(), 1440)})
    add("X06 만기 시각 양방향 스톱 ±0.5σ24", f)
    # ---------- O 옵션 흐름 ----------
    OF = {}
    for s in SYMS:
        cur = "BTC" if s == "BTCUSDT" else "ETH"; big = 25 if cur == "BTC" else 250; agg = []
        for fpath in sorted(glob.glob(str(store.PQ / "deribit_opt" / cur / "*.parquet"))):
            x = pd.read_parquet(fpath, columns=["ts", "expiry", "cp", "iv", "price", "index_price", "amount", "side"])
            x["prem"] = x.amount * x.price * x.index_price; x["h"] = x.ts.dt.floor("h")
            x["short"] = (x.expiry - x.ts) <= pd.Timedelta(days=2); x["blk"] = x.amount >= big
            g = x.groupby("h")
            a = pd.DataFrame({"cb": x.prem.where((x.cp == 1) & (x.side == 1), 0).groupby(x.h).sum(), "cs": x.prem.where((x.cp == 1) & (x.side != 1), 0).groupby(x.h).sum(),
                              "pb": x.prem.where((x.cp != 1) & (x.side == 1), 0).groupby(x.h).sum(), "ps": x.prem.where((x.cp != 1) & (x.side != 1), 0).groupby(x.h).sum(),
                              "bcb": x.prem.where(x.blk & (x.cp == 1) & (x.side == 1), 0).groupby(x.h).sum(), "bpb": x.prem.where(x.blk & (x.cp != 1) & (x.side == 1), 0).groupby(x.h).sum(),
                              "scb": x.prem.where(x.short & (x.cp == 1) & (x.side == 1), 0).groupby(x.h).sum(), "spb": x.prem.where(x.short & (x.cp != 1) & (x.side == 1), 0).groupby(x.h).sum(),
                              "iv": g.iv.median(), "nc": (x.cp == 1).groupby(x.h).sum(), "np": (x.cp != 1).groupby(x.h).sum()})
            agg.append(a)
        A = pd.concat(agg).groupby(level=0).sum(min_count=1); A.index = A.index + pd.Timedelta("1h")        # 그 시간이 끝난 뒤 사용
        A = A.reindex(HF[s].t).fillna({c_: 0 for c_ in ("cb", "cs", "pb", "ps", "bcb", "bpb", "scb", "spb", "nc", "np")})
        A.index = HF[s].index
        net = (A.cb - A.cs) - (A.pb - A.ps); n24 = net.rolling(24).sum()
        z = lambda v: (v - v.rolling(720, min_periods=240).mean()) / v.rolling(720, min_periods=240).std()
        OF[s] = pd.DataFrame({"nz": z(n24), "nz4": z(net.rolling(4).sum()), "bc": A.bcb.rolling(1).sum(), "bp": A.bpb.rolling(1).sum(),
                              "ivz": z(A.iv.ffill()), "pcr": z((A.np.rolling(24).sum() + 1) / (A.nc.rolling(24).sum() + 1)),
                              "sc": z(A.scb.rolling(4).sum()), "sp": z(A.spb.rolling(4).sum()),
                              "bcq": A.bcb.rolling(1).sum().rolling(2160, min_periods=500).quantile(0.99), "bpq": A.bpb.rolling(1).sum().rolling(2160, min_periods=500).quantile(0.99)})
    def oadd(nm, desc, cond, side, hold, tgst=None):
        def f(s, cond=cond, side=side, hold=hold):
            O = OF[s]; H = HF[s]; m = cond(O, H).fillna(False); m = m & ~m.shift(1).fillna(False).astype(bool); T = H[m]
            if tgst: return entries_at(D, s, T.t, side, (T.sig24 * tgst * 100).values, (T.sig24 * tgst * 100).values, hold)
            return entries_at(D, s, T.t, side, BIG, BIG, hold)
        add(f"{nm} {desc}", f)
    oadd("O01", "옵션 순매수(콜−풋) 24h z>2 → 롱 24h", lambda O, H: O.nz > 2, 1, 24)
    oadd("O02", "옵션 순매수 z<−2 → 숏 24h", lambda O, H: O.nz < -2, -1, 24)
    oadd("O03", "옵션 순매수 4h z>2 → 롱 4h", lambda O, H: O.nz4 > 2, 1, 4)
    oadd("O04", "옵션 순매수 4h z<−2 → 숏 4h", lambda O, H: O.nz4 < -2, -1, 4)
    oadd("O05", "대량 콜 매수(상위 1%) → 롱 24h", lambda O, H: O.bc > O.bcq, 1, 24)
    oadd("O06", "대량 풋 매수(상위 1%) → 숏 24h", lambda O, H: O.bp > O.bpq, -1, 24)
    oadd("O07", "IV 급등(z>2) + 24h 하락 → 롱(공포 정점) ±1σ", lambda O, H: (O.ivz > 2) & (H.r24 < 0), 1, 48, 1.0)
    oadd("O08", "IV 급락(z<−2) + 24h 상승 → 숏(안도 과열) ±1σ", lambda O, H: (O.ivz < -2) & (H.r24 > 0), -1, 48, 1.0)
    oadd("O09", "풋/콜 건수비 z>2(공포) → 롱 24h", lambda O, H: O.pcr > 2, 1, 24)
    oadd("O10", "풋/콜 건수비 z<−2(탐욕) → 숏 24h", lambda O, H: O.pcr < -2, -1, 24)
    oadd("O11", "단기 만기 콜 매수 급증 z>2 → 롱 4h", lambda O, H: O.sc > 2, 1, 4)
    oadd("O12", "단기 만기 풋 매수 급증 z>2 → 숏 4h", lambda O, H: O.sp > 2, -1, 4)
    oadd("O13", "옵션 순매수 z>2 + 가격 24h 하락(역행 정보) → 롱 48h", lambda O, H: (O.nz > 2) & (H.r24 < 0), 1, 48)
    # ---------- L 코인베이스 선행 ----------
    for th, hold, sd_, nm in ((3, 5, 1, "L01"), (3, 15, 1, "L02"), (5, 5, 1, "L03"), (5, 15, 1, "L04"),
                              (3, 5, -1, "L05"), (3, 15, -1, "L06"), (5, 5, -1, "L07"), (5, 15, -1, "L08")):
        def f(s, th=th, hold=hold, sd_=sd_):
            k = D.K[s]; cb = D.cb[s].reindex(k.index)
            d = np.log(cb).diff() - np.log(k.close).diff(); sg = d.rolling(1440, min_periods=300).std()
            m = (sd_ * d > th * sg); m = m & ~m.shift(1).fillna(False).astype(bool)
            i = np.where(m.values)[0] + 1; i = i[i < len(k) - 2]
            return pd.DataFrame({"i": i, "side": sd_, "tg": BIG, "st": BIG, "hold": hold})
        add(f"{nm} 코인베이스가 바이낸스보다 {'위' if sd_ > 0 else '아래'}로 {th}σ 앞섬 → 바이낸스 {'롱' if sd_ > 0 else '숏'} {hold}분", f)
    # ---------- S 스테이블코인 ----------
    sc = pd.read_parquet(EXT / "stablecoins.parquet").stable_total; sc.index = pd.DatetimeIndex(sc.index)
    sc.index = (sc.index.tz_localize("UTC") if sc.index.tz is None else sc.index) + pd.Timedelta(days=1)
    for look, hold, q, sd_, nm in ((7, 168, 0.8, 1, "S01"), (7, 168, 0.2, -1, "S02"), (3, 72, 0.8, 1, "S03"), (3, 72, 0.2, -1, "S04")):
        def f(s, look=look, hold=hold, q=q, sd_=sd_):
            g = np.log(sc).diff(look); qq = g.rolling(365, min_periods=120).rank(pct=True)
            m = (qq >= q) if sd_ > 0 else (qq <= q); m = m & ~m.shift(1).fillna(False).astype(bool)
            return entries_at(D, s, qq.index[m.values], sd_, BIG, BIG, hold)
        add(f"{nm} 스테이블코인 {look}일 증가 {'상위' if sd_ > 0 else '하위'} 20% → {'롱' if sd_ > 0 else '숏'} {hold}h", f)
    # ---------- M 거시 발표 ----------
    txt = (R / "research" / "r59_macro_events.py").read_text()
    blocks = re.findall(r'"""([0-9\-\s]+)"""', txt)
    ev = {"FOMC": [pd.Timestamp(f"{d} 14:00", tz="America/New_York").tz_convert("UTC") for d in blocks[0].split()],
          "CPI": [pd.Timestamp(f"{d} 08:30", tz="America/New_York").tz_convert("UTC") for d in blocks[1].split()]}
    for name in ("FOMC", "CPI"):
        for x_, nm in ((0.003, "a"), (0.006, "b")):
            def f(s, name=name, x_=x_):
                k = D.K[s]; st_ = k.index.searchsorted(pd.DatetimeIndex(ev[name])); ok = st_ < len(k) - 2; st_ = st_[ok]
                ent, sd = straddle(st_.astype(np.int64), k.close.values[st_ - 1], np.full(len(st_), x_), np.full(len(st_), 120, np.int64), k.high.values, k.low.values)
                g = ent >= 0
                return pd.DataFrame({"i": ent[g], "side": sd[g], "tg": np.full(g.sum(), x_ * 300), "st": np.full(g.sum(), x_ * 100), "hold": np.full(g.sum(), 360)})
            add(f"M{'1' if name == 'FOMC' else '2'}{nm} {name} 발표 양방향 스톱 ±{x_*100:.1f}%, 목표 3배", f)
    # ---------- H 추세·변동성 국면 ----------
    for nm, desc, cond, side in (("H01", "200일선 위 + 저변동 → 롱", lambda H: (H.close > H.sma200d) & (H.q < 1), 1),
                                 ("H02", "200일선 아래 + 고변동 → 숏", lambda H: (H.close < H.sma200d) & (H.q > 1), -1),
                                 ("H03", "200일선 위 + 30일↑ → 롱", lambda H: (H.close > H.sma200d) & (np.log(H.close).diff(720) > 0), 1),
                                 ("H04", "200일선 아래 + 30일↓ → 숏", lambda H: (H.close < H.sma200d) & (np.log(H.close).diff(720) < 0), -1),
                                 ("H05", "7·30·90일 추세 모두 상승 → 롱", lambda H: (np.log(H.close).diff(168) > 0) & (np.log(H.close).diff(720) > 0) & (np.log(H.close).diff(2160) > 0), 1),
                                 ("H06", "7·30·90일 추세 모두 하락 → 숏", lambda H: (np.log(H.close).diff(168) < 0) & (np.log(H.close).diff(720) < 0) & (np.log(H.close).diff(2160) < 0), -1)):
        def f(s, cond=cond, side=side):
            H = HF[s]; m = cond(H).fillna(False) & (H.index.hour == 0); T = H[m]
            return entries_at(D, s, T.t, side, BIG, BIG, 24)
        add(f"{nm} {desc} (매일 00시, 24h)", f)
    # ---------- Q 분포 모양 ----------
    for nm, desc, cond, side, hold in (("Q01", "7일 왜도 < −1 → 롱 24h", lambda H: H.skew < -1, 1, 24), ("Q02", "7일 왜도 > 1 → 숏 24h", lambda H: H.skew > 1, -1, 24),
                                       ("Q03", "첨도 > 8 + 24h 하락 → 롱 24h", lambda H: (H.kurt > 8) & (H.r24 < 0), 1, 24),
                                       ("Q04", "첨도 > 8 + 24h 상승 → 숏 24h", lambda H: (H.kurt > 8) & (H.r24 > 0), -1, 24),
                                       ("Q05", "왜도 < −1 + 24h 저점권 → 롱 72h", lambda H: (H.skew < -1) & (H.close < H.lo24 * 1.01), 1, 72),
                                       ("Q06", "왜도 > 1 + 24h 고점권 → 숏 72h", lambda H: (H.skew > 1) & (H.close > H.hi24 * 0.99), -1, 72)):
        def f(s, cond=cond, side=side, hold=hold):
            H = HF[s]; m = cond(H).fillna(False); m = m & ~m.shift(1).fillna(False).astype(bool); T = H[m]
            return entries_at(D, s, T.t, side, BIG, BIG, hold)
        add(f"{nm} {desc}", f)
    # ---------- R 분산비 국면 ----------
    for nm, th, mode in (("R01", 1.2, "f"), ("R02", 1.3, "f"), ("R03", 1.5, "f"), ("R04", 0.8, "m"), ("R05", 0.7, "m"), ("R06", 0.6, "m")):
        def f(s, th=th, mode=mode):
            H = HF[s]; m = ((H.VR > th) if mode == "f" else (H.VR < th)) & (H.index.hour == 0) & (H.r24.abs() > 0.5 * H.sig24); T = H[m]
            sd = np.sign(T.r24).astype(int) * (1 if mode == "f" else -1)
            return entries_at(D, s, T.t, sd.values, BIG, BIG, 24)
        add(f"{nm} 분산비 {'>' if mode == 'f' else '<'} {th} → 24h 움직임 {'추종' if mode == 'f' else '역행'} 24h", f)
    return S


if __name__ == "__main__":
    t_ = time.time(); D = Data(); S = strategies(D)
    P(f"전략 {len(S)}개 준비 ({time.time()-t_:.0f}s)")
    rows = []; daily = {}
    IDX = pd.date_range("2020-01-01", "2026-09-25", freq="D", tz="UTC")
    for name, fn in S.items():
        try:
            T = pd.concat([run(D, s, fn(s)) for s in SYMS])
        except Exception as e:
            P(f"  {name}: 오류 {type(e).__name__} {e}"); continue
        if len(T) == 0: rows.append((name, 0) + (np.nan,) * 6); continue
        T["t"] = pd.to_datetime(T.t, utc=True)
        a = T[T.t < "2024-01-01"].r; b = T[T.t >= "2024-01-01"].r
        tt = T.r.mean() / (T.r.std() / np.sqrt(len(T))) if len(T) > 1 else np.nan
        rows.append((name, len(T), len(T) / 6.7, T.r.mean() * 1e4, tt, a.mean() * 1e4 if len(a) else np.nan, b.mean() * 1e4 if len(b) else np.nan, (T.r > 0).mean()))
        daily[name] = (T.groupby(T.t.dt.floor("D")).r.sum() * 0.5).reindex(IDX).fillna(0)
    O = pd.DataFrame(rows, columns=["전략", "n", "연", "거래당bp", "t", "20~23", "24~26", "승률"])
    O["통과"] = (O.n >= 40) & (O.t >= 1.5) & (O["20~23"] > 0) & (O["24~26"] > 0)
    O.to_csv(R / "logs" / "r87.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200); pd.set_option("display.max_colwidth", 70)
    P(f"\n===== 전략 {len(O)}개, 통과 {O.통과.sum()}개 ({time.time()-t_:.0f}s)")
    P(O.round(2).to_string(index=False))
    # 결합
    import r40_regime_leverage as RL, r80_max_return as R80
    RL.MMR = 0.0065; W = R80.dip_frame(0.0025)
    mult = np.minimum(np.clip(W.vreg.values ** 1.5, 0.3, 4.0) * np.clip(W.pred.values / 0.001, 0.5, 3.0), 6.0)
    dip = R80.daily_from_eq(RL.sim(W, 1, 4, mult, X=0.03, H=720)[0]).reindex(IDX).fillna(0)
    pas = O[O.통과].전략.tolist()
    P(f"\n===== 결합: 통과 {len(pas)}개 + z0.31")
    if pas:
        M = pd.DataFrame({n: daily[n] for n in pas}); M["z0.31"] = dip
        tr = M["2022-01-01":"2023-12-31"]; te = M["2024-01-01":]
        cr = te.corr()["z0.31"].drop("z0.31")
        P("  2024~26 z0.31 상관: " + " ".join(f"{c[:4]}:{v:+.2f}" for c, v in cr.items()))
        for how in ("rp", "ms"):
            if how == "rp": w = 1 / tr.std().replace(0, np.nan); w = w.fillna(0) / w.fillna(0).sum()
            else:
                mu = tr.mean().values; C = tr.cov().values; C = 0.5 * C + 0.5 * np.diag(np.diag(C))
                w = np.clip(np.linalg.pinv(C) @ mu, 0, None); w = pd.Series(w / w.sum() if w.sum() > 0 else np.ones(len(mu)) / len(mu), index=tr.columns)
            r = (te * w).sum(axis=1); sh = r.mean() / r.std() * np.sqrt(365)
            P(f"  [{'위험 균등' if how == 'rp' else '최대 샤프'}] 2024~26 결합 샤프 {sh:.2f} (z0.31 단독 {te['z0.31'].mean()/te['z0.31'].std()*np.sqrt(365):.2f}), z0.31 비중 {w['z0.31']:.2f}")
    (R / "logs" / "r87.summary").write_text(f"R87 새 관점 {len(O)}개 전략, 통과 {O.통과.sum()}개\n")
