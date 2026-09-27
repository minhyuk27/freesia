"""z0.31 실시간 모의 운용(페이퍼) — 실제 시장 데이터로 매 분 결정, 실제 체결 틱으로 체결 판정, 주문은 내지 않는다.

설정(R81·R83 확정, 공격 단계): 예측 문턱 25bp, V3 배율 = min(clip(국면^1.5, 0.3, 4) × clip(예측/10bp, 0.5, 3), 6),
  거래당 명목 = 자본 × w × 배율 (w = 2.5), 총명목 ≤ 자본 × N (N = 10), 킬스위치: 최근 120분 청산 손익 합 < −3% → 720분 신규 중지.
주문 규칙(z0.31): 봉 B ∈ {5,10,15,30,60}분 × 깊이 k ∈ {3,4,5,6,8}, 지정가 T = 봉 시가 × (1 − k·rv·√B), rv = 봉 시가 직전 300분 1분 수익 표준편차.
  매 분 시작에 직전 분까지의 특징으로 예측 > 문턱이면 그 분 동안 주문 유지. 봉 안 첫 터치에서 주문이 꺼져 있었으면 그 봉은 끝.
  체결 = 주문을 켠 시각 이후 실제 체결 틱 중 T × (1 − 2bp) 이하가 있을 때(1분봉 저가가 아니라 틱으로 확인 — 판단 지연 반영).
  청산 = 봉 종가(모의: 봉 마지막 분 종가), 비용 = 메이커 2bp + 테이커 5bp + 청산 지연 1.2bp.
타이밍: 매 분 6초에 실행(마감 봉 집계 대기). 로그: logs/live/{decisions,fills,trades}.csv, 상태 logs/live/state.json, 심장박동 logs/live/heartbeat
사용: python -u scripts/live_paper.py   (cron에서 flock으로 상시 유지)"""
import sys, json, time, math, traceback
from pathlib import Path
import numpy as np, pandas as pd, joblib
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R))
from freesia import rest
from freesia.live import features as LF, feed

SYMS = ("BTCUSDT", "ETHUSDT")
TH, W_, N_, KX, KH, KL = 0.0025, 2.5, 10.0, 0.03, 720, 120
CM, CT, DELAY = 2e-4, 5e-4, 1.2e-4
WINDOW = 10080 + 300 + 200
LOG = R / "logs" / "live"; LOG.mkdir(parents=True, exist_ok=True)
MODEL = joblib.load(R / "data" / "models" / "z031_live_2026.joblib")


def log(name, row):
    p = LOG / f"{name}.csv"; new = not p.exists()
    pd.DataFrame([row]).to_csv(p, mode="a", header=new, index=False)


class Engine:
    def __init__(self):
        st = LOG / "state.json"
        self.S = json.loads(st.read_text()) if st.exists() else {"cap": 1.0, "positions": [], "closed": [], "halt_until": 0, "bars": {}}
        self.K, self.PR, self.MT, self.FU, self.VREG = {}, {}, {}, {}, {}
        self.last_daily = None

    def save(self):
        (LOG / "state.json").write_text(json.dumps(self.S, default=str))

    # ---------- 데이터 ----------
    def refresh(self, M):
        end_ms = int((LF.EPOCH + pd.Timedelta(minutes=M)).timestamp() * 1000)
        for s in SYMS:
            if s not in self.K or len(self.K[s]) < WINDOW - 5:
                self.K[s] = feed.klines_1m(s, WINDOW, end_ms); self.PR[s] = feed.premium_1m(s, WINDOW, end_ms)
            else:
                k = feed.klines_1m(s, 15, end_ms); self.K[s] = pd.concat([self.K[s], k]); self.K[s] = self.K[s][~self.K[s].index.duplicated(keep="last")].iloc[-WINDOW:]
                p = feed.premium_1m(s, 15, end_ms); self.PR[s] = pd.concat([self.PR[s], p]); self.PR[s] = self.PR[s][~self.PR[s].index.duplicated(keep="last")].iloc[-WINDOW:]
            self.MT[s] = feed.metrics_5m(s, 500) if (s not in self.MT or M % 5 == 1) else self.MT[s]
            if s not in self.FU or M % 30 == 0: self.FU[s] = feed.funding(s, 3)
        day = (LF.EPOCH + pd.Timedelta(minutes=M)).normalize()
        if self.last_daily != day:
            for s in SYMS:
                d = feed.daily_close(s, 500); r = np.log(d).diff(); v30 = r.rolling(30).std()
                reg = (v30 / v30.rolling(365, min_periods=120).median()).shift(1)
                self.VREG[s] = float(reg.reindex([day]).fillna(1.0).iloc[0]) if day in reg.index else float(reg.iloc[-1] if pd.notna(reg.iloc[-1]) else 1.0)
            self.last_daily = day

    def contiguous(self, s, M):
        k = self.K[s]; last = LF.minute_index(k.index[-1:])[0]
        return last == M - 1 and len(k) >= WINDOW - 5 and (k.index[-1] - k.index[0]) / pd.Timedelta("1min") == len(k) - 1

    # ---------- 체결·청산 ----------
    def ticks_below(self, s, t_from_ms, t_to_ms, price):
        """[t_from, t_to] 실제 체결 틱 중 price 이하가 있었나. 1,000건씩 fromId로 넘기며 끝까지(최대 60쪽)."""
        js = rest._get("/fapi/v1/aggTrades", {"symbol": s, "startTime": t_from_ms, "endTime": t_to_ms, "limit": 1000})
        mn, n = math.inf, 0
        for _ in range(60):
            if not js: break
            n += len(js); mn = min(mn, min(float(x["p"]) for x in js))
            if mn <= price or len(js) < 1000 or js[-1]["T"] >= t_to_ms: break
            js = [x for x in rest._get("/fapi/v1/aggTrades", {"symbol": s, "fromId": js[-1]["a"] + 1, "limit": 1000}) if x["T"] <= t_to_ms]
        return mn <= price, (None if mn == math.inf else mn), n

    def settle(self, M):
        """방금 끝난 분 M−1: 터치·체결 판정, 봉 끝 청산."""
        m1 = M - 1; now_ms = int(time.time() * 1000)
        for key, b in list(self.S["bars"].items()):
            if b["status"] not in ("pending",) or not (b["s0"] <= m1 < b["s0"] + b["B"]): continue
            s = b["sym"]; row = self.K[s].iloc[-1]
            if LF.minute_index(self.K[s].index[-1:])[0] != m1: continue
            thr = b["T"] * (1 - LF.EPS)
            if row.low <= thr:                                             # 1분봉상 첫 터치
                live = b.get("live_min") == m1
                if not live:
                    b["status"] = "dead"; log("fills", {"ts": str(self.K[s].index[-1]), "sym": s, "B": b["B"], "k": b["k"], "T": b["T"], "result": "터치(주문 꺼짐)"})
                    continue
                t0 = b["live_since_ms"]; t1 = int((LF.EPOCH + pd.Timedelta(minutes=M)).timestamp() * 1000) - 1
                hit, mn, nt = self.ticks_below(s, t0, t1, thr)
                if hit:
                    notional = b["notional"]
                    used = sum(p["notional"] for p in self.S["positions"])
                    notional = min(notional, max(self.S["cap"] * N_ - used, 0.0))
                    b["status"] = "filled"
                    if notional > 0:
                        self.S["positions"].append({"sym": s, "B": b["B"], "k": b["k"], "T": b["T"], "notional": notional, "exit_min": b["s0"] + b["B"] - 1,
                                                    "ts": str(self.K[s].index[-1]), "pred": b["pred"]})
                    log("fills", {"ts": str(self.K[s].index[-1]), "sym": s, "B": b["B"], "k": b["k"], "T": b["T"], "result": "체결", "tick_min": mn,
                                  "ticks": nt, "notional": notional, "pred": b["pred"], "delay_s": (t0 - (t1 + 1 - 60000)) / 1000})
                else:
                    b["status"] = "dead"
                    log("fills", {"ts": str(self.K[s].index[-1]), "sym": s, "B": b["B"], "k": b["k"], "T": b["T"], "result": "주문 후 미도달(지연으로 놓침)", "tick_min": mn})
        keep = []
        for p in self.S["positions"]:
            if p["exit_min"] == m1:
                px = float(self.K[p["sym"]].iloc[-1].close); r = px / p["T"] - 1 - CM - CT - DELAY
                pnl = p["notional"] * r; self.S["cap"] += pnl
                self.S["closed"].append([m1, pnl / max(self.S["cap"] - pnl, 1e-9)])
                log("trades", {"exit_ts": str(self.K[p["sym"]].index[-1]), **{k_: p[k_] for k_ in ("sym", "B", "k", "T", "notional", "ts", "pred")},
                               "exit_px": px, "ret_bp": r * 1e4, "pnl": pnl, "cap": self.S["cap"]})
            else: keep.append(p)
        self.S["positions"] = keep
        self.S["closed"] = [c for c in self.S["closed"] if c[0] > m1 - KL]
        if sum(c[1] for c in self.S["closed"]) < -KX and self.S["halt_until"] <= M:
            self.S["halt_until"] = M + KH; log("fills", {"ts": str(LF.EPOCH + pd.Timedelta(minutes=M)), "result": f"킬스위치 발동 → {KH}분 중지"})
        # 끝난 봉 정리
        self.S["bars"] = {k_: b for k_, b in self.S["bars"].items() if b["s0"] + b["B"] > M}

    # ---------- 결정 ----------
    def decide(self, M):
        now_ms = int(time.time() * 1000); halted = self.S["halt_until"] > M
        for s in SYMS:
            if not self.contiguous(s, M): log("fills", {"ts": str(LF.EPOCH + pd.Timedelta(minutes=M)), "sym": s, "result": "데이터 불연속 — 이번 분 건너뜀"}); continue
            k = self.K[s]; t0 = k.index[0]
            F, rv1, piv = LF.minute_features(k, self.PR[s], self.MT[s][self.MT[s].index >= t0 - pd.Timedelta("2D")],
                                             self.FU[s][self.FU[s].index >= t0 - pd.Timedelta("3D")])
            lr = np.log(k.close).diff(); rvn = lr.rolling(300).std().values
            j = len(k) - 1                                                  # 분 M−1
            base = {kk: v[j] for kk, v in F.items()}
            mi = LF.minute_index(k.index)
            for B in LF.BARS:
                s0 = (M // B) * B
                if s0 == M:
                    o_s0 = feed.current_open(s)[1]; rv_s0 = rvn[j]
                else:
                    js0 = int(np.searchsorted(mi, s0)); o_s0 = float(k.open.values[js0]); rv_s0 = float(rv1[js0])
                for kx in LF.KS:
                    key = f"{s}|{B}|{kx}|{s0}"
                    b = self.S["bars"].get(key)
                    if b is None:
                        T, _ = LF.level_features(M, B, kx, o_s0, rv_s0, k.close.values[j], piv[j], s == "BTCUSDT")
                        b = {"sym": s, "B": B, "k": kx, "s0": s0, "T": float(T), "status": "pending"}; self.S["bars"][key] = b
                    if b["status"] != "pending" or not np.isfinite(b["T"]): continue
                    T, lv = LF.level_features(M, B, kx, o_s0, rv_s0, k.close.values[j], piv[j], s == "BTCUSDT")
                    x = np.array([[{**base, **lv}[c] for c in MODEL["feats"]]], dtype=np.float64)
                    pred = float(MODEL["model"].predict(x)[0])
                    live = (pred > TH) and not halted
                    if live:
                        vreg = self.VREG.get(s, 1.0)
                        mult = min(float(np.clip(vreg ** 1.5, 0.3, 4.0) * np.clip(pred / 0.001, 0.5, 3.0)), 6.0)
                        b.update({"live_min": M, "live_since_ms": now_ms, "pred": pred, "notional": self.S["cap"] * W_ * mult})
                        log("decisions", {"ts": str(LF.EPOCH + pd.Timedelta(minutes=M)), "sym": s, "B": B, "k": kx, "T": b["T"],
                                          "px": float(k.close.values[j]), "gap_bp": (b["T"] / k.close.values[j] - 1) * 1e4, "pred_bp": pred * 1e4,
                                          "mult": mult, "vreg": vreg, "decide_s": (time.time() * 1000 - (M * 60000 + LF.EPOCH.timestamp() * 1000)) / 1000})

    def step(self, M):
        self.refresh(M)
        if all(s in self.K for s in SYMS): self.settle(M)
        self.decide(M); self.save()
        (LOG / "heartbeat").write_text(f"{pd.Timestamp.now(tz='UTC')} M={M} cap={self.S['cap']:.4f} pos={len(self.S['positions'])} halt={self.S['halt_until'] > M}\n")


def main():
    E = Engine()
    while True:
        now = time.time(); nxt = (math.floor(now / 60) + 1) * 60 + 6
        time.sleep(max(0, nxt - now))
        M = int((nxt - 6 - LF.EPOCH.timestamp()) // 60)
        try:
            E.step(M)
        except Exception:
            (LOG / "errors.log").open("a").write(f"{pd.Timestamp.now(tz='UTC')} M={M}\n{traceback.format_exc()}\n")


if __name__ == "__main__":
    main()
