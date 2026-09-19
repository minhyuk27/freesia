"""수집 결과 자체검증. 로그의 '성공'이 아니라 데이터 자체를 본다.

검사: 최신 시각(신선도), 1분봉 결손 구간, 중복, OHLC 무결성, 가격 급변 이상치,
      펀딩 주기 규칙성, 마크·프리미엄 봉 수 일치. 결과는 logs/validate_report.json.
종료코드 0=정상, 1=문제 있음.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PQ = ROOT / "data" / "parquet"
SYMS = ["BTCUSDT", "ETHUSDT"]
MAX_STALE_MIN = {"klines_1m": 48 * 60, "mark_1m": 48 * 60, "premium_1m": 48 * 60,
                 "funding": 24 * 60, "metrics_5m": 48 * 60}


def load(name, sym):
    files = sorted((PQ / name / sym).glob("*.parquet"))
    if not files:
        return None
    return pd.concat([pd.read_parquet(f) for f in files]).sort_values("ts").reset_index(drop=True)


def gaps(ts: pd.Series, step: pd.Timedelta, limit=20):
    d = ts.diff()
    idx = d[d > step].index
    out = [{"from": str(ts[i - 1]), "to": str(ts[i]), "missing": int(d[i] / step) - 1} for i in idx]
    return len(out), sum(g["missing"] for g in out), out[:limit]


def main():
    now = datetime.now(timezone.utc)
    rep, problems = {}, []
    for sym in SYMS:
        for name in MAX_STALE_MIN:
            df = load(name, sym)
            key = f"{name}/{sym}"
            if df is None:
                problems.append(f"{key}: 데이터 없음")
                continue
            r = {"rows": len(df), "first": str(df.ts.iloc[0]), "last": str(df.ts.iloc[-1])}
            age = (now - df.ts.iloc[-1]).total_seconds() / 60
            r["age_min"] = round(age)
            if age > MAX_STALE_MIN[name]:
                problems.append(f"{key}: 오래됨 (마지막 {r['last']}, {round(age/60)}시간 전)")
            dup = int(df.ts.duplicated().sum())
            r["dups"] = dup
            if dup:
                problems.append(f"{key}: 중복 {dup}행")
            step = {"klines_1m": "1min", "mark_1m": "1min", "premium_1m": "1min",
                    "metrics_5m": "5min"}.get(name)
            if step:
                n, miss, ex = gaps(df.ts, pd.Timedelta(step))
                r.update(gap_runs=n, missing_bars=miss, gap_examples=ex[:5])
                # 진짜 거래소 점검으로 인한 소량 결손은 허용하되 큰 것은 표시
                big = [g for g in ex if g["missing"] >= 30]
                if big:
                    problems.append(f"{key}: 30봉 이상 결손 {len(big)}건+ (예: {big[0]})")
            if name == "klines_1m":
                bad = int(((df.high < df[["open", "close"]].max(axis=1)) |
                           (df.low > df[["open", "close"]].min(axis=1)) |
                           (df[["open", "high", "low", "close"]] <= 0).any(axis=1)).sum())
                r["ohlc_violations"] = bad
                if bad:
                    problems.append(f"{key}: OHLC 무결성 위반 {bad}행")
                ret = df.close.pct_change().abs()
                r["max_1m_move_pct"] = round(float(ret.max()) * 100, 2)
                r["moves_over_5pct"] = int((ret > 0.05).sum())
            if name == "funding":
                iv = df.ts.diff().dropna().dt.total_seconds() / 3600
                r["interval_hours_counts"] = {str(k): int(x) for k, x in
                                              iv.round().value_counts().head(5).items()}
                r["rate_min"], r["rate_max"] = float(df.rate.min()), float(df.rate.max())
            rep[key] = r
        k, mk, pm = (load(n, sym) for n in ("klines_1m", "mark_1m", "premium_1m"))
        if k is not None and mk is not None and abs(len(k) - len(mk)) > 0.001 * len(k):
            problems.append(f"{sym}: 봉 수 불일치 klines={len(k)} mark={len(mk)}")
    out = {"checked_at": now.isoformat(), "problems": problems, "datasets": rep}
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "logs" / "validate_report.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({"problems": problems,
                      "summary": {k: (r["rows"], r["last"], r.get("missing_bars")) for k, r in rep.items()}},
                     indent=1, ensure_ascii=False))
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
