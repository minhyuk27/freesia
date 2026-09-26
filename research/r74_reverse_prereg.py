"""R74 — R73에서 사후 발견한 '최근 180봉 학습 신호의 반전'(24시간 목표)을 오염 없이 검증.

재료: R73 24시간 목표 거래 기록(data/r73_trades_fW_{real,plc}.parquet) — 56개 구성(지표 묶음 14 × 모델 2 × 문턱 2).
반대 거래 순수익 = −방향 × 24시간 수익 − 비용(왕복 14bp).
A 선택 없는 검증: 56개 구성을 전부 반대로 → 연도별 평균·흑자 구성 비율. 반전이 일반 현상이면 매년 흑자여야 한다.
B 워크포워드 선택: 평가 연도 Y(2022~2026)마다 Y 이전 연도들의 반대 성과로 최선 1개·상위 3개 구성을 고르고 Y에 거래.
C 사후 선택 구성(S1+S6 릿지 10%)은 참고로만 표시(발견에 쓴 데이터라 증거 아님).
위약: 미래수익을 시간 이동한 R73 위약 거래에 같은 절차.
비용 민감도: 왕복 14bp(테이커) / 20bp / 8bp(메이커 가정).
계좌(B 최선 1개): 레버리지 1·3·5, 24시간 보유 겹침 → 포지션 = 자본 × 레버리지 / (6 × 2종목).
사용: python research/r74_reverse_prereg.py"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
R = Path(__file__).resolve().parents[1]
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
C0 = 0.0014


def load(kind):
    d = pd.read_parquet(R / "data" / f"r73_trades_fW_{kind}.parquet")
    d["t"] = pd.to_datetime(d.t, utc=True); d["yr"] = d.t.dt.year
    d["cfg"] = d.set + "|" + d.model + "|" + (d.q * 100).astype(int).astype(str)
    d["g"] = -d.side * d.fwd                                      # 반대 거래 총수익
    return d


def account(T, lev, cost):
    per = (T.g - cost).groupby(T.t).sum() * lev / 12
    eq = (1 + per.sort_index().clip(lower=-0.999)).cumprod()
    mon = eq.resample("ME").last().pct_change().dropna(); yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 0.1)
    return (eq.iloc[-1] ** (1 / 12 / yrs) - 1) * 100, (eq / eq.cummax() - 1).min() * 100, (mon > 0).mean()


if __name__ == "__main__":
    for kind in ("real", "plc"):
        D = load(kind); lab = "실제" if kind == "real" else "위약"
        P(f"\n===== [{lab}] 반대 거래, 구성 {D.cfg.nunique()}개")
        # A
        by = D.groupby(["cfg", "yr"]).g.mean().unstack() - C0
        P("  A 선택 없음 — 56개 구성 평균(비용 14bp 후, bp)과 흑자 구성 비율, 연도별:")
        P("     " + " | ".join(f"{y}: {by[y].mean()*1e4:+6.1f} ({(by[y] > 0).mean():.0%})" for y in by.columns))
        allm = (D.g - C0).mean() * 1e4
        P(f"     전 기간 전 거래 평균 {allm:+.1f}bp, 전 기간 흑자 구성 {((D.groupby('cfg').g.mean() - C0) > 0).mean():.0%}")
        # B
        for top in (1, 3):
            parts = []; picks = []
            for Y in range(2022, 2027):
                past = D[D.yr < Y].groupby("cfg").agg(m=("g", "mean"), n=("g", "size"))
                past = past[past.n >= 300].sort_values("m", ascending=False).head(top)
                picks.append(f"{Y}:{','.join(past.index)}")
                parts.append(D[(D.yr == Y) & D.cfg.isin(past.index)])
            T = pd.concat(parts)
            yr = {y: (G.g - C0).mean() * 1e4 for y, G in T.groupby("yr")}
            costs = " / ".join(f"비용 {int(c*1e4)}bp {(T.g - c).mean()*1e4:+.1f}" for c in (0.0008, 0.0014, 0.0020))
            P(f"  B 워크포워드 상위 {top}: 거래당 {(T.g - C0).mean()*1e4:+.1f}bp, 하루 {len(T)/((T.t.max()-T.t.min()).days):.2f}건, 적중 {(T.g > 0).mean():.3f} | "
              + " ".join(f"{y%100}:{v:+.1f}" for y, v in yr.items()) + f" | {costs}")
            if top == 1:
                P("     선택: " + " ".join(picks))
                if kind == "real":
                    for lev, nm in ((1, "무레버"), (3, "중간 3배"), (5, "공격 5배")):
                        cm, dd, wp = account(T, lev, C0); P(f"     {nm}: 월 복리 {cm:+.2f}%, MDD {dd:.0f}%, 흑자 달 {wp:.0%}")
        # C
        c = D[D.cfg == "S1+S6|RG|10"]
        P(f"  C 사후 선택 S1+S6 릿지 10% (참고, 증거 아님): {(c.g - C0).mean()*1e4:+.1f}bp | " +
          " ".join(f"{y%100}:{(G.g - C0).mean()*1e4:+.0f}" for y, G in c.groupby('yr')))
    (R / "logs" / "r74.summary").write_text("R74 반전 사전등록 검증 완료. 세부 logs/r74.log\n")
