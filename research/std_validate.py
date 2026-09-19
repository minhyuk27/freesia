"""freesia 전략 검증 기준 v1 적용 도구.
- 연도별 테스트 성과(순수익·Sharpe·MDD)와 '유의미' 판정, train/test(앵커드 워크포워드 + 연도 하나씩 빼기) 비교
- 일·주·월·분기·연 단위 승률(노출 기간 기준/전체 기간 기준)과 거래(포지션 구간) 승률·손익비
- 기준 판정표. BTC·ETH만. 결과는 logs/std_report.md 에 마크다운으로 저장."""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from freesia.bt import core  # noqa: E402
import screen as S  # noqa: E402
import screen2 as S2  # noqa: E402
import screen6 as S6  # noqa: E402
import combo as CB  # noqa: E402

MIN_RET, MIN_SH = 0.05, 0.5                       # 연 유의미 기준: 순수익 ≥ 5%, Sharpe ≥ 0.5 (부분 연도는 기간 비례)
TARGET = {"일": 0.50, "주": 0.55, "월": 0.65, "분기": 0.75, "연": 1.00, "거래": 0.50}
RULE = {"일": "1D", "주": "W-SUN", "월": "MS", "분기": "QS", "연": "YS"}


def compound(net, idx, rule):
    s = pd.Series(net, index=idx)
    return (1 + s).groupby(pd.Grouper(freq=rule)).prod() - 1


def win(r):
    a = r[np.abs(r) > 1e-9]
    return (float((a > 0).mean()) if len(a) else np.nan, float((r > 0).mean()), int(len(a)), int(len(r)))


def episodes(pos, net):
    """포지션이 자본의 25% 이상 바뀔 때마다 구간을 끊고, 노출이 있는 구간의 복리 수익을 거래 1건으로 본다."""
    pos = np.nan_to_num(np.asarray(pos, float)); p_hold = np.concatenate([[0.0], pos[:-1]])
    cuts = np.where(np.abs(np.diff(p_hold, prepend=0.0)) >= 0.25)[0]
    bounds = np.concatenate([[0], cuts, [len(net)]])
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b <= a or np.abs(p_hold[a:b]).mean() < 0.05: continue
        out.append(float(np.prod(1 + net[a:b]) - 1))
    return np.array(out)


def yearly(net, idx, ppy):
    rows = []
    for Y in sorted(set(idx.year)):
        m = idx.year == Y
        a = net[m]
        if len(a) < 20: continue
        frac = min(1.0, m.sum() / ppy)
        ret = float(np.prod(1 + a) - 1)
        sh = float(a.mean() / a.std() * np.sqrt(ppy)) if a.std() > 0 else 0.0
        eq = np.cumprod(1 + a); mdd = float((eq / np.maximum.accumulate(eq) - 1).min())
        ok = ret >= MIN_RET * frac and sh >= MIN_SH
        rows.append((int(Y), ret, sh, mdd, ok, frac))
    return rows


def scorecard(name, net, idx, ppy, pos=None, note=""):
    yr = yearly(net, idx, ppy)
    lines = [f"### {name}{(' — ' + note) if note else ''}", "", "| 연도 | 순수익 | Sharpe | 연내 MDD | 유의미(≥5%·Sharpe≥0.5) | 월 승률 |", "|---|---|---|---|---|---|"]
    mo = compound(net, idx, "MS")
    for Y, ret, sh, mdd, ok, frac in yr:
        w = win(mo[mo.index.year == Y])[0]
        lines.append(f"| {Y}{'*' if frac < 0.95 else ''} | {ret*100:+.1f}% | {sh:.2f} | {mdd*100:.0f}% | {'✅' if ok else '❌'} | {w*100:.0f}% |" if not np.isnan(w) else f"| {Y} | {ret*100:+.1f}% | {sh:.2f} | {mdd*100:.0f}% | {'✅' if ok else '❌'} | - |")
    n_ok = sum(r[4] for r in yr); n = len(yr)
    lines += ["", f"연도 통과 **{n_ok}/{n}** (* 부분 연도는 기간 비례 기준)", "", "| 단위 | 승률(노출 기간) | 승률(전체 기간) | 목표 | 판정 | 평균 승/평균 패 | 최악 |", "|---|---|---|---|---|---|---|"]
    res = {"name": name, "years": (n_ok, n)}
    for k in ("일", "주", "월", "분기", "연"):
        r = compound(net, idx, RULE[k]); wa, wall, na, nt = win(r)
        aw, al = r[r > 1e-9], r[r < -1e-9]
        po = (aw.mean() / -al.mean()) if len(al) and len(aw) else np.nan
        lines.append(f"| {k} | {wa*100:.0f}% (n={na}) | {wall*100:.0f}% | {TARGET[k]*100:.0f}% | {'✅' if wa >= TARGET[k] else '❌'} | {po:.2f} | {r.min()*100:.1f}% |")
        res[k] = wa
    if pos is not None:
        ep = episodes(pos, net)
        if len(ep) >= 5:
            w_ = float((ep > 0).mean()); aw, al = ep[ep > 0], ep[ep < 0]
            po = (aw.mean() / -al.mean()) if len(al) and len(aw) else np.nan
            lines.append(f"| 거래(구간) | {w_*100:.0f}% (n={len(ep)}) | - | {TARGET['거래']*100:.0f}% | {'✅' if w_ >= TARGET['거래'] else '❌'} | {po:.2f} | {ep.min()*100:.1f}% |")
            res["거래"] = w_
    lines.append("")
    return "\n".join(lines), res


def fixed_strategies():
    out = []
    HS = {"기준(24~2160h)": CB.HS, "저속(168~2160h)": [168, 336, 720, 1080, 1440, 2160]}
    for nm, hs in HS.items():
        series, poss = {}, {}
        for sym in S.SYMS:
            x = S2.CarryCtx(sym, "4h"); x.r_price = x.C.pct_change().fillna(0).to_numpy()
            sg = np.mean([np.sign((x.C / x.C.shift(x.nb(h)) - 1).fillna(0).to_numpy()) for h in hs], axis=0)
            e = np.clip(sg, 0, 1)
            H = CB.series(x, e)[0]
            series[sym] = pd.Series(H, index=x.idx); poss[sym] = pd.Series(e, index=x.idx)
            out.append((f"결합 H {nm} {sym[:3]} (4시간)", H, x.idx, 6 * 365, e, "고정 규칙(파라미터 학습 없음, 규칙은 사전 설계)"))
        h = pd.concat(series, axis=1).dropna(); pe = pd.concat(poss, axis=1).dropna().mean(axis=1)
        out.append((f"결합 H {nm} BTC+ETH 혼합 (4시간)", h.mean(axis=1).to_numpy(), h.index, 6 * 365, pe.to_numpy(), "고정 규칙"))
    for sym in S.SYMS:
        x = S2.CarryCtx(sym, "4h")
        c = core.run_ret(np.ones(len(x.idx)), x.ret, np.zeros(len(x.idx)), cost=0.0017 / 1.2); c[0] = 0
        out.append((f"perp 캐리(상시) {sym[:3]} (4시간)", c, x.idx, 6 * 365, None, "델타중립, 거래 없음"))
    for sym in S.SYMS:
        x = S.Ctx(sym, "1d"); e = CB.trend_exposure(x, "ALL", None)
        out.append((f"추세 앙상블(현금 대기) {sym[:3]} (일봉)", core.run(e, x.C.to_numpy(), x.fund), x.idx, 365, e, "고정 규칙, 헤지 없이 현금"))
    return out


def param_strategy(group, tf, sym):
    x = S.Ctx(sym, tf); x.other = x.C; x.otherName = "X"
    N, P = S6.GROUPS[group](x)
    R = np.array([x.run(np.asarray(p, float)) for p in P]).T
    yrs = x.idx.year.to_numpy(); ppy = x.ppy
    sh = lambda a: (a.mean(0) / np.where(a.std(0) > 0, a.std(0), 1)) * np.sqrt(ppy)  # noqa: E731
    outs = []
    for scheme in ("앵커드 워크포워드(직전 연도들로 학습)", "연도 하나 빼기(나머지 연도로 학습, 미래 정보 포함)"):
        net = np.zeros(len(yrs)); pos = np.zeros(len(yrs)); used = np.zeros(len(yrs), bool); picks = {}; gap = []
        for Y in sorted(set(yrs)):
            te = yrs == Y
            tr = (yrs < Y) if scheme.startswith("앵커드") else (yrs != Y)
            if tr.sum() < 0.9 * ppy or te.sum() < 20: continue
            j = int(np.argmax(sh(R[tr]))); net[te] = R[te, j]; pos[te] = np.asarray(P[j], float)[te]; used |= te; picks[Y] = N[j]
            gap.append((Y, float(sh(R[tr])[j]), float(sh(R[te][:, [j]])[0])))
        note = scheme + " | 학습→테스트 Sharpe: " + ", ".join(f"{Y}: {a:.2f}→{b:.2f}" for Y, a, b in gap)
        outs.append((f"{group} {sym[:3]} ({tf})", net[used], x.idx[used], ppy, pos[used], note))
    return outs


if __name__ == "__main__":
    md = ["# 검증 기준 v1 적용 결과 (자동 생성)", ""]
    summary = []
    cases = fixed_strategies()
    for g, tf, sym in [("추세추종", "1d", "BTCUSDT"), ("추세추종", "1d", "ETHUSDT"), ("돌파", "4h", "ETHUSDT"), ("모멘텀", "1d", "BTCUSDT"), ("레짐전환", "1d", "BTCUSDT")]:
        cases += param_strategy(g, tf, sym)
    for name, net, idx, ppy, pos, note in cases:
        txt, res = scorecard(name, np.asarray(net, float), idx, ppy, pos, note)
        md.append(txt); summary.append(res)
    head = ["## 요약 (연도 통과 / 승률 목표 통과 수)", "", "| 전략 | 연도 통과 | 일 | 주 | 월 | 분기 | 연 | 거래 |", "|---|---|---|---|---|---|---|---|"]
    def f(v, k): return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v*100:.0f}%{'✅' if v >= TARGET[k] else ''}"
    for r in summary:
        head.append(f"| {r['name']} | {r['years'][0]}/{r['years'][1]} | " + " | ".join(f(r.get(k), k) for k in ("일", "주", "월", "분기", "연", "거래")) + " |")
    Path("logs/std_report.md").write_text("\n".join(head + [""] + md))
    print("\n".join(head))
