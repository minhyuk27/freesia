"""볼록성 추가 — z0.3 진입분의 일부를 '러너'로 장기 보유.
패닉 바닥에서 산 포지션을 봉 종가에 전부 털지 않고 일부를 추적손절/장기목표로 들고 간다.
오목(승률90%·상방절단) → 볼록(꼬리 확장) 전환 시도.
사용: python research/runner.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
import exit_tier as X
from absfloor import met
from anchor import build_a
from exit_tier import RAW, SYM
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
TE = pd.Timestamp("2024-01-01", tz="UTC"); REC = pd.Timestamp("2025-01-01", tz="UTC")
Z = build_a("open", ext=0.10)
P(f"z0.3 기본 이벤트 {len(Z)}건")


@numba.njit(cache=True)
def runner_leg(h, l, c, entry_i, entry_px, base_exit, trail, maxdays, fee_bp):
    """러너: 진입가에서 시작, 최고가 대비 trail 비율 하락 시 청산. maxdays 초과 시 강제 청산.
    반환: 수익률, 청산 인덱스, 최대역행"""
    n = len(c)
    out_r = np.zeros(len(entry_i)); out_x = np.zeros(len(entry_i), np.int64); out_m = np.zeros(len(entry_i))
    for i in range(len(entry_i)):
        f = entry_i[i]; px = entry_px[i]
        lim = f + maxdays*1440
        if lim >= n: lim = n-1
        pk = px; end = lim; lo_ = 1e18
        for t in range(base_exit[i], lim+1):
            if h[t] > pk: pk = h[t]
            if l[t] < lo_: lo_ = l[t]
            if l[t] <= pk*(1.0-trail):
                end = t
                out_r[i] = (pk*(1.0-trail))/px - 1.0 - fee_bp/1e4
                break
        else:
            end = lim
            out_r[i] = c[lim]/px - 1.0 - fee_bp/1e4
        if end == lim and out_r[i] == 0.0:
            out_r[i] = c[lim]/px - 1.0 - fee_bp/1e4
        out_x[i] = end
        out_m[i] = 1.0 - lo_/px
    return out_r, out_x, out_m


def build_runner(frac, trail, maxdays, base=Z):
    """frac: 러너 비중. 나머지(1-frac)는 기존대로 봉 종가 청산."""
    E = []
    by = {}
    for ev in base:
        by.setdefault(ev[5][0], []).append(ev)
    for s, evs in by.items():
        d = RAW[s]
        ei = np.array([e[0] for e in evs], np.int64)
        bx = np.array([e[1] for e in evs], np.int64)
        # 진입가 = 종가기준 역산 (r = px_out/px_in -1 - fee) → px_in = c[bx]/(r+1+fee)
        px = np.array([d["c"][e[1]]/(e[2] + 1.0 + 7.0/1e4) for e in evs])
        rr, xx, mm = runner_leg(d["h"], d["l"], d["c"], ei, px, bx, trail, maxdays, 7.0)
        for j, e in enumerate(evs):
            if frac < 1.0:
                E.append((e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7]*(1.0-frac)))
            if frac > 0.0:
                E.append((e[0], int(xx[j]), float(rr[j]), float(max(mm[j], e[3])), e[4],
                          (e[5][0], e[5][1], e[5][2]+1000), e[6], e[7]*frac))
    E.sort(key=lambda z: z[0])
    return E


def tiers(E, lab):
    r = np.array([x[2] for x in E])
    row = f"  {lab:30s} n={len(r):4d} 평균 {r.mean()*1e4:+6.0f}bp 최대 {r.max()*1e4:+7.0f}bp |"
    for w, Nm, tag in ((1.0, 1.0, "무레버"), (3.0, 5.0, "중간"), (5.0, 8.0, "공격")):
        eq, fl, liq = X.sim(E, w, Nm); m = met(eq, fl, liq)
        row += f" {tag} {'파산' if m is None else f'{m[0]:+5.0f}%({m[2]:5.1f}%)'}"
    P(row)
    return


P("\n[기준] 러너 없음")
tiers(Z, "frac=0")

P("\n[1] 러너 비중 × 추적손절 폭 (최대 30일)")
for frac in (0.2, 0.33, 0.5):
    for trail in (0.05, 0.10, 0.15, 0.25, 0.40):
        tiers(build_runner(frac, trail, 30), f"frac={frac} trail={trail*100:.0f}%")

P("\n[2] 최대 보유일수 (frac=0.33, trail=15%)")
for md in (1, 3, 7, 14, 30, 60, 120):
    tiers(build_runner(0.33, 0.15, md), f"최대 {md}일")

P("\n[3] 러너 수익 분포 (frac=1.0으로 러너만 측정)")
for trail, md in ((0.10, 30), (0.15, 30), (0.25, 30), (0.15, 90)):
    E = build_runner(1.0, trail, md)
    r = np.array([x[2] for x in E]); hd = np.array([x[1]-x[0] for x in E])/1440
    P(f"  trail={trail*100:.0f}% {md}일: 평균 {r.mean()*1e4:+7.0f}bp 중앙 {np.median(r)*1e4:+6.0f}bp "
      f"승률 {(r>0).mean():.3f} 최대 {r.max()*1e4:+7.0f}bp 최소 {r.min()*1e4:+7.0f}bp 보유중앙 {np.median(hd):.1f}일")
    P(f"     상위5개: " + " ".join(f"{v*1e4:+.0f}" for v in np.sort(r)[-5:]) +
      f" | 상위10%가 총합의 {np.sort(r)[-len(r)//10:].sum()/max(r.sum(),1e-9)*100:.0f}%")
