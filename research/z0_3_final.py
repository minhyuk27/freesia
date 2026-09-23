"""z0.3 최종 검증: 안전여유(최악 floor) 제약 하 최적점 + 워크포워드 + 플라시보 + 비용/체결 민감도."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from z0_3b import build, sim, rep, D, SYM, BARS, TE, REC, EPS
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
DEP = {8.0: 1.0, 12.0: 3.0}

P("[1] 안전여유 제약 — 최악 floor가 −70%보다 얕은 것만 채택")
rows = []
for kg in (0.0, 3.0, 5.0, 8.0):
    E = build(DEP, kg)
    for w in (1.0, 1.5, 2.0, 3.0, 5.0):
        for Nm in (3.0, 5.0, 8.0):
            if Nm < w: continue
            dly, fld, liq = sim(E, w, Nm)
            if liq or dly is None: continue
            r = rep(dly, fld, liq, "", quiet=True)
            rows.append((kg, w, Nm, r[0], r[1], r[2], r[3], r[4]))
t = pd.DataFrame(rows, columns=["kg", "w", "Nmax", "연%", "MDD%", "최악%", "24~%", "25~%"]).round(1)
P(t[t["최악%"] > -70].sort_values("연%", ascending=False).head(14).to_string(index=False))
P("\n  (참고) 안전제약 없이 상위:")
P(t.sort_values("연%", ascending=False).head(6).to_string(index=False))

sel = t[t["최악%"] > -70].sort_values("연%", ascending=False)
KG, W, NM = (float(sel.iloc[0].kg), float(sel.iloc[0].w), float(sel.iloc[0].Nmax)) if len(sel) else (8.0, 2.0, 5.0)
P(f"\n→ 채택: kg={KG:.0f} w={W:.1f} Nmax={NM:.0f}")
E = build(DEP, KG)

P("\n[2] 워크포워드 (2년 학습 → 1년 검증)")
def seg(E, s, e):
    E2 = [x for x in E if (s is None or x[4] >= s) and (e is None or x[4] < e)]
    return E2
tot = []
for y in pd.date_range("2022-01-01", "2026-01-01", freq="YS", tz="UTC"):
    tr = seg(E, y-pd.DateOffset(years=2), y); te = seg(E, y, y+pd.DateOffset(years=1))
    if len(tr) < 20 or len(te) < 5: continue
    bw = None
    for w in (1.0, 2.0, 3.0, 5.0):
        for Nm in (3.0, 5.0, 8.0):
            if Nm < w: continue
            d_, f_, l_ = sim(tr, w, Nm)
            if l_ or d_ is None: continue
            r = rep(d_, f_, l_, "", quiet=True)
            if r and r[2] > -70 and (bw is None or r[0] > bw[0]): bw = (r[0], w, Nm)
    if bw is None: continue
    d_, f_, l_ = sim(te, bw[1], bw[2])
    r = rep(d_, f_, l_, "", quiet=True)
    c = r[0] if r else -100.0
    tot.append(c)
    P(f"  {y.year} 검증: 학습최적 w={bw[1]:.0f} Nmax={bw[2]:.0f} → 연 {c:+8.0f}% {'청산' if l_ else ''}")
if tot: P(f"  → 평균 {np.mean(tot):+.0f}% 양수 {sum(1 for x in tot if x>0)}/{len(tot)}")

P("\n[3] 플라시보 — 진입 시점을 무작위 분으로 옮김 (깊이·보유길이 동일, 200회)")
rng = np.random.default_rng(7)
real = np.mean([e[2] for e in E])*1e4
nul = []
for _ in range(200):
    sh = int(rng.integers(500, 200000))
    Esh = [(a, b, r, m, ts, k, rv, wg) for (a, b, r, m, ts, k, rv, wg) in E]
    # 수익률만 시간축에서 순환 이동 → 조건과 결과의 연결을 끊음
    rr = np.array([e[2] for e in E]); rr = np.roll(rr, sh)
    nul.append(rr.mean()*1e4)
nul = np.array(nul)
P(f"  실제 평균 {real:+.1f}bp | 플라시보 평균 {nul.mean():+.1f}bp 95분위 {np.quantile(nul,0.95):+.1f} 최대 {nul.max():+.1f}")
P(f"  → 실제가 플라시보 분포에서 차지하는 백분위: {(nul < real).mean()*100:.1f}%")

P("\n[4] 체결·비용 민감도")
for eps, fm, ft, lab in ((0.0, 2.0, 5.0, "낙관 eps0 메2/테5"), (0.0002, 2.0, 5.0, "기준 eps2bp 메2/테5"),
                          (0.0005, 3.0, 6.0, "보수 eps5bp 메3/테6"), (0.001, 4.0, 7.0, "비관 eps10bp 메4/테7"),
                          (0.002, 5.0, 8.0, "최악 eps20bp 메5/테8")):
    E2 = build(DEP, KG, fm=fm, ft=ft, eps=eps)
    d_, f_, l_ = sim(E2, W, NM)
    rep(d_, f_, l_, f"  {lab} (n={len(E2)})")

P("\n[5] 최종 성과")
d_, f_, l_ = sim(E, W, NM)
r = rep(d_, f_, l_, "  z0.3")
dly = r[5]
yr = dly.resample("YE").last()/dly.resample("YE").first()-1
mon = dly.resample("ME").last()/dly.resample("ME").first()-1
P("  연도별: " + " | ".join(f"{i.year} {v*100:+.0f}%" for i, v in yr.items()))
P("  최근 12개월: " + " | ".join(f"{i.strftime('%y-%m')} {v*100:+.1f}%" for i, v in mon.tail(12).items()))
l12 = mon.tail(12)
P(f"  최근 12개월 누적 {(np.prod(1+l12)-1)*100:+.1f}% | 월승률 {float((mon>0).mean()):.3f} | 거래 {len(E)}건 ({len(E)/6.7:.0f}건/년)")
rr = np.array([e[2] for e in E])
P(f"  거래당 평균 {rr.mean()*1e4:+.0f}bp 중앙 {np.median(rr)*1e4:+.0f}bp 승률 {(rr>0).mean():.3f} 최악 {rr.min()*1e4:+.0f}bp")
