"""z8의 C 다리(4시간딥) 검증 — 문서화된 z0.3과 왜 20배 차이가 나는가."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R/"research"))
import z2_lib as L2
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
S = "BTCUSDT"
d = L2.D[S]; rv = d["rv"]; O, H, Lo, C = d["o"], d["h"], d["l"], d["c"]
N = len(C); warm = 200000

P("z2_lib의 rv 정체 확인")
P(f"  rv 중앙 {np.nanmedian(rv)*1e4:.2f}bp  평균 {np.nanmean(rv)*1e4:.2f}bp")
P(f"  → 이것은 '1분 로그수익의 1440분 표준편차' = 분 단위 변동성\n")

for barmin, name, ks in ((15, "z1.4(15분)", (6, 8, 12)), (240, "z0.3(4시간)", (10, 14, 20))):
    P(f"── {name}: rv×{ks} ──")
    med_rv = np.nanmedian(rv[warm:])
    P(f"  내 구현 깊이 = 분단위rv × k = {med_rv*ks[0]*100:.3f}% ~ {med_rv*ks[2]*100:.3f}%")
    scale = np.sqrt(barmin)
    P(f"  문서 구현 깊이 = 봉단위rv × k ≈ 분rv×√{barmin} × k = "
      f"{med_rv*scale*ks[0]*100:.3f}% ~ {med_rv*scale*ks[2]*100:.3f}%")
    P(f"  → 내 구현이 {scale:.1f}배 얕다\n")

# 체결률 비교
nb = (N-warm)//240
fills_mine = 0; fills_doc = 0
for b in range(nb):
    t0 = warm+b*240; t1 = t0+239
    if t1 >= N-1: break
    if np.isnan(rv[t0]) or rv[t0] <= 0: continue
    lo = Lo[t0:t1+1].min()
    if lo <= O[t0]*(1-rv[t0]*10): fills_mine += 1
    if lo <= O[t0]*(1-rv[t0]*np.sqrt(240)*10): fills_doc += 1
P(f"4시간봉 {nb:,}개 중 1층 체결:")
P(f"  내 구현(분rv×10 = {np.nanmedian(rv[warm:])*10*100:.2f}%):  {fills_mine:,}회 ({fills_mine/nb*100:.1f}%)")
P(f"  문서 구현(봉rv×10 = {np.nanmedian(rv[warm:])*np.sqrt(240)*10*100:.2f}%): {fills_doc:,}회 ({fills_doc/nb*100:.1f}%)")
P(f"\n문서 z0.3 실측: 6년간 117건 (연 17건). 내 구현: 연 877건 → {877/17:.0f}배")
P("→ 서로 다른 전략이다. C 다리의 연 +477%는 z0.3이 아니며,")
P("  볼트에 이미 '얕은 딥은 무레버 +24%→+15%로 악화, 레버리지 태우면 파산'으로 기각돼 있다.")
