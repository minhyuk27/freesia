"""z1.5b — 배리어 전략의 유일한 생존 가능성: 동시보유 억제(쿨다운 확대).
계산 절약: ±2%/최대2일, ±5%/최대5일로 제한. step 크게.
사용: python research/z15b.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research"))
from z15_barrier import build, D, SYM, rep
import exit_tier as X
from absfloor import met
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)

P("[1] 쿨다운 확대 — 동시 보유를 줄이면 레버리지가 열리나 (추세4320분 상위10%)")
for bx, mh in ((0.02, 2880), (0.05, 7200)):
    for cool in (60, 720, 1440, 4320, 10080):
        E = sorted(sum([build(s, 4320, 0.90, bx, bx, mh, mkt=True, step=240, cool=cool)
                        for s in SYM], []), key=lambda z: z[0])
        rep(E, f"±{bx*100:.0f}% 쿨다운{cool}분({cool/1440:.1f}일)")

P("\n[2] 지정가 진입 결합 (쿨다운 1일)")
for bx, mh in ((0.02, 2880), (0.05, 7200)):
    for kx in (0.3, 0.6, 1.0):
        E = sorted(sum([build(s, 4320, 0.90, bx, bx, mh, mkt=False, kx=kx, W=240, step=240, cool=1440)
                        for s in SYM], []), key=lambda z: z[0])
        rep(E, f"±{bx*100:.0f}% 지정가k={kx} 쿨다운1일")

P("\n[3] 최선 후보 상세")
BEST = []
for bx, mh, cool, mkt, kx in ((0.05, 7200, 1440, True, 1.0), (0.05, 7200, 4320, True, 1.0),
                              (0.02, 2880, 1440, True, 1.0), (0.05, 7200, 1440, False, 0.6)):
    E = sorted(sum([build(s, 4320, 0.90, bx, bx, mh, mkt=mkt, kx=kx, W=240, step=240, cool=cool)
                    for s in SYM], []), key=lambda z: z[0])
    b = rep(E, f"±{bx*100:.0f}% 쿨{cool/1440:.0f}일 {'시장가' if mkt else f'지정가k{kx}'}", full=True)
    if b: BEST.append((b[0], bx, mh, cool, mkt, kx))
