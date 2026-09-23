"""R50b — R50 결과(±1.5% 상위 0.1% 결판 적중 65.8%)의 시드 견고성 점검.

같은 특징·모델·선택(양 꼬리 각각 상위/하위 q)에서 random_state만 0~4로 바꿔 적중률 분포를 본다.
HistGradientBoosting은 특징 분할 후보 표본 추출(대용량에서 서브샘플 빈 경계)과 조기 종료 없이도 시드에 따라 결과가 달라질 수 있다.
사용: python research/r50b_seed_check.py"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R / "research"))
from r50_direction_model import walk, select
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)
y = "y15"
res = {}
for sd in range(5):
    W = walk(y, seed=sd)
    for q in (0.003, 0.001):
        S = select(W, q); dec = S[S[y] != 0]
        acc = (dec[y] == dec.side).mean()
        res.setdefault(q, []).append(acc)
        P(f"시드 {sd} q={q*100:.1f}%: 주 {len(S)/((W.ts.max()-W.ts.min()).days/7):.1f}회 결판 적중 {acc:.3f} (n={len(dec)}) 무결판 {(S[y]==0).mean():.2f}")
for q, v in res.items():
    P(f"q={q*100:.1f}%: 시드 5개 평균 {np.mean(v):.3f} 범위 {min(v):.3f}~{max(v):.3f}")
