"""상위 후보 강건성: 비용 배수, 첫 검증연도(분할 경계), 파라미터 고원."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from freesia.bt import core  # noqa: E402
import screen as S  # noqa: E402
import screen3 as S3  # noqa: E402

CASES = [("volbreak", S3.f_volbreak, "4h", "BTCUSDT"), ("volbreak", S3.f_volbreak, "4h", "ETHUSDT"),
         ("donchian", S.f_donchian, "4h", "BTCUSDT"), ("voltsmom", S.f_voltsmom, "4h", "ETHUSDT"),
         ("volregime", S3.f_volregime, "1d", "BTCUSDT"), ("tsmom", S.f_tsmom, "1d", "BTCUSDT")]
out = {}
for fam, fn, tf, sym in CASES:
    x = S.Ctx(sym, tf)
    N, P = fn(x)
    row = {}
    for cm in (1, 2, 4):
        core.run.__defaults__ = (core.COST * cm,)
        for fy, mt in ((2021, 1.0), (2022, 1.5), (2023, 2.0), (2024, 3.0)):
            r = S.evaluate(fam, x, N, P, fy, mt)
            if r:
                row[f"cost×{cm} test≥{fy}"] = {"oosSh": r["oos_sharpe"], "ann": r["oos_ann"], "mdd": r["oos_mdd"], "BHSh": r["bh_sharpe"], "pos": r["all_params_oos_pos_frac"]}
    core.run.__defaults__ = (core.COST,)
    out[f"{fam} {tf} {sym[:3]}"] = row
    print(f"== {fam} {tf} {sym[:3]}")
    for k, v in row.items():
        print(f"   {k:22} OOS Sh {v['oosSh']:5} ann {v['ann']:6} mdd {v['mdd']:6} | BH Sh {v['BHSh']:5} | 모든파라미터 양수비율 {v['pos']}")
Path("logs/robust.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
