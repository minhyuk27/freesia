"""주 2회 이상 거래(자본 25% 이상 포지션 변경 연 104회 이상) 조건 스크리닝. BTC·ETH만.
사용: python research/screen5.py <tf,...> <tag> [min_events_per_year]"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import screen as S  # noqa: E402
import screen3 as S3  # noqa: E402
import screen4 as S4  # noqa: E402

FAM = dict(S.FAMILIES)
FAM.update({"volbreak": S3.f_volbreak, "volregime": S3.f_volregime, "nearhigh": S4.f_nearhigh, "ddregime": S4.f_ddregime,
            "obv": S4.f_obv, "bollinger": S4.f_bollinger, "clv": S4.f_clv})


def events_per_year(p, tf):
    d = np.abs(np.diff(np.nan_to_num(np.asarray(p, float)), prepend=0.0))
    return float((d >= 0.25).sum() / (len(p) / S.core.PPY[tf]))


def main():
    tfs, tag = sys.argv[1].split(","), sys.argv[2]
    mn = float(sys.argv[3]) if len(sys.argv) > 3 else 104
    out, t0 = [], time.time()
    for fam, fn in FAM.items():
        for tf in tfs:
            for sym in S.SYMS:
                try:
                    x = S.Ctx(sym, tf)
                    N, P = fn(x)
                    keep = [i for i, p in enumerate(P) if events_per_year(p, tf) >= mn]
                    if len(keep) < 3:
                        continue
                    N2, P2 = [N[i] for i in keep], [P[i] for i in keep]
                    fy, mt = (2023, 1.0) if fam == "metrics" else (2022, 1.5)
                    r = S.evaluate(fam, x, N2, P2, fy, mt)
                    if r:
                        r["kept"] = f"{len(keep)}/{len(P)}"
                        out.append(r)
                        print(fam, tf, sym, "kept", r["kept"], "OOS Sh", r["oos_sharpe"], flush=True)
                except Exception as e:  # noqa: BLE001
                    print("ERR", fam, tf, sym, repr(e), flush=True)
    Path(f"logs/screen_{tag}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
