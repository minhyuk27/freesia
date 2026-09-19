"""screen_*.json 결과를 OOS Sharpe 순으로 출력."""
import glob
import json
import sys

rows = []
for f in sorted(glob.glob("logs/screen_*.json")):
    if len(sys.argv) > 1 and sys.argv[1] not in f:
        continue
    rows += json.load(open(f))
err = [x for x in rows if "error" in x]
rows = [x for x in rows if "error" not in x]
for e in err[:5]:
    print("ERR", e)
print(f"{'family':12}{'sym':4}{'tf':3} {'OOS Sh':>6} {'ann':>6} {'mdd':>6} | {'BH Sh':>5} | yrs+ | med  pos | n    years")
for x in sorted(rows, key=lambda x: -x["oos_sharpe"])[: int(sys.argv[2]) if len(sys.argv) > 2 else 40]:
    print(f"{x['family']:12}{x['sym'][:3]:4}{x['tf']:3} {x['oos_sharpe']:6} {x['oos_ann']:6} {x['oos_mdd']:6} | {x['bh_sharpe']:5} | {x['years_pos']}/{x['years']}  | {x['all_params_oos_sharpe_median']:5} {x['all_params_oos_pos_frac']:4} | {x['n_params']:4} {x['oos_years']}")
