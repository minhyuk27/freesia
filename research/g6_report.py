import glob, json, sys
rows = []
for f in sorted(glob.glob("logs/screen_g6*.json")):
    rows += [x for x in json.load(open(f)) if "error" not in x]
errs = [x for f in glob.glob("logs/screen_g6*.json") for x in json.load(open(f)) if "error" in x]
for e in errs: print("ERR", e)
print(f"{'그룹':10}{'tf':3}{'sym':4}{'WF OOS':>7}{'ann':>6}{'mdd':>7}{'BH':>5} {'yrs+':>4} {'고원':>5} {'빈도/년(선택)':>10} n")
for x in sorted(rows, key=lambda r: (r["group"], -r["oos_sharpe"])):
    print(f"{x['group']:10}{x['tf']:3}{x['sym'][:3]:4}{x['oos_sharpe']:7}{x['oos_ann']:6}{x['oos_mdd']:7}{x['bh_sharpe']:5} {x['years_pos']}/{x['years']} {x['all_params_oos_pos_frac']:5} {x['ev_picks_mean']:10} {x['n_params']}")
print("\n[개별 기법 상위: 전체 표본 Sharpe·2022+ Sharpe·빈도] (재검증 전 탐색용)")
best = {}
for x in rows:
    for t in x["top_full"]:
        key = (x["group"], x["sym"][:3], x["tf"], t["name"])
        best[key] = t
lst = sorted(best.items(), key=lambda kv: -kv[1]["sh_2022+"])
for (g, s, tf, n), t in lst[: int(sys.argv[1]) if len(sys.argv) > 1 else 30]:
    print(f"{s} {tf} 전체 {t['sh_all']:5} 2022+ {t['sh_2022+']:5} 빈도 {t['ev']:6}/년 | {n}")
