import json, sys
rows = [x for x in json.load(open(sys.argv[1])) if "error" not in x]
print("그룹 tf sym | WF표본밖Sharpe ann mdd | BH | 양수연도 | 고원 | 빈도/년 | n")
for x in sorted(rows, key=lambda r: -r["oos_sharpe"]):
    print(f"{x['group']:8} {x['tf']} {x['sym'][:3]} | {x['oos_sharpe']:5} {x['oos_ann']:6} {x['oos_mdd']:7} | {x['bh_sharpe']:5} | {x['years_pos']}/{x['years']} | {x['all_params_oos_pos_frac']:4} | {x['ev_picks_mean']:6} | {x['n_params']}")
best = []
for x in rows:
    for t in x["top_full"]:
        best.append((t["sh_2022+"], x["sym"][:3], x["tf"], t))
print()
for s22, sym, tf, t in sorted(best, key=lambda z: -z[0])[:22]:
    print(f"{sym} {tf} 전체 {t['sh_all']:5} 2022+ {t['sh_2022+']:5} 빈도 {t['ev']:6}/년 | {t['name']}")
