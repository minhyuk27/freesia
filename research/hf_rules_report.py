import json, sys
import numpy as np
B = {(r["rule"], r["geom"]): r for r in json.load(open("logs/hf_rules_BTC.json"))}
E = {(r["rule"], r["geom"]): r for r in json.load(open("logs/hf_rules_ETH.json"))}
keys = sorted(set(B) & set(E))
print("규칙×기하 조합", len(keys), "(종목당)")
ok = lambda r: r["train"]["bps"] is not None and r["test"]["bps"] is not None and r["train"]["n"] >= 100 and r["test"]["n"] >= 50  # noqa: E731
pos_b = [k for k in keys if ok(B[k]) and B[k]["train"]["bps"] > 0 and B[k]["test"]["bps"] > 0]
pos_e = [k for k in keys if ok(E[k]) and E[k]["train"]["bps"] > 0 and E[k]["test"]["bps"] > 0]
both = [k for k in keys if k in pos_b and k in pos_e]
print(f"학습·테스트 모두 비용 후 양수: BTC {len(pos_b)}, ETH {len(pos_e)}, 두 종목 모두 {len(both)}")
allrows = [B[k] for k in keys if ok(B[k])] + [E[k] for k in keys if ok(E[k])]
print("전체 행 중 테스트 t≥3:", sum(1 for r in allrows if (r["test"]["t"] or 0) >= 3), "/", len(allrows))
print("총수익(수수료 전) 평균 bps: %.1f, 비용 12bp 차감 후 평균 bps: %.1f" % (np.mean([r["gross_bps"] for r in allrows]), np.mean([r["gross_bps"] for r in allrows]) - 12))
print("\n[두 종목 모두 양수인 규칙] 테스트 bps(BTC/ETH), 승률, 연도별(BTC)")
for k in sorted(both, key=lambda k: -min(B[k]["test"]["bps"], E[k]["test"]["bps"]))[:25]:
    b, e = B[k], E[k]
    print(f"  {k[0]:34} {k[1]:5} 학습 {b['train']['bps']:6.1f}/{e['train']['bps']:6.1f} 테스트 {b['test']['bps']:6.1f}/{e['test']['bps']:6.1f} bps  t {b['test']['t']}/{e['test']['t']}  승률 {b['test']['win']}/{e['test']['win']}  n {b['test']['n']}/{e['test']['n']}  BTC연도 {b['years']}")
print("\n[BTC 테스트 t 상위 10 (종목 무관)]")
for r in sorted([B[k] for k in keys if ok(B[k])], key=lambda r: -(r["test"]["t"] or -9))[:10]:
    print(f"  {r['rule']:34} {r['geom']:5} 학습 {r['train']['bps']:6.1f} 테스트 {r['test']['bps']:6.1f} t {r['test']['t']} 승률 {r['test']['win']} n {r['test']['n']}")
print("\n[ETH 테스트 t 상위 10]")
for r in sorted([E[k] for k in keys if ok(E[k])], key=lambda r: -(r["test"]["t"] or -9))[:10]:
    print(f"  {r['rule']:34} {r['geom']:5} 학습 {r['train']['bps']:6.1f} 테스트 {r['test']['bps']:6.1f} t {r['test']['t']} 승률 {r['test']['win']} n {r['test']['n']}")
