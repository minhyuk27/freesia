import json
import numpy as np
J = {s: json.load(open(f"logs/hf_battery_{s}.json")) for s in ("BTC", "ETH")}
rows = {s: {(r["rule"], r["geom"]): r for r in J[s]["rows"]} for s in J}
keys = sorted(set(rows["BTC"]) & set(rows["ETH"]))
ok = lambda r: r["train"]["bps"] is not None and r["test"]["bps"] is not None and r["train"]["n"] >= 100 and r["test"]["n"] >= 50  # noqa: E731
is_pl = lambda k: k[0].startswith("대조군")  # noqa: E731
real = [k for k in keys if not is_pl(k) and ok(rows["BTC"][k]) and ok(rows["ETH"][k])]
pl = [k for k in keys if is_pl(k) and ok(rows["BTC"][k]) and ok(rows["ETH"][k])]
pos = lambda r: r["train"]["bps"] > 0 and r["test"]["bps"] > 0  # noqa: E731
both = lambda k: pos(rows["BTC"][k]) and pos(rows["ETH"][k])  # noqa: E731
print("접근법(기본) 수:", len(J["BTC"]["bases"]) - 1, "| 실제 규칙×기하:", len(real), "| 대조군 규칙×기하:", len(pl))
print("학습·테스트 모두 비용 후 양수(종목별) 비율: 실제 BTC %.1f%% ETH %.1f%% / 대조군 BTC %.1f%% ETH %.1f%%" % (
    100 * np.mean([pos(rows["BTC"][k]) for k in real]), 100 * np.mean([pos(rows["ETH"][k]) for k in real]),
    100 * np.mean([pos(rows["BTC"][k]) for k in pl]), 100 * np.mean([pos(rows["ETH"][k]) for k in pl])))
print("두 종목 모두 양수: 실제 %d개(%.1f%%) / 대조군 %d개(%.1f%%)" % (sum(both(k) for k in real), 100 * np.mean([both(k) for k in real]), sum(both(k) for k in pl), 100 * np.mean([both(k) for k in pl])))
for s in ("BTC", "ETH"):
    tp = [rows[s][k]["test"]["t"] or 0 for k in pl]; tr_ = [rows[s][k]["test"]["t"] or 0 for k in real]
    gp = [rows[s][k]["gross_bps"] for k in pl]; gr = [rows[s][k]["gross_bps"] for k in real]
    print(f"{s} 테스트 t: 대조군 95% {np.percentile(tp,95):.2f} 99% {np.percentile(tp,99):.2f} 최대 {max(tp):.2f} | 실제 95% {np.percentile(tr_,95):.2f} 99% {np.percentile(tr_,99):.2f} 최대 {max(tr_):.2f} | 수수료 전 평균 bps 대조군 {np.mean(gp):.1f} 실제 {np.mean(gr):.1f}, 실제 상위 95% {np.percentile(gr,95):.1f}")
print("\n[두 종목 모두 학습·테스트 양수인 실제 규칙 — 테스트 t(BTC/ETH) 상위]")
cand = sorted([k for k in real if both(k)], key=lambda k: -min(rows["BTC"][k]["test"]["t"] or -9, rows["ETH"][k]["test"]["t"] or -9))
for k in cand[:20]:
    b, e = rows["BTC"][k], rows["ETH"][k]
    print(f"  {k[0][:60]:60} {k[1]:5} 테스트 bps {b['test']['bps']:6.1f}/{e['test']['bps']:6.1f} t {b['test']['t']}/{e['test']['t']} 승률 {b['test']['win']}/{e['test']['win']} n {b['test']['n']}/{e['test']['n']} 총(수수료전) {b['gross_bps']}/{e['gross_bps']}")
print("\n[전체 실제 규칙 중 (BTC 또는 ETH) 테스트 t 상위 10]")
allr = [(s, k) for s in ("BTC", "ETH") for k in keys if not is_pl(k) and ok(rows[s][k])]
for s, k in sorted(allr, key=lambda x: -(rows[x[0]][x[1]]["test"]["t"] or -9))[:10]:
    r = rows[s][k]; print(f"  {s} {k[0][:58]:58} {k[1]:5} 학습 {r['train']['bps']:6.1f} 테스트 {r['test']['bps']:6.1f} t {r['test']['t']} 승률 {r['test']['win']} n {r['test']['n']}")
# 접근법 단위 요약
ap = {}
for k in real:
    base = k[0].split(" | ")[0]
    b, e = rows["BTC"][k], rows["ETH"][k]
    score = min(b["test"]["t"] or -9, e["test"]["t"] or -9)
    a = ap.setdefault(base, {"n": 0, "both": 0, "best": None, "bs": -99})
    a["n"] += 1; a["both"] += int(both(k))
    if score > a["bs"]: a["bs"] = score; a["best"] = (k, b["test"]["bps"], e["test"]["bps"], b["test"]["t"], e["test"]["t"], b["gross_bps"], e["gross_bps"])
json.dump({"approach": {b: {"variants": v["n"], "both_positive": v["both"], "best_minT": v["bs"], "best": v["best"]} for b, v in ap.items()}}, open("logs/hf_battery_approach.json", "w"), ensure_ascii=False, default=float)
print("\n접근법 단위: 두 종목 모두 양수 규칙이 1개 이상인 접근법 %d / %d" % (sum(1 for v in ap.values() if v["both"] > 0), len(ap)))
