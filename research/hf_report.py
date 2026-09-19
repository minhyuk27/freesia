import json, sys
f = sys.argv[1]
for s in ("BTC", "ETH"):
    r = json.load(open(f"logs/hf_{s}_{f}.json"))
    print("=====", s)
    for c, v in r.items():
        print(f"[{c}] TP {v['tp']*100:.2f}% SL {v['sl']*100:.2f}% H {v['H']}분 손익분기 승률 {v['breakeven_win']}")
        agg = {}
        for Y, row in v["years"].items():
            for tau, t in row["taus"].items():
                if t.get("거래수"):
                    a = agg.setdefault(tau, [0, 0, 0.0, []]); a[0] += t["거래수"]; a[1] += round(t["승률(목표)"] * t["거래수"]); a[2] += t["평균 순가격수익%"] * t["거래수"]
                    a[3].append((int(Y), round(t["레버리지·30% 최종배수"], 2)))
        for tau, (n, w, pr, fin) in agg.items():
            print(f"   τ={tau}: 거래 {n}회(연평균 {n/len(v['years']):.0f}), 승률 {w/n*100:.1f}%, 평균 순가격수익 {pr/n:+.3f}%, 연도별 최종배수 {fin}")
        top = [row["보정(예측확률 구간→실제 승률, n)"][-1] for row in v["years"].values()]
        print("   상위1% 예측구간 실제 승률:", [(a[0], a[1], a[2]) for a in top])
