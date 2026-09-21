"""z4-C 최대 월수익 경계면 — 빈도·승률·명목의 삼각 절충에서 진짜 최대를 찾는다.

지금까지:
  단일 포지션 고승률 저빈도 → 월 +0.12%(고정베팅) / 사이징 포함 월 +2.7%
  동시 보유 저승률 고빈도  → 월 +5.98% (하루 36.8건)

여기서 (배리어 × 슬롯 × 총명목 × 문턱 × 사이징계단)을 함께 훑어 경계면 최대를 찾는다."""
import sys, warnings, itertools
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R/"research"))
from z4_concurrent import sim, Y, P


def fmt(v):
    return "파산" if (v is None or v["ruin"]) else f"{v['mon']:+.2f}%"


P("="*132); P("[1] 경계면 스윕 — 배리어 × 슬롯 × 총명목 (문턱 상위10%, 스캔1분)"); P("="*132)
ALL = []
for bar in (0.01, 0.015, 0.02, 0.03, 0.04):
    for ns in (16, 32, 64, 128):
        for tot in (1, 2, 3, 4, 6, 8):
            mh = min(20160, max(1440, int(bar*600000)))
            v = sim(scan=1, nslot=ns, tot=tot, up=bar, dn=bar, th=0.90, maxh=mh)
            if v and not v["ruin"]:
                ALL.append((v["mon"], bar, ns, tot, 0.90, v))
ALL.sort(key=lambda x: -x[0])
P(f"  {'월%':>8s} {'연%':>12s} {'MDD%':>8s} {'연거래':>8s} {'하루':>7s} {'승률%':>7s} "
  f"{'배리어':>8s} {'슬롯':>5s} {'총명목':>6s} {'최악미실현%':>11s}")
for mon, bar, ns, tot, th, v in ALL[:12]:
    P(f"  {mon:+8.2f} {v['ann']:12.1f} {v['mdd']:8.1f} {v['n']/Y:8.0f} {v['n']/Y/365:7.2f} "
      f"{v['wr']:7.2f} ±{bar*100:6.2f}% {ns:5d} {tot:6.1f}x {v['worst']:11.1f}")

P("\n"+"="*132); P("[2] 문턱 × 사이징 계단 (최고 배리어·슬롯 고정)"); P("="*132)
if ALL:
    _, BAR, NS, TOT, _, _ = ALL[0]
    P(f"  고정: 배리어 ±{BAR*100:.2f}%, 슬롯 {NS}, 총명목 {TOT}배")
    P(f"\n  {'문턱':>10s} " + " ".join(f"{str(lo)+'/'+str(hi):>16s}"
                                     for lo, hi in ((1.0, 1.0), (0.5, 2.0), (0.3, 3.0), (0.2, 4.0))))
    for th in (0.95, 0.90, 0.84, 0.75, 0.60):
        row = []
        for lo, hi in ((1.0, 1.0), (0.5, 2.0), (0.3, 3.0), (0.2, 4.0)):
            mh = min(20160, max(1440, int(BAR*600000)))
            v = sim(scan=1, nslot=NS, tot=TOT, up=BAR, dn=BAR, th=th, maxh=mh,
                    s_lo=lo, s_mid=1.0, s_hi=hi)
            row.append("      파산      " if (v is None or v["ruin"])
                       else f"{v['mon']:+7.2f}%/{v['n']/Y/365:5.1f}")
            if v and not v["ruin"]: ALL.append((v["mon"], BAR, NS, TOT, th, v))
        P(f"  상위{(1-th)*100:5.0f}% " + " ".join(f"{x:>16s}" for x in row))
    P("  (셀 = 월% / 하루 거래수)")

P("\n"+"="*132); P("[3] 비대칭 배리어도 — 익절 좁게·손절 넓게가 동시보유에서도 되나"); P("="*132)
P(f"  {'익절/손절':>12s} " + " ".join(f"{'슬롯'+str(n):>18s}" for n in (16, 32, 64, 128)))
for up, dn in ((0.01, 0.02), (0.015, 0.03), (0.02, 0.04), (0.01, 0.03), (0.02, 0.02), (0.03, 0.03)):
    row = []
    for ns in (16, 32, 64, 128):
        v = sim(scan=1, nslot=ns, tot=3.0, up=up, dn=dn, th=0.90, maxh=8640)
        row.append("       파산       " if (v is None or v["ruin"])
                   else f"{v['mon']:+7.2f}%/{v['wr']:5.1f}%")
        if v and not v["ruin"]: ALL.append((v["mon"], up, ns, 3.0, 0.90, v))
    P(f"  {up*100:5.2f}/{dn*100:5.2f}% " + " ".join(f"{x:>18s}" for x in row))

P("\n"+"="*132); P("[4] 전체 최고 + 3단 레버리지 보고"); P("="*132)
ALL.sort(key=lambda x: -x[0])
if ALL:
    mon, bar, ns, tot, th, v = ALL[0]
    P(f"  최고 구성: 배리어 ±{bar*100:.2f}%(또는 익절값) / 슬롯 {ns} / 총명목 {tot}배 / 문턱 상위{(1-th)*100:.0f}%")
    P(f"    월 {mon:+.2f}%  연 {v['ann']:+.1f}%  MDD {v['mdd']:.1f}%")
    P(f"    연거래 {v['n']/Y:,.0f} (하루 {v['n']/Y/365:.2f}), 승률 {v['wr']:.2f}%, "
      f"거래당 원시 {v['bp']:+.1f}bp, 평균보유 {v['hold']:.0f}분")
    P(f"    최대 동시보유 {v['maxopen']}개, 평균 {v['avgopen']:.2f}개, 최악 미실현 역행 {v['worst']:.1f}%")
    P(f"\n  {'구간':>12s} {'총명목':>7s} {'월%':>9s} {'연%':>12s} {'MDD%':>8s} {'최종배수':>12s}")
    mh = min(20160, max(1440, int(bar*600000)))
    for lb, t_ in (("무레버리지", 1.0), ("중간", 2.0), ("중간+", 3.0), ("공격", 4.0), ("최대공격", 6.0)):
        vv = sim(scan=1, nslot=ns, tot=t_, up=bar, dn=bar, th=th, maxh=mh)
        P(f"  {lb:>12s} {t_:6.1f}x " + ("        파산" if (vv is None or vv["ruin"])
                                         else f"{vv['mon']:+9.2f} {vv['ann']:12.1f} {vv['mdd']:8.1f} {vv['fin']:12.2f}"))

P("\n"+"="*132); P("[5] 월 +80%까지 남은 거리 — 무엇을 얼마나 더 얻어야 하나"); P("="*132)
if ALL:
    mon, bar, ns, tot, th, v = ALL[0]
    cur_mon = mon/100
    P(f"  현재 최고: 월 {mon:+.2f}% (목표 +80% 대비 {(1+cur_mon)/1.8:.4f}배)")
    P(f"  연 로그성장 현재 {np.log((1+cur_mon)**12):.3f} vs 목표 {np.log(1.8)*12:.3f} "
      f"→ {np.log(1.8)*12/max(np.log((1+cur_mon)**12), 1e-9):.1f}배 필요")
    N = v["n"]/Y
    g_now = np.log((1+cur_mon)**12)/N
    g_need = np.log(1.8)*12/N
    P(f"\n  같은 빈도(연 {N:,.0f}건)를 유지한다면:")
    P(f"    거래당 로그성장 {g_now:.6f} → {g_need:.6f} 필요 ({g_need/max(g_now,1e-9):.1f}배)")
    P(f"    거래당 자본증가율 {(np.exp(g_now)-1)*1e4:.2f}bp → {(np.exp(g_need)-1)*1e4:.2f}bp 필요")
    P(f"\n  거래당 엣지를 유지하고 빈도로만 메운다면:")
    P(f"    연 {N*g_need/max(g_now,1e-9):,.0f}건 필요 (하루 {N*g_need/max(g_now,1e-9)/365:,.0f}건)")
    P(f"\n  승률로만 메운다면 (±{bar*100:.1f}%, 비용 10bp, 켈리 최적):")
    for pw in (0.52, 0.55, 0.58, 0.60, 0.65, 0.70):
        best = None
        for Lx in np.arange(0.5, 1.0/(bar+0.004), 0.5):
            win = 1+Lx*(bar-0.001); lose = 1-Lx*(bar+0.001)
            if lose <= 0: break
            g = pw*np.log(win)+(1-pw)*np.log(lose)
            if best is None or g > best[1]: best = (Lx, g)
        if best is None: continue
        Lx, g = best
        m = np.exp(N*g/12)-1
        tag = " ★" if m >= 0.8 else ""
        P(f"    승률 {pw*100:.0f}% → 켈리 {Lx:4.1f}배, 월 {m*100:+10.2f}%{tag}")
