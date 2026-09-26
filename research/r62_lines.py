"""R62 — 차트 선(수평 지지·저항, 사선 추세선) 검출과 사건 카탈로그. 호가 결합(R63)의 바탕.

선 (전부 인과적: 피벗 i는 i+k 봉 종가 뒤에야 사용)
  H 수평: 확정 피벗 저점(지지)/고점(저항) 가격. 새 피벗이 살아 있는 같은 쪽 선의 띠(±δ) 안이면 새 선을 만들지 않는다
          (그 선의 반등으로 이미 집계됨). 생성 시 이전 반등 수 = 1(피벗 자체).
  T 사선: 새 확정 피벗 저점 p2와 W봉 안의 이전 피벗 저점 p1을 잇는 로그가격 직선 중, 사이 저가가 선을 0.1·ATR 넘게
          뚫지 않는 것 가운데 가장 긴 것. 기울기 |s| ≤ 0.3 ATR/봉. 생성 시 이전 반등 수 = 2 + 사이 피벗 터치 수.
  저항은 가격 부호를 뒤집어 같은 지지 엔진으로 처리.
사건 (선을 앞으로 추적, δ = 0.25·ATR 띠, B = 1·ATR)
  무장: 봉 전체가 띠 위(저가 > 선+δ) → 다음에 저가가 선+δ 이하로 오면 '접근'.
  접근 후 결과: 종가 < 선−δ → 돌파(선 소멸) / 이후 봉 고가 > 선+B → 반등(반등 수 +1, 재무장 필요) / TO봉 안에 둘 다 없음 → 미결.
  돌파 뒤 4·12·48봉 수익(돌파 방향, 로그)과 접근~결판 사이 최대 관통 깊이(스윕 판단용)를 기록.
위약: 실제 선을 생성 시각·기울기·수명 그대로 두고 위치만 ±0.5~1.5 ATR 이동(생성 시 가격 아래쪽 유지). 같은 규칙으로 판정.
질문: ① p(반등 | 이전 반등 수)가 위약보다 높은가, 반등 수가 늘수록 오르는가(Garzarelli 재현)
      ② 돌파 뒤 지속(4·12·48봉)이 위약보다 큰가, 돌파 전 터치 수가 많을수록 큰가("건드리다 돌파")
출력: data/r62_events.parquet
사용: python research/r62_lines.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, numba
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R))
from freesia import store
P = lambda *a: print(" ".join(str(x) for x in a), flush=True)

K, DLT, BB, TO, AGE, WIN, SLOPE = 5, 0.25, 1.0, 48, 500, 150, 0.3
FWD = (4, 12, 48)
MAXE = 64


@numba.njit(cache=True)
def scan(i0, iend, ia, a, s, d, B, h, l, c, TO, out):
    """지지선 하나 추적. out[e] = (접근봉, 결과(1 반등, -1 돌파, 0 미결), 결판봉, 이전 반등 수, 최대 관통(δ단위)). 반환: 사건 수, 소멸봉."""
    armed = False; nb = 0; e = 0; j = i0; n = len(c)
    while j < min(iend, n):
        L = a + s * (j - ia)
        if not armed:
            if c[j] < L - d: return e, j                        # 무장 전 이탈 → 소멸(사건 아님)
            if l[j] > L + d: armed = True
            j += 1; continue
        if l[j] <= L + d:                                        # 접근
            res = 0; m = j; pen = 0.0
            while m < n and m - j <= TO:
                Lm = a + s * (m - ia)
                pen = max(pen, (Lm - l[m]) / d)
                if c[m] < Lm - d: res = -1; break
                if m > j and h[m] > Lm + B: res = 1; break
                m += 1
            m = min(m, n - 1)
            if e < out.shape[0]:
                out[e, 0] = j; out[e, 1] = res; out[e, 2] = m; out[e, 3] = nb; out[e, 4] = pen; e += 1
            if res == -1: return e, m
            if res == 1: nb += 1
            armed = False; j = m + 1; continue
        j += 1
    return e, j


def pivots(x, k, low):
    n = len(x); p = np.zeros(n, bool)
    for i in range(k, n - k):
        w = x[i - k:i + k + 1]
        p[i] = x[i] == (w.min() if low else w.max())
    return p


def bars(sym, tf):
    k = store.load("klines_1m", sym).set_index("ts")
    g = k.resample(tf).agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    lg = np.log(g)
    tr = pd.concat([lg.high - lg.low, (lg.high - lg.close.shift()).abs(), (lg.low - lg.close.shift()).abs()], axis=1).max(axis=1)
    lg["atr"] = tr.rolling(14).mean()
    return lg.dropna()


def lines_support(h, l, c, atr, kind, rng):
    """지지선 후보 생성 → (i0 사용시작, iend, ia 기준봉, a, s, d, n0). 저항은 호출 쪽에서 부호 반전."""
    n = len(c); pl = np.where(pivots(l, K, True))[0]; out = []
    if kind == "H":
        for p in pl:
            i0 = p + K
            if i0 >= n: continue
            out.append((i0, min(n, i0 + AGE), p, l[p], 0.0, DLT * atr[p], 1))
    else:
        for q, p2 in enumerate(pl):
            i0 = p2 + K
            if i0 >= n: continue
            best = None
            for p1 in pl[:q][::-1]:
                if p2 - p1 > WIN: break
                if p2 - p1 < 10: continue
                s = (l[p2] - l[p1]) / (p2 - p1)
                if abs(s) > SLOPE * atr[p2]: continue
                seg = l[p1:p2 + 1] - (l[p1] + s * np.arange(p2 - p1 + 1))
                if seg.min() < -0.1 * atr[p2]: continue
                best = (p1, s, seg)                              # 더 먼 p1이 오면 덮어씀 → 가장 긴 선
            if best is None: continue
            p1, s, seg = best
            mid = [p for p in pl[q - 1::-1] if p1 < p < p2]
            n0 = 2 + sum(1 for p in mid if seg[p - p1] < DLT * atr[p2])
            out.append((i0, min(n, i0 + min(AGE, 2 * (p2 - p1) + 50)), p2, l[p2], s, DLT * atr[p2], n0))
    return out


def run_lines(Ls, h, l, c, placebo, rng, B_atr):
    rec = []; buf = np.zeros((MAXE, 5)); kept = []
    alive = []                                                   # (i0, death, level_at_fn) 중복 제거용
    for (i0, iend, ia, a, s, d, n0) in Ls:
        B = B_atr * d / DLT
        if placebo:
            u = rng.uniform(0.5, 1.5) * (d / DLT) * rng.choice([-1, 1])
            a2 = a + u
            if a2 + s * (i0 - ia) >= c[i0] - d: a2 = a - abs(u)  # 생성 시 가격 아래쪽 유지
            a = a2
        e, death = scan(i0, iend, ia, a, s, d, B, h, l, c, TO, buf)
        if not placebo:
            L0 = a + s * (i0 - ia)
            dup = any(j0 < i0 < dj and abs((aa + ss * (i0 - iaa)) - L0) < d for j0, dj, iaa, aa, ss in alive)
            alive = [z for z in alive if z[1] > i0] + [(i0, death, ia, a, s)]
            if dup: continue
            kept.append((i0, iend, ia, a, s, d, n0))
        for x in buf[:e]:
            j, res, m, nb, pen = int(x[0]), int(x[1]), int(x[2]), int(x[3]), x[4]
            fw = [(c[m] - c[min(m + f, len(c) - 1)]) if (res == -1 and m + f < len(c)) else np.nan for f in FWD]
            rec.append((j, res, m, nb + n0, pen, *fw, i0, ia, s / (d / DLT)))
    return rec, kept


if __name__ == "__main__":
    rng = np.random.default_rng(0); allr = []
    for tf in ("15min", "1h", "4h"):
        for sym in ("BTCUSDT", "ETHUSDT"):
            g = bars(sym, tf); ix = g.index
            for side, sg in (("지지", 1), ("저항", -1)):
                h, l, c = (g.high.values, g.low.values, g.close.values) if sg == 1 else (-g.low.values, -g.high.values, -g.close.values)
                atr = g.atr.values
                for kind in ("H", "T"):
                    Ls = lines_support(h, l, c, atr, kind, rng)
                    for plc in (False, True):
                        r, kept = run_lines(Ls, h, l, c, plc, rng, BB)
                        if not plc: Ls = kept                    # 위약은 중복 제거 후 남은 실제 선에서만
                        df = pd.DataFrame(r, columns=["j", "res", "m", "nprev", "pen", "f4", "f12", "f48", "i0", "ia", "slope"])
                        df["t"] = ix[df.j.values]; df["tm"] = ix[df.m.values]; df["atr"] = atr[df.j.values]
                        df = df.assign(tf=tf, sym=sym, side=side, kind=kind, plc=plc)
                        allr.append(df)
                    P(f"{tf} {sym} {side} {kind}: 선(중복 제거 후) {len(Ls):,} 사건 {len(allr[-2]):,} (위약 {len(allr[-1]):,})")
    E = pd.concat(allr, ignore_index=True)
    E.to_parquet(R / "data" / "r62_events.parquet", index=False)

    def tab(E, title):
        P(f"\n===== {title}")
        P("  p(반등|이전 반등 수) [결판 사건만] — 실제 / 위약 (n)")
        for (tf, kind), G in E.groupby(["tf", "kind"], sort=False):
            parts = []
            for nb in (1, 2, 3, 4, 5):
                for plc in (False, True):
                    x = G[(G.plc == plc) & (G.res != 0) & ((G.nprev == nb) if nb < 5 else (G.nprev >= 5))]
                    parts.append(f"{(x.res == 1).mean():.3f}({len(x)})" if len(x) >= 30 else "  -  ")
            P(f"  {tf:5s} {kind} " + " | ".join(f"n={nb}{'+' if nb == 5 else ''}: {parts[2*i]} / {parts[2*i+1]}" for i, nb in enumerate((1, 2, 3, 4, 5))))
        P("  돌파 뒤 돌파 방향 수익(bp, 4/12/48봉) — 실제 / 위약 · 터치 수별(실제)")
        for (tf, kind), G in E.groupby(["tf", "kind"], sort=False):
            br = G[G.res == -1]; a = br[~br.plc]; b = br[br.plc]
            f = lambda x: " ".join(f"{x[c].mean()*1e4:+6.1f}" for c in ("f4", "f12", "f48"))
            by = " ".join(f"n{nb}{'+' if nb == 4 else ''}:{a[(a.nprev == nb) if nb < 4 else (a.nprev >= 4)].f12.mean()*1e4:+.1f}({((a.nprev == nb) if nb < 4 else (a.nprev >= 4)).sum()})"
                          for nb in (1, 2, 3, 4))
            P(f"  {tf:5s} {kind} 실제 [{f(a)}] n={len(a):,} / 위약 [{f(b)}] n={len(b):,} | 12봉 by 터치 {by}")

    tab(E, "전 기간 2020~2026")
    tab(E[E.t >= pd.Timestamp("2023-01-01", tz="UTC")], "2023~ (호가 결합 구간)")
    P("\n  결판 비율·미결 비율 (실제/위약): " + " ".join(
        f"{tf}{k}:{(E[(E.tf==tf)&(E.kind==k)&~E.plc].res==0).mean():.2f}/{(E[(E.tf==tf)&(E.kind==k)&E.plc].res==0).mean():.2f}"
        for tf in ("15min", "1h", "4h") for k in ("H", "T")))
    (R / "logs" / "r62.summary").write_text("R62 선·사건 카탈로그 완료. 세부 logs/r62.log\n")
