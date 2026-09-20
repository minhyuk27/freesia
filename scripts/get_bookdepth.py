"""bookDepth(±1~5% 호가 잔량) 수집. data.binance.vision 일별, 체크섬 검증.
사용: python scripts/get_bookdepth.py [시작YYYY-MM-DD] [종료YYYY-MM-DD]"""
import sys, io, zipfile, hashlib, time
from pathlib import Path
import pandas as pd, requests
R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from freesia import store
BASE = "https://data.binance.vision/data/futures/um/daily/bookDepth"
S = requests.Session()
SYM = ("BTCUSDT", "ETHUSDT")


def get(url, retries=4):
    for i in range(retries):
        try:
            r = S.get(url, timeout=90)
            if r.status_code == 404: return None
            r.raise_for_status(); return r.content
        except requests.RequestException:
            time.sleep(1.5*(i+1))
    return None


def fetch_day(sym, day):
    u = f"{BASE}/{sym}/{sym}-bookDepth-{day}.zip"
    b = get(u)
    if b is None: return None
    ck = get(u + ".CHECKSUM")
    if ck is not None:
        want = ck.decode().split()[0].lower()
        if hashlib.sha256(b).hexdigest() != want:
            print(f"  체크섬 불일치 {sym} {day}", flush=True); return None
    with zipfile.ZipFile(io.BytesIO(b)) as z:
        df = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])))
    df["ts"] = pd.to_datetime(df.timestamp, utc=True)
    # 넓은 형식으로: 각 시점별 ±1~5% 잔량
    df = df.drop_duplicates(subset=["ts", "percentage"], keep="last")
    w = df.pivot(index="ts", columns="percentage", values="notional")
    w.columns = [f"n{int(c):+d}" for c in w.columns]
    d2 = df.pivot(index="ts", columns="percentage", values="depth")
    d2.columns = [f"d{int(c):+d}" for c in d2.columns]
    out = pd.concat([w, d2], axis=1).reset_index()
    out = out.loc[:, ~out.columns.duplicated()]
    return out


def main():
    a = sys.argv[1] if len(sys.argv) > 1 else "2023-01-01"
    b = sys.argv[2] if len(sys.argv) > 2 else pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    days = pd.date_range(a, b, freq="D")
    for sym in SYM:
        buf = []; cur_ym = None; got = 0
        for d in days:
            ym = d.strftime("%Y-%m")
            if cur_ym is None: cur_ym = ym
            if ym != cur_ym and buf:
                df = pd.concat(buf).sort_values("ts").reset_index(drop=True)
                n = store.save("bookdepth", sym, cur_ym, df, partial=False)
                print(f"  {sym} {cur_ym} 저장 {n:,}행", flush=True)
                buf = []; cur_ym = ym
            x = fetch_day(sym, d.strftime("%Y-%m-%d"))
            if x is not None: buf.append(x); got += 1
        if buf:
            df = pd.concat(buf).sort_values("ts").reset_index(drop=True)
            n = store.save("bookdepth", sym, cur_ym, df, partial=True)
            print(f"  {sym} {cur_ym} 저장 {n:,}행", flush=True)
        print(f"{sym} 완료: {got}일", flush=True)


if __name__ == "__main__":
    main()
