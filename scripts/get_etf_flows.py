"""R118d 데이터 — 미국 현물 비트코인·이더리움 ETF 일별 순유입액(Total, 백만 달러) 수집.

1순위 출처: Farside Investors 공식 표
  https://farside.co.uk/bitcoin-etf-flow-all-data/ , https://farside.co.uk/ethereum-etf-flow-all-data/
  2026-10-01 확인: oracle1·Odroid 양쪽에서 브라우저 User-Agent로도 HTTP 403(Cloudflare 챌린지 "Just a moment...")
  → 직접 수집 불가. 아래 2순위로 자동 대체.
2순위(실사용) 출처: 같은 Farside 표를 30분마다 그대로 미러링하는 공개 GitHub 저장소
  https://github.com/haturatu/crypto-etf-flow (헤드리스 크롬으로 Farside HTML을 파싱해 git push, README 확인)
  원본 CSV: raw.githubusercontent.com/haturatu/crypto-etf-flow/main/etf_btc.csv, etf_eth.csv
  raw.githubusercontent.com은 Cloudflare 차단이 없어 oracle1에서 직접 200 OK 확인(2026-10-01).
각 행 = Farside 기준 그날(미국 거래일, 날짜 라벨만 있고 시간대 표기 없음) 전 펀드 합산 Total(USD 백만).
이 Total을 그대로 btc_total/eth_total로 쓴다. date는 그 날짜 라벨(그 날 00:00 UTC로 저장) — 이 값은 미국 장
마감(약 21:00 UTC) 이후에야 알려지므로, 연구에서는 date D의 값을 D+1 00:00 UTC부터 사용해야 한다(당일 사용 = 룩어헤드).
ETH CSV 첫 행 "Seed"(상장 전 시드 자금, 날짜 아님)는 제외한다.
저장: data/ext/etf_flows.parquet (date, btc_total, eth_total; 해당 자산이 아직 상장 전이면 NaN)
사용: python scripts/get_etf_flows.py
"""
import sys
from io import StringIO
from pathlib import Path
import numpy as np, pandas as pd, requests

OUT = Path(__file__).resolve().parents[1] / "data" / "ext" / "etf_flows.parquet"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
FARSIDE = dict(btc="https://farside.co.uk/bitcoin-etf-flow-all-data/", eth="https://farside.co.uk/ethereum-etf-flow-all-data/")
MIRROR = dict(btc="https://raw.githubusercontent.com/haturatu/crypto-etf-flow/main/etf_btc.csv",
              eth="https://raw.githubusercontent.com/haturatu/crypto-etf-flow/main/etf_eth.csv")
# 알려진 값과 대조할 스팟체크(상장 첫날, 널리 보도된 수치)
SPOT = dict(btc=[("2024-01-11", 655.3)], eth=[("2024-07-23", 106.6)])


def from_farside(kind):
    """Farside 표 직접 긁기(1순위). 실패(차단·구조 변경 등)하면 None."""
    try:
        r = requests.get(FARSIDE[kind], headers={"User-Agent": UA}, timeout=20)
        if r.status_code != 200 or "Total" not in r.text:
            return None
        tabs = pd.read_html(StringIO(r.text))
        for t in tabs:
            cols = [str(c) for c in t.columns]
            if "Total" in cols and len(t) > 50:
                df = t.rename(columns={t.columns[0]: "Date"})
                dt = pd.to_datetime(df.Date.astype(str), format="%d %b %Y", errors="coerce")
                tot = pd.to_numeric(df.Total.astype(str).str.replace(",", "", regex=False)
                                     .str.replace("(", "-", regex=False).str.replace(")", "", regex=False), errors="coerce")
                out = pd.DataFrame({"date": dt, "total": tot}).dropna(subset=["date"])
                if len(out) > 50:
                    return out
    except Exception as e:
        print(f"  {kind}: Farside 직접 수집 예외 ({type(e).__name__}: {e})", flush=True)
    return None


def from_mirror(kind):
    """GitHub 미러 CSV(2순위, 실사용 경로)."""
    r = requests.get(MIRROR[kind], headers={"User-Agent": UA}, timeout=20)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    dt = pd.to_datetime(df.Date.astype(str), format="%d %b %Y", errors="coerce")  # "Seed" 등 날짜 아닌 행 → NaT
    out = pd.DataFrame({"date": dt, "total": pd.to_numeric(df.Total, errors="coerce")}).dropna(subset=["date"])
    return out


def load_total(kind):
    out = from_farside(kind)
    if out is not None:
        print(f"  {kind}: Farside 직접 수집 성공 ({len(out)}행)", flush=True)
        return out, "farside direct"
    out = from_mirror(kind)
    print(f"  {kind}: Farside 직접 수집 실패(403/Cloudflare로 추정) → GitHub 미러로 대체, "
          f"{len(out)}행 {out.date.min():%Y-%m-%d}~{out.date.max():%Y-%m-%d}", flush=True)
    return out, "github mirror (haturatu/crypto-etf-flow, Farside 원본 미러)"


if __name__ == "__main__":
    btc, src_b = load_total("btc")
    eth, src_e = load_total("eth")
    D = pd.merge(btc.rename(columns={"total": "btc_total"}), eth.rename(columns={"total": "eth_total"}), on="date", how="outer")
    D = D.sort_values("date").reset_index(drop=True)
    D["date"] = pd.DatetimeIndex(D.date).tz_localize("UTC").normalize()

    assert D.date.is_unique, "중복 날짜 발견"
    b0 = D.loc[D.btc_total.notna(), "date"].min(); e0 = D.loc[D.eth_total.notna(), "date"].min()
    assert b0 == pd.Timestamp("2024-01-11", tz="UTC"), f"BTC 시작일 불일치: {b0}"
    assert e0 == pd.Timestamp("2024-07-23", tz="UTC"), f"ETH 시작일 불일치: {e0}"

    # 스팟체크
    ok = True
    for kind, col in (("btc", "btc_total"), ("eth", "eth_total")):
        for ds, exp in SPOT[kind]:
            v = D.loc[D.date == pd.Timestamp(ds, tz="UTC"), col]
            got = v.iloc[0] if len(v) else np.nan
            hit = np.isfinite(got) and abs(got - exp) < 1.0
            ok &= bool(hit)
            print(f"  스팟체크 {kind} {ds}: 수집값 {got:+.1f} vs 보도값 {exp:+.1f} {'OK' if hit else '불일치!'}", flush=True)
    if not ok:
        print("  경고: 스팟체크 불일치 — 소스·파싱을 다시 확인할 것", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    D.to_parquet(OUT, index=False)
    print(f"저장 {OUT} {D.shape}")
    print(f"  BTC 출처: {src_b} | {D.btc_total.notna().sum()}일 {D.loc[D.btc_total.notna(),'date'].min():%Y-%m-%d}~{D.loc[D.btc_total.notna(),'date'].max():%Y-%m-%d}")
    print(f"  ETH 출처: {src_e} | {D.eth_total.notna().sum()}일 {D.loc[D.eth_total.notna(),'date'].min():%Y-%m-%d}~{D.loc[D.eth_total.notna(),'date'].max():%Y-%m-%d}")
