"""미국 예정 경제지표 발표 시각 달력(2020~) → data/ext/macro_calendar.parquet (R133).

발표일: ALFRED(세인트루이스 연준) release/downloaddates — 실제 발표일 기록(정부 셧다운 지연 반영).
발표 시각(미국 동부, 서머타임 자동 반영 → UTC): 지표별 공식 정시. FOMC 성명 14:00, 기자회견 14:30(2021~ 매 회의).
  ※ ALFRED 날짜 목록은 '그 발표에 속한 시계열이 개정된 날'이라 정규 발표 외 날짜가 섞일 수 있다 → 월간 지표는 월 1회만(그 달 첫 날짜) 남긴다.
사용: python scripts/get_macro_calendar.py"""
import sys, re
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd, requests
R = Path(__file__).resolve().parents[1]
OUT = R / "data" / "ext" / "macro_calendar.parquet"
ET = ZoneInfo("America/New_York")
# (이름, ALFRED release id, 동부 시각, 월간 여부)
REL = [("CPI", 10, "08:30", True), ("PPI", 46, "08:30", True), ("NFP", 50, "08:30", True), ("CLAIMS", 180, "08:30", False),
       ("RETAIL", 9, "08:30", True), ("GDP", 53, "08:30", True), ("PCE", 54, "08:30", True), ("JOLTS", 192, "10:00", True)]
FOMC = """2020-01-29 2020-03-03 2020-03-15 2020-04-29 2020-06-10 2020-07-29 2020-09-16 2020-11-05 2020-12-16
2021-01-27 2021-03-17 2021-04-28 2021-06-16 2021-07-28 2021-09-22 2021-11-03 2021-12-15
2022-01-26 2022-03-16 2022-05-04 2022-06-15 2022-07-27 2022-09-21 2022-11-02 2022-12-14
2023-02-01 2023-03-22 2023-05-03 2023-06-14 2023-07-26 2023-09-20 2023-11-01 2023-12-13
2024-01-31 2024-03-20 2024-05-01 2024-06-12 2024-07-31 2024-09-18 2024-11-07 2024-12-18
2025-01-29 2025-03-19 2025-05-07 2025-06-18 2025-07-30 2025-09-17 2025-10-29 2025-12-10
2026-01-28 2026-03-18 2026-04-29 2026-06-17 2026-07-29 2026-09-16""".split()
FOMC_IRREG = {"2020-03-03": "10:00", "2020-03-15": "17:00"}          # 2020년 긴급 인하(정시 아님)


def alfred_dates(rid):
    t = requests.get(f"https://alfred.stlouisfed.org/release/downloaddates?rid={rid}&ff=txt", timeout=60).text
    body = t.split("Release Dates:")[1]
    return [d for d in re.findall(r"\d{4}-\d{2}-\d{2}", body) if d >= "2020-01-01"]


def to_utc(d, hm):
    return pd.Timestamp(f"{d} {hm}").tz_localize(ET).tz_convert("UTC")


if __name__ == "__main__":
    rows = []
    for name, rid, hm, monthly in REL:
        ds = sorted(set(alfred_dates(rid)))
        ds = [d for d in ds if pd.Timestamp(d).dayofweek < 5]
        if monthly:
            s = pd.Series(ds); ds = s.groupby(s.str[:7]).first().tolist()
        rows += [dict(ev=name, ts=to_utc(d, hm)) for d in ds]
        print(f"{name}: {len(ds)}개 ({ds[0]} ~ {ds[-1]})", flush=True)
    for d in FOMC:
        hm = FOMC_IRREG.get(d, "14:00"); rows.append(dict(ev="FOMC", ts=to_utc(d, hm)))
        if d not in FOMC_IRREG and d >= "2021-01-01": rows.append(dict(ev="FOMC_PC", ts=to_utc(d, "14:30")))
    C = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    C.to_parquet(OUT, index=False)
    print(C.groupby("ev").size().to_string()); print(f"저장 {OUT} ({len(C)}행)")
