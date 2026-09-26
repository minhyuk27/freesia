"""위키백과(영어) 문서 일간 조회수 수집 — 관심도 보조 지표. Wikimedia REST API, 사람 조회(user)만.

값은 그날 하루 전체 합계라 연구에서는 다음 날부터 사용해야 한다.
저장: data/ext/wiki.parquet (date, 문서별 열)
사용: python scripts/get_wiki.py"""
from pathlib import Path
import pandas as pd, requests

ART = ["Bitcoin", "Ethereum", "Cryptocurrency"]
OUT = Path(__file__).resolve().parents[1] / "data" / "ext" / "wiki.parquet"
UA = {"User-Agent": "freesia-research/1.0 (personal research)"}

if __name__ == "__main__":
    end = (pd.Timestamp.utcnow() - pd.Timedelta(days=1)).strftime("%Y%m%d")
    cols = {}
    for a in ART:
        u = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/{a}/daily/20191201/{end}"
        js = requests.get(u, headers=UA, timeout=30).json()["items"]
        s = pd.Series({pd.Timestamp(x["timestamp"][:8]): float(x["views"]) for x in js}); cols[a] = s
        print(f"  {a}: {len(s)}일")
    D = pd.DataFrame(cols); D.index.name = "date"
    OUT.parent.mkdir(parents=True, exist_ok=True); D.reset_index().to_parquet(OUT, index=False)
    print(f"저장 {OUT} {D.shape}")
