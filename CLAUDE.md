# freesia — BTC/ETH 퀀트 자동매매 (Binance USDⓈ-M 선물)

기억(왜·어디까지)은 볼트 `~/obsidian-vault/1 Project/freesia/` (MOC → STATE 순으로 읽는다).
이 레포에는 코드만 둔다.

- 편집은 Odroid, **데이터 수집·백테스트·거래 실행은 oracle1** (`git push oracle main`으로 배포).
- SSH는 반드시 alias `oracle1`.
- 데이터는 `data/`(gitignore). 수집 결과는 `scripts/validate.py`가 스스로 신선도·결손을 검증한다.
  "수집 성공" 로그를 믿지 말고 validate 리포트를 본다.
- 모든 비용(수수료·펀딩·청산)은 `freesia/costs.py` 한 곳에서만 정의한다.
