"""Binance USDⓈ-M 선물 비용·청산 모델 — 비용 상수의 단일 출처.

출처(2026-09-19 확인):
- 수수료 VIP0 maker 0.02% / taker 0.05%, BNB 결제 시 10% 할인 (Binance 지원 문서, 2026-05-01 갱신본)
- 펀딩: BTCUSDT·ETHUSDT 8시간 주기, 상·하한 ±0.30% (fapi/v1/fundingInfo)
- 청산 수수료 1.25% (fapi/v1/exchangeInfo liquidationFee=0.0125)
- 유지증거금률 tier1 0.40% 는 2차 출처 → 미검증. API 키 준비 후
  /fapi/v1/leverageBracket 으로 확인해 MMR_TIER1 을 확정할 것.
"""
from dataclasses import dataclass

MAKER = 0.0002
TAKER = 0.0005
BNB_DISCOUNT = 0.10
LIQUIDATION_FEE = 0.0125  # 청산 시 유지증거금 위에 추가로 잃음(보험기금 귀속)
MMR_TIER1 = 0.004         # 미검증 (위 주석)


@dataclass(frozen=True)
class Fees:
    maker: float = MAKER
    taker: float = TAKER

    @classmethod
    def with_bnb(cls) -> "Fees":
        k = 1 - BNB_DISCOUNT
        return cls(MAKER * k, TAKER * k)


def round_trip_cost(fees: Fees, entry: str = "taker", exit_: str = "taker") -> float:
    """명목가치 대비 왕복 수수료 비율 (슬리피지·펀딩 제외)."""
    return getattr(fees, entry) + getattr(fees, exit_)


def liquidation_price(entry: float, leverage: float, side: str,
                      mmr: float = MMR_TIER1, cum: float = 0.0, qty: float = 1.0) -> float:
    """격리마진 청산가 (마크가격 기준, 진입 수수료 제외한 이상적 값).

    유지증거금 = 명목가치*mmr - cum. 격리증거금 WB = 진입명목/leverage.
    long : qty*(P-E) + WB = qty*P*mmr - cum
    short: qty*(E-P) + WB = qty*P*mmr - cum
    """
    wb = qty * entry / leverage
    if side == "long":
        return (qty * entry - wb - cum) / (qty * (1 - mmr))
    if side == "short":
        return (qty * entry + wb + cum) / (qty * (1 + mmr))
    raise ValueError(side)


def liquidation_distance(leverage: float, mmr: float = MMR_TIER1) -> float:
    """진입가 대비 청산까지 불리하게 움직여야 하는 비율(long 기준)."""
    return 1 - liquidation_price(1.0, leverage, "long", mmr)
