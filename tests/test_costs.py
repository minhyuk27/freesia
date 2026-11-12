import pytest
from freesia.costs import Fees, round_trip_cost, liquidation_price

def test_fees_default():
    fees = Fees()
    assert fees.maker == 0.0002
    assert fees.taker == 0.0005
    assert round_trip_cost(fees, "maker", "taker") == pytest.approx(0.0007)

def test_fees_with_bnb():
    fees = Fees.with_bnb()
    assert fees.maker == pytest.approx(0.0002 * 0.9)
    assert fees.taker == pytest.approx(0.0005 * 0.9)

def test_liquidation_price_long():
    entry = 50000.0
    leverage = 10.0
    liq_p = liquidation_price(entry, leverage, "long", mmr=0.004)
    # With 10x leverage, liquidation is below entry price
    assert liq_p < entry
    assert liq_p > 40000.0

def test_liquidation_price_short():
    entry = 50000.0
    leverage = 10.0
    liq_p = liquidation_price(entry, leverage, "short", mmr=0.004)
    # With 10x leverage, liquidation is above entry price
    assert liq_p > entry
    assert liq_p < 60000.0
