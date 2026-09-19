import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freesia.bt import core
for s in ("BTCUSDT", "ETHUSDT"):
    for t in ("1h", "4h", "1d"):
        core.bars(s, t)
print("bars ok")
