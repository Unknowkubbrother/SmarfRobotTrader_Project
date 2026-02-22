# Profitable 3-window validation

Train window:
- 2020-01-01 00:00:00 -> 2025-12-31 23:59:59

Default gate settings used:
- OPEN_PROB_THRESHOLD=0.86
- OPEN_EDGE_THRESHOLD=0.16
- MIN_ACTION_MARGIN=0.20
- HOLD_EDGE_THRESHOLD=0.08
- TRADE_COOLDOWN_BARS=10
- ADAPTIVE_GATE=1
- EMBED_TEST_MODE=knn_map

Validation results:
1) 2025-01-01 -> 2025-12-31
- Final Equity: 11253.20
- Return: +12.53%
- Trades: 294

2) 2026-01-01 -> 2026-02-20
- Final Equity: 10023.40
- Return: +0.23%
- Trades: 29

3) 2026-02-01 -> 2026-02-20
- Final Equity: 10010.00
- Return: +0.10%
- Trades: 7
