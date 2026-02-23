# Final default (CLS + Autoencoder projector, tuned gate)

Train window:
- 2020-01-01 00:00:00 -> 2025-12-31 23:59:59

Default source/projector:
- EMBED_SOURCE_MODE=cls
- EMBED_PROJECTOR_MODE=autoencoder

Default test gate:
- EMBED_TEST_MODE=knn_map
- ADAPTIVE_GATE=1
- OPEN_PROB_THRESHOLD=0.83
- OPEN_EDGE_THRESHOLD=0.14
- MIN_ACTION_MARGIN=0.24
- HOLD_EDGE_THRESHOLD=0.10
- TRADE_COOLDOWN_BARS=6
- EMBED_QUALITY_MIN=0.25

Validation (all positive):
1) 2025-01-01 -> 2025-12-31
- Final Equity: 12812.60
- Return: +28.13%
- Trades: 422

2) 2026-01-01 -> 2026-02-20
- Final Equity: 10064.00
- Return: +0.64%
- Trades: 36

3) 2026-02-01 -> 2026-02-20
- Final Equity: 10200.60
- Return: +2.01%
- Trades: 9
