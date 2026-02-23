# RL Folder Layout

## Structure
- `train/`
  - `train_ppo.py` (training entrypoint)
- `test/`
  - `test_ppo.py` (backtest entrypoint)
  - `backtest_config.py`, `backtest_semantic.py`, `backtest_features.py`, `backtest_bridge.py`
- `models/`
  - `ppo_trading.zip`
  - `vec_normalize.pkl`
  - `semantic_map.joblib`
  - `embedding_projector_meta.joblib`, `embedding_projector.pt`
  - `chroma_db/`
- `datasets/`
  - `h1_ohlc_delta.csv`
  - `h1_ohlc_delta1.csv`
  - `prepareData.py`
- `core/`
  - `env_trading.py`
  - `semantic_embedding.py`
  - `embedding_projector.py`
  - `chroma_client.py`

## Runtime Flow
1. `train/train_ppo.py` reads data from `datasets/`.
2. Missing embedding is filled by semantic fallback (`knn_map` only).
3. Embedding source is fixed to `cls` and reduced by `autoencoder` to `sem_pca_*` features.
4. PPO model is trained and saved to `models/`.
5. `test/test_ppo.py` loads model + semantic artifacts from `models/` and runs bar-by-bar backtest.

## Commands
- Train:
  - `python train/train_ppo.py`
- Backtest:
  - `python test/test_ppo.py`

## Notes
- `SEM_LATENT_DIM` controls number of semantic latent features (default 16).
- `TEST_DATA_FILE` should be just filename inside `datasets/` (e.g. `h1_ohlc_delta1.csv`).
