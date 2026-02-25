import os
import random
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RL_ROOT = os.path.dirname(SCRIPT_DIR)
CORE_DIR = os.path.join(RL_ROOT, "core")
DATASETS_DIR = os.path.join(RL_ROOT, "datasets")
MODELS_DIR = os.path.join(RL_ROOT, "models")
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)
os.makedirs(MODELS_DIR, exist_ok=True)

from embedding_source import load_time_to_embedding_map, normalize_source_mode
from embedding_projector import fit_and_save_projector
from env_trading import TradingEnv
from risk_lot import calc_auto_lot, normalize_risk_level, resolve_risk_percent
from semantic_embedding import build_semantic_map, compute_regime_frame, resolve_from_semantic_map


MODEL_SAVE_PATH = os.getenv("TRAIN_MODEL_PATH", os.path.join(MODELS_DIR, "ppo_trading")).strip() or os.path.join(
    MODELS_DIR, "ppo_trading"
)
VEC_SAVE_PATH = os.getenv("TRAIN_VEC_PATH", os.path.join(MODELS_DIR, "vec_normalize.pkl")).strip() or os.path.join(
    MODELS_DIR, "vec_normalize.pkl"
)


def _sb3_zip_path(path: str) -> str:
    return path if str(path).lower().endswith(".zip") else f"{path}.zip"


@dataclass(frozen=True)
class TrainConfig:
    sem_latent_dim: int
    projector_mode: str
    embed_source_mode: str
    random_seed: int
    ae_hidden_dim: int
    ae_epochs: int
    ae_batch_size: int
    ae_lr: float
    ae_weight_decay: float
    total_timesteps: int
    recent_bias: float
    recent_lookback: int
    min_episode_bars: int
    max_open_orders: int
    initial_balance: float
    lot_size: float
    lot_risk_pips: float
    risk_level: str
    risk_percent: float
    lot_source: str
    resume: bool
    deterministic: bool
    device: str


def _build_config() -> TrainConfig:
    sem_latent_dim = int(os.getenv("SEM_LATENT_DIM", "16"))
    projector_mode = "autoencoder"
    embed_source_mode = normalize_source_mode("cls")
    random_seed = int(os.getenv("TRAIN_RANDOM_SEED", "11"))
    ae_hidden_dim = int(os.getenv("TRAIN_AE_HIDDEN_DIM", "256"))
    ae_epochs = int(os.getenv("TRAIN_AE_EPOCHS", "8"))
    ae_batch_size = int(os.getenv("TRAIN_AE_BATCH_SIZE", "256"))
    ae_lr = float(os.getenv("TRAIN_AE_LR", "0.001"))
    ae_weight_decay = float(os.getenv("TRAIN_AE_WEIGHT_DECAY", "0.00001"))
    total_timesteps = int(os.getenv("TRAIN_TIMESTEPS", "80000"))
    recent_bias = float(os.getenv("TRAIN_RECENT_BIAS", "0.65"))
    recent_lookback = int(os.getenv("TRAIN_RECENT_LOOKBACK", "1200"))
    min_episode_bars = int(os.getenv("TRAIN_MIN_EPISODE_BARS", "250"))
    max_open_orders = int(os.getenv("TRAIN_MAX_OPEN_ORDERS", os.getenv("MAX_OPEN_ORDERS", "1")))
    initial_balance = float(os.getenv("TRAIN_INITIAL_BALANCE", "10000"))
    lot_risk_pips = float(os.getenv("LOT_RISK_PIPS", "50"))
    risk_percent_low = float(os.getenv("RISK_PERCENT_LOW", "0.5"))
    risk_percent_medium = float(os.getenv("RISK_PERCENT_MEDIUM", "1.0"))
    risk_percent_high = float(os.getenv("RISK_PERCENT_HIGH", "2.0"))
    risk_level_raw = os.getenv("TRAIN_RISK_LEVEL", os.getenv("LIVE_RISK_LEVEL", os.getenv("RISK_LEVEL", "2")))
    risk_level = normalize_risk_level(risk_level_raw)
    risk_pct_override_raw = os.getenv("TRAIN_RISK_PERCENT", "").strip()
    risk_pct_override = float(risk_pct_override_raw) if risk_pct_override_raw else None
    risk_percent = resolve_risk_percent(
        risk_level=risk_level,
        risk_pct=risk_pct_override,
        risk_percent_low=risk_percent_low,
        risk_percent_medium=risk_percent_medium,
        risk_percent_high=risk_percent_high,
    )
    train_lot_raw = os.getenv("TRAIN_LOT_SIZE", "").strip()
    if train_lot_raw:
        lot_size = float(train_lot_raw)
        lot_source = "manual(TRAIN_LOT_SIZE)"
    else:
        lot_size = calc_auto_lot(
            balance=initial_balance,
            risk_level=risk_level,
            risk_pct=risk_percent,
            lot_risk_pips=lot_risk_pips,
            pip_value_per_lot=10.0,
            risk_percent_low=risk_percent_low,
            risk_percent_medium=risk_percent_medium,
            risk_percent_high=risk_percent_high,
        )
        lot_source = f"auto(risk_level={risk_level}, risk_pct={risk_percent}%)"
    resume = os.getenv("TRAIN_RESUME", "0").strip().lower() not in {"0", "false", "no"}
    deterministic = os.getenv("TRAIN_DETERMINISTIC", "1").strip().lower() not in {"0", "false", "no"}
    device = os.getenv("TRAIN_DEVICE", "cpu").strip().lower() or "cpu"

    recent_bias = float(np.clip(recent_bias, 0.0, 1.0))

    return TrainConfig(
        sem_latent_dim=sem_latent_dim,
        projector_mode=projector_mode,
        embed_source_mode=embed_source_mode,
        random_seed=random_seed,
        ae_hidden_dim=ae_hidden_dim,
        ae_epochs=ae_epochs,
        ae_batch_size=ae_batch_size,
        ae_lr=ae_lr,
        ae_weight_decay=ae_weight_decay,
        total_timesteps=total_timesteps,
        recent_bias=recent_bias,
        recent_lookback=recent_lookback,
        min_episode_bars=min_episode_bars,
        max_open_orders=max(1, max_open_orders),
        initial_balance=initial_balance,
        lot_size=lot_size,
        lot_risk_pips=lot_risk_pips,
        risk_level=risk_level,
        risk_percent=risk_percent,
        lot_source=lot_source,
        resume=resume,
        deterministic=deterministic,
        device=device,
    )


def set_global_seed(seed: int, deterministic: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


class TradingMetricsCallback(BaseCallback):

    def __init__(self, verbose=0, initial_balance=10000.0):
        super().__init__(verbose)
        self.initial_balance = float(initial_balance)
        self.episode_equities = []
        self.episode_trades = []
        self.episode_wins = []
        self.episode_drawdowns = []
        self.episode_accuracies = []
        self.action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        self.total_actions = 0
        self.current_equity = self.initial_balance
        self.current_accuracy = 0
        self.step_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])
        actions = self.locals.get("actions", [])

        for i, info in enumerate(infos):
            if info:
                self.current_equity = info.get("equity", self.initial_balance)
                self.current_accuracy = info.get("accuracy", 0)
                if len(actions) > i:
                    action = int(actions[i])
                    self.action_counts[action] = self.action_counts.get(action, 0) + 1
                    self.total_actions += 1

        self.step_count += 1
        if self.step_count % 500 == 0:
            self._log_metrics()
        return True

    def _on_rollout_end(self) -> None:
        infos = self.locals.get("infos", [{}])
        for info in infos:
            if info:
                self.episode_equities.append(info.get("equity", self.initial_balance))
                self.episode_trades.append(info.get("trades", 0))
                self.episode_wins.append(info.get("wins", 0))
                self.episode_drawdowns.append(info.get("drawdown", 0))
                self.episode_accuracies.append(info.get("accuracy", 0))
        self._log_metrics()

    def _log_metrics(self):
        self.logger.record("trading/equity", self.current_equity)
        self.logger.record("trading/profit_loss", self.current_equity - self.initial_balance)
        self.logger.record("trading/return_pct", ((self.current_equity / max(self.initial_balance, 1e-9)) - 1) * 100)
        self.logger.record("trading/accuracy", self.current_accuracy)

        if self.total_actions > 0:
            hold_pct = (self.action_counts.get(0, 0) / self.total_actions) * 100
            buy_pct = (self.action_counts.get(1, 0) / self.total_actions) * 100
            sell_pct = (self.action_counts.get(2, 0) / self.total_actions) * 100
            close_pct = (self.action_counts.get(3, 0) / self.total_actions) * 100
            self.logger.record("actions/hold_pct", hold_pct)
            self.logger.record("actions/buy_pct", buy_pct)
            self.logger.record("actions/sell_pct", sell_pct)
            self.logger.record("actions/close_pct", close_pct)

        if self.episode_equities:
            self.logger.record("trading/avg_equity", np.mean(self.episode_equities[-100:]))
            self.logger.record("trading/max_equity", np.max(self.episode_equities[-100:]))

        if self.episode_trades:
            total_trades = sum(self.episode_trades[-100:])
            total_wins = sum(self.episode_wins[-100:])
            win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
            self.logger.record("trading/total_trades", total_trades)
            self.logger.record("trading/win_rate", win_rate)

        if self.episode_accuracies:
            self.logger.record("trading/avg_accuracy", np.mean(self.episode_accuracies[-100:]))

        if self.episode_drawdowns:
            self.logger.record("trading/max_drawdown", np.max(self.episode_drawdowns[-100:]) * 100)

def load_training_data() -> pd.DataFrame:
    base_file = os.getenv("TRAIN_DATA_FILE", "h1_ohlc_delta.csv").strip() or "h1_ohlc_delta.csv"
    train_date_from = os.getenv("TRAIN_DATE_FROM", "2025-01-01 00:00:00").strip()
    train_date_to = os.getenv("TRAIN_DATE_TO", "2025-12-31 23:00:00").strip()
    data_path = os.path.join(DATASETS_DIR, base_file)
    df = pd.read_csv(data_path)

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").drop_duplicates(subset=["time"], keep="last").reset_index(drop=True)
    df_full = df.copy()
    df = df[df["has_delta"] == 1].sort_values("time").reset_index(drop=True)
    if train_date_from:
        df = df[df["time"] >= pd.to_datetime(train_date_from)]
    if train_date_to:
        df = df[df["time"] <= pd.to_datetime(train_date_to)]
    df = df.reset_index(drop=True)
    df.drop(columns=["has_delta"], inplace=True)

    print("=" * 50)
    print(" DATA FILTERING (Recent Only)")
    print("=" * 50)
    print(f"ไฟล์ที่ใช้:         {os.path.basename(data_path)}")
    print(f"ช่วงเทรนที่เลือก:   from={train_date_from or '-'} to={train_date_to or '-'}")
    print(f"ข้อมูลทั้งหมด:     {len(df_full):,} rows")
    print(f"ข้อมูลที่ใช้ train: {len(df):,} rows (ตามช่วงที่เลือก, has_delta)")
    print(f"ช่วงเวลา:          {df['time'].iloc[0]} → {df['time'].iloc[-1]}")
    print("=" * 50 + "\n")
    return df


def _load_time_to_embedding(source_mode: str) -> tuple[dict, str]:
    return load_time_to_embedding_map(
        models_dir=MODELS_DIR,
        source_mode=source_mode,
        force_rebuild=False,
    )


def build_semantic_features(df: pd.DataFrame, cfg: TrainConfig) -> tuple[pd.DataFrame, int]:
    import joblib

    time_to_vec, resolved_source_mode = _load_time_to_embedding(cfg.embed_source_mode)
    df = df.copy()
    df["raw_embedding"] = df["time"].dt.strftime("%Y-%m-%d %H:%M:%S").map(time_to_vec)

    matched_count = int(df["raw_embedding"].notna().sum())
    missing_count = int(len(df) - matched_count)
    print(f" Embedding source mode: {resolved_source_mode}")
    print(f" Embedding overlap: {matched_count}/{len(df)} rows matched ChromaDB dates")
    if missing_count > 0:
        print(f" {missing_count} rows have no embedding — mode=knn_map")

    valid_mask = df["raw_embedding"].notna()
    semantic_map = build_semantic_map(df, raw_embedding_col="raw_embedding")
    semantic_map_path = os.path.join(MODELS_DIR, "semantic_map.joblib")
    joblib.dump(semantic_map, semantic_map_path)
    print(f" Saved semantic map: {semantic_map_path}")

    if matched_count > 0:
        regime_frame = compute_regime_frame(df)
        filled_embeddings = []
        for i, raw in enumerate(df["raw_embedding"].values):
            if isinstance(raw, (list, np.ndarray)):
                filled_embeddings.append(np.asarray(raw, dtype=np.float32))
            else:
                filled_embeddings.append(resolve_from_semantic_map(regime_frame.iloc[i], semantic_map))
        projector_source = np.vstack(filled_embeddings).astype(np.float32)
        source_mask = pd.Series(True, index=df.index)

        emb_dim = int(projector_source.shape[1])
        target_components = min(cfg.sem_latent_dim, emb_dim, len(projector_source))
        print(
            f" Fitting embedding projector mode={cfg.projector_mode} "
            f"({target_components}) from {emb_dim}-dim embeddings..."
        )
        _, projected_embeddings, projector_meta = fit_and_save_projector(
            embeddings=projector_source,
            artifact_dir=MODELS_DIR,
            mode=cfg.projector_mode,
            latent_dim=target_components,
            random_seed=cfg.random_seed,
            ae_hidden_dim=cfg.ae_hidden_dim,
            ae_epochs=cfg.ae_epochs,
            ae_batch_size=cfg.ae_batch_size,
            ae_lr=cfg.ae_lr,
            ae_weight_decay=cfg.ae_weight_decay,
        )
        actual_comps = int(projector_meta.get("latent_dim", target_components))
        print(f" Saved projector mode={projector_meta.get('mode', 'autoencoder')} latent_dim={actual_comps}")

        sem_cols = [f"sem_pca_{i + 1}" for i in range(actual_comps)]
        sem_df = pd.DataFrame(0.0, index=df.index, columns=sem_cols, dtype=np.float32)
        sem_df.loc[source_mask, sem_cols] = projected_embeddings.astype(np.float32)
        df = pd.concat([df, sem_df], axis=1)
    else:
        actual_comps = cfg.sem_latent_dim
        print(f" No valid embeddings found! Filling {actual_comps} semantic features with zeros.")
        sem_cols = [f"sem_pca_{i + 1}" for i in range(actual_comps)]
        for col in sem_cols:
            df[col] = 0.0

    df = df.drop(columns=["raw_embedding"])
    print(f" Semantic embeddings loaded as {actual_comps} semantic features.")
    print(f" Semantic feature columns: {', '.join(sem_cols)}")
    print("\n Data ready for training")
    print("=" * 50 + "\n")
    return df, actual_comps


def build_market_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    print(" Creating features...")
    df = df.copy()

    df["return"] = df["close"].pct_change().fillna(0)
    df["range"] = (df["high"] - df["low"]) / df["close"]
    df["raw_return"] = df["return"]

    full_range = df["high"] - df["low"]
    df["body_ratio"] = np.where(full_range > 0, abs(df["close"] - df["open"]) / full_range, 0)
    df["momentum"] = df["return"].rolling(window=5).sum().fillna(0)

    sma20 = df["close"].rolling(20).mean()
    sma50 = df["close"].rolling(50).mean()
    df["sma_cross"] = np.where(sma20 > sma50, 1, np.where(sma20 < sma50, -1, 0))
    df["sma_cross"] = df["sma_cross"].fillna(0)

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    df["rsi_norm"] = ((rsi - 50) / 50).fillna(0)

    tr = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            abs(df["high"] - df["close"].shift(1)),
            abs(df["low"] - df["close"].shift(1)),
        ),
    )
    df["atr_norm"] = (tr.rolling(14).mean() / df["close"]).fillna(0)

    df["trend"] = (sma20.pct_change(5) * 100).fillna(0)
    df["trend"] = df["trend"].clip(-2, 2)

    tr_adx = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            abs(df["high"] - df["close"].shift(1)),
            abs(df["low"] - df["close"].shift(1)),
        ),
    )
    plus_dm = np.where(
        (df["high"] - df["high"].shift(1)) > (df["low"].shift(1) - df["low"]),
        np.maximum(df["high"] - df["high"].shift(1), 0),
        0,
    )
    minus_dm = np.where(
        (df["low"].shift(1) - df["low"]) > (df["high"] - df["high"].shift(1)),
        np.maximum(df["low"].shift(1) - df["low"], 0),
        0,
    )
    atr14_adx = pd.Series(tr_adx).rolling(14).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(14).mean() / (atr14_adx + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).rolling(14).mean() / (atr14_adx + 1e-10)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_raw = dx.rolling(14).mean()
    df["adx"] = ((adx_raw - 25) / 25).fillna(0).clip(-1, 1)

    df = df.iloc[50:].reset_index(drop=True)

    base_cols = [
        "return",
        "range",
        "delta_tick",
        "delta_price",
        "body_ratio",
        "momentum",
        "sma_cross",
        "rsi_norm",
        "atr_norm",
        "trend",
        "adx",
    ]
    feature_list = base_cols + [c for c in df.columns if str(c).startswith("sem_pca_")]
    print(f" Features ({len(feature_list)}): {', '.join(feature_list)}")
    print("=" * 50 + "\n")
    return df, feature_list


def build_train_env(df: pd.DataFrame, cfg: TrainConfig):
    def _make_env():
        env = TradingEnv(
            df,
            random_start=True,
            initial_balance=cfg.initial_balance,
            lot_size=cfg.lot_size,
            max_open_orders=cfg.max_open_orders,
            recent_bias=cfg.recent_bias,
            recent_lookback=cfg.recent_lookback,
            min_episode_bars=cfg.min_episode_bars,
        )
        env.seed(cfg.random_seed)
        env.action_space.seed(cfg.random_seed)
        return env

    train_env = DummyVecEnv(
        [_make_env]
    )
    train_env.seed(cfg.random_seed)
    return VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)


def build_model(train_env, cfg: TrainConfig):
    model_path = _sb3_zip_path(MODEL_SAVE_PATH)
    if cfg.resume and os.path.exists(model_path):
        print(f" Resuming from existing model: {model_path}")
        return PPO.load(model_path, env=train_env)

    lr = float(os.getenv("PPO_LEARNING_RATE", "0.00008"))
    n_steps = int(os.getenv("PPO_N_STEPS", "1024"))
    batch_size = int(os.getenv("PPO_BATCH_SIZE", "256"))
    n_epochs = int(os.getenv("PPO_N_EPOCHS", "12"))
    gamma = float(os.getenv("PPO_GAMMA", "0.999"))
    gae_lambda = float(os.getenv("PPO_GAE_LAMBDA", "0.95"))
    clip_range = float(os.getenv("PPO_CLIP_RANGE", "0.2"))
    ent_coef = float(os.getenv("PPO_ENT_COEF", "0.0"))
    vf_coef = float(os.getenv("PPO_VF_COEF", "0.5"))
    max_grad_norm = float(os.getenv("PPO_MAX_GRAD_NORM", "0.5"))

    return PPO(
        "MlpPolicy",
        train_env,
        learning_rate=lr,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        seed=cfg.random_seed,
        device=cfg.device,
        policy_kwargs=dict(net_arch=[512, 256, 128]),
        verbose=1,
        tensorboard_log=os.path.join(SCRIPT_DIR, "tensorboard"),
    )


def print_training_banner(cfg: TrainConfig, feature_list: list[str]):
    print("\n" + "=" * 50)
    print(" Starting Training (Standard PPO Mode)")
    print("=" * 50)
    print(" TensorBoard: tensorboard --logdir=./tensorboard/")
    print(" No forced auto-close (RL handles close/hold) | RandomStart=ON")
    print(f" Multi-order mode: max_open_orders={cfg.max_open_orders} (action buy/sell = add 1 order)")
    print(
        " Lot config: "
        f"source={cfg.lot_source} | initial_balance={cfg.initial_balance:.2f} | "
        f"risk_level={cfg.risk_level} ({cfg.risk_percent}%) | "
        f"lot={cfg.lot_size} | lot_risk_pips={cfg.lot_risk_pips}"
    )
    print(
        " Projector: "
        f"mode={cfg.projector_mode}, latent={cfg.sem_latent_dim}, source={cfg.embed_source_mode}, "
        f"ae_epochs={cfg.ae_epochs}, ae_hidden={cfg.ae_hidden_dim}"
    )
    print(
        " Recent focus: "
        f"bias={cfg.recent_bias:.2f}, lookback={cfg.recent_lookback}, min_episode={cfg.min_episode_bars}"
    )
    print(f" Resume: {cfg.resume}")
    print(f" Seed: {cfg.random_seed} | Deterministic: {cfg.deterministic} | Device: {cfg.device}")
    print(f" Features: {len(feature_list)} (incl. SMA, RSI, ATR, Trend, ADX)")
    print(f"⏱ Timesteps: {cfg.total_timesteps:,}")
    print(f" Save model: {MODEL_SAVE_PATH}")
    print(f" Save vec:   {VEC_SAVE_PATH}")
    print("=" * 50 + "\n")


def main():
    cfg = _build_config()
    set_global_seed(cfg.random_seed, cfg.deterministic)
    df = load_training_data()
    df, _ = build_semantic_features(df, cfg)
    df, feature_list = build_market_features(df)

    train_env = build_train_env(df, cfg)
    model = build_model(train_env, cfg)
    callback = TradingMetricsCallback(verbose=1, initial_balance=cfg.initial_balance)

    print_training_banner(cfg, feature_list)

    try:
        model.learn(total_timesteps=cfg.total_timesteps, callback=callback)
    except KeyboardInterrupt:
        print("\n Training interrupted. Saving current progress...")
    finally:
        model.save(MODEL_SAVE_PATH)
        train_env.save(VEC_SAVE_PATH)
        print(f" Saved model: {_sb3_zip_path(MODEL_SAVE_PATH)}")
        print(f" Saved vec:   {VEC_SAVE_PATH}")


if __name__ == "__main__":
    main()
