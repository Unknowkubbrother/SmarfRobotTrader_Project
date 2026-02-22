import os
from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


META_FILENAME = "embedding_projector_meta.joblib"
TORCH_FILENAME = "embedding_projector.pt"
PCA_FILENAME = "pca_model.joblib"


def _as_2d_float32(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={arr.shape}")
    return arr


class NonlinearAutoEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decoder(z)


class LinearAutoEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decoder(z)


@dataclass
class EmbeddingProjector:
    mode: str
    input_dim: int
    latent_dim: int
    torch_model: nn.Module | None = None
    pca_model: object | None = None
    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    def transform(self, raw_vectors: np.ndarray) -> np.ndarray:
        x = _as_2d_float32(raw_vectors)
        mode = str(self.mode).strip().lower()

        if mode == "pca":
            if self.pca_model is None:
                raise RuntimeError("PCA projector is not loaded.")
            return np.asarray(self.pca_model.transform(x), dtype=np.float32)

        if self.torch_model is None or self.mean is None or self.std is None:
            raise RuntimeError("Torch projector is not loaded.")

        x_norm = (x - self.mean) / (self.std + 1e-6)
        with torch.no_grad():
            tensor = torch.from_numpy(x_norm).float()
            z = self.torch_model.encode(tensor).cpu().numpy()
        return np.asarray(z, dtype=np.float32)


def _train_torch_projector(
    model: nn.Module,
    x_norm: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> nn.Module:
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = TensorDataset(torch.from_numpy(x_norm).float())
    safe_batch = max(1, min(batch_size, len(dataset)))
    loader = DataLoader(dataset, batch_size=safe_batch, shuffle=True, drop_last=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()

    for _ in range(max(1, epochs)):
        for (batch,) in loader:
            recon = model(batch)
            z = model.encode(batch)
            loss = F.mse_loss(recon, batch) + (1e-4 * torch.mean(z.pow(2)))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def fit_and_save_projector(
    embeddings: np.ndarray,
    artifact_dir: str,
    mode: str = "autoencoder",
    latent_dim: int = 16,
    random_seed: int = 42,
    ae_hidden_dim: int = 256,
    ae_epochs: int = 35,
    ae_batch_size: int = 256,
    ae_lr: float = 1e-3,
    ae_weight_decay: float = 1e-5,
    **_kwargs,
) -> tuple[EmbeddingProjector, np.ndarray, dict]:
    x = _as_2d_float32(embeddings)
    if len(x) == 0:
        raise ValueError("No embeddings to fit projector.")

    projector_mode = str(mode).strip().lower()
    if projector_mode not in {"autoencoder", "linear", "pca"}:
        raise ValueError(f"Unsupported projector mode: {mode}")

    input_dim = int(x.shape[1])
    latent_dim = int(max(1, min(latent_dim, input_dim, len(x))))

    os.makedirs(artifact_dir, exist_ok=True)
    meta_path = os.path.join(artifact_dir, META_FILENAME)
    torch_path = os.path.join(artifact_dir, TORCH_FILENAME)
    pca_path = os.path.join(artifact_dir, PCA_FILENAME)

    if projector_mode == "pca":
        pca_model = PCA(n_components=latent_dim, random_state=random_seed)
        latent = np.asarray(pca_model.fit_transform(x), dtype=np.float32)

        meta = {
            "mode": "pca",
            "input_dim": input_dim,
            "latent_dim": latent_dim,
            "pca_file": PCA_FILENAME,
        }
        joblib.dump(pca_model, pca_path)
        joblib.dump(meta, meta_path)
        if os.path.exists(torch_path):
            os.remove(torch_path)

        projector = EmbeddingProjector(
            mode="pca",
            input_dim=input_dim,
            latent_dim=latent_dim,
            pca_model=pca_model,
        )
        return projector, latent, meta

    mean = x.mean(axis=0).astype(np.float32)
    std = (x.std(axis=0) + 1e-6).astype(np.float32)
    x_norm = ((x - mean) / std).astype(np.float32)

    hidden_dim = int(max(latent_dim * 4, min(ae_hidden_dim, max(latent_dim * 2, input_dim // 2))))
    if projector_mode == "linear":
        model = LinearAutoEncoder(input_dim=input_dim, latent_dim=latent_dim)
        hidden_dim = 0
    else:
        model = NonlinearAutoEncoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)

    model = _train_torch_projector(
        model=model,
        x_norm=x_norm,
        epochs=ae_epochs,
        batch_size=ae_batch_size,
        lr=ae_lr,
        weight_decay=ae_weight_decay,
        seed=random_seed,
    )

    with torch.no_grad():
        latent = np.asarray(model.encode(torch.from_numpy(x_norm).float()).cpu().numpy(), dtype=np.float32)

    meta = {
        "mode": projector_mode,
        "input_dim": input_dim,
        "latent_dim": latent_dim,
        "hidden_dim": hidden_dim,
        "torch_file": TORCH_FILENAME,
    }
    checkpoint = {
        "mode": projector_mode,
        "input_dim": input_dim,
        "latent_dim": latent_dim,
        "hidden_dim": hidden_dim,
        "mean": mean,
        "std": std,
        "state_dict": model.state_dict(),
    }

    joblib.dump(meta, meta_path)
    torch.save(checkpoint, torch_path)
    if os.path.exists(pca_path):
        os.remove(pca_path)

    projector = EmbeddingProjector(
        mode=projector_mode,
        input_dim=input_dim,
        latent_dim=latent_dim,
        torch_model=model,
        mean=mean,
        std=std,
    )
    return projector, latent, meta


def load_projector(artifact_dir: str) -> EmbeddingProjector | None:
    meta_path = os.path.join(artifact_dir, META_FILENAME)
    if not os.path.exists(meta_path):
        return None

    meta = joblib.load(meta_path)
    mode = str(meta.get("mode", "autoencoder")).strip().lower()

    if mode == "pca":
        pca_file = meta.get("pca_file", PCA_FILENAME)
        pca_path = os.path.join(artifact_dir, pca_file)
        if not os.path.exists(pca_path):
            return None
        pca_model = joblib.load(pca_path)
        input_dim = int(getattr(pca_model, "n_features_in_", meta.get("input_dim", 1024)))
        latent_dim = int(getattr(pca_model, "n_components_", meta.get("latent_dim", 16)))
        return EmbeddingProjector(
            mode="pca",
            input_dim=input_dim,
            latent_dim=latent_dim,
            pca_model=pca_model,
        )

    torch_file = meta.get("torch_file", TORCH_FILENAME)
    torch_path = os.path.join(artifact_dir, torch_file)
    if not os.path.exists(torch_path):
        return None

    try:
        checkpoint = torch.load(torch_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(torch_path, map_location="cpu")

    ckpt_mode = str(checkpoint.get("mode", mode)).strip().lower()
    input_dim = int(checkpoint.get("input_dim", meta.get("input_dim", 1024)))
    latent_dim = int(checkpoint.get("latent_dim", meta.get("latent_dim", 16)))
    hidden_dim = int(checkpoint.get("hidden_dim", meta.get("hidden_dim", max(latent_dim * 2, 64))))

    if ckpt_mode == "linear":
        model = LinearAutoEncoder(input_dim=input_dim, latent_dim=latent_dim)
    else:
        model = NonlinearAutoEncoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    mean = np.asarray(checkpoint.get("mean", np.zeros(input_dim, dtype=np.float32)), dtype=np.float32)
    std = np.asarray(checkpoint.get("std", np.ones(input_dim, dtype=np.float32)), dtype=np.float32)

    return EmbeddingProjector(
        mode=ckpt_mode,
        input_dim=input_dim,
        latent_dim=latent_dim,
        torch_model=model,
        mean=mean,
        std=std,
    )
