"""
models/autoencoder/conv_ae.py — Convolutional Autoencoder v2.

Key improvements over v1:
- Strided convolutions (stride=2) force real compression
- RGB input (3 channels) instead of grayscale
- Smaller latent dim (256) = tighter bottleneck
- Threshold calibrated at p95 and saved to checkpoint
- num_workers=0 (Windows)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

DEFAULT_CFG = {
    "img_size": 128,
    "latent_dim": 256,
    "batch_size": 32,
    "epochs": 50,
    "lr": 0.001,
    "threshold_percentile": 95,
}


class _PathListDataset(Dataset):
    def __init__(self, frame_paths: list, size: tuple) -> None:
        self.paths = [Path(p) for p in frame_paths]
        self.size  = size
        self.transform = transforms.Compose([
            transforms.Resize((size[0], size[1])),
            transforms.ToTensor(),
        ])
        print(f"[_PathListDataset] {len(self.paths)} frames")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


class ConvEncoder(nn.Module):
    def __init__(self, latent_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc   = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        h = self.pool(h)
        return self.fc(h.view(h.size(0), -1))


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim: int = 256) -> None:
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64,  4, stride=2, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
            nn.ConvTranspose2d(64,  32,  4, stride=2, padding=1), nn.BatchNorm2d(32),  nn.ReLU(),
            nn.ConvTranspose2d(32,  3,   4, stride=2, padding=1), nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, 256, 4, 4)
        return self.net(h)


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 256) -> None:
        super().__init__()
        self.encoder = ConvEncoder(latent_dim)
        self.decoder = ConvDecoder(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W  = x.shape[2], x.shape[3]
        recon = self.decoder(self.encoder(x))
        if recon.shape[2] != H or recon.shape[3] != W:
            recon = F.interpolate(recon, size=(H, W), mode="bilinear", align_corners=False)
        return recon


class ConvAETrainer:

    def __init__(self, cfg: dict) -> None:
        self.cfg         = {**DEFAULT_CFG, **cfg}
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model       = ConvAutoencoder(self.cfg["latent_dim"]).to(self.device)
        self.threshold: Optional[float] = None
        self._frame_size: tuple         = (128, 128)

    def _make_transform(self, size: tuple) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((size[0], size[1])),
            transforms.ToTensor(),
        ])

    def fit(self, frame_paths, frame_size: tuple = (128, 128)) -> None:
        self._frame_size = frame_size
        if isinstance(frame_paths, str):
            frame_paths = sorted(Path(frame_paths).rglob("*.jpg"))
        frame_paths = [Path(p) for p in frame_paths]
        if len(frame_paths) == 0:
            raise ValueError("No frames found for training.")

        dataset = _PathListDataset(frame_paths, size=frame_size)
        loader  = DataLoader(dataset, batch_size=self.cfg["batch_size"],
                             shuffle=True, num_workers=0, pin_memory=False)

        optimiser = optim.Adam(self.model.parameters(), lr=self.cfg["lr"])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode="min", factor=0.5, patience=5)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(1, self.cfg["epochs"] + 1):
            total = 0.0
            for batch in loader:
                batch = batch.to(self.device)
                recon = self.model(batch)
                loss  = criterion(recon, batch)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                total += loss.item()
            avg = total / len(loader)
            scheduler.step(avg)
            if epoch % 5 == 0 or epoch == 1:
                logger.info(f"[ConvAE] Epoch {epoch:3d}/{self.cfg['epochs']}  loss={avg:.6f}")

        self._set_threshold(frame_paths, frame_size)
        logger.info(f"[ConvAE] Training complete. Threshold={self.threshold:.6f}")

    def _set_threshold(self, frame_paths: list, frame_size: tuple) -> None:
        logger.info("[ConvAE] Calibrating threshold on training frames ...")
        transform = self._make_transform(frame_size)
        errors = []
        self.model.eval()
        with torch.no_grad():
            for path in frame_paths:
                img   = Image.open(path).convert("RGB")
                t     = transform(img).unsqueeze(0).to(self.device)
                recon = self.model(t)
                errors.append(F.mse_loss(recon, t).item())
        pct            = self.cfg.get("threshold_percentile", 95)
        self.threshold = float(np.percentile(errors, pct))
        logger.info(f"[ConvAE] Threshold p{pct}: {self.threshold:.6f}")

    def score(self, frame_paths) -> dict:
        if isinstance(frame_paths, str):
            frame_paths = sorted(Path(frame_paths).rglob("*.jpg"))
        frame_paths = [Path(p) for p in frame_paths]
        transform   = self._make_transform(self._frame_size)
        errors = []
        self.model.eval()
        with torch.no_grad():
            for path in frame_paths:
                img   = Image.open(path).convert("RGB")
                t     = transform(img).unsqueeze(0).to(self.device)
                recon = self.model(t)
                errors.append(F.mse_loss(recon, t).item())
        return {"paths": frame_paths, "scores": np.array(errors, dtype=np.float32)}

    def score_degraded(self, frame_paths, degradation_fn: Callable) -> dict:
        if isinstance(frame_paths, str):
            frame_paths = sorted(Path(frame_paths).rglob("*.jpg"))
        frame_paths = [Path(p) for p in frame_paths]
        transform   = self._make_transform(self._frame_size)
        errors = []
        self.model.eval()
        with torch.no_grad():
            for path in frame_paths:
                img   = Image.open(path).convert("RGB")
                img   = degradation_fn(img)
                t     = transform(img).unsqueeze(0).to(self.device)
                recon = self.model(t)
                errors.append(F.mse_loss(recon, t).item())
        return {"paths": frame_paths, "scores": np.array(errors, dtype=np.float32)}

    def score_single(self, img) -> float:
        """Score a single image; returns MSE reconstruction error as a float.

        Accepts:
            img: PIL.Image, a file path (str or Path), or a numpy array (H x W x C).
        """
        transform = self._make_transform(self._frame_size)
        if isinstance(img, (str, Path)):
            img = Image.open(img).convert("RGB")
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img).convert("RGB")
        elif not isinstance(img, Image.Image):
            raise TypeError(f"score_single: unsupported image type {type(img)}")
        self.model.eval()
        with torch.no_grad():
            t     = transform(img).unsqueeze(0).to(self.device)
            recon = self.model(t)
            return F.mse_loss(recon, t).item()

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "threshold":   self.threshold,
            "frame_size":  self._frame_size,
            "cfg":         self.cfg,
        }, path)
        logger.info(f"[ConvAE] Saved → {path}  (threshold={self.threshold:.6f})")

    def load(self, path: str) -> None:
        ckpt             = torch.load(path, map_location=self.device)
        self.cfg         = {**DEFAULT_CFG, **ckpt.get("cfg", DEFAULT_CFG)}
        self._frame_size = ckpt.get("frame_size", (128, 128))
        self.threshold   = ckpt.get("threshold")
        self.model       = ConvAutoencoder(self.cfg["latent_dim"]).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        print(f"[ConvAETrainer] Loaded from '{path}'")
        if self.threshold is not None:
            logger.info(f"[ConvAE] Threshold loaded: {self.threshold:.6f}")
        else:
            logger.warning("[ConvAE] No threshold in checkpoint — retrain to calibrate.")