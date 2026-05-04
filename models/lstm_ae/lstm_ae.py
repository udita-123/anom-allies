"""
models/lstm_ae/lstm_ae.py — LSTM Sequence Autoencoder for trajectory anomaly detection.

Input : (batch, seq_len, input_dim) trajectory sequences
Output: reconstructed sequence of same shape
Loss  : MSE reconstruction error — high error = abnormal motion pattern
"""

from __future__ import annotations

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

DEFAULT_CFG = {
    "input_dim":             2,
    "hidden_dim":           64,
    "num_layers":            2,
    "dropout":             0.2,
    "seq_len":              12,   # MUST match min_trajectory_length in config
    "batch_size":           64,
    "epochs":               25,
    "lr":                0.001,
    "threshold_percentile": 95,
}


# ─────────────────────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────────────────────

class TrajectoryDataset(Dataset):
    """
    Accepts a list of (seq_len, input_dim) numpy arrays.
    Each array is one training sequence — no internal windowing.
    """

    def __init__(self, sequences: list[np.ndarray]) -> None:
        valid = [s for s in sequences if s.ndim == 2]
        self.data = [torch.tensor(s, dtype=torch.float32) for s in valid]
        logger.info(f"[TrajectoryDataset] {len(self.data)} sequences loaded.")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


# ─────────────────────────────────────────────────────────────
#  Model
# ─────────────────────────────────────────────────────────────

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, input_dim)
        _, (hidden, cell) = self.encoder(x)

        # Repeat last hidden state T times as decoder input
        B, T = x.shape[0], x.shape[1]
        dec_input = hidden[-1].unsqueeze(1).repeat(1, T, 1)   # (B, T, hidden_dim)

        dec_out, _ = self.decoder(dec_input, (hidden, cell))   # (B, T, hidden_dim)
        return self.output_layer(dec_out)                       # (B, T, input_dim)


# ─────────────────────────────────────────────────────────────
#  Trainer
# ─────────────────────────────────────────────────────────────

class LSTMAETrainer:
    """
    Called by pipeline.py:
        trainer = LSTMAETrainer(cfg["lstm_ae"])
        trainer.fit(sequences)           # list of (seq_len, 2) np.ndarray
        trainer.save("models/lstm_ae/lstm_ae_ucsd_ped2.pt")
        trainer.load("models/lstm_ae/lstm_ae_ucsd_ped2.pt")
        score = trainer.score(sequence)  # single (seq_len, 2) → float
    """

    def __init__(self, cfg: dict = None) -> None:
        self.cfg = {**DEFAULT_CFG, **(cfg or {})}

        device_str = self.cfg.get("device", "auto")
        if device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)

        self.model:     LSTMAutoencoder | None = None
        self.threshold: float                  = 0.0
        self._build_model()

    def _build_model(self) -> None:
        self.model = LSTMAutoencoder(
            input_dim  = self.cfg["input_dim"],
            hidden_dim = self.cfg["hidden_dim"],
            num_layers = self.cfg["num_layers"],
            dropout    = self.cfg["dropout"],
        ).to(self.device)

    # ── fit ───────────────────────────────────────────────────

    def fit(self, sequences: list[np.ndarray]) -> list[float]:
        """
        Train on a list of (seq_len, input_dim) numpy arrays.
        Returns per-epoch losses.
        """
        if not sequences:
            logger.warning("[LSTM-AE] No sequences provided — skipping training.")
            return []

        # Filter to correct shape
        seq_len   = self.cfg["seq_len"]
        input_dim = self.cfg["input_dim"]
        valid = [s for s in sequences
                 if isinstance(s, np.ndarray)
                 and s.ndim == 2
                 and s.shape[0] == seq_len
                 and s.shape[1] == input_dim]

        if not valid:
            logger.warning(
                f"[LSTM-AE] No valid sequences after filtering "
                f"(need shape ({seq_len}, {input_dim})). "
                f"Got {len(sequences)} sequences with shapes: "
                f"{[s.shape for s in sequences[:5]]}"
            )
            return []

        logger.info(
            f"[LSTM-AE] Training on {len(valid)} sequences "
            f"(seq_len={seq_len}, input_dim={input_dim}) "
            f"on {self.device}"
        )

        dataset    = TrajectoryDataset(valid)
        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg["batch_size"],
            shuffle=True,
            num_workers=0,
        )

        optimizer = optim.Adam(self.model.parameters(), lr=self.cfg["lr"])
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=0.5
        )

        self.model.train()
        losses = []
        epochs = self.cfg["epochs"]

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            for batch in dataloader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                recon = self.model(batch)
                loss  = criterion(recon, batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item() * len(batch)

            epoch_loss /= len(dataset)
            scheduler.step(epoch_loss)
            losses.append(epoch_loss)

            if epoch == 1 or epoch % 5 == 0:
                logger.info(f"[LSTM-AE] Epoch {epoch:3d}/{epochs}  loss={epoch_loss:.6f}")

        # Calibrate threshold on training data
        self.threshold = self._calibrate(valid)
        logger.info(
            f"[LSTM-AE] Training complete. "
            f"Threshold p{self.cfg['threshold_percentile']}={self.threshold:.6f}"
        )
        return losses

    def _calibrate(self, sequences: list[np.ndarray]) -> float:
        self.model.eval()
        errors = []
        with torch.no_grad():
            for seq in sequences:
                err = self._reconstruction_error(seq)
                errors.append(err)
        pct = self.cfg.get("threshold_percentile", 95)
        return float(np.percentile(errors, pct))

    # ── score ─────────────────────────────────────────────────

    def score(self, sequence: np.ndarray) -> float:
        """
        Score a single (seq_len, input_dim) sequence.
        Returns MSE reconstruction error as float.
        """
        return self._reconstruction_error(sequence)

    def score_batch(self, sequences: np.ndarray) -> np.ndarray:
        """
        Score a batch (B, seq_len, input_dim).
        Returns (B,) array of errors.
        """
        self.model.eval()
        t = torch.tensor(sequences, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            recon = self.model(t)
        errors = ((recon - t) ** 2).mean(dim=[1, 2])
        return errors.cpu().numpy()

    def _reconstruction_error(self, seq: np.ndarray) -> float:
        self.model.eval()
        t = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            recon = self.model(t)
        return float(((recon - t) ** 2).mean().item())

    # ── save / load ───────────────────────────────────────────

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "cfg":         self.cfg,
            "threshold":   self.threshold,
        }, path)
        logger.info(f"[LSTM-AE] Saved → {path}")

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        saved_cfg = ckpt.get("cfg", {})
        # Rebuild model with saved architecture
        self.cfg = {**self.cfg, **saved_cfg}
        self._build_model()
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.threshold = ckpt.get("threshold", 0.0)
        logger.info(
            f"[LSTM-AE] Loaded from '{path}' "
            f"(threshold={self.threshold:.6f})"
        )


# ─────────────────────────────────────────────────────────────
#  Self-test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    logging.basicConfig(level=logging.INFO)
    logger.info("Running LSTM-AE self-test ...")

    SEQ_LEN   = 12
    INPUT_DIM = 2
    N_SEQS    = 80

    # Synthetic normal walking trajectories
    normal_seqs = []
    for _ in range(N_SEQS):
        t    = np.linspace(0, 1, SEQ_LEN)
        x    = t + np.random.randn(SEQ_LEN) * 0.02
        y    = 0.5 * t + np.random.randn(SEQ_LEN) * 0.02
        seq  = np.stack([x, y], axis=1).astype(np.float32)
        normal_seqs.append(seq)

    trainer = LSTMAETrainer({"seq_len": SEQ_LEN, "input_dim": INPUT_DIM, "epochs": 10})
    trainer.fit(normal_seqs)

    # Score a normal sequence
    normal_score = trainer.score(normal_seqs[0])

    # Score an anomalous sequence (random walk)
    anomaly = np.random.randn(SEQ_LEN, INPUT_DIM).astype(np.float32)
    anomaly_score = trainer.score(anomaly)

    logger.info(f"Normal score:  {normal_score:.6f}")
    logger.info(f"Anomaly score: {anomaly_score:.6f}")
    assert anomaly_score > normal_score, "Anomaly should score higher than normal!"
    logger.info("Self-test PASSED.")