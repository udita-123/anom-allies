code = """from __future__ import annotations
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

DEFAULT_CFG = {
    "input_dim": 2,
    "hidden_dim": 64,
    "num_layers": 2,
    "dropout": 0.2,
    "seq_len": 12,
    "batch_size": 64,
    "epochs": 25,
    "lr": 0.001,
    "threshold_percentile": 95,
}

class TrajectoryDataset(Dataset):
    def __init__(self, sequences):
        self.data = [torch.tensor(s, dtype=torch.float32) for s in sequences if s.ndim == 2]
        logger.info(f"[TrajectoryDataset] {len(self.data)} sequences loaded.")
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
                               dropout=dropout if num_layers > 1 else 0.0)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True,
                               dropout=dropout if num_layers > 1 else 0.0)
        self.output_layer = nn.Linear(hidden_dim, input_dim)
    def forward(self, x):
        _, (hidden, cell) = self.encoder(x)
        dec_input = hidden[-1].unsqueeze(1).repeat(1, x.shape[1], 1)
        dec_out, _ = self.decoder(dec_input, (hidden, cell))
        return self.output_layer(dec_out)

class LSTMAETrainer:
    def __init__(self, cfg=None):
        self.cfg = {**DEFAULT_CFG, **(cfg or {})}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = 0.0
        self._build_model()

    def _build_model(self):
        self.model = LSTMAutoencoder(
            self.cfg["input_dim"], self.cfg["hidden_dim"],
            self.cfg["num_layers"], self.cfg["dropout"],
        ).to(self.device)

    def fit(self, sequences):
        if not sequences:
            logger.warning("[LSTM-AE] No sequences provided.")
            return []
        seq_len = self.cfg["seq_len"]
        input_dim = self.cfg["input_dim"]
        valid = [s for s in sequences
                 if isinstance(s, np.ndarray) and s.ndim == 2
                 and s.shape[0] == seq_len and s.shape[1] == input_dim]
        if not valid:
            logger.warning(f"[LSTM-AE] No valid sequences (need ({seq_len},{input_dim})). "
                           f"Got shapes: {[s.shape for s in sequences[:5]]}")
            return []
        logger.info(f"[LSTM-AE] Training on {len(valid)} sequences "
                    f"(seq_len={seq_len}, input_dim={input_dim}) on {self.device}")
        loader = DataLoader(TrajectoryDataset(valid), batch_size=self.cfg["batch_size"],
                            shuffle=True, num_workers=0)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg["lr"])
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        self.model.train()
        losses = []
        for epoch in range(1, self.cfg["epochs"] + 1):
            epoch_loss = 0.0
            for batch in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(batch), batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item() * len(batch)
            epoch_loss /= len(valid)
            scheduler.step(epoch_loss)
            losses.append(epoch_loss)
            if epoch == 1 or epoch % 5 == 0:
                logger.info(f"[LSTM-AE] Epoch {epoch:3d}/{self.cfg['epochs']}  loss={epoch_loss:.6f}")
        self.threshold = self._calibrate(valid)
        logger.info(f"[LSTM-AE] Training complete. Threshold={self.threshold:.6f}")
        return losses

    def _calibrate(self, sequences):
        self.model.eval()
        return float(np.percentile([self._reconstruction_error(s) for s in sequences],
                                   self.cfg["threshold_percentile"]))

    def score(self, sequence):
        return self._reconstruction_error(sequence)

    def score_batch(self, sequences):
        self.model.eval()
        t = torch.tensor(sequences, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            recon = self.model(t)
        return ((recon - t) ** 2).mean(dim=[1, 2]).cpu().numpy()

    def _reconstruction_error(self, seq):
        self.model.eval()
        t = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            recon = self.model(t)
        return float(((recon - t) ** 2).mean().item())

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": self.model.state_dict(),
                    "cfg": self.cfg, "threshold": self.threshold}, path)
        logger.info(f"[LSTM-AE] Saved -> {path}")

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.cfg = {**self.cfg, **ckpt.get("cfg", {})}
        self._build_model()
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.threshold = ckpt.get("threshold", 0.0)
        logger.info(f"[LSTM-AE] Loaded from {path} (threshold={self.threshold:.6f})")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
"""

with open("models/lstm_ae/lstm_ae.py", "w") as f:
    f.write(code)

print("File written successfully.")
print("First 300 chars:")
print(open("models/lstm_ae/lstm_ae.py").read()[:300])