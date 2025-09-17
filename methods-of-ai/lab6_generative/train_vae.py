"""Training entrypoint for Lab 6 (Variational Autoencoder on Fashion-MNIST)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, utils

from common.seed import set_seed


class VAE(nn.Module):
    """Simple MLP-based VAE for 28×28 grayscale images."""

    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(128, latent_dim)
        self.logvar_head = nn.Linear(128, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor, beta: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
        hidden = self.encoder(inputs)
        mu = self.mu_head(hidden)
        logvar = self.logvar_head(hidden)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        latent = mu + eps * std
        recon = self.decoder(latent).view(-1, 1, 28, 28)
        recon_loss = nn.functional.binary_cross_entropy(recon, inputs, reduction="sum") / inputs.size(0)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / inputs.size(0)
        total = recon_loss + beta * kl_loss
        return recon, total, float(recon_loss.item()), float(kl_loss.item())


@dataclass
class TrainingConfig:
    batch_size: int = 128
    max_epochs: int = 8
    learning_rate: float = 1e-3
    latent_dim: int = 16
    kl_warmup_epochs: int = 3
    train_subset: int = 10000
    eval_subset: int = 2000
    seed: int = 42
    log_dir: Optional[Path] = Path("runs/lab6_generative")
    artifacts_dir: Optional[Path] = Path("artifacts/lab6_generative")
    device: str = "cpu"


@dataclass
class TrainingResult:
    loss_history: List[float]
    recon_history: List[float]
    kl_history: List[float]
    eval_metrics: Dict[str, float]
    loss_drop_pct: float
    samples_path: Optional[Path]


def prepare_loaders(config: TrainingConfig) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.ToTensor()
    train_dataset = datasets.FashionMNIST(".data/fashion_mnist", train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(".data/fashion_mnist", train=False, download=True, transform=transform)
    train_subset = Subset(train_dataset, list(range(min(config.train_subset, len(train_dataset)))))
    eval_subset = Subset(test_dataset, list(range(min(config.eval_subset, len(test_dataset)))))
    train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True, num_workers=0, drop_last=True)
    eval_loader = DataLoader(eval_subset, batch_size=256, shuffle=False, num_workers=0)
    return train_loader, eval_loader


def evaluate(model: VAE, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    recon_losses: List[float] = []
    kl_losses: List[float] = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            _, _, recon, kl = model(xb, beta=1.0)
            recon_losses.append(recon)
            kl_losses.append(kl)
    return {
        "recon": float(sum(recon_losses) / max(1, len(recon_losses))),
        "kl": float(sum(kl_losses) / max(1, len(kl_losses))),
    }


def train(config: TrainingConfig = TrainingConfig()) -> TrainingResult:
    set_seed(config.seed)
    device = torch.device(config.device)
    train_loader, eval_loader = prepare_loaders(config)
    model = VAE(latent_dim=config.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    writer: Optional[SummaryWriter] = None
    if config.log_dir is not None:
        config.log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(config.log_dir))

    loss_history: List[float] = []
    recon_history: List[float] = []
    kl_history: List[float] = []

    for epoch in range(config.max_epochs):
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        batches = 0
        beta = float(min(1.0, (epoch + 1) / max(1, config.kl_warmup_epochs)))
        for xb, _ in train_loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            _, loss, recon, kl = model(xb, beta=beta)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_recon += recon
            total_kl += kl
            batches += 1
        avg_loss = total_loss / max(1, batches)
        avg_recon = total_recon / max(1, batches)
        avg_kl = total_kl / max(1, batches)
        loss_history.append(avg_loss)
        recon_history.append(avg_recon)
        kl_history.append(avg_kl)
        if writer is not None:
            writer.add_scalar("train/loss", avg_loss, epoch)
            writer.add_scalar("train/recon", avg_recon, epoch)
            writer.add_scalar("train/kl", avg_kl, epoch)

    if writer is not None:
        writer.close()

    initial_loss = loss_history[0]
    final_loss = loss_history[-1]
    loss_drop_pct = float((initial_loss - final_loss) / max(initial_loss, 1e-6) * 100.0)

    eval_metrics = evaluate(model, eval_loader, device)

    samples_path: Optional[Path] = None
    if config.artifacts_dir is not None:
        config.artifacts_dir.mkdir(parents=True, exist_ok=True)
        samples_path = config.artifacts_dir / "samples.png"
        with torch.no_grad():
            z = torch.randn(64, config.latent_dim, device=device)
            samples = model.decoder(z).view(-1, 1, 28, 28)
            utils.save_image(samples, samples_path, nrow=8)

    return TrainingResult(
        loss_history=loss_history,
        recon_history=recon_history,
        kl_history=kl_history,
        eval_metrics=eval_metrics,
        loss_drop_pct=loss_drop_pct,
        samples_path=samples_path,
    )


def main() -> None:  # pragma: no cover - CLI helper
    result = train()
    print("Loss drop %", result.loss_drop_pct)
    print("Eval metrics", result.eval_metrics)
    if result.samples_path:
        print("Samples saved to", result.samples_path)


if __name__ == "__main__":  # pragma: no cover
    main()
