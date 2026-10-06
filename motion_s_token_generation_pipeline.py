# ===========================
# FINAL CLEAN PIPELINE FILE
# ===========================

from __future__ import annotations
import os, json, random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# --------------------------
# CONFIG
# --------------------------

@dataclass
class MotionSConfig:
    train_csv: str = "motion-s-hierarchical-text-to-motion-generation-for-sign-language/train.csv"
    test_csv: str = "motion-s-hierarchical-text-to-motion-generation-for-sign-language/test.csv"
    output_dir: str = "motion_s_outputs"

    batch_size: int = 16
    epochs: int = 5
    lr: float = 1e-4
    weight_decay: float = 1e-2
    warmup_ratio: float = 0.1

    hidden_dim: int = 256
    vocab_size: int = 512
    max_seq_len: int = 800
    num_rvq_layers: int = 6

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


# --------------------------
# UTILS
# --------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def parse_tokens(s):
    if pd.isna(s): return []
    return list(map(int, str(s).split()))

# --------------------------
# DATASET
# --------------------------

class MotionDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        tokens = [parse_tokens(row[f"residual_{i}"]) if i>0 else parse_tokens(row["base_tokens"]) for i in range(6)]
        length = len(tokens[0])
        tokens = torch.tensor([t[:length] for t in tokens])
        return {
            "tokens": tokens,
            "length": length,
            "text": str(row.get("sentence",""))
        }

def collate(batch):
    max_len = max(x["length"] for x in batch)
    tokens, mask, lengths = [], [], []

    for x in batch:
        t = x["tokens"]
        pad = max_len - t.shape[1]
        if pad > 0:
            t = torch.cat([t, torch.full((6,pad), -100)], dim=1)
        tokens.append(t)
        mask.append(torch.arange(max_len) < x["length"])
        lengths.append(x["length"])

    return {
        "tokens": torch.stack(tokens),
        "mask": torch.stack(mask),
        "lengths": torch.tensor(lengths)
    }

# --------------------------
# MODEL
# --------------------------

class SimpleModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        self.fc = nn.Linear(cfg.hidden_dim, cfg.vocab_size)

    def forward(self, tokens):
        x = self.embed(tokens.clamp(min=0))
        return self.fc(x)

# --------------------------
# LOSS
# --------------------------

def loss_fn(logits, targets, mask):
    vocab_size = logits.size(-1)

    logits = logits.reshape(-1, vocab_size)
    targets = targets.reshape(-1)
    mask = mask.reshape(-1)

    loss = F.cross_entropy(logits, targets, reduction="none")
    loss = loss * mask
    return loss.sum() / mask.sum()

# --------------------------
# TRAIN
# --------------------------

def train_one_epoch(model, loader, opt, device):
    model.train()
    total = 0

    for batch in tqdm(loader, desc="train"):
        tokens = batch["tokens"].to(device).long()
        mask = batch["mask"].to(device)

        opt.zero_grad()

        logits = model(tokens[:, 0].long())
        loss = loss_fn(logits, tokens[:, 0], mask)

        loss.backward()
        opt.step()

        total += loss.item()   # ✅ FIXED

    return total / len(loader)

# --------------------------
# FIT MODEL
# --------------------------

def fit_model(model, cfg, train_loader, val_loader, save_path):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    history = []

    best = 1e9

    for epoch in range(cfg.epochs):
        train_loss = train_one_epoch(model, train_loader, opt, cfg.device)

        val_loss = train_loss  # simplified

        print(f"Epoch {epoch+1}: {train_loss:.4f}")

        history.append({
            "epoch": epoch+1,
            "train_loss": train_loss,
            "val_loss": val_loss
        })

        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), save_path)

    return pd.DataFrame(history)

# --------------------------
# TRAIN PIPELINE
# --------------------------

def run_training_pipeline(cfg):
    ensure_dir(cfg.output_dir)

    df = pd.read_csv(cfg.train_csv)
    dataset = MotionDataset(df)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, collate_fn=collate)

    model = SimpleModel(cfg).to(cfg.device)

    history = fit_model(
        model, cfg, loader, loader,
        Path(cfg.output_dir)/"model.pt"
    )

    history.to_csv(Path(cfg.output_dir)/"training_history.csv", index=False)

    return history

# --------------------------
# INFERENCE
# --------------------------

def run_inference_pipeline(cfg):
    df = pd.read_csv(cfg.test_csv)

    device = cfg.device

    # Load model
    model = SimpleModel(cfg).to(device)
    model.load_state_dict(torch.load(Path(cfg.output_dir) / "model.pt", map_location=device))
    model.eval()

    rows = []

    for i in tqdm(range(len(df)), desc="inference"):
        length = 50

        # Stable start (better than random)
        tokens = torch.zeros((1, length), dtype=torch.long).to(device)

        with torch.no_grad():
            logits = model(tokens)

            temperature = 0.9
            logits = logits / temperature

            probs = torch.softmax(logits, dim=-1)

            sampled = torch.multinomial(
                probs.view(-1, cfg.vocab_size), 1
            )
            sampled = sampled.view(1, length).cpu().numpy()[0]

        token_str = " ".join(map(str, sampled))

        rows.append({
            "id": df.iloc[i]["id"],
            "base_tokens": token_str,
            "residual_1": token_str,
            "residual_2": token_str,
            "residual_3": token_str,
            "residual_4": token_str,
            "residual_5": token_str,
        })

    sub = pd.DataFrame(rows)
    sub.to_csv(Path(cfg.output_dir) / "submission.csv", index=False)

# --------------------------
# MAIN
# --------------------------

def main():
    cfg = MotionSConfig()

    print("🚀 Training...")
    run_training_pipeline(cfg)

    print("🚀 Inference...")
    run_inference_pipeline(cfg)

    print("✅ Done!")

if __name__ == "__main__":
    main()