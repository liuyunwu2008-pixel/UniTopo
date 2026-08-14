"""Pre-training loop for MMG-MPP."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from .losses import TotalPretrainLoss


class PretrainTrainer:
    """Pre-training trainer with InfoNCE + GMAE objectives."""

    def __init__(self, model, config, device="cuda"):
        self.model = model.to(device)
        self.config = config
        self.device = device

        self.loss_fn = TotalPretrainLoss(
            temperature=config.info_nce_tau,
            gmae_weight=config.gmae_loss_weight,
        )
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.num_epochs,
        )

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        total_cont = 0.0
        total_gmae = 0.0

        pbar = tqdm(dataloader, desc="Pre-training")
        for batch in pbar:
            # Move to device
            data_me = batch["atomic"].to(self.device)
            data_mt = batch["tree"].to(self.device)
            data_fm = batch["fragment"].to(self.device)
            mph_maps = batch.get("mph_maps")
            if mph_maps is not None:
                mph_maps = mph_maps.to(self.device)

            # Forward
            outputs = self.model(data_me, data_mt, data_fm, mph_maps)

            # Loss
            loss, l_cont, l_gmae = self.loss_fn(
                outputs["z_frag"], outputs["z_gt"], outputs["gmae_loss"],
            )

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_cont += l_cont.item()
            total_gmae += l_gmae.item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "cont": f"{l_cont.item():.4f}",
                "gmae": f"{l_gmae.item():.4f}",
            })

        n = len(dataloader)
        self.scheduler.step()
        return total_loss / n, total_cont / n, total_gmae / n

    @torch.no_grad()
    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0.0

        for batch in dataloader:
            data_me = batch["atomic"].to(self.device)
            data_mt = batch["tree"].to(self.device)
            data_fm = batch["fragment"].to(self.device)
            mph_maps = batch.get("mph_maps")
            if mph_maps is not None:
                mph_maps = mph_maps.to(self.device)

            outputs = self.model(data_me, data_mt, data_fm, mph_maps)
            loss, _, _ = self.loss_fn(
                outputs["z_frag"], outputs["z_gt"], outputs["gmae_loss"],
            )
            total_loss += loss.item()

        return total_loss / len(dataloader)

    def train(self, train_dl, val_dl, save_path="pretrained_mmg_mpp.pt"):
        best_val_loss = float("inf")

        for epoch in range(self.config.num_epochs):
            train_loss, train_cont, train_gmae = self.train_epoch(train_dl)
            val_loss = self.validate(val_dl)

            print(f"Epoch {epoch+1}/{self.config.num_epochs} | "
                  f"Train: {train_loss:.4f} (C:{train_cont:.4f} G:{train_gmae:.4f}) | "
                  f"Val: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_path)
                print(f"  -> Saved checkpoint to {save_path}")

        print(f"Pre-training complete. Best val loss: {best_val_loss:.4f}")
