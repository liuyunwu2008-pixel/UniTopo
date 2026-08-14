"""Fine-tuning loop for downstream molecular property prediction."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, mean_squared_error


class FinetuneTrainer:
    """Fine-tuning trainer for molecular property prediction."""

    def __init__(self, model, config, task_type="classification", device="cuda"):
        self.model = model.to(device)
        self.config = config
        self.task_type = task_type
        self.device = device

        if task_type == "classification":
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.MSELoss()

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for batch in tqdm(dataloader, desc="Fine-tuning", leave=False):
            data_me = batch["atomic"].to(self.device)
            data_mt = batch["tree"].to(self.device)
            data_fm = batch["fragment"].to(self.device)
            labels = batch["label"].to(self.device)
            mph = batch.get("mph_maps")
            if mph is not None:
                mph = mph.to(self.device)

            preds = self.model(data_me, data_mt, data_fm, mph)
            loss = self.criterion(preds, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            all_preds.append(preds.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

        preds = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)
        return total_loss / len(dataloader), preds, labels

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for batch in dataloader:
            data_me = batch["atomic"].to(self.device)
            data_mt = batch["tree"].to(self.device)
            data_fm = batch["fragment"].to(self.device)
            labels = batch["label"].to(self.device)
            mph = batch.get("mph_maps")
            if mph is not None:
                mph = mph.to(self.device)

            preds = self.model(data_me, data_mt, data_fm, mph)
            loss = self.criterion(preds, labels)
            total_loss += loss.item()
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        preds = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)

        if self.task_type == "classification":
            try:
                metric = roc_auc_score(labels, preds)
            except ValueError:
                metric = float('nan')
        else:
            metric = np.sqrt(mean_squared_error(labels, preds))

        return total_loss / len(dataloader), metric, preds, labels

    def train(self, train_dl, val_dl, test_dl, save_path="finetuned_mmg_mpp.pt"):
        best_val_metric = -float("inf") if self.task_type == "classification" else float("inf")
        patience = self.config.early_stop_patience
        patience_counter = 0

        for epoch in range(self.config.num_epochs):
            train_loss, _, _ = self.train_epoch(train_dl)
            val_loss, val_metric, _, _ = self.evaluate(val_dl)

            is_better = (self.task_type == "classification" and val_metric > best_val_metric) or \
                        (self.task_type == "regression" and val_metric < best_val_metric)

            if is_better:
                best_val_metric = val_metric
                patience_counter = 0
                torch.save(self.model.state_dict(), save_path)
            else:
                patience_counter += 1

            metric_name = "AUC" if self.task_type == "classification" else "RMSE"
            print(f"Epoch {epoch+1}/{self.config.num_epochs} | "
                  f"Train loss: {train_loss:.4f} | "
                  f"Val {metric_name}: {val_metric:.4f}")

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Load best model and evaluate on test set
        self.model.load_state_dict(torch.load(save_path))
        _, test_metric, test_preds, test_labels = self.evaluate(test_dl)
        metric_name = "AUC" if self.task_type == "classification" else "RMSE"
        print(f"\nTest {metric_name}: {test_metric:.4f}")
        return test_metric, test_preds, test_labels
