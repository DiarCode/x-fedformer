import flwr as fl
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from typing import Dict, Optional
from torch.utils.data import random_split 

from config import (
    log, DEVICE, BATCH_SIZE, LR, LOCAL_EPOCHS, PROX_MU, DP_ENABLED,
    NOISE_MULTIPLIER, MAX_GRAD_NORM, SEQ_LEN, HORIZON
)
from data_utils import TransitDataset, collate_fn
from xfedformer_model import XFedFormer

class FedProxClient(fl.client.NumPyClient):
    def __init__(self, city: str, df: pd.DataFrame, days_data: int):
        self.city = city
        self.model: Optional[XFedFormer] = None
        self.dataset_params = {"seq_len": SEQ_LEN, "horizon": HORIZON}

        full_ds = TransitDataset(df, city_name=city, **self.dataset_params)

        if len(full_ds) == 0:
            log.warning(f"Client {city}: Dataset is empty. Skipping client.")
            self.tr_ld, self.va_ld = None, None
            self.num_train_samples = 0
            self.num_val_samples = 0
            # Initialize a dummy model to prevent AttributeError if model is accessed
            self.model = XFedFormer(input_dim=19, n_routes=10, n_static_feats=0, seq_len=SEQ_LEN, horizon=HORIZON)
            self.model_initialized_correctly = False
            return

        self.model_initialized_correctly = True
        self.n_routes = full_ds.n_routes
        self.input_dim = full_ds.input_dim
        self.n_static_feats = full_ds.static_features_norm.shape[1] if full_ds.static_features_norm is not None else 0


        self.model = XFedFormer(
            input_dim=self.input_dim,
            n_routes=self.n_routes,
            n_static_feats=self.n_static_feats,
            seq_len=SEQ_LEN, horizon=HORIZON
        ).to(DEVICE)

        n_total = len(full_ds)
        n_train = int(n_total * 0.8)
        n_val = n_total - n_train

        if n_train == 0 or n_val == 0:
            log.warning(f"Client {city}: Not enough samples for train/val split. Train: {n_train}, Val: {n_val}")
            if n_total > 0 and n_train == 0:
                n_train = n_total
                n_val = 0
            if n_total > 0 and n_val == 0 and n_train > 0:
                pass
            else:
                self.tr_ld, self.va_ld = None, None
                self.num_train_samples = 0
                self.num_val_samples = 0
                self.model_initialized_correctly = False
                return

        self.ds_train, self.ds_val = random_split(
            full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42)
        )
        self.num_train_samples = len(self.ds_train)
        self.num_val_samples = len(self.ds_val)

        pin_memory = DEVICE == 'cuda'
        num_workers = 0

        self.tr_ld = torch.utils.data.DataLoader(
            self.ds_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn,
            num_workers=num_workers, pin_memory=pin_memory
        )
        self.va_ld = torch.utils.data.DataLoader(
            self.ds_val, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn,
            num_workers=num_workers, pin_memory=pin_memory
        )

        log.info(f"Client {city}: Train {self.num_train_samples}, Val {self.num_val_samples} samples. "
                 f"Model input_dim: {self.input_dim}, n_routes: {self.n_routes}")

        self.optimizer = None
        self.privacy_engine = None

    def _init_optimizer_and_dp(self):
        if not self.model_initialized_correctly or self.model is None:
            return

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=LR, weight_decay=1e-4)

        if DP_ENABLED:
            model_opacus_compatible = ModuleValidator.is_valid(self.model)
            if not model_opacus_compatible:
                log.warning(f"Client {self.city}: Model is not Opacus compatible. Fixing...")
                self.model = ModuleValidator.fix(self.model)
                self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=LR, weight_decay=1e-4)

            if self.num_train_samples > 0:
                self.privacy_engine = PrivacyEngine(
                    module=self.model,
                    sample_rate=BATCH_SIZE / self.num_train_samples,
                    noise_multiplier=NOISE_MULTIPLIER,
                    max_grad_norm=MAX_GRAD_NORM,
                    target_delta=1e-5
                )
                self.privacy_engine.attach(self.optimizer)
                log.info(f"Client {self.city}: Opacus PrivacyEngine attached.")
            else:
                log.warning(f"Client {self.city}: DP_ENABLED but no training samples, cannot attach PrivacyEngine.")
                self.privacy_engine = None

    def get_parameters(self, config):
        if not self.model_initialized_correctly or self.model is None:
            log.warning(f"Client {self.city}: get_parameters called but model not initialized.")
            # Return empty list of parameters if model not initialized
            return []
        return [p.cpu().detach().numpy() for p in self.model.parameters()]

    def fit(self, parameters, config):
        if not self.model_initialized_correctly or self.model is None or self.tr_ld is None or self.num_train_samples == 0:
            log.warning(f"Client {self.city}: fit called but not properly initialized or no data. Skipping.")
            # Return current parameters, 0 examples, and empty metrics
            return self.get_parameters(config), 0, {}

        if self.optimizer is None:
            self._init_optimizer_and_dp()
            if self.optimizer is None and DP_ENABLED and self.privacy_engine is None:
                log.error(f"Client {self.city}: Failed to initialize optimizer/DP. Cannot train.")
                return self.get_parameters(config), 0, {"error": "optimizer/DP init failed"}

        for p_global, p_local in zip(parameters, self.model.parameters()):
            p_local.data.copy_(torch.tensor(p_global, device=DEVICE))

        global_params_tensors = [p.clone().detach() for p in self.model.parameters()]

        self.model.train()
        epoch_losses = []
        for epoch in range(LOCAL_EPOCHS):
            batch_losses = []
            for xb, yb, static_fb, _ in self.tr_ld: # Unpack 4 items, ignore the 4th (datetimes)
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                static_fb = static_fb.to(DEVICE) if static_fb is not None else None

                self.optimizer.zero_grad()
                preds_scaled = self.model(xb, static_feats=static_fb)
                yb_target_scaled = yb[:, 0, :] # Predicting for first horizon step

                loss = F.l1_loss(preds_scaled, yb_target_scaled)

                prox_term = 0.0
                if PROX_MU > 0:
                    for w_local, w_global in zip(self.model.parameters(), global_params_tensors):
                        prox_term += (w_local - w_global).norm(2)**2
                    loss += (PROX_MU / 2) * prox_term

                loss.backward()
                self.optimizer.step()
                batch_losses.append(loss.item())
            
            if batch_losses: # Ensure batch_losses is not empty before calculating mean
                epoch_loss = np.mean(batch_losses)
                epoch_losses.append(epoch_loss)
                log.debug(f"Client {self.city} Epoch {epoch+1}/{LOCAL_EPOCHS} Avg Loss: {epoch_loss:.4f}")
            else:
                log.warning(f"Client {self.city} Epoch {epoch+1}/{LOCAL_EPOCHS}: No batches processed, skipping loss calculation for this epoch.")

        avg_fit_loss = np.mean(epoch_losses) if epoch_losses else float('nan') # Handle case where epoch_losses might be empty
        metrics = {"loss": avg_fit_loss}
        if self.privacy_engine and DP_ENABLED:
            epsilon = self.privacy_engine.get_epsilon(delta=1e-5)
            metrics["epsilon"] = epsilon
            log.info(f"Client {self.city} Fit complete. Avg Loss: {avg_fit_loss:.4f}, Epsilon: {epsilon:.2f}")
        else:
            log.info(f"Client {self.city} Fit complete. Avg Loss: {avg_fit_loss:.4f}")

        return self.get_parameters(None), self.num_train_samples, metrics

    def evaluate(self, parameters, config):
        if not self.model_initialized_correctly or self.model is None or self.va_ld is None or self.num_val_samples == 0:
            log.warning(f"Client {self.city}: evaluate called but not properly initialized or no val data. Skipping.")
            return 0.0, 0, {"mae": 0.0}

        for p_global, p_local in zip(parameters, self.model.parameters()):
            p_local.data.copy_(torch.tensor(p_global, device=DEVICE))

        self.model.eval()
        total_mae = 0.0
        total_loss = 0.0

        if self.ds_val and hasattr(self.ds_val.dataset, 'scaler_means') and hasattr(self.ds_val.dataset, 'scaler_stds'):
            scaler_means_inflow = torch.tensor(self.ds_val.dataset.scaler_means[self.ds_val.dataset.route_ids].values, device=DEVICE).float()
            scaler_stds_inflow = torch.tensor(self.ds_val.dataset.scaler_stds[self.ds_val.dataset.route_ids].values, device=DEVICE).float()
            can_inverse_transform = True
        else:
            log.warning(f"Client {self.city}: Scaler info not found in validation dataset. MAE will be on scaled data.")
            can_inverse_transform = False

        with torch.no_grad():
            for xb, yb, static_fb, _ in self.va_ld: # Unpack 4 items, ignore the 4th (datetimes)
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                static_fb = static_fb.to(DEVICE) if static_fb is not None else None

                preds_scaled = self.model(xb, static_feats=static_fb)
                yb_target_scaled = yb[:, 0, :] # Predicting for first horizon step

                loss = F.l1_loss(preds_scaled, yb_target_scaled)
                total_loss += loss.item() * xb.size(0)

                if can_inverse_transform:
                    preds_orig_scale = preds_scaled * scaler_stds_inflow + scaler_means_inflow
                    yb_target_orig_scale = yb_target_scaled * scaler_stds_inflow + scaler_means_inflow
                    total_mae += np.mean(np.abs(
                        yb_target_orig_scale.cpu().numpy() - preds_orig_scale.cpu().numpy()
                    )) * xb.size(0)
                else:
                    total_mae += np.mean(np.abs(
                        yb_target_scaled.cpu().numpy() - preds_scaled.cpu().numpy()
                    )) * xb.size(0)

        avg_loss = total_loss / self.num_val_samples if self.num_val_samples > 0 else float('nan')
        avg_mae = total_mae / self.num_val_samples if self.num_val_samples > 0 else float('nan')
        
        log.info(f"Client {self.city} Evaluate: MAE={avg_mae:.4f}, Avg Loss (scaled L1)={avg_loss:.4f}")
        return float(avg_loss), self.num_val_samples, {"mae": float(avg_mae)}
