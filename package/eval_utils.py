import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader
from typing import Dict

from config import log, BATCH_SIZE, DEVICE, SEQ_LEN, HORIZON # Import globals
from data_utils import TransitDataset, collate_fn # Import data utilities
from xfedformer_model import XFedFormer # Import the model

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation Helper
# ─────────────────────────────────────────────────────────────────────────────

def quick_metrics(model: XFedFormer, city_dataset: TransitDataset, device: torch.device) -> Dict[str, float]:
    if len(city_dataset) == 0:
        log.warning(
            f"quick_metrics: Dataset for {city_dataset.city_name} is empty.")
        return {"mae": float('nan'), "r2": float('nan'), "rmse": float('nan')}

    data_loader = DataLoader(
        city_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model.eval()
    all_preds_scaled, all_targets_scaled = [], []

    with torch.no_grad():
        # Changed: Unpack 4 values from data_loader, similar to client.py
        for xb, yb, static_fb, _ in data_loader:
            xb, yb = xb.to(device), yb.to(device)
            static_fb = static_fb.to(device)

            preds_s = model(xb, static_feats=static_fb)
            targets_s = yb[:, 0, :]

            all_preds_scaled.append(preds_s.cpu())
            all_targets_scaled.append(targets_s.cpu())

    if not all_preds_scaled:
        return {"mae": float('nan'), "r2": float('nan'), "rmse": float('nan')}

    preds_scaled_np = torch.cat(all_preds_scaled).numpy()
    targets_scaled_np = torch.cat(all_targets_scaled).numpy()

    scaler_means_inflow = city_dataset.scaler_means[city_dataset.route_ids].values
    scaler_stds_inflow = city_dataset.scaler_stds[city_dataset.route_ids].values

    preds_orig_scale = preds_scaled_np * scaler_stds_inflow + scaler_means_inflow
    targets_orig_scale = targets_scaled_np * \
        scaler_stds_inflow + scaler_means_inflow

    mae = float(mean_absolute_error(
        targets_orig_scale.flatten(), preds_orig_scale.flatten()))
    r2 = float(r2_score(targets_orig_scale.flatten(),
               preds_orig_scale.flatten()))
    rmse = float(np.sqrt(((targets_orig_scale - preds_orig_scale)**2).mean()))

    return {"mae": mae, "r2": r2, "rmse": rmse}
