#!/usr/bin/env python3
"""
X-FedFormer — Cross-City Federated Transformer with Differential Privacy
Refactored with modular layers, synthetic data generator, FedProx, and DP.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Requirements: pip install flwr torch torchvision pandas numpy opacus rich scikit-learn
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from rich.console import Console
from rich.table import Table
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader, Dataset, random_split

# ─────────────────────────────────────────────────────────────────────────────
# Globals & Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("xfedformer")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Model dims
D_MODEL = 128  # Reduced for faster example, original: 256
N_HEADS = 4   # Reduced, original: 8
N_LAYERS = 2  # Reduced, original: 4
# Sequence
SEQ_LEN = 24  # Reduced, original: 60
HORIZON = 12
# Training
BATCH_SIZE = 32  # Renamed from BATCH, original: 64
LR = 1e-4       # Original: 3e-4
LOCAL_EPOCHS = 1  # Original: 2
PROX_MU = 0.01
DP_ENABLED = False  # Set to False for quicker debugging, can be True
NOISE_MULTIPLIER = 1.0
MAX_GRAD_NORM = 1.0

# Directories
DATA_DIR = Path("data_cache")
DATA_DIR.mkdir(exist_ok=True)
CKPT_DIR = Path("checkpoints")
CKPT_DIR.mkdir(exist_ok=True)
RESULT_DIR = Path("results")
RESULT_DIR.mkdir(exist_ok=True)

# Holiday definition (example for KZ)
_KZ_HOLIDAYS = {(3, 21), (3, 22), (3, 23), (12, 16)
                }  # Nauryz, Independence Day


def _is_kz_holiday(dt):
    return (dt.month, dt.day) in _KZ_HOLIDAYS

# ─────────────────────────────────────────────────────────────────────────────
# 1) Synthetic Data Generation
# ─────────────────────────────────────────────────────────────────────────────


def generate_synthetic_kz(cities: List[str], days: int, routes_per_city: int = 5) -> Dict[str, pd.DataFrame]:
    """
    Generates per-city synthetic bus passenger flow with weather,
    holiday flags, inflow/outflow, and route metadata.
    """
    out = {}
    log.info(f"Generating synthetic data for cities: {cities}, days: {days}")

    for city_idx, city in enumerate(cities):
        records = []
        # Generate geozones
        zones = [f"zone_{i}" for i in range(1, 4)]  # Reduced for simplicity
        # Create routes
        routes = []
        for rid in range(routes_per_city):
            route_type = np.random.choice(["urban_core", "suburban_feeder"])
            length_km = float(np.random.uniform(5, 30))
            num_stops = int(np.random.uniform(8, 25))
            zone = np.random.choice(zones)
            routes.append({
                # More distinct IDs
                "route_id": f"{city[:3].upper()}_R{rid:02d}",
                "route_type": route_type,
                "length_km": length_km,
                "num_stops": num_stops,
                "zone": zone
            })

        # Simulation start date fixed for reproducibility across cities if days is same
        # Stagger start dates slightly
        sim_start_date = datetime(2023, 1, 1) + timedelta(days=city_idx*days)
        idx = pd.date_range(sim_start_date,
                            periods=days*24, freq="h")

        for r_idx, r_meta in enumerate(routes):
            # Base daily profile (two peaks)
            base_profile = (
                50 +
                # Slight variation in peak
                100 * np.exp(-((idx.hour - (8 + r_idx % 2)) % 24)**2 / 8) +
                80 * np.exp(-((idx.hour - (18 + r_idx % 2)) % 24)**2 / 8)
            )
            # Scale by route popularity (more deterministic based on length/stops)
            popularity_factor = (
                r_meta["num_stops"] / 15.0) * (r_meta["length_km"] / 15.0)
            base_profile *= popularity_factor * \
                (1.2 if r_meta["route_type"] == "urban_core" else 0.8)

            # Day-of-week factor
            dow_factor = np.select(
                [idx.dayofweek < 5, idx.dayofweek == 5, idx.dayofweek == 6],
                [1.0, 0.8, 0.7],  # Weekday, Sat, Sun
                default=1.0
            )
            base_profile *= dow_factor

            # Holiday factor
            holiday_flags = np.array([_is_kz_holiday(dt) for dt in idx])
            # Lower on holidays
            base_profile *= np.where(holiday_flags, 0.5, 1.0)

            # Random events (e.g., festivals, disruptions)
            event_multiplier = np.ones(len(idx))
            for _ in range(days // 10):  # ~1 event per 10 days
                event_start = np.random.randint(0, len(idx) - 24)
                event_duration = np.random.randint(6, 24)
                event_impact = np.random.uniform(
                    1.5, 2.5) if np.random.rand() > 0.3 else np.random.uniform(0.4, 0.7)
                event_multiplier[event_start: event_start +
                                 event_duration] = event_impact
            base_profile *= event_multiplier

            # Weather generator
            # Seasonal temp
            temp_base = 10 * np.sin(2 * np.pi * (idx.dayofyear - 80) / 365)
            temp = temp_base + np.random.normal(0, 3, len(idx)) \
                - 10 * (city_idx % 2)  # Basic city differentiation
            # More rain in some seasons
            precip_prob = 0.05 + 0.1 * \
                (np.sin(2 * np.pi * idx.dayofyear / 365)**2)
            precip = (np.random.rand(len(idx)) < precip_prob).astype(float)

            # Weather effect
            weather_effect = np.ones(len(idx))
            weather_effect[temp < -5] *= 0.8  # Cold
            weather_effect[temp > 30] *= 0.9  # Hot
            weather_effect[precip > 0] *= 0.85  # Precipitation

            final_inflow = base_profile * weather_effect * \
                np.random.uniform(0.9, 1.1, len(idx))
            final_inflow = np.maximum(0, final_inflow).astype(int)

            # Outflow (simplified: roughly proportional to inflow, slightly lagged)
            outflow_ratio = np.random.uniform(0.85, 0.95)
            final_outflow = np.roll(
                final_inflow, shift=np.random.randint(1, 3)) * outflow_ratio
            final_outflow = np.maximum(0, final_outflow).astype(int)

            for i, dt_val in enumerate(idx):
                records.append({
                    "datetime": dt_val,
                    "route_id": r_meta["route_id"],
                    "inflow_count": final_inflow[i],
                    "outflow_count": final_outflow[i],  # Added outflow
                    "temperature": round(float(temp[i]), 2),
                    "precip_flag": int(precip[i]),
                    # Route metadata duplicated per record for easier initial join
                    "route_length_km": r_meta["length_km"],
                    "num_stops": r_meta["num_stops"],
                    "route_type": r_meta["route_type"],
                    "zone": r_meta["zone"]
                })
        out[city] = pd.DataFrame(records)
        log.info(
            f"Generated {len(records)} records for {city} with {routes_per_city} routes.")
    return out

# ─────────────────────────────────────────────────────────────────────────────
# 2) Dataset & Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────


class TransitDataset(Dataset):
    def __init__(self, df: pd.DataFrame, city_name: str, seq_len=SEQ_LEN, horizon=HORIZON):
        self.city_name = city_name
        self.seq_len = seq_len
        self.horizon = horizon

        df = df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"])

        # Pivot route-specific features (inflow is primary target)
        self.route_ids = sorted(list(df["route_id"].unique()))
        self.n_routes = len(self.route_ids)

        # Inflow data
        pivot_inflow = df.pivot(
            index="datetime", columns="route_id", values="inflow_count")[self.route_ids]

        # Exogenous: weather (shared across routes in a city)
        # Take mean if multiple routes reported at same time (should be same)
        weather_feats = df.groupby("datetime")[
            ["temperature", "precip_flag"]].mean()

        # Combine inflows and weather
        processed_df = pd.concat([pivot_inflow, weather_feats], axis=1)

        # Time features derived from the main index
        dt_index = processed_df.index
        processed_df["sin_hour"] = np.sin(2 * np.pi * dt_index.hour / 24.0)
        processed_df["cos_hour"] = np.cos(2 * np.pi * dt_index.hour / 24.0)
        processed_df["day_of_week_sin"] = np.sin(
            2 * np.pi * dt_index.dayofweek / 7.0)
        processed_df["day_of_week_cos"] = np.cos(
            2 * np.pi * dt_index.dayofweek / 7.0)
        processed_df["month_sin"] = np.sin(2 * np.pi * dt_index.month / 12.0)
        processed_df["month_cos"] = np.cos(2 * np.pi * dt_index.month / 12.0)
        processed_df["is_holiday"] = dt_index.to_series().apply(
            _is_kz_holiday).astype(int).values

        # Fill NaNs that might result from pivot/joins (e.g., if a route starts later)
        processed_df.ffill(inplace=True)
        processed_df.bfill(inplace=True)  # For NaNs at the beginning
        processed_df.fillna(0, inplace=True)  # If all are NaN

        self.feature_names = processed_df.columns.tolist()
        # Total features including inflows, weather, time
        self.input_dim = len(self.feature_names)

        # Normalize each feature column (z-score)
        self.scaler_means = processed_df.mean()
        self.scaler_stds = processed_df.std() + 1e-6  # Avoid division by zero
        normalized_arr = (processed_df - self.scaler_means) / self.scaler_stds

        X_data = normalized_arr.values.astype(np.float32)

        # Target data (only inflows, use their original scaled values for Y)
        # We need to find indices of inflow columns in X_data for target extraction
        inflow_indices_in_X = [self.feature_names.index(
            rid) for rid in self.route_ids]
        Y_data_scaled = X_data[:, inflow_indices_in_X]  # Scaled inflows

        # Sliding windows
        self.X, self.Y = [], []
        num_samples = len(X_data) - seq_len - horizon + 1
        if num_samples <= 0:
            log.error(f"Not enough data for {city_name} to create sequences. "
                      f"Data length: {len(X_data)}, SeqLen: {seq_len}, Horizon: {horizon}. "
                      f"Need at least {seq_len + horizon} records.")
            # Create empty tensors to avoid crashing downstream if this dataset is used
            self.X = torch.empty(
                0, seq_len, self.input_dim, dtype=torch.float32)
            self.Y = torch.empty(
                0, horizon, self.n_routes, dtype=torch.float32)
            return

        for i in range(num_samples):
            self.X.append(X_data[i: i + seq_len])
            self.Y.append(Y_data_scaled[i + seq_len: i + seq_len + horizon])

        self.X = torch.tensor(np.stack(self.X), dtype=torch.float32)
        self.Y = torch.tensor(np.stack(self.Y), dtype=torch.float32)

        # Static features per route (length, num_stops)
        static_df = df.drop_duplicates("route_id").set_index(
            "route_id").loc[self.route_ids]
        static_feats_raw = static_df[[
            "route_length_km", "num_stops"]].astype(np.float32)
        self.static_scaler_means = static_feats_raw.mean()
        self.static_scaler_stds = static_feats_raw.std() + 1e-6
        self.static_features_norm = torch.tensor(
            ((static_feats_raw - self.static_scaler_means) /
             self.static_scaler_stds).values,
            dtype=torch.float32
        )  # Shape: (n_routes, n_static_features)

        log.info(f"[{city_name}] Dataset created: X shape {self.X.shape}, Y shape {self.Y.shape}, "
                 f"Static feats shape: {self.static_features_norm.shape}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        # static_features_norm is shared for all samples in this city's dataset
        return self.X[i], self.Y[i], self.static_features_norm


def collate_fn(batch):
    # Batch now contains tuples of (x_sample, y_sample, static_features_for_city)
    # static_features_for_city is the same for all samples in a batch from the same client
    # We only need one copy of static_features for the batch.
    xs, ys, static_features_list = zip(*batch)
    # Assuming all are same for this client's batch
    static_features_batch = static_features_list[0]
    return torch.stack(xs), torch.stack(ys), static_features_batch

# ─────────────────────────────────────────────────────────────────────────────
# 3) Model Components
# ─────────────────────────────────────────────────────────────────────────────


class SeasonalTrendDecomp(nn.Module):
    def __init__(self, kernel_size=7):  # Kernel size should be odd
        super().__init__()
        self.pool = nn.AvgPool1d(kernel_size, stride=1, padding=kernel_size//2)

    def forward(self, x):  # x: (B, T, D)
        trend = self.pool(x.transpose(1, 2)).transpose(1, 2)
        resid = x - trend
        return trend, resid


class SpatialEncoder(nn.Module):
    def __init__(self, n_routes, static_feat_dim, emb_dim):  # emb_dim is D_MODEL
        super().__init__()
        # Simple MLP for static features, as route_id itself isn't used for embedding index here
        # If you had many more routes than D_MODEL, an Embedding layer for route_id might be useful.
        self.mlp = nn.Sequential(
            nn.Linear(static_feat_dim, emb_dim),  # e.g. 2 -> D_MODEL
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )
        log.info(
            f"SpatialEncoder init: n_routes={n_routes}, static_feat_dim={static_feat_dim}, emb_dim={emb_dim}")

    def forward(self, static_route_features):
        # static_route_features: [N_ROUTES, static_feat_dim] (e.g., normalized length, stops)
        # Output: [N_ROUTES, emb_dim]
        return self.mlp(static_route_features)


class CrossModalFusion(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(
            d_model, d_model*2), nn.ReLU(), nn.Linear(d_model*2, d_model), nn.Dropout(0.1))
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x_q, x_kv):  # x_q: [B,T,D], x_kv: [B,T,D]
        attn_out, _ = self.cross_attn(x_q, x_kv, x_kv)
        x_q = self.norm(x_q + attn_out)
        ff_out = self.ff(x_q)
        x_q = self.norm2(x_q + ff_out)
        return x_q


class MoEBlock(nn.Module):
    def __init__(self, d_model, n_experts=4, top_k=2):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_model*2),  # Wider experts
                          nn.ReLU(),
                          nn.Linear(d_model*2, d_model),
                          nn.Dropout(0.1))
            for _ in range(n_experts)
        ])
        self.gating = nn.Linear(d_model, n_experts)
        self.top_k = top_k
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):  # x: [B,T,D]
        res_conn = x
        gating_logits = self.gating(x)  # [B,T,E]

        # Sparsely select top_k experts
        top_k_weights, top_k_indices = torch.topk(
            gating_logits, self.top_k, dim=-1)  # [B,T,k], [B,T,k]
        # Softmax over top_k experts
        top_k_weights = F.softmax(top_k_weights, dim=-1)

        expert_outputs_list = []
        for i in range(self.top_k):
            indices_i = top_k_indices[..., i]  # [B,T]
            # Create a flat batch of inputs for selected experts
            flat_x = x.reshape(-1, x.size(-1))  # [B*T, D]
            flat_indices_i = indices_i.reshape(-1)  # [B*T]

            # Dispatch: collect inputs for each expert
            # This is a simplified dispatch; for performance, more advanced methods exist
            current_expert_outputs = torch.zeros_like(flat_x)  # [B*T, D]
            for exp_idx in range(len(self.experts)):
                mask = (flat_indices_i == exp_idx)
                if mask.any():
                    # Apply expert exp_idx to inputs x[mask]
                    selected_inputs = flat_x[mask]
                    current_expert_outputs[mask] = self.experts[exp_idx](
                        selected_inputs)

            # Weight and sum
            expert_outputs_list.append(current_expert_outputs.reshape_as(
                x) * top_k_weights[..., i].unsqueeze(-1))

        mixed = sum(expert_outputs_list)  # [B,T,D]
        return self.norm(mixed + res_conn)


class XFedFormer(nn.Module):
    def __init__(self, input_dim: int, n_routes: int, n_static_feats: int,
                 d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
                 seq_len=SEQ_LEN, horizon=HORIZON):
        super().__init__()
        self.n_routes = n_routes
        self.input_dim = input_dim  # Full input dimension from dataset
        log.info(
            f"XFedFormer init: input_dim={input_dim}, n_routes={n_routes}, d_model={d_model}")

        # Kernel size should be odd, e.g., 7 or 25 for daily seasonality on hourly data
        self.decomp = SeasonalTrendDecomp(kernel_size=7)

        # Projection from input_dim (e.g., N_ROUTES_inflow + N_WEATHER + N_TIME_FEATS) to D_MODEL
        self.input_projection = nn.Linear(input_dim, d_model)

        # SpatialEncoder for static route features (length, stops, etc.)
        # Output of spatial encoder will be [N_ROUTES, D_MODEL]
        # self.spatial_encoder = SpatialEncoder(n_routes, n_static_feats, d_model)
        # Note: Proper integration of spatial_encoder output requires careful thought
        # on how [N_ROUTES, D_MODEL] combines with [B, T, D_MODEL] temporal features.
        # For now, we won't use its output directly in the main temporal path to avoid dim issues.

        # Positional encoding for sequence length T
        self.pos_enc = nn.Parameter(torch.randn(seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * 4, batch_first=True, dropout=0.1, activation='gelu'
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer, n_layers, norm=nn.LayerNorm(d_model))

        # Example for CrossModalFusion: if you had separate exogenous features projected to d_model
        # self.exog_projection = nn.Linear(n_exog_features, d_model)
        # self.cross_modal_fusion = CrossModalFusion(d_model, n_heads)

        self.moe_block = MoEBlock(d_model, n_experts=4, top_k=2)

        # Decoder to forecast N_ROUTES from D_MODEL representation
        # It predicts the *scaled* values. Inverse transform will be outside.
        # Predicts N_ROUTES (e.g., inflows)
        self.decoder = nn.Linear(d_model, n_routes)

    def forward(self, x_series: torch.Tensor, static_feats: Optional[torch.Tensor] = None,
                route_indices: Optional[torch.Tensor] = None):
        # x_series: [B, T, FullInputDim] (contains inflows, weather, time feats)
        # static_feats: [N_ROUTES, N_STATIC_FEATS] (e.g. length, num_stops for all routes in city)
        # route_indices: [N_ROUTES] (e.g. torch.arange(N_ROUTES)) - not used in this simplified version

        B, T, _ = x_series.shape

        # 1. Decomposition
        trend_full, resid_full = self.decomp(
            x_series)  # Both are [B, T, FullInputDim]

        # 2. Input Projection of residual part
        # We project the residual of the *full* input series to d_model
        z = self.input_projection(resid_full)  # [B, T, D_MODEL]

        # 3. Add Positional Encoding
        z = z + self.pos_enc[:T]  # Add PE to the D_MODEL representation

        # (Optional) Spatial Encoding / Conditioning - Placeholder for future refinement
        # if static_feats is not None and hasattr(self, 'spatial_encoder'):
        #     sp_emb = self.spatial_encoder(static_feats) # [N_ROUTES, D_MODEL]
        #     # How to combine sp_emb with z?
        #     # Example: Average spatial embedding and add as a bias
        #     # global_sp_bias = sp_emb.mean(dim=0) # [D_MODEL]
        #     # z = z + global_sp_bias # Add to all tokens and batches
        #     pass # Needs careful design

        # (Optional) Cross-Modal Fusion - Placeholder
        # if x_exog is not None and hasattr(self, 'cross_modal_fusion'):
        #     # Assuming x_exog was projected to exog_repr [B,T,D_MODEL]
        #     # z = self.cross_modal_fusion(z, exog_repr)
        #     pass

        # 4. Transformer Temporal Encoding
        z = self.temporal_encoder(z)  # [B, T, D_MODEL]

        # 5. Mixture-of-Experts Block
        z = self.moe_block(z)  # [B, T, D_MODEL]

        # 6. Decoder: Forecast N_ROUTES from the last time step's D_MODEL representation
        # Taking representation from the last sequence token
        forecast_scaled = self.decoder(z[:, -1, :])  # [B, N_ROUTES]

        # 7. Trend Add-Back
        # The trend component should be for the N_ROUTES we are predicting (e.g., inflows)
        # Assumes the first N_ROUTES features in x_series (and thus in trend_full) are the target series
        trend_to_add = trend_full[:, -1, :self.n_routes]  # [B, N_ROUTES]

        final_forecast_scaled = forecast_scaled + trend_to_add  # [B, N_ROUTES]

        return final_forecast_scaled  # This is still in scaled domain

# ─────────────────────────────────────────────────────────────────────────────
# 4) Federated Client with FedProx & DP
# ─────────────────────────────────────────────────────────────────────────────


class FedProxClient(fl.client.NumPyClient):
    def __init__(self, city: str, df: pd.DataFrame, days_data: int):
        self.city = city
        self.model: Optional[XFedFormer] = None
        self.dataset_params = {"seq_len": SEQ_LEN, "horizon": HORIZON}

        full_ds = TransitDataset(df, city_name=city, **self.dataset_params)

        if len(full_ds) == 0:
            log.warning(f"Client {city}: Dataset is empty. Skipping client.")
            # Flower client needs to be able to return empty parameters if it can't train
            self.tr_ld, self.va_ld = None, None
            self.num_train_samples = 0
            self.num_val_samples = 0
            # Minimal model for parameter exchange if absolutely necessary, but training/eval won't work
            self.model = XFedFormer(
                input_dim=10, n_routes=2, n_static_feats=2)  # Dummy params
            self.model_initialized_correctly = False
            return

        self.model_initialized_correctly = True
        self.n_routes = full_ds.n_routes
        self.input_dim = full_ds.input_dim
        self.n_static_feats = full_ds.static_features_norm.shape[1]

        # Initialize model here to get its structure based on data
        self.model = XFedFormer(
            input_dim=self.input_dim,
            n_routes=self.n_routes,
            n_static_feats=self.n_static_feats,
            d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
            seq_len=SEQ_LEN, horizon=HORIZON
        ).to(DEVICE)

        # Split data
        n_total = len(full_ds)
        n_train = int(n_total * 0.8)
        n_val = n_total - n_train

        if n_train == 0 or n_val == 0:
            log.warning(
                f"Client {city}: Not enough samples for train/val split. Train: {n_train}, Val: {n_val}")
            # Adjust to use all available for training if validation is impossible
            if n_total > 0 and n_train == 0:
                n_train = n_total
                n_val = 0
            if n_total > 0 and n_val == 0 and n_train > 0:
                pass  # Use all for training, no val
            else:  # Still problematic
                self.tr_ld, self.va_ld = None, None
                self.num_train_samples = 0
                self.num_val_samples = 0
                self.model_initialized_correctly = False  # Mark as not properly usable
                return

        self.ds_train, self.ds_val = random_split(
            full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42)
        )
        self.num_train_samples = len(self.ds_train)
        self.num_val_samples = len(self.ds_val)

        pin_memory = DEVICE.type == 'cuda'
        num_workers = 0  # Keep 0 for simplicity, especially with smaller datasets / debugging

        self.tr_ld = DataLoader(
            self.ds_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn,
            num_workers=num_workers, pin_memory=pin_memory
        )
        self.va_ld = DataLoader(
            self.ds_val, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn,
            num_workers=num_workers, pin_memory=pin_memory
        )

        log.info(f"Client {city}: Train {self.num_train_samples}, Val {self.num_val_samples} samples. "
                 f"Model input_dim: {self.input_dim}, n_routes: {self.n_routes}")

        # Optimizer and Privacy Engine (if enabled) are initialized before first fit
        self.optimizer = None
        self.privacy_engine = None

    def _init_optimizer_and_dp(self):
        if not self.model_initialized_correctly or self.model is None:
            return

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=LR, weight_decay=1e-4)

        if DP_ENABLED:
            # Opacus validation
            model_opacus_compatible = ModuleValidator.is_valid(self.model)
            if not model_opacus_compatible:
                log.warning(
                    f"Client {self.city}: Model is not Opacus compatible. Fixing...")
                self.model = ModuleValidator.fix(self.model)
                # Re-init optimizer with potentially fixed model params
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(), lr=LR, weight_decay=1e-4)

            if self.num_train_samples > 0:  # sample_rate must be > 0
                self.privacy_engine = PrivacyEngine(
                    module=self.model,
                    sample_rate=BATCH_SIZE / self.num_train_samples,  # sample_rate per batch
                    noise_multiplier=NOISE_MULTIPLIER,
                    max_grad_norm=MAX_GRAD_NORM,
                    target_delta=1e-5  # Common delta value
                )
                self.privacy_engine.attach(self.optimizer)
                log.info(f"Client {self.city}: Opacus PrivacyEngine attached.")
            else:
                log.warning(
                    f"Client {self.city}: DP_ENABLED but no training samples, cannot attach PrivacyEngine.")
                self.privacy_engine = None  # Ensure it's None

    def get_parameters(self, config):
        if not self.model_initialized_correctly or self.model is None:
            log.warning(
                f"Client {self.city}: get_parameters called but model not initialized.")
            return []  # Return empty list if model isn't there
        return [p.cpu().detach().numpy() for p in self.model.parameters()]

    def fit(self, parameters, config):
        if not self.model_initialized_correctly or self.model is None or self.tr_ld is None or self.num_train_samples == 0:
            log.warning(
                f"Client {self.city}: fit called but not properly initialized or no data. Skipping.")
            # Return current (dummy) params, 0 samples
            return self.get_parameters(config), 0, {}

        if self.optimizer is None:  # First time fit is called
            self._init_optimizer_and_dp()
            if self.optimizer is None and DP_ENABLED and self.privacy_engine is None:
                log.error(
                    f"Client {self.city}: Failed to initialize optimizer/DP. Cannot train.")
                return self.get_parameters(config), 0, {"error": "optimizer/DP init failed"}

        for p_global, p_local in zip(parameters, self.model.parameters()):
            p_local.data.copy_(torch.tensor(p_global, device=DEVICE))

        global_params_tensors = [p.clone().detach()
                                 for p in self.model.parameters()]

        self.model.train()
        epoch_losses = []
        for epoch in range(LOCAL_EPOCHS):
            batch_losses = []
            # static_fb is [N_ROUTES, N_STATIC_FEATS]
            for xb, yb, static_fb in self.tr_ld:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                # Pass static features if model uses them
                static_fb = static_fb.to(DEVICE)

                self.optimizer.zero_grad()

                # Model expects x_series, static_feats (optional)
                preds_scaled = self.model(
                    xb, static_feats=static_fb)  # [B, N_ROUTES]

                # Target: use the first step of the horizon
                # yb is [B, HORIZON, N_ROUTES], so yb_target is [B, N_ROUTES]
                yb_target_scaled = yb[:, 0, :]

                loss = F.l1_loss(preds_scaled, yb_target_scaled)

                # FedProx proximal term
                prox_term = 0.0
                if PROX_MU > 0:
                    for w_local, w_global in zip(self.model.parameters(), global_params_tensors):
                        # L2 norm squared
                        prox_term += (w_local - w_global).norm(2)**2
                    loss += (PROX_MU / 2) * prox_term

                loss.backward()
                self.optimizer.step()
                batch_losses.append(loss.item())
            epoch_loss = np.mean(batch_losses)
            epoch_losses.append(epoch_loss)
            log.debug(
                f"Client {self.city} Epoch {epoch+1}/{LOCAL_EPOCHS} Avg Loss: {epoch_loss:.4f}")

        avg_fit_loss = np.mean(epoch_losses)
        metrics = {"loss": avg_fit_loss}
        if self.privacy_engine and DP_ENABLED:  # Check privacy_engine exists
            epsilon = self.privacy_engine.get_epsilon(
                delta=1e-5)  # Use same delta
            metrics["epsilon"] = epsilon
            log.info(
                f"Client {self.city} Fit complete. Avg Loss: {avg_fit_loss:.4f}, Epsilon: {epsilon:.2f}")
        else:
            log.info(
                f"Client {self.city} Fit complete. Avg Loss: {avg_fit_loss:.4f}")

        return self.get_parameters(None), self.num_train_samples, metrics

    def evaluate(self, parameters, config):
        if not self.model_initialized_correctly or self.model is None or self.va_ld is None or self.num_val_samples == 0:
            log.warning(
                f"Client {self.city}: evaluate called but not properly initialized or no val data. Skipping.")
            return 0.0, 0, {"mae": 0.0}  # Return 0 loss, 0 samples, 0 mae

        for p_global, p_local in zip(parameters, self.model.parameters()):
            p_local.data.copy_(torch.tensor(p_global, device=DEVICE))

        self.model.eval()
        total_mae = 0.0
        total_loss = 0.0  # L1 loss for consistency with training

        # Get dataset scalers for inverse transform
        # Assuming full_ds was split into self.ds_train and self.ds_val, they share the same underlying full_ds object
        # Need to access the scaler from the original TransitDataset object
        # This is a bit tricky with random_split Subsets.
        # A cleaner way would be to pass scaler info through config or store it more accessibly.
        # For now, try to access from one of the Subset's dataset attribute.
        if self.ds_val and hasattr(self.ds_val.dataset, 'scaler_means') and hasattr(self.ds_val.dataset, 'scaler_stds'):
            scaler_means_inflow = self.ds_val.dataset.scaler_means[
                self.ds_val.dataset.route_ids].values
            scaler_stds_inflow = self.ds_val.dataset.scaler_stds[self.ds_val.dataset.route_ids].values
            scaler_means_inflow = torch.tensor(
                scaler_means_inflow, device=DEVICE).float()
            scaler_stds_inflow = torch.tensor(
                scaler_stds_inflow, device=DEVICE).float()
            can_inverse_transform = True
        else:
            log.warning(
                f"Client {self.city}: Scaler info not found in validation dataset. MAE will be on scaled data.")
            can_inverse_transform = False

        with torch.no_grad():
            for xb, yb, static_fb in self.va_ld:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                static_fb = static_fb.to(DEVICE)

                preds_scaled = self.model(
                    xb, static_feats=static_fb)  # [B, N_ROUTES]
                yb_target_scaled = yb[:, 0, :]  # [B, N_ROUTES]

                loss = F.l1_loss(preds_scaled, yb_target_scaled)
                total_loss += loss.item() * xb.size(0)

                if can_inverse_transform:
                    preds_orig_scale = preds_scaled * scaler_stds_inflow + scaler_means_inflow
                    yb_target_orig_scale = yb_target_scaled * \
                        scaler_stds_inflow + scaler_means_inflow
                    total_mae += mean_absolute_error(
                        yb_target_orig_scale.cpu().numpy().flatten(),
                        preds_orig_scale.cpu().numpy().flatten()
                    ) * xb.size(0)  # MAE on original scale
                else:  # Fallback to MAE on scaled data
                    total_mae += mean_absolute_error(
                        yb_target_scaled.cpu().numpy().flatten(),
                        preds_scaled.cpu().numpy().flatten()
                    ) * xb.size(0)

        avg_loss = total_loss / self.num_val_samples
        avg_mae = total_mae / self.num_val_samples
        log.info(
            f"Client {self.city} Evaluate: MAE={avg_mae:.4f}, Avg Loss (scaled L1)={avg_loss:.4f}")
        return float(avg_loss), self.num_val_samples, {"mae": float(avg_mae)}


# ─────────────────────────────────────────────────────────────────────────────
# 5) Server Strategy
# ─────────────────────────────────────────────────────────────────────────────

class FedProxStrategy(fl.server.strategy.FedAvg):
    def __init__(self, initial_parameters: Optional[fl.common.Parameters] = None, **kwargs):
        super().__init__(
            initial_parameters=initial_parameters,
            fraction_fit=1.0,       # Sample all clients for training
            fraction_evaluate=1.0,  # Sample all clients for validation
            min_fit_clients=1,      # Minimum clients to proceed with training
            min_evaluate_clients=1,  # Minimum clients for validation
            min_available_clients=1,  # Wait for at least this many clients
            **kwargs
        )
        log.info("FedProxStrategy initialized with FedAvg base.")

    def configure_fit(self, server_round: int, parameters: fl.common.Parameters, client_manager: fl.server.client_manager.ClientManager):
        # This is where server can send round-specific configs to clients
        config = {"server_round": server_round,
                  "prox_mu": PROX_MU, "local_epochs": LOCAL_EPOCHS}
        fit_ins = super().configure_fit(server_round, parameters, client_manager)
        # Update config for each client instruction
        for _, ins in fit_ins:
            ins.config.update(config)
        return fit_ins

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:

        aggregated_parameters, aggregated_metrics = super(
        ).aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            log.info(
                f"Round {server_round}: Aggregation complete. Saving global model.")
            try:
                # Convert Parameters to list of NumPy arrays
                weights_list = fl.common.parameters_to_ndarrays(
                    aggregated_parameters)

                # Save model (example: save as PyTorch state_dict if structure is known,
                # or just raw weights. For raw weights, need to load carefully)
                # Here, we save the raw list of ndarrays.
                model_path = CKPT_DIR / f"global_model_round_{server_round}.pt"
                # For the 'evaluate' script, let's also save a consistent 'global_model.pt'
                latest_model_path = CKPT_DIR / "global_model.pt"

                torch.save(weights_list, model_path)  # Save as list of arrays
                torch.save(weights_list, latest_model_path)
                log.info(
                    f"Global model saved to {model_path} and {latest_model_path}")

            except Exception as e:
                log.error(
                    f"Could not save global model in round {server_round}: {e}")

        # Aggregate custom metrics from clients (e.g., epsilon if DP is on)
        if results:
            epsilons = [r.metrics.get("epsilon", float('inf'))
                        for _, r in results if r.metrics]
            losses = [r.metrics.get("loss", float(
                'nan')) * r.num_examples for _, r in results if r.metrics]
            num_examples_total = sum(r.num_examples for _, r in results)

            if num_examples_total > 0:
                aggregated_metrics["avg_client_loss"] = sum(
                    losses) / num_examples_total

            if DP_ENABLED and epsilons:
                finite_epsilons = [e for e in epsilons if e != float('inf')]
                if finite_epsilons:
                    aggregated_metrics["max_epsilon"] = max(finite_epsilons)
                    aggregated_metrics["avg_epsilon"] = np.mean(
                        finite_epsilons)

        return aggregated_parameters, aggregated_metrics

    def configure_evaluate(self, server_round: int, parameters: fl.common.Parameters, client_manager: fl.server.client_manager.ClientManager):
        config = {"server_round": server_round}
        evaluate_ins = super().configure_evaluate(
            server_round, parameters, client_manager)
        for _, ins in evaluate_ins:
            ins.config.update(config)
        return evaluate_ins

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, fl.common.Scalar]]:

        # Default aggregation for loss (usually weighted average of client losses)
        loss_aggregated, metrics_aggregated = super(
        ).aggregate_evaluate(server_round, results, failures)

        # Aggregate custom metrics like MAE
        if results:
            maes = [r.metrics.get("mae", float('nan')) *
                    r.num_examples for _, r in results if r.metrics]
            num_examples_total = sum(r.num_examples for _, r in results)
            if num_examples_total > 0 and not np.isnan(maes).all():
                metrics_aggregated["avg_mae"] = sum(
                    m for m in maes if not np.isnan(m)) / num_examples_total
            else:
                metrics_aggregated["avg_mae"] = float('nan')
            log.info(
                f"Round {server_round} evaluation: Loss Aggregated={loss_aggregated:.4f}, Avg MAE={metrics_aggregated.get('avg_mae', float('nan')):.4f}")

        return loss_aggregated, metrics_aggregated


# ─────────────────────────────────────────────────────────────────────────────
# 6) Evaluation Helper (for `evaluate` CLI command)
# ─────────────────────────────────────────────────────────────────────────────

def quick_metrics(model: XFedFormer, city_dataset: TransitDataset, device: torch.device) -> Dict[str, float]:
    if len(city_dataset) == 0:
        log.warning(
            f"quick_metrics: Dataset for {city_dataset.city_name} is empty.")
        return {"mae": float('nan'), "r2": float('nan'), "rmse": float('nan')}

    # Use a DataLoader for consistency, even if batch_size is large
    data_loader = DataLoader(
        city_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model.eval()
    all_preds_scaled, all_targets_scaled = [], []

    with torch.no_grad():
        for xb, yb, static_fb in data_loader:
            xb, yb = xb.to(device), yb.to(device)
            static_fb = static_fb.to(device)

            preds_s = model(xb, static_feats=static_fb)  # [B, N_ROUTES]
            # [B, N_ROUTES] (first step of horizon)
            targets_s = yb[:, 0, :]

            all_preds_scaled.append(preds_s.cpu())
            all_targets_scaled.append(targets_s.cpu())

    if not all_preds_scaled:  # No data processed
        return {"mae": float('nan'), "r2": float('nan'), "rmse": float('nan')}

    preds_scaled_np = torch.cat(all_preds_scaled).numpy()
    targets_scaled_np = torch.cat(all_targets_scaled).numpy()

    # Inverse transform to original scale for metrics
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


# ─────────────────────────────────────────────────────────────────────────────
# 7) CLI Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="X-FedFormer CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser(
        "generate-data", help="Generate synthetic multi-city transit data.")
    g.add_argument("--cities", nargs="+", default=["Almaty", "Astana", "Karaganda",
                   "Shymkent", "Aktobe", "Pavlodar", "Taraz", "Atyrau", "Kostanay", "Aktau"])
    g.add_argument("--days", type=int, default=30,
                   help="Number of days for synthetic data per city.")
    g.add_argument("--routes-per-city", type=int, default=10,
                   help="Number of routes per city.")

    srv = sub.add_parser("server", help="Start Federated Learning server.")
    srv.add_argument("--rounds", type=int, default=5,
                     help="Number of federated rounds.")
    srv.add_argument("--initial_model_path", type=str, default=None,
                     help="Path to initial global model (list of ndarrays).")

    cli_p = sub.add_parser(
        "client", help="Launch a federated client for a specific city.")
    cli_p.add_argument("--city", required=True,
                       help="City name for this client.")
    cli_p.add_argument("--days-data", type=int, default=30,
                       help="Number of days of data to load (must match generated).")
    cli_p.add_argument("--server_address", type=str, default="127.0.0.1:8080")

    ev = sub.add_parser(
        "evaluate", help="Evaluate a global model checkpoint on specified cities.")
    ev.add_argument("--cities", nargs="+", default=["Almaty", "Astana", "Karaganda",
                   "Shymkent", "Aktobe", "Pavlodar", "Taraz", "Atyrau", "Kostanay", "Aktau"])
    ev.add_argument("--days-data", type=int, default=30,
                    help="Number of days of data to load.")
    ev.add_argument("--model-path", type=str, default=str(CKPT_DIR / "global_model.pt"),
                    help="Path to saved global model weights (list of ndarrays).")
    ev.add_argument("--report-file",
                    default=str(RESULT_DIR / "evaluation_report.json"))

    args = ap.parse_args()
    log.info(f"Executing command: {args.cmd} with args: {vars(args)}")

    if args.cmd == "generate-data":
        console.rule(f"[bold cyan]Generating Synthetic Data for {args.cities}")
        synth_data = generate_synthetic_kz(
            args.cities, args.days, args.routes_per_city)
        for city, df_city in synth_data.items():
            out_path = DATA_DIR / \
                f"{city}_{args.days}days_routes{args.routes_per_city}.csv"
            df_city.to_csv(out_path, index=False)
            log.info(
                f"Generated data for {city} → {out_path} ({len(df_city)} rows)")

    elif args.cmd == "server":
        console.rule("[bold cyan]Flower Server Starting")
        initial_params = None
        if args.initial_model_path:
            try:
                weights_list = torch.load(
                    args.initial_model_path, map_location=torch.device('cpu'))
                initial_params = fl.common.ndarrays_to_parameters(weights_list)
                log.info(
                    f"Loaded initial model from {args.initial_model_path}")
            except Exception as e:
                log.error(
                    f"Could not load initial model from {args.initial_model_path}: {e}. Starting with random init.")

        strategy = FedProxStrategy(initial_parameters=initial_params)
        fl.server.start_server(
            server_address="0.0.0.0:8080",  # Listen on all interfaces
            config=fl.server.ServerConfig(num_rounds=args.rounds),
            strategy=strategy,
        )
        log.info("Flower server finished.")

    elif args.cmd == "client":
        # Determine filename based on typical generation parameters if not perfectly matched
        # This is a simplification; robust solution would query available files or use exact params.
        # For now, assume user provides matching --days-data for a generated file.
        # We need to know routes_per_city to load the correct file. This is a small gap.
        # Let's assume a common routes_per_city or require it as arg for client if files are named with it.
        # For now, I'll assume a default or try to find one.
        # A better way: list files and pick one matching city and days.

        possible_files = list(DATA_DIR.glob(
            f"{args.city}_{args.days_data}days_routes*.csv"))
        if not possible_files:
            log.error(f"No data file found for {args.city} with {args.days_data} days. "
                      f"Looked for: {DATA_DIR}/{args.city}_{args.days_data}days_routes*.csv")
            return

        data_file_path = possible_files[0]  # Take the first match
        log.info(f"Loading data for client {args.city} from {data_file_path}")

        try:
            df_city = pd.read_csv(data_file_path, parse_dates=["datetime"])
        except FileNotFoundError:
            log.error(f"Data file not found: {data_file_path}")
            return

        client = FedProxClient(city=args.city, df=df_city,
                               days_data=args.days_data)
        if not client.model_initialized_correctly or client.num_train_samples == 0:
            log.error(
                f"Client {args.city} could not be initialized properly or has no training data. Aborting.")
            return

        console.rule(f"[bold green]Flower Client: {args.city}")
        # Use .to_client() for NumPyClient
        fl.client.start_client(
            server_address=args.server_address, client=client.to_client())
        log.info(f"Client {args.city} finished.")

    elif args.cmd == "evaluate":
        console.rule(
            f"[bold yellow]Evaluating Global Model: {args.model_path}")
        model_file_path = Path(args.model_path)
        if not model_file_path.exists():
            log.error(
                f"Global model checkpoint not found at: {args.model_path}")
            return

        try:
            # ******** THE FIX IS HERE ********
            global_weights_list = torch.load(
                model_file_path, map_location=DEVICE, weights_only=False)
            # *******************************
        except Exception as e:
            log.error(
                f"Error loading model weights from {args.model_path}: {e}")
            return

        results_summary = {}
        table = Table(
            title=f"Global Model Evaluation ({model_file_path.name})")
        table.add_column("City", style="cyan")
        table.add_column("MAE", style="magenta")
        table.add_column("R²", style="green")
        table.add_column("RMSE", style="yellow")

        for city_to_eval in args.cities:
            log.info(f"Evaluating city: {city_to_eval}")

            possible_files = list(DATA_DIR.glob(
                f"{city_to_eval}_{args.days_data}days_routes*.csv"))
            if not possible_files:
                log.warning(
                    f"No data file found for evaluation of {city_to_eval} with {args.days_data} days. Skipping.")
                table.add_row(city_to_eval, "N/A (no data)", "N/A", "N/A")
                continue

            data_file_path_eval = possible_files[0]
            df_eval_city = pd.read_csv(
                data_file_path_eval, parse_dates=["datetime"])

            temp_eval_dataset = TransitDataset(
                df_eval_city, city_name=city_to_eval, seq_len=SEQ_LEN, horizon=HORIZON)
            if len(temp_eval_dataset) == 0:
                log.warning(
                    f"Evaluation dataset for {city_to_eval} is empty after processing. Skipping.")
                table.add_row(
                    city_to_eval, "N/A (empty dataset)", "N/A", "N/A")
                continue

            eval_model = XFedFormer(
                input_dim=temp_eval_dataset.input_dim,
                n_routes=temp_eval_dataset.n_routes,
                n_static_feats=temp_eval_dataset.static_features_norm.shape[1],
                d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
                seq_len=SEQ_LEN, horizon=HORIZON
            ).to(DEVICE)

            current_model_params = list(eval_model.parameters())
            if len(global_weights_list) != len(current_model_params):
                log.error(
                    f"Mismatched parameter count for {city_to_eval}. Model has {len(current_model_params)}, loaded weights have {len(global_weights_list)}. Skipping.")
                table.add_row(
                    city_to_eval, "N/A (param mismatch)", "N/A", "N/A")
                continue

            try:
                for i, (p_loaded, p_model) in enumerate(zip(global_weights_list, current_model_params)):
                    # Ensure p_loaded is converted to tensor
                    p_model.data.copy_(torch.tensor(p_loaded, device=DEVICE))
            except Exception as e:
                log.error(
                    f"Error setting model parameters for {city_to_eval}: {e}. Check model architecture consistency.")
                table.add_row(
                    city_to_eval, "N/A (param set error)", "N/A", "N/A")
                continue

            city_metrics = quick_metrics(eval_model, temp_eval_dataset, DEVICE)
            results_summary[city_to_eval] = city_metrics
            table.add_row(
                city_to_eval,
                f"{city_metrics['mae']:.3f}" if not np.isnan(
                    city_metrics['mae']) else "N/A",
                f"{city_metrics['r2']:.3f}" if not np.isnan(
                    city_metrics['r2']) else "N/A",
                f"{city_metrics['rmse']:.3f}" if not np.isnan(
                    city_metrics['rmse']) else "N/A"
            )

        console.print(table)
        report_path = Path(args.report_file)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(results_summary, f, indent=2)
        log.info(f"Evaluation report saved to → {report_path}")

    else:
        log.error(f"Unknown command: {args.cmd}")
        ap.print_help()


if __name__ == "__main__":
    # For Windows compatibility with multiprocessing, if num_workers > 0 in DataLoader
    from multiprocessing import freeze_support
    freeze_support()
    main()
