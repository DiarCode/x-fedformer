import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader, Dataset, random_split

from config import log, SEQ_LEN, HORIZON, KZ_HOLIDAYS

def _is_kz_holiday(dt):
    return (dt.month, dt.day) in KZ_HOLIDAYS

def generate_synthetic_kz(cities: List[str], days: int, routes_per_city: int = 5) -> Dict[str, pd.DataFrame]:
    """Generates per-city synthetic bus passenger flow with various features."""
    out = {}
    log.info(f"Generating synthetic data for cities: {cities}, days: {days}")

    for city_idx, city in enumerate(cities):
        records = []
        zones = [f"zone_{i}" for i in range(1, 4)]
        routes = []
        for rid in range(routes_per_city):
            route_type = np.random.choice(["urban_core", "suburban_feeder"])
            length_km = float(np.random.uniform(5, 30))
            num_stops = int(np.random.uniform(8, 25))
            zone = np.random.choice(zones)
            routes.append({
                "route_id": f"{city[:3].upper()}_R{rid:02d}",
                "route_type": route_type,
                "length_km": length_km,
                "num_stops": num_stops,
                "zone": zone
            })

        sim_start_date = datetime(2023, 1, 1) + timedelta(days=city_idx*days)
        idx = pd.date_range(sim_start_date, periods=days*24, freq="h")

        for r_idx, r_meta in enumerate(routes):
            base_profile = (
                50 +
                100 * np.exp(-((idx.hour - (8 + r_idx % 2)) % 24)**2 / 8) +
                80 * np.exp(-((idx.hour - (18 + r_idx % 2)) % 24)**2 / 8)
            )
            popularity_factor = (r_meta["num_stops"] / 15.0) * (r_meta["length_km"] / 15.0)
            base_profile *= popularity_factor * (1.2 if r_meta["route_type"] == "urban_core" else 0.8)

            dow_factor = np.select([idx.dayofweek < 5, idx.dayofweek == 5, idx.dayofweek == 6], [1.0, 0.8, 0.7], default=1.0)
            base_profile *= dow_factor

            holiday_flags = np.array([_is_kz_holiday(dt) for dt in idx])
            base_profile *= np.where(holiday_flags, 0.5, 1.0)

            event_multiplier = np.ones(len(idx))
            for _ in range(days // 10):
                event_start = np.random.randint(0, len(idx) - 24)
                event_duration = np.random.randint(6, 24)
                event_impact = np.random.uniform(1.5, 2.5) if np.random.rand() > 0.3 else np.random.uniform(0.4, 0.7)
                event_multiplier[event_start: event_start + event_duration] = event_impact
            base_profile *= event_multiplier

            temp_base = 10 * np.sin(2 * np.pi * (idx.dayofyear - 80) / 365)
            temp = temp_base + np.random.normal(0, 3, len(idx)) - 10 * (city_idx % 2)
            precip_prob = 0.05 + 0.1 * (np.sin(2 * np.pi * idx.dayofyear / 365)**2)
            precip = (np.random.rand(len(idx)) < precip_prob).astype(float)

            weather_effect = np.ones(len(idx))
            weather_effect[temp < -5] *= 0.8
            weather_effect[temp > 30] *= 0.9
            weather_effect[precip > 0] *= 0.85

            final_inflow = base_profile * weather_effect * np.random.uniform(0.9, 1.1, len(idx))
            final_inflow = np.maximum(0, final_inflow).astype(int)

            outflow_ratio = np.random.uniform(0.85, 0.95)
            final_outflow = np.roll(final_inflow, shift=np.random.randint(1, 3)) * outflow_ratio
            final_outflow = np.maximum(0, final_outflow).astype(int)

            for i, dt_val in enumerate(idx):
                records.append({
                    "datetime": dt_val,
                    "route_id": r_meta["route_id"],
                    "inflow_count": final_inflow[i],
                    "outflow_count": final_outflow[i],
                    "temperature": round(float(temp[i]), 2),
                    "precip_flag": int(precip[i]),
                    "route_length_km": r_meta["length_km"],
                    "num_stops": r_meta["num_stops"],
                    "route_type": r_meta["route_type"],
                    "zone": r_meta["zone"]
                })
        out[city] = pd.DataFrame(records)
        log.info(f"Generated {len(records)} records for {city} with {routes_per_city} routes.")
    return out


class TransitDataset(Dataset):
    """PyTorch Dataset for transit data, preparing sequences for time series prediction."""
    def __init__(self, df: pd.DataFrame, city_name: str, seq_len: int = SEQ_LEN, horizon: int = HORIZON):
        self.city_name = city_name
        self.seq_len = seq_len
        self.horizon = horizon

        df = df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"])

        self.route_ids = sorted(list(df["route_id"].unique()))
        self.n_routes = len(self.route_ids)

        pivot_inflow = df.pivot(index="datetime", columns="route_id", values="inflow_count")[self.route_ids]
        weather_feats = df.groupby("datetime")[["temperature", "precip_flag"]].mean()
        processed_df = pd.concat([pivot_inflow, weather_feats], axis=1)

        dt_index = processed_df.index
        processed_df["sin_hour"] = np.sin(2 * np.pi * dt_index.hour / 24.0)
        processed_df["cos_hour"] = np.cos(2 * np.pi * dt_index.hour / 24.0)
        processed_df["day_of_week_sin"] = np.sin(2 * np.pi * dt_index.dayofweek / 7.0)
        processed_df["day_of_week_cos"] = np.cos(2 * np.pi * dt_index.dayofweek / 7.0)
        processed_df["month_sin"] = np.sin(2 * np.pi * dt_index.month / 12.0)
        processed_df["month_cos"] = np.cos(2 * np.pi * dt_index.month / 12.0)
        processed_df["is_holiday"] = dt_index.to_series().apply(_is_kz_holiday).astype(int).values

        processed_df.ffill(inplace=True)
        processed_df.bfill(inplace=True)
        processed_df.fillna(0, inplace=True)

        self.feature_names = processed_df.columns.tolist()
        self.input_dim = len(self.feature_names)

        self.scaler_means = processed_df.mean()
        self.scaler_stds = processed_df.std() + 1e-6
        normalized_arr = (processed_df - self.scaler_means) / self.scaler_stds

        X_data = normalized_arr.values.astype(np.float32)

        inflow_indices_in_X = [self.feature_names.index(rid) for rid in self.route_ids]
        Y_data_scaled = X_data[:, inflow_indices_in_X]

        self.X, self.Y, self.original_datetimes_for_Y = [], [], []
        num_samples = len(X_data) - seq_len - horizon + 1
        if num_samples <= 0:
            log.error(f"Not enough data for {city_name} to create sequences. "
                      f"Data length: {len(X_data)}, SeqLen: {seq_len}, Horizon: {horizon}. "
                      f"Need at least {seq_len + horizon} records.")
            self.X = torch.empty(0, seq_len, self.input_dim, dtype=torch.float32)
            self.Y = torch.empty(0, horizon, self.n_routes, dtype=torch.float32)
            self.original_datetimes_for_Y = []
            return

        for i in range(num_samples):
            self.X.append(X_data[i: i + seq_len])
            self.Y.append(Y_data_scaled[i + seq_len: i + seq_len + horizon])
            self.original_datetimes_for_Y.append(dt_index[i + seq_len])

        self.X = torch.tensor(np.stack(self.X), dtype=torch.float32)
        self.Y = torch.tensor(np.stack(self.Y), dtype=torch.float32)

        static_df = df.drop_duplicates("route_id").set_index("route_id").loc[self.route_ids]
        static_feats_raw = static_df[["route_length_km", "num_stops"]].astype(np.float32)
        self.static_scaler_means = static_feats_raw.mean()
        self.static_scaler_stds = static_feats_raw.std() + 1e-6
        self.static_features_norm = torch.tensor(
            ((static_feats_raw - self.static_scaler_means) / self.static_scaler_stds).values,
            dtype=torch.float32
        )

        log.info(f"[{city_name}] Dataset created: X shape {self.X.shape}, Y shape {self.Y.shape}, "
                 f"Static feats shape: {self.static_features_norm.shape}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        # Returns 4 items: X, Y, static_features_norm, original_datetimes_for_Y
        return self.X[i], self.Y[i], self.static_features_norm, self.original_datetimes_for_Y[i]

def collate_fn(batch):
    """Custom collate function for the DataLoader."""
    # Unpack 4 items from each sample
    xs, ys, static_features_list, original_datetimes_list = zip(*batch)
    # static_features are per-client, not per-sample, so take the first one.
    static_features_batch = static_features_list[0]
    # Stack and return 4 items for the batch
    return torch.stack(xs), torch.stack(ys), static_features_batch, original_datetimes_list