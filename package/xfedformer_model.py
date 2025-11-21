import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# Assuming config is available and defines these globals
# from config import D_MODEL, N_HEADS, N_LAYERS, SEQ_LEN, HORIZON, log

# Placeholder for config values if config.py is not provided or accessible in this context
# In a real scenario, these would be imported from config.py
D_MODEL = 512
N_HEADS = 8
N_LAYERS = 3
SEQ_LEN = 1 # Assuming a sequence length of 1 for point predictions based on current features
HORIZON = 1 # Assuming a prediction horizon of 1 for a single future step
class Log:
    def info(self, message):
        print(f"INFO: {message}")
    def warning(self, message):
        print(f"WARNING: {message}")
    def error(self, message, exc_info=False):
        print(f"ERROR: {message}")
        if exc_info:
            import traceback
            traceback.print_exc()
    def debug(self, message): # Added debug method
        print(f"DEBUG: {message}")
log = Log()


# ─────────────────────────────────────────────────────────────────────────────
# Model Components
# ─────────────────────────────────────────────────────────────────────────────

class SeasonalTrendDecomp(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.pool = nn.AvgPool1d(kernel_size, stride=1, padding=kernel_size//2)

    def forward(self, x):  # x: (B, T, D)
        # Ensure x is 3D (Batch, Sequence Length, Features)
        if x.ndim == 2:
            x = x.unsqueeze(1) # Add sequence length dimension if missing (B, 1, D)

        # Transpose for AvgPool1d: (B, D, T)
        trend = self.pool(x.transpose(1, 2)).transpose(1, 2)
        resid = x - trend
        
        # FIX: Ensure seasonal component is always returned, even if it's a zero tensor.
        # This resolves the "not enough values to unpack (expected 3, got 2)" error.
        seasonal = torch.zeros_like(x) # Placeholder: assumes seasonal component has same shape as input
        
        log.info(f"SeasonalTrendDecomp returning: {len([trend, seasonal, resid])} values")
        return trend, seasonal, resid # This must return 3 values


class SpatialEncoder(nn.Module):
    def __init__(self, n_routes, static_feat_dim, emb_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(static_feat_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )
        log.info(
            f"SpatialEncoder init: n_routes={n_routes}, static_feat_dim={static_feat_dim}, emb_dim={emb_dim}")

    def forward(self, static_route_features):
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

    def forward(self, x_q, x_kv):
        attn_out, _ = self.cross_attn(x_q, x_kv, x_kv)
        x_q = self.norm(x_q + attn_out)
        ff_out = self.ff(x_q)
        x_q = self.norm2(x_q + ff_out)
        return x_q


class MoEBlock(nn.Module):
    def __init__(self, d_model, n_experts=4, top_k=2):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_model*2),
                          nn.ReLU(),
                          nn.Linear(d_model*2, d_model),
                          nn.Dropout(0.1))
            for _ in range(n_experts)
        ])
        self.gating = nn.Linear(d_model, n_experts)
        self.top_k = top_k
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        res_conn = x
        gating_logits = self.gating(x)

        top_k_weights, top_k_indices = torch.topk(
            gating_logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_weights, dim=-1)

        expert_outputs_list = []
        for i in range(self.top_k):
            indices_i = top_k_indices[..., i]
            flat_x = x.reshape(-1, x.size(-1))
            flat_indices_i = indices_i.reshape(-1)

            current_expert_outputs = torch.zeros_like(flat_x)
            for exp_idx in range(len(self.experts)):
                mask = (flat_indices_i == exp_idx)
                if mask.any():
                    selected_inputs = flat_x[mask]
                    current_expert_outputs[mask] = self.experts[exp_idx](
                        selected_inputs)

            expert_outputs_list.append(current_expert_outputs.reshape_as(
                x) * top_k_weights[..., i].unsqueeze(-1))

        mixed = sum(expert_outputs_list)
        return self.norm(mixed + res_conn)


class XFedFormer(nn.Module):
    def __init__(self, input_dim: int, n_routes: int, n_static_feats: int,
                 d_model: int = D_MODEL, n_heads: int = N_HEADS, n_layers: int = N_LAYERS,
                 seq_len: int = SEQ_LEN, horizon: int = HORIZON):
        super().__init__()
        self.n_routes = n_routes
        self.input_dim = input_dim
        # Store seq_len and horizon as attributes for easy access in app.py
        self.seq_len = seq_len
        self.horizon = horizon
        log.info(
            f"XFedFormer init: input_dim={input_dim}, n_routes={n_routes}, d_model={d_model}, seq_len={seq_len}, horizon={horizon}")

        self.decomp = SeasonalTrendDecomp(kernel_size=7)

        self.input_projection = nn.Linear(input_dim, d_model)

        self.pos_enc = nn.Parameter(torch.randn(seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * 4, batch_first=True, dropout=0.1, activation='gelu'
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer, n_layers, norm=nn.LayerNorm(d_model))

        self.moe_block = MoEBlock(d_model, n_experts=4, top_k=2)

        self.decoder = nn.Linear(d_model, n_routes)

    def forward(self, x_series: torch.Tensor, static_feats: Optional[torch.Tensor] = None):
        # IMPORTANT DEBUGGING STEP: Print the shape of x_series as it enters the forward method
        log.debug(f"XFedFormer forward received x_series shape: {x_series.shape}")

        # Ensure x_series is 3D (Batch, Sequence Length, Features)
        # This check is crucial if the input might sometimes be 2D (e.g., when seq_len=1)
        if x_series.ndim == 2:
            x_series = x_series.unsqueeze(1) # Add sequence length dimension if missing (B, 1, D)
            log.debug(f"XFedFormer forward reshaped x_series to: {x_series.shape}")

        B, T, _ = x_series.shape

        # FIX: Unpack 3 values from decomposition. SeasonalTrendDecomp now always returns 3.
        trend_full, seasonal_full, resid_full = self.decomp(x_series)

        z = self.input_projection(resid_full)

        # Ensure positional encoding is correctly applied based on actual sequence length T
        z = z + self.pos_enc[:T]

        z = self.temporal_encoder(z)

        z = self.moe_block(z)

        forecast_scaled = self.decoder(z[:, -1, :])

        trend_to_add = trend_full[:, -1, :self.n_routes]

        final_forecast_scaled = forecast_scaled + trend_to_add

        # If the model was designed to also add back the seasonal component, it would be here:
        # seasonal_to_add = seasonal_full[:, -1, :self.n_routes]
        # final_forecast_scaled = forecast_scaled + trend_to_add + seasonal_to_add

        return final_forecast_scaled
