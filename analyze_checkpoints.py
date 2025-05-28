import argparse
import json
import logging
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo
import seaborn as sns
import torch
from plotly.subplots import make_subplots
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler

# Interpretability libraries
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Install with: pip install shap")

try:
    import lime
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    warnings.warn("LIME not available. Install with: pip install lime")

# Assuming these are imported from your xfedformer module
from xfedformer import (
    D_MODEL,
    DEVICE,
    HORIZON,
    N_HEADS,
    N_LAYERS,
    SEQ_LEN,
    TransitDataset,
    XFedFormer,
    quick_metrics,
)

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class EnhancedCheckpointAnalyzer:
    """Enhanced analyzer for X-FedFormer checkpoints with comprehensive evaluation."""
    
    def __init__(self, ckpt_dir: Path, data_dir: Path, results_dir: Path, 
                 cities: List[str], days_data: int):
        self.ckpt_dir = ckpt_dir
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.cities = cities
        self.days_data = days_data
        self.results = {}
        self.models = {}
        self.datasets = {}
        self.predictions = defaultdict(dict)
        self.feature_importance = defaultdict(dict)
        
        # Create results directories
        self.results_dir.mkdir(exist_ok=True, parents=True)
        (self.results_dir / "plots").mkdir(exist_ok=True)
        (self.results_dir / "interpretability").mkdir(exist_ok=True)
        (self.results_dir / "detailed_analysis").mkdir(exist_ok=True)
    
    def load_weights(self, path: Path) -> List[torch.Tensor]:
        """Load model weights from checkpoint."""
        return torch.load(path, map_location=DEVICE, weights_only=False)
    
    def prepare_datasets(self) -> Dict[str, TransitDataset]:
        """Prepare datasets for all cities."""
        datasets = {}
        for city in self.cities:
            files = list(self.data_dir.glob(f"{city}_{self.days_data}days_routes*.csv"))
            if not files:
                logging.warning(f"No data found for {city}")
                continue
            
            df = pd.read_csv(files[0], parse_dates=["datetime"])
            ds = TransitDataset(df, city_name=city, seq_len=SEQ_LEN, horizon=HORIZON)
            
            if len(ds) > 0:
                datasets[city] = ds
                logging.info(f"Loaded dataset for {city}: {len(ds)} samples")
            else:
                logging.warning(f"Empty dataset for {city}")
        
        return datasets
    
    def evaluate_checkpoints(self) -> Dict[int, Dict[str, Dict]]:
        """Evaluate all checkpoints with detailed metrics."""
        results = {}
        self.datasets = self.prepare_datasets()
        
        ckpts = sorted(self.ckpt_dir.glob("global_model_round_*.pt"),
                      key=lambda p: int(p.stem.split("_")[-1]))
        
        for ckpt in ckpts:
            round_num = int(ckpt.stem.split("_")[-1])
            logging.info(f"Evaluating Round {round_num}")
            
            weights = self.load_weights(ckpt)
            results[round_num] = {}
            
            for city, dataset in self.datasets.items():
                model = self._create_model(dataset)
                self._load_model_weights(model, weights, city)
                
                # Store model and dataset for later analysis
                self.models[f"{round_num}_{city}"] = model
                
                # Evaluate with comprehensive metrics
                metrics = self._comprehensive_evaluation(model, dataset, city, round_num)
                results[round_num][city] = metrics
                
                logging.info(f"  {city}: MAE={metrics['mae']:.3f}, "
                           f"R2={metrics['r2']:.3f}, RMSE={metrics['rmse']:.3f}")
        
        return results
    
    def _create_model(self, dataset: TransitDataset) -> XFedFormer:
        """Create model instance."""
        return XFedFormer(
            input_dim=dataset.input_dim,
            n_routes=dataset.n_routes,
            n_static_feats=dataset.static_features_norm.shape[1],
            d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
            seq_len=SEQ_LEN, horizon=HORIZON
        ).to(DEVICE)
    
    def _load_model_weights(self, model: XFedFormer, weights: List[torch.Tensor], city: str):
        """Load weights into model."""
        params = list(model.parameters())
        if len(weights) != len(params):
            logging.error(f"Parameter mismatch for {city}")
            return False
        
        for w, p in zip(weights, params):
            p.data.copy_(torch.tensor(w, device=DEVICE))
        return True
    
    def _comprehensive_evaluation(self, model: XFedFormer, dataset: TransitDataset, 
                                city: str, round_num: int) -> Dict[str, float]:
        """Comprehensive model evaluation with detailed metrics."""
        model.eval()
        
        # Basic metrics
        basic_metrics = quick_metrics(model, dataset, DEVICE)
        
        # Additional detailed metrics
        detailed_metrics = self._calculate_detailed_metrics(model, dataset)
        
        # Store predictions for later analysis
        predictions = self._get_predictions(model, dataset)
        self.predictions[round_num][city] = predictions
        
        # Combine all metrics
        all_metrics = {**basic_metrics, **detailed_metrics}
        return all_metrics
    
    def _calculate_detailed_metrics(self, model: XFedFormer, dataset: TransitDataset) -> Dict[str, float]:
        """Calculate additional detailed metrics."""
        model.eval()
        predictions = []
        targets = []

        # 1) Run model over the entire dataset and collect numpy outputs
        with torch.no_grad():
            for i in range(len(dataset)):
                x, y, _ = dataset[i]
                x = x.unsqueeze(0).to(DEVICE)
                y = y.unsqueeze(0).to(DEVICE)

                pred = model(x)
                predictions.append(pred.cpu().numpy())
                targets.append(y.cpu().numpy())

        # 2) Concatenate into arrays of shape (total_samples, HORIZON?, N_ROUTES)
        predictions = np.concatenate(predictions, axis=0)
        targets     = np.concatenate(targets,     axis=0)

        # ======================================================
        # 3) ALIGNMENT PATCH: handle the (685, 12, 3) vs (685, 3) mismatch by
        #    trimming the extra horizon dimension when only the final step is predicted
        original_predictions_shape = predictions.shape
        original_targets_shape     = targets.shape

        # Case A: model only predicts the last step (2D) but targets are full-horizon (3D)
        if (predictions.ndim == 2 and targets.ndim == 3 and
            predictions.shape[0] == targets.shape[0] and
            predictions.shape[1] == targets.shape[2]):
            logging.info(
                f"Aligning targets from {original_targets_shape} to predictions {original_predictions_shape} by taking last horizon step."
            )
            targets = targets[:, -1, :]

        # Case B: model predicts full-horizon (3D) but targets only have the last step (2D)
        elif (predictions.ndim == 3 and targets.ndim == 2 and
              predictions.shape[0] == targets.shape[0] and
              predictions.shape[2] == targets.shape[1]):
            logging.info(
                f"Aligning predictions from {original_predictions_shape} to targets {original_targets_shape} by taking last horizon step."
            )
            predictions = predictions[:, -1, :]

        # Case C: still mismatched after those two possibilities → bail out
        elif predictions.reshape(predictions.shape[0], -1).shape != targets.reshape(targets.shape[0], -1).shape:
            logging.error(
                f"Unable to align prediction and target shapes: {original_predictions_shape} vs {original_targets_shape}. "
                "Cannot calculate detailed metrics."
            )
            return {
                'mape': np.nan,
                'smape': np.nan,
                'directional_accuracy': np.nan,
                'quantile_loss_0.1': np.nan,
                'quantile_loss_0.5': np.nan,
                'quantile_loss_0.9': np.nan
            }
        # ======================================================

        # 4) Flatten for element-wise metrics
        predictions_final = predictions.flatten()
        targets_final     = targets.flatten()

        # 5) Core metrics
        mape  = np.mean(np.abs((targets_final - predictions_final) / (targets_final + 1e-8))) * 100
        smape = np.mean(
            2 * np.abs(predictions_final - targets_final) /
            (np.abs(predictions_final) + np.abs(targets_final) + 1e-8)
        ) * 100

        # 6) Directional accuracy & quantile losses
        if predictions.ndim == 2 and targets.ndim == 2:
            # Single-step case → directional accuracy isn't defined over time
            directional_accuracy = np.nan

            quantiles = [0.1, 0.5, 0.9]
            quantile_losses = {}
            for q in quantiles:
                loss = np.mean(
                    np.maximum(
                        q * (targets_final - predictions_final),
                        (q - 1) * (targets_final - predictions_final)
                    )
                )
                quantile_losses[f'quantile_loss_{q}'] = loss

        else:
            # Multi-step case → you could compute sequence-based directional accuracy here
            directional_accuracy = np.nan

            quantiles = [0.1, 0.5, 0.9]
            quantile_losses = {}
            for q in quantiles:
                loss = np.mean(
                    np.maximum(
                        q * (targets_final - predictions_final),
                        (q - 1) * (targets_final - predictions_final)
                    )
                )
                quantile_losses[f'quantile_loss_{q}'] = loss

        # 7) Return all metrics
        return {
            'mape': mape,
            'smape': smape,
            'directional_accuracy': directional_accuracy,
            **quantile_losses
        }

    def _get_predictions(self, model: XFedFormer, dataset: TransitDataset) -> Dict[str, np.ndarray]:
        """Get model predictions for interpretability analysis."""
        model.eval()
        inputs, predictions, targets = [], [], []
        
        with torch.no_grad():
            for i in range(min(len(dataset), 1000)):  # Limit for memory efficiency
                x, y, _ = dataset[i]
                x_tensor = x.unsqueeze(0).to(DEVICE)
                
                pred = model(x_tensor)
                
                inputs.append(x.cpu().numpy()) # Store x as (SEQ_LEN, input_dim)
                predictions.append(pred.cpu().numpy()) # Store pred as (1, HORIZON, N_ROUTES)
                targets.append(y.cpu().numpy()) # Store y as (HORIZON, N_ROUTES)
        
        # Concatenate along the batch dimension (axis 0)
        # inputs will be (total_samples, SEQ_LEN, input_dim)
        # predictions will be (total_samples, HORIZON, N_ROUTES)
        # targets will be (total_samples, HORIZON, N_ROUTES)
        
        # It's crucial that `y` from `TransitDataset` has the shape `(HORIZON, N_ROUTES)`
        # If `y` is `(N_ROUTES)` then this is where the `(685,3)` shape originates for `targets`
        # in `_calculate_detailed_metrics`. You need to verify `TransitDataset`'s `__getitem__` method
        # for the shape of `y`. If `y` is indeed `(N_ROUTES)`, and your model predicts `HORIZON` steps,
        # then you must align the `predictions` to `y` by selecting the relevant horizon step (e.g., the last one).
        
        # Assuming y from dataset is consistently (HORIZON, N_ROUTES)
        return {
            'inputs': np.concatenate(inputs, axis=0),
            'predictions': np.concatenate(predictions, axis=0),
            'targets': np.concatenate(targets, axis=0)
        }
    
    def generate_comprehensive_plots(self):
        """Generate comprehensive visualization suite."""
        logging.info("Generating comprehensive visualizations...")
        
        # 1. Basic metric evolution plots
        self._plot_metric_evolution()
        
        # 2. Performance comparison heatmaps
        self._plot_performance_heatmaps()
        
        # 3. Convergence analysis
        self._plot_convergence_analysis()
        
        # 4. City-wise performance comparison
        self._plot_city_comparison()
        
        # 5. Error distribution analysis
        self._plot_error_distributions()
        
        # 6. Prediction vs actual scatter plots
        self._plot_prediction_scatter()
        
        # 7. Time series forecasting visualization
        self._plot_time_series_forecasts()
        
        # 8. Model stability analysis
        self._plot_stability_analysis()
        
        # 9. Feature importance analysis
        if SHAP_AVAILABLE or LIME_AVAILABLE:
            self._generate_interpretability_analysis()
    
    def _plot_metric_evolution(self):
        """Plot evolution of metrics over training rounds."""
        metrics = ['mae', 'rmse', 'r2', 'mape', 'directional_accuracy']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        rounds = sorted(self.results.keys())
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            for city in self.cities:
                values = []
                for round_num in rounds:
                    city_results = self.results[round_num].get(city, {})
                    values.append(city_results.get(metric, np.nan))
                
                ax.plot(rounds, values, marker='o', linewidth=2, markersize=6, label=city)
            
            ax.set_xlabel('Training Round')
            ax.set_ylabel(metric.upper().replace('_', ' '))
            ax.set_title(f'{metric.upper().replace("_", " ")} Evolution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplot
        if len(metrics) < len(axes):
            fig.delaxes(axes[-1])
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "plots" / "metric_evolution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_heatmaps(self):
        """Generate performance heatmaps across cities and rounds."""
        metrics = ['mae', 'rmse', 'r2', 'mape']
        
        for metric in metrics:
            # Prepare data matrix
            rounds = sorted(self.results.keys())
            cities = self.cities
            
            data_matrix = np.zeros((len(cities), len(rounds)))
            
            for i, city in enumerate(cities):
                for j, round_num in enumerate(rounds):
                    city_results = self.results[round_num].get(city, {})
                    data_matrix[i, j] = city_results.get(metric, np.nan)
            
            # Create heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(data_matrix, 
                       xticklabels=[f'Round {r}' for r in rounds],
                       yticklabels=cities,
                       annot=True, fmt='.3f', cmap='viridis',
                       cbar_kws={'label': metric.upper()})
            
            plt.title(f'{metric.upper()} Across Cities and Training Rounds')
            plt.xlabel('Training Round')
            plt.ylabel('City')
            plt.tight_layout()
            plt.savefig(self.results_dir / "plots" / f"{metric}_heatmap.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_convergence_analysis(self):
        """Analyze and plot convergence patterns."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        rounds = sorted(self.results.keys())
        
        # 1. Learning curves with confidence intervals
        ax = axes[0, 0]
        for city in self.cities:
            mae_values = [self.results[r].get(city, {}).get('mae', np.nan) for r in rounds]
            ax.plot(rounds, mae_values, marker='o', label=f'{city} MAE')
        
        ax.set_xlabel('Training Round')
        ax.set_ylabel('MAE')
        ax.set_title('Learning Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Convergence rate analysis
        ax = axes[0, 1]
        for city in self.cities:
            mae_values = [self.results[r].get(city, {}).get('mae', np.nan) for r in rounds]
            # Calculate improvement rate
            improvements = [mae_values[i-1] - mae_values[i] if i > 0 else 0 
                          for i in range(len(mae_values))]
            ax.plot(rounds, improvements, marker='s', label=f'{city} Improvement Rate')
        
        ax.set_xlabel('Training Round')
        ax.set_ylabel('MAE Improvement')
        ax.set_title('Convergence Rate Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # 3. Stability analysis (coefficient of variation)
        ax = axes[1, 0]
        window_size = min(5, len(rounds))
        for city in self.cities:
            mae_values = [self.results[r].get(city, {}).get('mae', np.nan) for r in rounds]
            stability = []
            for i in range(window_size-1, len(mae_values)):
                window_vals = mae_values[i-window_size+1:i+1]
                cv = np.std(window_vals) / np.mean(window_vals) if np.mean(window_vals) > 0 else 0
                stability.append(cv)
            
            ax.plot(rounds[window_size-1:], stability, marker='^', label=f'{city} CV')
        
        ax.set_xlabel('Training Round')
        ax.set_ylabel('Coefficient of Variation')
        ax.set_title('Model Stability (Lower is Better)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Multi-metric convergence
        ax = axes[1, 1]
        for city in self.cities:
            # Normalize metrics for comparison
            mae_norm = [(1 - self.results[r].get(city, {}).get('mae', 1)) for r in rounds]
            r2_values = [self.results[r].get(city, {}).get('r2', 0) for r in rounds]
            
            ax.plot(rounds, mae_norm, '--', label=f'{city} MAE (normalized)')
            ax.plot(rounds, r2_values, '-', label=f'{city} R²')
        
        ax.set_xlabel('Training Round')
        ax.set_ylabel('Normalized Score')
        ax.set_title('Multi-Metric Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "plots" / "convergence_analysis.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_city_comparison(self):
        """Compare performance across cities."""
        # Radar chart for latest round
        latest_round = max(self.results.keys())
        metrics = ['mae', 'rmse', 'r2', 'mape', 'directional_accuracy']
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Radar chart
        ax = axes[0]
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for city in self.cities:
            values = []
            for metric in metrics:
                val = self.results[latest_round].get(city, {}).get(metric, 0)
                # Normalize values for radar chart
                if metric in ['mae', 'rmse', 'mape']:
                    val = 1 / (1 + val)  # Invert error metrics
                elif metric == 'directional_accuracy':
                    val = val / 100
                values.append(val)
            
            values += values[:1]  # Complete the circle
            ax.plot(angles, values, 'o-', linewidth=2, label=city)
            ax.fill(angles, values, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.upper().replace('_', ' ') for m in metrics])
        ax.set_ylim(0, 1)
        ax.set_title(f'Performance Radar Chart (Round {latest_round})')
        ax.legend()
        ax.grid(True)
        
        # Box plots for metric distributions
        ax = axes[1]
        all_data = []
        labels = []
        
        for metric in ['mae', 'rmse', 'mape']:
            for city in self.cities:
                city_values = [self.results[r].get(city, {}).get(metric, np.nan) 
                              for r in self.results.keys()]
                all_data.append([v for v in city_values if not np.isnan(v)])
                labels.append(f'{city}\n{metric.upper()}')
        
        ax.boxplot(all_data, labels=labels)
        ax.set_title('Metric Distribution Across Training')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "plots" / "city_comparison.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_error_distributions(self):
        """Analyze error distributions."""
        latest_round = max(self.results.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        for idx, city in enumerate(self.cities[:4]):  # Limit to 4 cities
            if idx >= 4:
                break
                
            ax = axes[idx // 2, idx % 2]
            
            if f"{latest_round}_{city}" in self.predictions:
                preds = self.predictions[latest_round][city]
                # Flatten for error calculation if needed, ensuring consistent shape
                # The error plot should ideally use the same values as the metrics
                predictions_flat = preds['predictions'].flatten()
                targets_flat = preds['targets'].flatten()
                
                # If during metric calculation, predictions were aligned to a single horizon step,
                # then you might want to do the same here for consistency.
                # Re-applying the logic from _calculate_detailed_metrics:
                if preds['predictions'].ndim == 3 and preds['targets'].ndim == 2 and \
                   preds['predictions'].shape[0] == preds['targets'].shape[0] and \
                   preds['predictions'].shape[2] == preds['targets'].shape[1]:
                    predictions_flat = preds['predictions'][:, -1, :].flatten()
                    targets_flat = preds['targets'].flatten()
                else: # Otherwise, assume they are already compatible or will be flattened
                    predictions_flat = preds['predictions'].flatten()
                    targets_flat = preds['targets'].flatten()

                errors = predictions_flat - targets_flat
                
                # Error distribution histogram
                ax.hist(errors, bins=50, alpha=0.7, density=True, 
                       color=f'C{idx}', label='Error Distribution')
                
                # Fit normal distribution
                mu, sigma = stats.norm.fit(errors)
                x = np.linspace(errors.min(), errors.max(), 100)
                ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', 
                       label=f'Normal Fit (μ={mu:.3f}, σ={sigma:.3f})')
                
                ax.set_xlabel('Prediction Error')
                ax.set_ylabel('Density')
                ax.set_title(f'{city} - Error Distribution')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "plots" / "error_distributions.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_prediction_scatter(self):
        """Generate prediction vs actual scatter plots."""
        latest_round = max(self.results.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, city in enumerate(self.cities[:4]):
            if idx >= 4:
                break
                
            ax = axes[idx]
            
            if f"{latest_round}_{city}" in self.predictions:
                preds = self.predictions[latest_round][city]
                
                # Apply the same flattening/alignment logic as in _calculate_detailed_metrics
                y_pred = preds['predictions']
                y_true = preds['targets']

                if y_pred.ndim == 3 and y_true.ndim == 2 and \
                   y_pred.shape[0] == y_true.shape[0] and \
                   y_pred.shape[2] == y_true.shape[1]:
                    y_pred = y_pred[:, -1, :] # Take the last horizon step
                
                y_pred = y_pred.flatten()
                y_true = y_true.flatten()
                
                # Scatter plot
                ax.scatter(y_true, y_pred, alpha=0.6, s=10)
                
                # Perfect prediction line
                min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction')
                
                # Calculate R²
                r2 = self.results[latest_round][city]['r2']
                ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                ax.set_xlabel('Actual Values')
                ax.set_ylabel('Predicted Values')
                ax.set_title(f'{city} - Predictions vs Actual')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "plots" / "prediction_scatter.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_time_series_forecasts(self):
        """Visualize time series forecasting performance."""
        latest_round = max(self.results.keys())
        
        for city in self.cities:
            if f"{latest_round}_{city}" not in self.predictions:
                continue
                
            preds = self.predictions[latest_round][city]
            
            # This function needs to plot actual time series vs predicted time series.
            # This means `preds['predictions']` and `preds['targets']` should ideally
            # be 3D: (samples, HORIZON, N_ROUTES).
            # If `targets` was flattened in `_calculate_detailed_metrics`, that was for *metric calculation*,
            # but for *plotting actual series*, you need the original sequence structure.
            # The `_get_predictions` method collects them as (total_samples, HORIZON, N_ROUTES) for both,
            # so this part should be fine as long as `y` from `TransitDataset` truly provides `(HORIZON, N_ROUTES)`.

            # Plot first few samples
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            axes = axes.flatten()
            
            for i in range(min(4, len(preds['predictions']))):
                ax = axes[i]
                
                # Assuming predictions and targets are (num_samples, HORIZON, N_ROUTES)
                # Let's plot the first route's predictions/targets for simplicity
                # If you have multiple routes and want to plot each, you'll need more subplots or iteration.
                # For now, let's pick the first route (index 0) if N_ROUTES > 1
                
                pred_series = preds['predictions'][i, :, 0] if preds['predictions'].ndim == 3 else preds['predictions'][i]
                true_series = preds['targets'][i, :, 0] if preds['targets'].ndim == 3 else preds['targets'][i]
                
                # If targets is just (N_ROUTES) for a single step:
                # If targets is (batch_size, N_ROUTES) and predictions is (batch_size, HORIZON, N_ROUTES)
                # then true_series will be (N_ROUTES,) but pred_series will be (HORIZON, N_ROUTES).
                # This plot needs sequence. So `targets` must also be a sequence.
                # This implies `TransitDataset` must return `y` as `(HORIZON, N_ROUTES)`.
                # If not, this plot won't work correctly as a time series comparison over horizon.
                
                # Check consistency before plotting
                if pred_series.shape != true_series.shape:
                    logging.warning(f"Shape mismatch for time series plot sample {i}: pred_series {pred_series.shape}, true_series {true_series.shape}. Skipping.")
                    ax.set_title(f'{city} - Sample {i+1} (Skipped due to shape mismatch)')
                    continue

                time_steps = range(len(pred_series))
                
                ax.plot(time_steps, true_series, 'b-', label='Actual', linewidth=2)
                ax.plot(time_steps, pred_series, 'r--', label='Predicted', linewidth=2)
                
                # Standard deviation for uncertainty might need to be calculated across multiple prediction runs
                # or assumed from model's output if it provides uncertainty. Here, it's just std of the series itself,
                # which isn't true prediction uncertainty. For a single sample, std of its time series is not uncertainty.
                # Let's simplify this or remove if actual uncertainty isn't available.
                # For plotting, assume some simple error bounds if not available.
                # Currently: `np.std(pred_series)` is standard deviation *of the predicted sequence itself*.
                # This is not uncertainty. Removing for clarity unless proper uncertainty is available.
                # ax.fill_between(time_steps, pred_series - np.std(pred_series),
                #                pred_series + np.std(pred_series),
                #                alpha=0.2, color='red', label='Prediction Uncertainty')
                
                ax.set_xlabel('Time Step (Horizon)')
                ax.set_ylabel('Passenger Flow')
                ax.set_title(f'{city} - Sample {i+1} Forecast (Route 1)')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.suptitle(f'{city} - Time Series Forecasting Examples')
            plt.tight_layout()
            plt.savefig(self.results_dir / "plots" / f"{city.lower()}_forecasts.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_stability_analysis(self):
        """Analyze model stability across rounds."""
        metrics = ['mae', 'rmse', 'r2']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            # Calculate stability metrics
            rounds = sorted(self.results.keys())
            
            for city in self.cities:
                values = [self.results[r].get(city, {}).get(metric, np.nan) for r in rounds]
                
                # Moving average and standard deviation
                window = min(5, len(values))
                if len(values) >= window:
                    moving_avg = pd.Series(values).rolling(window=window).mean()
                    moving_std = pd.Series(values).rolling(window=window).std()
                    
                    ax.plot(rounds, values, 'o-', alpha=0.6, label=f'{city} Raw')
                    ax.plot(rounds, moving_avg, '-', linewidth=2, label=f'{city} MA({window})')
                    # Only plot fill_between if moving_std is not all NaN/zero
                    if not pd.isna(moving_std).all() and not (moving_std == 0).all():
                        ax.fill_between(rounds, moving_avg - moving_std, moving_avg + moving_std,
                                       alpha=0.2)
            
            ax.set_xlabel('Training Round')
            ax.set_ylabel(metric.upper())
            ax.set_title(f'{metric.upper()} Stability Analysis')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "plots" / "stability_analysis.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_interpretability_analysis(self):
        """Generate SHAP and LIME interpretability analysis."""
        latest_round = max(self.results.keys())
        
        for city in self.cities:
            if f"{latest_round}_{city}" not in self.models:
                continue
                
            model = self.models[f"{latest_round}_{city}"]
            
            if f"{latest_round}_{city}" in self.predictions:
                preds = self.predictions[latest_round][city]
                
                # SHAP Analysis
                if SHAP_AVAILABLE:
                    self._generate_shap_analysis(model, preds, city, latest_round)
                
                # LIME Analysis
                if LIME_AVAILABLE:
                    self._generate_lime_analysis(model, preds, city, latest_round)
    
    def _generate_shap_analysis(self, model, predictions, city, round_num):
        """Generate SHAP explanations."""
        try:
            # Prepare data for SHAP
            # X should be (num_samples, total_features_for_model_input_flattened)
            # The model_predict function expects `x` to be `(batch_size, SEQ_LEN, input_dim)`
            X = predictions['inputs'] # This is (num_samples, SEQ_LEN, input_dim)
            
            # Reshape for KernelExplainer if it expects 2D array, and then reshape back in wrapper
            # For KernelExplainer, it's often more robust if the background data is 2D,
            # and the wrapper function handles reshaping back to the model's expected 3D input.
            original_input_shape = X.shape[1:] # (SEQ_LEN, input_dim)
            X_2d = X.reshape(X.shape[0], -1) # Flatten to (num_samples, SEQ_LEN * input_dim)
            
            # Create a wrapper function for the model
            def model_predict_shap(x_2d_array):
                # Reshape input back to original format (batch_size, SEQ_LEN, input_dim)
                x_reshaped = x_2d_array.reshape(x_2d_array.shape[0], original_input_shape[0], original_input_shape[1])
                x_tensor = torch.FloatTensor(x_reshaped).to(DEVICE)
                
                model.eval()
                with torch.no_grad():
                    output = model(x_tensor)
                # SHAP expects a 1D array of predictions or (batch_size, num_outputs)
                # Assuming output is (batch_size, HORIZON, N_ROUTES) and we want to explain a single output.
                # Let's take the first element of the first route at the last horizon as the target output for SHAP.
                # This needs careful consideration if your model has multiple outputs for explanation.
                # For simplicity, taking the sum or a specific output if the model predicts multiple features/horizons.
                # For multi-output models, SHAP typically requires explaining one output at a time or the sum.
                # Let's sum across horizon and routes to get a single score per sample for simplicity.
                return output.cpu().numpy().sum(axis=(1,2)) # Sum over horizon and routes
            
            # Use KernelExplainer for complex models
            # Using a small subset for background and explanation for performance
            background_data = X_2d[:min(100, X_2d.shape[0])] # Use a small subset as background
            explain_data = X_2d[:min(50, X_2d.shape[0])] # Explain a small subset
            
            explainer = shap.KernelExplainer(model_predict_shap, background_data)
            shap_values = explainer.shap_values(explain_data)
            
            # Plot SHAP summary
            plt.figure(figsize=(12, 8))
            # Feature names: difficult to derive directly. Use generic names or try to map.
            # If input_dim contains temporal features, static features etc.
            # For now, generic feature_0, feature_1...
            feature_names = [f'feature_{i}' for i in range(X_2d.shape[1])]
            shap.summary_plot(shap_values, explain_data, feature_names=feature_names, show=False)
            plt.title(f'{city} - SHAP Feature Importance (Round {round_num})')
            plt.tight_layout()
            plt.savefig(self.results_dir / "interpretability" / f"{city.lower()}_shap_summary.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Generated SHAP analysis for {city}")
            
        except Exception as e:
            logging.warning(f"SHAP analysis failed for {city}: {str(e)}")
    
    def _generate_lime_analysis(self, model, predictions, city, round_num):
        """Generate LIME explanations."""
        try:
            # Prepare data for LIME
            # X should be (num_samples, total_features_for_model_input_flattened)
            X = predictions['inputs'] # This is (num_samples, SEQ_LEN, input_dim)
            original_input_shape = X.shape[1:] # (SEQ_LEN, input_dim)
            X_2d = X.reshape(X.shape[0], -1) # Flatten to (num_samples, SEQ_LEN * input_dim)
            
            # Create LIME explainer
            # LIME explainer background data
            lime_background_data = X_2d[:min(1000, X_2d.shape[0])]
            
            # Feature names for LIME
            feature_names = [f'feature_{i}' for i in range(X_2d.shape[1])]

            explainer = LimeTabularExplainer(
                lime_background_data,
                mode='regression',
                feature_names=feature_names,
                discretize_continuous=True
            )
            
            # Create model wrapper
            def model_predict_lime(x_2d_array):
                # Reshape input back to original format (batch_size, SEQ_LEN, input_dim)
                x_reshaped = x_2d_array.reshape(x_2d_array.shape[0], original_input_shape[0], original_input_shape[1])
                x_tensor = torch.FloatTensor(x_reshaped).to(DEVICE)
                
                model.eval()
                with torch.no_grad():
                    output = model(x_tensor)
                # LIME expects a 1D array of predictions or (batch_size, num_outputs) for regression.
                # Summing across horizon and routes to get a single score per sample.
                return output.cpu().numpy().sum(axis=(1,2)).flatten()
            
            # Explain a few instances
            feature_importance_scores = []
            
            for i in range(min(5, len(X_2d))):
                explanation = explainer.explain_instance(
                    X_2d[i], model_predict_lime, num_features=min(20, X_2d.shape[1])
                )
                
                # Extract feature importance
                importance_dict = dict(explanation.as_list())
                feature_importance_scores.append(importance_dict)
                
                # Save individual explanation
                explanation.save_to_file(
                    self.results_dir / "interpretability" / f"{city.lower()}_lime_explanation_{i}.html"
                )
            
            # Aggregate feature importance
            all_features = set()
            for score_dict in feature_importance_scores:
                all_features.update(score_dict.keys())
            
            avg_importance = {}
            for feature in all_features:
                scores = [score_dict.get(feature, 0) for score_dict in feature_importance_scores]
                avg_importance[feature] = np.mean(scores)
            
            # Plot aggregated feature importance
            sorted_features = sorted(avg_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:20]
            
            plt.figure(figsize=(12, 8))
            features, importances = zip(*sorted_features)
            colors = ['red' if imp < 0 else 'blue' for imp in importances]
            
            plt.barh(range(len(features)), importances, color=colors, alpha=0.7)
            plt.yticks(range(len(features)), features)
            plt.xlabel('Average Feature Importance')
            plt.title(f'{city} - LIME Feature Importance (Round {round_num})')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.results_dir / "interpretability" / f"{city.lower()}_lime_importance.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Generated LIME analysis for {city}")
            
        except Exception as e:
            logging.warning(f"LIME analysis failed for {city}: {str(e)}")
    
    def generate_statistical_analysis(self):
        """Generate comprehensive statistical analysis."""
        logging.info("Generating statistical analysis...")
        
        # 1. Performance statistical tests
        self._statistical_significance_tests()
        
        # 2. Correlation analysis
        self._correlation_analysis()
        
        # 3. Clustering analysis
        self._clustering_analysis()
        
        # 4. Anomaly detection
        self._anomaly_detection()
    
    def _statistical_significance_tests(self):
        """Perform statistical significance tests."""
        results_data = []
        
        # Prepare data for statistical tests
        for round_num in self.results.keys():
            for city in self.cities:
                city_results = self.results[round_num].get(city, {})
                if city_results:
                    results_data.append({
                        'round': round_num,
                        'city': city,
                        'mae': city_results.get('mae', np.nan),
                        'rmse': city_results.get('rmse', np.nan),
                        'r2': city_results.get('r2', np.nan),
                        'mape': city_results.get('mape', np.nan)
                    })
        
        df = pd.DataFrame(results_data)
        
        # ANOVA test for city differences
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        metrics = ['mae', 'rmse', 'r2', 'mape']
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            # Box plot
            city_groups = [df[df['city'] == city][metric].dropna().values for city in self.cities]
            
            bp = ax.boxplot(city_groups, labels=self.cities, patch_artist=True)
            
            # Color boxes
            colors = plt.cm.Set3(np.linspace(0, 1, len(self.cities)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Perform ANOVA
            if len(city_groups) > 1 and all(len(group) > 1 for group in city_groups):
                f_stat, p_value = stats.f_oneway(*city_groups)
                ax.text(0.02, 0.98, f'ANOVA: F={f_stat:.3f}, p={p_value:.4f}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_ylabel(metric.upper())
            ax.set_title(f'{metric.upper()} Distribution Across Cities')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "detailed_analysis" / "statistical_tests.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _correlation_analysis(self):
        """Analyze correlations between metrics and cities."""
        # Prepare correlation matrix
        correlation_data = []
        
        for round_num in self.results.keys():
            round_data = {'round': round_num}
            for city in self.cities:
                city_results = self.results[round_num].get(city, {})
                for metric in ['mae', 'rmse', 'r2', 'mape']:
                    round_data[f'{city}_{metric}'] = city_results.get(metric, np.nan)
            correlation_data.append(round_data)
        
        df = pd.DataFrame(correlation_data)
        
        # Calculate correlation matrix
        corr_matrix = df.select_dtypes(include=[np.number]).corr()
        
        # Plot correlation heatmap
        plt.figure(figsize=(16, 12))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                   cmap='coolwarm', center=0, square=True,
                   cbar_kws={'label': 'Correlation Coefficient'})
        plt.title('Correlation Matrix: Metrics Across Cities and Rounds')
        plt.tight_layout()
        plt.savefig(self.results_dir / "detailed_analysis" / "correlation_matrix.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _clustering_analysis(self):
        """Perform clustering analysis on city performance patterns."""
        # Prepare data for clustering
        city_profiles = {}
        
        for city in self.cities:
            city_data = []
            for round_num in sorted(self.results.keys()):
                city_results = self.results[round_num].get(city, {})
                city_data.extend([
                    city_results.get('mae', np.nan),
                    city_results.get('rmse', np.nan),
                    city_results.get('r2', np.nan),
                    city_results.get('mape', np.nan)
                ])
            city_profiles[city] = city_data
        
        # Create feature matrix
        feature_matrix = np.array([city_profiles[city] for city in self.cities])
        
        # Handle NaN values
        feature_matrix = np.nan_to_num(feature_matrix, nan=0)
        
        # Standardize features
        scaler = StandardScaler()
        feature_matrix_scaled = scaler.fit_transform(feature_matrix)
        
        # Hierarchical clustering
        linkage_matrix = linkage(feature_matrix_scaled, method='ward')
        
        # Plot dendrogram
        plt.figure(figsize=(12, 8))
        dendrogram(linkage_matrix, labels=self.cities, leaf_rotation=90)
        plt.title('Hierarchical Clustering of Cities Based on Performance Patterns')
        plt.ylabel('Distance')
        plt.tight_layout()
        plt.savefig(self.results_dir / "detailed_analysis" / "city_clustering.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _anomaly_detection(self):
        """Detect anomalous performance patterns."""
        anomalies = {}
        
        for city in self.cities:
            city_anomalies = []
            
            for metric in ['mae', 'rmse', 'r2', 'mape']:
                values = []
                rounds = []
                
                for round_num in sorted(self.results.keys()):
                    city_results = self.results[round_num].get(city, {})
                    if metric in city_results:
                        values.append(city_results[metric])
                        rounds.append(round_num)
                
                if len(values) > 3:
                    # Use IQR method for anomaly detection
                    Q1 = np.percentile(values, 25)
                    Q3 = np.percentile(values, 75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    for i, (round_num, value) in enumerate(zip(rounds, values)):
                        if value < lower_bound or value > upper_bound:
                            city_anomalies.append({
                                'round': round_num,
                                'metric': metric,
                                'value': value,
                                'bounds': (lower_bound, upper_bound),
                                'type': 'low' if value < lower_bound else 'high'
                            })
            
            if city_anomalies:
                anomalies[city] = city_anomalies
        
        # Plot anomalies
        if anomalies:
            # Determine number of rows needed based on number of cities with anomalies
            num_anomaly_cities = len(anomalies)
            if num_anomaly_cities == 0: # No anomalies to plot
                logging.info("No anomalies detected to plot.")
                return

            fig, axes = plt.subplots(num_anomaly_cities, 1, figsize=(14, 4 * num_anomaly_cities), squeeze=False)
            
            plot_idx = 0
            for city in self.cities:
                if city not in anomalies: # Only plot cities with detected anomalies
                    continue

                ax = axes[plot_idx, 0] # Access subplot via 2D index
                
                # Plot MAE with anomalies highlighted
                rounds = sorted(self.results.keys())
                mae_values = [self.results[r].get(city, {}).get('mae', np.nan) for r in rounds]
                
                ax.plot(rounds, mae_values, 'b-o', label='MAE')
                
                # Highlight anomalies
                for anomaly in anomalies[city]:
                    if anomaly['metric'] == 'mae':
                        color = 'red' if anomaly['type'] == 'high' else 'orange'
                        # Check if a label already exists to avoid duplicate labels in legend
                        if f"Anomaly ({anomaly['type']})" not in [l.get_text() for l in ax.get_legend().get_texts()]:
                            ax.scatter(anomaly['round'], anomaly['value'], 
                                    color=color, s=100, marker='x', 
                                    label=f"Anomaly ({anomaly['type']})", zorder=5) # zorder to ensure visibility
                        else:
                            ax.scatter(anomaly['round'], anomaly['value'], 
                                    color=color, s=100, marker='x', zorder=5)
                
                ax.set_xlabel('Training Round')
                ax.set_ylabel('MAE')
                ax.set_title(f'{city} - Anomaly Detection')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plot_idx += 1
            
            plt.tight_layout()
            plt.savefig(self.results_dir / "detailed_analysis" / "anomaly_detection.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # Save anomaly report
        with open(self.results_dir / "detailed_analysis" / "anomaly_report.json", 'w') as f:
            json.dump(anomalies, f, indent=2, default=str)
    
    def generate_interactive_dashboard(self):
        """Generate interactive Plotly dashboard."""
        logging.info("Generating interactive dashboard...")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('MAE Evolution', 'R² Evolution', 
                          'Performance Heatmap (MAE)', 'City Performance (Latest Round)',
                          'Error Distribution (Sample)', 'Prediction Accuracy (Sample)'), # Adjusted titles for clarity
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"colspan": 2}, None], # Heatmap spans two columns
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        rounds = sorted(self.results.keys())
        colors = px.colors.qualitative.Set1[:len(self.cities)]
        
        # 1. MAE Evolution
        for i, city in enumerate(self.cities):
            mae_values = [self.results[r].get(city, {}).get('mae', np.nan) for r in rounds]
            fig.add_trace(
                go.Scatter(x=rounds, y=mae_values, mode='lines+markers',
                          name=f'{city} MAE', line=dict(color=colors[i])),
                row=1, col=1
            )
        fig.update_yaxes(title_text="MAE", row=1, col=1)
        fig.update_xaxes(title_text="Training Round", row=1, col=1)

        # 2. R² Evolution
        for i, city in enumerate(self.cities):
            r2_values = [self.results[r].get(city, {}).get('r2', np.nan) for r in rounds]
            fig.add_trace(
                go.Scatter(x=rounds, y=r2_values, mode='lines+markers',
                          name=f'{city} R²', line=dict(color=colors[i]),
                          showlegend=False), # Legend already in MAE plot for cities
                row=1, col=2
            )
        fig.update_yaxes(title_text="R²", row=1, col=2)
        fig.update_xaxes(title_text="Training Round", row=1, col=2)

        # 3. Performance Heatmap (MAE)
        heatmap_data = []
        for city in self.cities:
            city_data = []
            for round_num in rounds:
                city_results = self.results[round_num].get(city, {})
                city_data.append(city_results.get('mae', np.nan)) # Using MAE for heatmap
            heatmap_data.append(city_data)
        
        # Handle cases where heatmap_data might be empty or all NaNs
        if heatmap_data and any(row for row in heatmap_data if any(not np.isnan(val) for val in row)):
            fig.add_trace(
                go.Heatmap(z=heatmap_data, x=[f'Round {r}' for r in rounds],
                           y=self.cities, colorscale='Viridis', showscale=True,
                           colorbar=dict(title='MAE')),
                row=2, col=1
            )
        else:
            fig.add_annotation(text="No MAE data for heatmap.",
                               xref="x2", yref="y2", showarrow=False, font=dict(size=16),
                               row=2, col=1)

        # 4. City Performance (Latest Round) - Bar Chart
        latest_round = max(rounds)
        metrics_for_bar = ['mae', 'rmse', 'mape'] # Metrics to show in bar chart
        
        bar_chart_data = []
        for metric in metrics_for_bar:
            values = [self.results[latest_round].get(city, {}).get(metric, np.nan) for city in self.cities]
            # Filter out NaN values if any city doesn't have data for this metric
            valid_cities = [city for city, val in zip(self.cities, values) if not np.isnan(val)]
            valid_values = [val for val in values if not np.isnan(val)]
            
            if valid_values: # Only add trace if there's data
                bar_chart_data.append(go.Bar(x=valid_cities, y=valid_values, name=metric.upper()))
        
        if bar_chart_data:
            for bar_trace in bar_chart_data:
                fig.add_trace(bar_trace, row=3, col=1)
            fig.update_yaxes(title_text="Metric Value", row=3, col=1)
            fig.update_xaxes(title_text="City", row=3, col=1)
        else:
            fig.add_annotation(text="No metric data for latest round city comparison.",
                               xref="x3", yref="y3", showarrow=False, font=dict(size=16),
                               row=3, col=1)
        
        # 5. Error Distribution (Sample) - Histogram
        # Pick a city and round to display a sample error distribution
        sample_city = self.cities[0] if self.cities else None
        if sample_city and f"{latest_round}_{sample_city}" in self.predictions:
            preds = self.predictions[latest_round][sample_city]
            # Apply same flattening/alignment as in _calculate_detailed_metrics for consistency
            y_pred_for_plot = preds['predictions']
            y_true_for_plot = preds['targets']
            
            if y_pred_for_plot.ndim == 3 and y_true_for_plot.ndim == 2 and \
               y_pred_for_plot.shape[0] == y_true_for_plot.shape[0] and \
               y_pred_for_plot.shape[2] == y_true_for_plot.shape[1]:
                y_pred_for_plot = y_pred_for_plot[:, -1, :]
            
            errors_for_plot = (y_pred_for_plot.flatten() - y_true_for_plot.flatten())
            fig.add_trace(go.Histogram(x=errors_for_plot, name='Error Distribution', marker_color='#00BFFF', opacity=0.7),
                          row=3, col=2)
            fig.update_yaxes(title_text="Count", row=3, col=2)
            fig.update_xaxes(title_text="Prediction Error", row=3, col=2)
        else:
            fig.add_annotation(text="No prediction data for error distribution.",
                               xref="x5", yref="y5", showarrow=False, font=dict(size=16),
                               row=3, col=2)

        # 6. Prediction Accuracy (Sample) - Scatter Plot (y_true vs y_pred)
        if sample_city and f"{latest_round}_{sample_city}" in self.predictions:
            preds = self.predictions[latest_round][sample_city]
            y_pred_scatter = preds['predictions']
            y_true_scatter = preds['targets']

            if y_pred_scatter.ndim == 3 and y_true_scatter.ndim == 2 and \
               y_pred_scatter.shape[0] == y_true_scatter.shape[0] and \
               y_pred_scatter.shape[2] == y_true_scatter.shape[1]:
                y_pred_scatter = y_pred_scatter[:, -1, :]
            
            y_pred_scatter = y_pred_scatter.flatten()
            y_true_scatter = y_true_scatter.flatten()

            fig.add_trace(go.Scatter(x=y_true_scatter, y=y_pred_scatter, mode='markers', name='Predictions',
                                     marker=dict(opacity=0.6, size=5, color='#32CD32')), # Lighter green
                          row=2, col=2) # Place it in the empty spot in row 2
            
            # Add y=x line
            min_val = min(y_true_scatter.min(), y_pred_scatter.min())
            max_val = max(y_true_scatter.max(), y_pred_scatter.max())
            fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines',
                                     name='Perfect Prediction', line=dict(color='red', dash='dash')),
                          row=2, col=2)
            
            fig.update_yaxes(title_text="Predicted Values", row=2, col=2)
            fig.update_xaxes(title_text="Actual Values", row=2, col=2)
        else:
             fig.add_annotation(text="No prediction data for accuracy scatter plot.",
                               xref="x4", yref="y4", showarrow=False, font=dict(size=16),
                               row=2, col=2) # This annotation is for the second column of row 2


        # Update layout
        fig.update_layout(
            height=1400, # Increased height to accommodate more plots
            title_text="X-FedFormer Comprehensive Analysis Dashboard",
            showlegend=True,
            hovermode="x unified" # For better hover experience
        )
        
        # Save interactive plot
        pyo.plot(fig, filename=str(self.results_dir / "interactive_dashboard.html"), auto_open=False)
        logging.info("Interactive dashboard saved as HTML")
    
    def generate_comprehensive_report(self):
        """Generate comprehensive analysis report."""
        logging.info("Generating comprehensive report...")
        
        report = {
            'analysis_summary': self._generate_summary_statistics(),
            'model_performance': self._analyze_model_performance(),
            'convergence_analysis': self._analyze_convergence(),
            'city_comparison': self._compare_cities(),
            'recommendations': self._generate_recommendations()
        }
        
        # Save detailed report
        with open(self.results_dir / "comprehensive_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate markdown report
        self._generate_markdown_report(report)
        
        logging.info("Comprehensive report generated")
    
    def _generate_summary_statistics(self) -> Dict:
        """Generate summary statistics."""
        summary = {
            'total_rounds': len(self.results),
            'cities_analyzed': len(self.cities),
            'best_performance': {},
            'worst_performance': {},
            'average_performance': {}
        }
        
        # Calculate best/worst performance per metric
        for metric in ['mae', 'rmse', 'r2', 'mape', 'directional_accuracy']: # Include directional_accuracy
            all_values = []
            for round_results in self.results.values():
                for city_results in round_results.values():
                    if metric in city_results and not np.isnan(city_results[metric]): # Ensure not NaN
                        all_values.append(city_results[metric])
            
            if all_values:
                if metric in ['mae', 'rmse', 'mape']:  # Lower is better
                    summary['best_performance'][metric] = min(all_values)
                    summary['worst_performance'][metric] = max(all_values)
                elif metric == 'r2' or metric == 'directional_accuracy':  # Higher is better (r2, directional_accuracy)
                    summary['best_performance'][metric] = max(all_values)
                    summary['worst_performance'][metric] = min(all_values)
                
                summary['average_performance'][metric] = np.mean(all_values)
            else: # If no valid values for a metric
                summary['best_performance'][metric] = 'N/A'
                summary['worst_performance'][metric] = 'N/A'
                summary['average_performance'][metric] = 'N/A'

        return summary
    
    def _analyze_model_performance(self) -> Dict:
        """Analyze overall model performance."""
        performance = {}
        
        for city in self.cities:
            city_performance = {
                'improvement_rate': {},
                'stability': {},
                'final_performance': {}
            }
            
            rounds = sorted(self.results.keys())
            
            for metric in ['mae', 'rmse', 'r2', 'mape', 'directional_accuracy']: # Include directional_accuracy
                values = []
                for round_num in rounds:
                    city_results = self.results[round_num].get(city, {})
                    if metric in city_results and not np.isnan(city_results[metric]): # Ensure not NaN
                        values.append(city_results[metric])
                
                if len(values) > 1:
                    # Calculate improvement rate
                    if metric in ['mae', 'rmse', 'mape']:  # Lower is better
                        # Avoid division by zero if initial value is 0 or very close to 0
                        improvement = (values[0] - values[-1]) / (values[0] + 1e-8) * 100
                    else:  # Higher is better (r2, directional_accuracy)
                        improvement = (values[-1] - values[0]) / (abs(values[0]) + 1e-8) * 100 # Use abs for initial if it can be negative (though R2 shouldn't be for initial)
                    
                    city_performance['improvement_rate'][metric] = improvement
                    # Calculate stability (Coefficient of Variation)
                    city_performance['stability'][metric] = np.std(values) / (np.mean(values) + 1e-8) if np.mean(values) != 0 else np.nan
                    city_performance['final_performance'][metric] = values[-1]
                elif len(values) == 1: # Only one data point
                    city_performance['improvement_rate'][metric] = np.nan
                    city_performance['stability'][metric] = np.nan
                    city_performance['final_performance'][metric] = values[-1]
                else: # No data points
                    city_performance['improvement_rate'][metric] = np.nan
                    city_performance['stability'][metric] = np.nan
                    city_performance['final_performance'][metric] = np.nan
            
            performance[city] = city_performance
        
        return performance
    
    def _analyze_convergence(self) -> Dict:
        """Analyze convergence patterns."""
        convergence = {}
        
        for city in self.cities:
            rounds = sorted(self.results.keys())
            mae_values = [self.results[r].get(city, {}).get('mae', np.nan) for r in rounds]
            mae_values = [v for v in mae_values if not np.isnan(v)]
            
            if len(mae_values) > 3: # Need at least 4 points to see some trend
                # Calculate convergence metrics
                improvements = [mae_values[i-1] - mae_values[i] for i in range(1, len(mae_values))]
                
                converged_status = False
                convergence_round = None
                # Check for convergence: if the absolute improvement in the last few rounds is very small
                if len(improvements) > 2: # Check last 3 improvements
                    if all(abs(imp) < 0.001 for imp in improvements[-3:]):
                        converged_status = True
                        # Find the first round where it started converging
                        for i, imp in enumerate(improvements):
                            if abs(imp) < 0.001:
                                convergence_round = rounds[i+1] # The round where it first reached minimal improvement
                                break
                
                convergence[city] = {
                    'converged': converged_status,
                    'convergence_round': convergence_round,
                    'total_improvement': mae_values[0] - mae_values[-1] if mae_values else 0,
                    'avg_improvement_rate': np.mean(improvements) if improvements else 0
                }
            else: # Not enough data for meaningful convergence analysis
                convergence[city] = {
                    'converged': False,
                    'convergence_round': None,
                    'total_improvement': np.nan,
                    'avg_improvement_rate': np.nan
                }
        
        return convergence
    
    def _compare_cities(self) -> Dict:
        """Compare performance across cities."""
        comparison = {}
        # Ensure there's at least one round in results
        if not self.results:
            return comparison # Return empty if no results

        latest_round = max(self.results.keys())
        
        for metric in ['mae', 'rmse', 'r2', 'mape', 'directional_accuracy']: # Include directional_accuracy
            city_scores = {}
            for city in self.cities:
                city_results = self.results[latest_round].get(city, {})
                if metric in city_results and not np.isnan(city_results[metric]): # Ensure not NaN
                    city_scores[city] = city_results[metric]
            
            if city_scores:
                if metric in ['mae', 'rmse', 'mape']:  # Lower is better
                    best_city = min(city_scores, key=city_scores.get)
                    worst_city = max(city_scores, key=city_scores.get)
                else:  # Higher is better (r2, directional_accuracy)
                    best_city = max(city_scores, key=city_scores.get)
                    worst_city = min(city_scores, key=city_scores.get)
                
                comparison[metric] = {
                    'best_city': best_city,
                    'best_score': city_scores[best_city],
                    'worst_city': worst_city,
                    'worst_score': city_scores[worst_city],
                    'all_scores': city_scores
                }
        
        return comparison
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Check if results exist
        if not self.results:
            recommendations.append("No analysis results available to generate recommendations.")
            return recommendations

        latest_round = max(self.results.keys())
        
        # Analyze performance based on latest round
        for city in self.cities:
            city_results = self.results[latest_round].get(city, {})
            
            if 'mae' in city_results and not np.isnan(city_results['mae']):
                if city_results['mae'] > 0.15: # Example threshold, adjust as needed
                    recommendations.append(f"For {city}: MAE ({city_results['mae']:.3f}) is relatively high. Consider increasing model capacity, exploring different architectures, or augmenting data for this specific city.")
                elif city_results['mae'] > 0.08:
                    recommendations.append(f"For {city}: MAE ({city_results['mae']:.3f}) suggests room for improvement. Fine-tuning hyperparameters or applying advanced regularization might help.")
            
            if 'r2' in city_results and not np.isnan(city_results['r2']):
                if city_results['r2'] < 0.6: # Example threshold
                    recommendations.append(f"For {city}: R² ({city_results['r2']:.3f}) is low, indicating poor fit. Re-evaluate feature engineering, data quality, or model complexity for {city}.")
                elif city_results['r2'] < 0.75:
                    recommendations.append(f"For {city}: R² ({city_results['r2']:.3f}) shows moderate fit. Further optimization of model parameters or ensemble methods could improve performance.")
            
            if 'directional_accuracy' in city_results and not np.isnan(city_results['directional_accuracy']):
                if city_results['directional_accuracy'] < 60: # Example threshold for directional accuracy
                    recommendations.append(f"For {city}: Directional accuracy ({city_results['directional_accuracy']:.2f}%) is poor. Investigate if the model captures temporal dynamics effectively. Consider sequence-to-sequence improvements.")


        # Analyze stability across rounds
        for city in self.cities:
            mae_values = [self.results[r].get(city, {}).get('mae', np.nan) for r in self.results.keys()]
            mae_values = [v for v in mae_values if not np.isnan(v)]
            
            if len(mae_values) > 1:
                # Calculate Coefficient of Variation (CV)
                mean_mae = np.mean(mae_values)
                std_mae = np.std(mae_values)
                if mean_mae > 1e-8: # Avoid division by zero
                    cv = std_mae / mean_mae
                    if cv > 0.15: # Example threshold for instability
                        recommendations.append(f"For {city}: Training instability detected (MAE CV: {cv:.3f}). Consider adjusting learning rate schedules, batch normalization, or adding more robust regularization.")
                else: # Mean is zero or very small, implying either perfect or NaN performance
                    if std_mae > 1e-8: # If mean is near zero but std is not, it's unstable
                         recommendations.append(f"For {city}: Training instability detected (MAE std: {std_mae:.3f}) despite near-zero mean. Investigate potential numerical issues or oscillations.")


        # Analyze convergence
        convergence_data = self._analyze_convergence()
        for city, conv_info in convergence_data.items():
            if not conv_info['converged'] and len(self.results) > 5: # If not converged after a few rounds
                recommendations.append(f"For {city}: Model does not appear to have fully converged (last round improvement: {conv_info['avg_improvement_rate']:.4f}). Consider extending training duration or re-evaluating optimizer settings.")
            if conv_info['total_improvement'] < 0 and len(self.results) > 1:
                recommendations.append(f"For {city}: Overall performance degraded during training (Total MAE improvement: {conv_info['total_improvement']:.4f}). This indicates potential overfitting or issues with later training rounds.")
        
        # General recommendations based on overall performance
        summary_stats = self._generate_summary_statistics()
        if summary_stats['average_performance'].get('mae', np.inf) > 0.1:
            recommendations.append("Overall average MAE is high across all cities. A global model architecture review or more comprehensive data preprocessing might be beneficial.")
        if summary_stats['average_performance'].get('r2', -np.inf) < 0.7:
            recommendations.append("Overall average R² is moderate. Consider exploring more advanced feature engineering techniques or a deeper model architecture.")

        if not recommendations:
            recommendations.append("Model performance is generally good across all cities. Focus on minor fine-tuning or exploring deployment optimization.")

        return recommendations
    
    def _generate_markdown_report(self, report: Dict):
        """Generate markdown report."""
        markdown_content = f"""# X-FedFormer Comprehensive Analysis Report

## Executive Summary

This report provides a comprehensive analysis of the X-FedFormer model performance across {report['analysis_summary']['cities_analyzed']} cities over {report['analysis_summary']['total_rounds']} training rounds.

## Performance Summary

### Best Performance Achieved
- **MAE**: {report['analysis_summary']['best_performance'].get('mae', 'N/A'):.4f}
- **RMSE**: {report['analysis_summary']['best_performance'].get('rmse', 'N/A'):.4f}
- **R²**: {report['analysis_summary']['best_performance'].get('r2', 'N/A'):.4f}
- **MAPE**: {report['analysis_summary']['best_performance'].get('mape', 'N/A'):.4f}%
- **Directional Accuracy**: {report['analysis_summary']['best_performance'].get('directional_accuracy', 'N/A'):.2f}%

### Average Performance
- **MAE**: {report['analysis_summary']['average_performance'].get('mae', 'N/A'):.4f}
- **RMSE**: {report['analysis_summary']['average_performance'].get('rmse', 'N/A'):.4f}
- **R²**: {report['analysis_summary']['average_performance'].get('r2', 'N/A'):.4f}
- **MAPE**: {report['analysis_summary']['average_performance'].get('mape', 'N/A'):.4f}%
- **Directional Accuracy**: {report['analysis_summary']['average_performance'].get('directional_accuracy', 'N/A'):.2f}%

## City-wise Performance Comparison

"""
        
        for metric, comparison in report['city_comparison'].items():
            markdown_content += f"""
### {metric.upper()}
- **Best Performance**: {comparison['best_city']} ({comparison['best_score']:.4f})
- **Worst Performance**: {comparison['worst_city']} ({comparison['worst_score']:.4f})
"""
        
        markdown_content += """
## Convergence Analysis

"""
        
        for city, conv_data in report['convergence_analysis'].items():
            status = "✅ Converged" if conv_data['converged'] else "⚠️ Not Converged"
            conv_round = conv_data['convergence_round'] if conv_data['convergence_round'] else "N/A"
            total_imp_str = f"{conv_data['total_improvement']:.4f}" if not np.isnan(conv_data['total_improvement']) else "N/A"
            avg_imp_rate_str = f"{conv_data['avg_improvement_rate']:.4f}" if not np.isnan(conv_data['avg_improvement_rate']) else "N/A"

            markdown_content += f"""
### {city}
- **Status**: {status}
- **Convergence Round**: {conv_round}
- **Total Improvement (MAE)**: {total_imp_str}
- **Average Improvement Rate (MAE)**: {avg_imp_rate_str}
"""
        
        markdown_content += """
## Recommendations

"""
        
        if report['recommendations']:
            for i, rec in enumerate(report['recommendations'], 1):
                markdown_content += f"{i}. {rec}\n"
        else:
            markdown_content += "No specific recommendations at this time. Model performance is robust.\n"

        markdown_content += """
## Files Generated

- `plots/`: Comprehensive visualization suite
- `interpretability/`: SHAP and LIME analysis results
- `detailed_analysis/`: Statistical analysis and anomaly detection
- `interactive_dashboard.html`: Interactive performance dashboard
- `comprehensive_report.json`: Detailed analysis results
- `checkpoint_metrics.json`: Raw evaluation metrics

---
*Report generated by Enhanced X-FedFormer Checkpoint Analyzer*
"""
        
        with open(self.results_dir / "analysis_report.md", 'w') as f:
            f.write(markdown_content)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Enhanced X-FedFormer Checkpoint Analysis")
    parser.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints"),
                       help="Directory containing model checkpoints")
    parser.add_argument("--data-dir", type=Path, default=Path("data_cache"),
                       help="Directory containing dataset files")
    parser.add_argument("--results-dir", type=Path, default=Path("analysis_results"),
                       help="Directory to save analysis results")
    parser.add_argument("--cities", nargs="+", default=["Almaty", "Astana", "Karaganda",
                   "Shymkent", "Aktobe", "Pavlodar", "Taraz", "Atyrau", "Kostanay", "Aktau"],
                       help="List of cities to analyze")
    parser.add_argument("--days-data", type=int, default=30,
                       help="Number of days of data to use")
    parser.add_argument("--skip-interpretability", action="store_true",
                       help="Skip SHAP/LIME interpretability analysis")
    parser.add_argument("--skip-interactive", action="store_true",
                       help="Skip interactive dashboard generation")
    
    args = parser.parse_args()
    
    # Setup logging
    args.results_dir.mkdir(exist_ok=True, parents=True) # Ensure results directory exists before logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        handlers=[
            logging.FileHandler(args.results_dir / "analysis.log"),
            logging.StreamHandler()
        ]
    )
    
    logging.info("Starting Enhanced X-FedFormer Analysis...")
    logging.info(f"Checkpoint directory: {args.ckpt_dir}")
    logging.info(f"Data directory: {args.data_dir}")
    logging.info(f"Results directory: {args.results_dir}")
    logging.info(f"Cities: {args.cities}")
    
    # Initialize analyzer
    analyzer = EnhancedCheckpointAnalyzer(
        ckpt_dir=args.ckpt_dir,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        cities=args.cities,
        days_data=args.days_data
    )
    
    try:
        # 1. Evaluate all checkpoints
        logging.info("Phase 1: Evaluating checkpoints...")
        analyzer.results = analyzer.evaluate_checkpoints()
        
        # 2. Save raw results
        with open(analyzer.results_dir / "checkpoint_metrics.json", 'w') as f:
            json.dump(analyzer.results, f, indent=2, default=str)
        
        # 3. Generate comprehensive visualizations
        logging.info("Phase 2: Generating visualizations...")
        analyzer.generate_comprehensive_plots()
        
        # 4. Statistical analysis
        logging.info("Phase 3: Statistical analysis...")
        analyzer.generate_statistical_analysis()
        
        # 5. Interactive dashboard (optional)
        if not args.skip_interactive:
            logging.info("Phase 4: Generating interactive dashboard...")
            try:
                analyzer.generate_interactive_dashboard()
            except Exception as e:
                logging.warning(f"Interactive dashboard generation failed: {e}")
        
        # 6. Comprehensive report
        logging.info("Phase 5: Generating comprehensive report...")
        analyzer.generate_comprehensive_report()
        
        logging.info("Analysis complete! Check the results directory for outputs.")
        logging.info(f"Main outputs:")
        logging.info(f"  - Analysis report: {analyzer.results_dir / 'analysis_report.md'}")
        logging.info(f"  - Visualizations: {analyzer.results_dir / 'plots'}")
        logging.info(f"  - Detailed analysis: {analyzer.results_dir / 'detailed_analysis'}")
        if not args.skip_interpretability:
            logging.info(f"  - Interpretability: {analyzer.results_dir / 'interpretability'}")
        
    except Exception as e:
        logging.error(f"Analysis failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()