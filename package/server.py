import flwr as fl
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict # Import OrderedDict for state_dict

from config import log, CKPT_DIR, PROX_MU, LOCAL_EPOCHS, DEVICE, SEQ_LEN, HORIZON, DATA_DIR, BATCH_SIZE
from data_utils import TransitDataset, collate_fn # Assuming TransitDataset contains route_ids and zone info
from xfedformer_model import XFedFormer
import monitor_db

class FedProxStrategy(fl.server.strategy.FedAvg):
    def __init__(self, initial_parameters: Optional[fl.common.Parameters] = None, **kwargs):
        super().__init__(
            initial_parameters=initial_parameters,
            fraction_fit=1.0, # All available clients will be sampled for fitting
            fraction_evaluate=1.0, # All available clients will be sampled for evaluation
            min_fit_clients=1, # Minimum number of clients required to be sampled for fit
            min_evaluate_clients=1, # Minimum number of clients required to be sampled for evaluate
            min_available_clients=1, # Minimum number of clients that need to be connected to the server
            **kwargs
        )
        log.info("FedProxStrategy initialized with FedAvg base.")
        monitor_db.init_db() # Ensure DB is initialized when strategy starts
        self.current_global_parameters: Optional[fl.common.Parameters] = initial_parameters # Store current global parameters

    def configure_fit(self, server_round: int, parameters: fl.common.Parameters, client_manager: fl.server.client_manager.ClientManager):
        # This method is called by the server to determine which clients should be selected for training
        # and what configuration they should receive.
        config = {
            "server_round": server_round,
            "prox_mu": PROX_MU,
            "local_epochs": LOCAL_EPOCHS
        }
        # Call the base class's configure_fit to get the default client instructions
        fit_ins = super().configure_fit(server_round, parameters, client_manager)
        # Update the configuration for each client instruction
        for _, ins in fit_ins:
            ins.config.update(config)
        return fit_ins

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:
        """Aggregate fit results and optionally evaluate model."""
        if not results:
            log.warning(f"Round {server_round}: No fit results received from clients.")
            return None, {}

        # Aggregate parameters (model weights) using the base FedAvg strategy
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            self.current_global_parameters = aggregated_parameters # Store the newly aggregated parameters
            log.info(f"Round {server_round}: Aggregation complete. Parameters stored for server-side evaluation and saving.")
            # Model saving will now happen within _server_side_prediction_evaluation,
            # which is called after aggregate_evaluate.
        else:
            log.warning(f"Round {server_round}: Aggregation failed, no parameters to save.")

        # Prepare client fit metrics for batch saving to the database
        fit_metrics_for_db = []
        for client, res in results:
            client_id = client.cid
            num_examples = res.num_examples
            # Extract metrics reported by the client during its fit phase
            fit_loss = res.metrics.get("loss", float('nan')) # Assuming client reports 'loss'
            mae = res.metrics.get("mae", float('nan'))       # Assuming client reports 'mae'
            epsilon = res.metrics.get("epsilon", float('nan')) # Assuming client reports 'epsilon'
            eval_loss = res.metrics.get("eval_loss", None) # Clients might report eval_loss during fit

            fit_metrics_for_db.append({
                "round": server_round,
                "client_id": client_id,
                "num_examples": num_examples,
                "fit_loss": fit_loss,
                "eval_loss": eval_loss,
                "epsilon": epsilon,
                "mae": mae
            })
        
        # Save collected client fit metrics to the database in a batch
        if fit_metrics_for_db:
            monitor_db.save_client_metrics_batch(fit_metrics_for_db)
            log.info(f"Client fit metrics for round {server_round} saved to DB.")
        else:
            log.warning(f"No client fit metrics to save for round {server_round}.")

        # Return the aggregated parameters and metrics to the Flower server
        return aggregated_parameters, aggregated_metrics

    def configure_evaluate(self, server_round: int, parameters: fl.common.Parameters, client_manager: fl.server.client_manager.ClientManager):
        # This method is called by the server to determine which clients should be selected for evaluation
        # and what configuration they should receive.
        config = {"server_round": server_round}
        # Call the base class's configure_evaluate to get the default client instructions
        evaluate_ins = super().configure_evaluate(server_round, parameters, client_manager)
        # Update the configuration for each client instruction
        for _, ins in evaluate_ins:
            ins.config.update(config)
        return evaluate_ins

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, fl.common.Scalar]]:
        """Aggregate evaluation results and save server metrics."""
        if not results:
            log.warning(f"Round {server_round}: No evaluation results received from clients.")
            # If no results, still attempt server-side prediction evaluation if parameters exist
            self._server_side_prediction_evaluation(server_round, self.current_global_parameters)
            return None, {}

        # Aggregate loss and other metrics from client evaluations using the base FedAvg strategy
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(server_round, results, failures)

        log.debug(f"Round {server_round}: Aggregated evaluation results from {len(results)} clients.")

        # Calculate average MAE and epsilon across clients for server-side metrics
        total_mae = 0.0
        total_epsilon = 0.0
        num_examples_total_mae = 0
        num_examples_total_epsilon = 0

        eval_metrics_for_db = [] # Collect client eval metrics for batch saving

        for client, eval_res in results:
            mae = eval_res.metrics.get("mae", float('nan'))
            epsilon = eval_res.metrics.get("epsilon", float('nan'))

            # Accumulate weighted MAE and Epsilon
            if not np.isnan(mae):
                total_mae += mae * eval_res.num_examples
                num_examples_total_mae += eval_res.num_examples
            if not np.isnan(epsilon):
                total_epsilon += epsilon * eval_res.num_examples
                num_examples_total_epsilon += eval_res.num_examples

            # Prepare client evaluation metrics for batch saving
            eval_metrics_for_db.append({
                "round": server_round,
                "client_id": client.cid,
                "num_examples": eval_res.num_examples,
                "fit_loss": None, # This is an eval aggregation, fit_loss is not directly here
                "eval_loss": eval_res.loss,
                "epsilon": epsilon,
                "mae": mae
            })

        # Calculate weighted averages
        avg_mae = total_mae / num_examples_total_mae if num_examples_total_mae > 0 else None
        avg_epsilon = total_epsilon / num_examples_total_epsilon if num_examples_total_epsilon > 0 else None
        
        # Determine the maximum epsilon reported by any client (for DP tracking)
        epsilon_values = [res.metrics.get("epsilon") for _, res in results if res.metrics.get("epsilon") is not None and not np.isnan(res.metrics.get("epsilon"))]
        max_epsilon = max(epsilon_values) if epsilon_values else None

        # Prepare server metrics dictionary to be saved
        server_metrics_to_save = {
            "loss_aggregated": loss_aggregated, # This is the aggregated evaluation loss
            "avg_mae": avg_mae,
            "max_epsilon": max_epsilon,
            "avg_epsilon": avg_epsilon
        }

        # Save server-side metrics to the database
        monitor_db.save_server_metrics(server_round, server_metrics_to_save)
        log.info(f"Server eval metrics for round {server_round} updated in DB.")

        # Save collected client evaluation metrics to the database in a batch
        if eval_metrics_for_db:
            monitor_db.save_client_metrics_batch(eval_metrics_for_db)
            log.info(f"Client eval metrics for round {server_round} saved to DB.")
        else:
            log.warning(f"No client eval metrics to save for round {server_round}.")


        # Perform server-side prediction evaluation and save the global model
        self._server_side_prediction_evaluation(server_round, self.current_global_parameters)

        # Return the aggregated loss and metrics to the Flower server
        return loss_aggregated, metrics_aggregated

    def _server_side_prediction_evaluation(self, server_round: int, global_parameters: fl.common.Parameters):
        """
        Performs server-side evaluation of the global model on selected cities
        and saves predictions and route metadata to the database.
        Also saves the global model checkpoint.
        """
        if global_parameters == None:
            log.warning("No global parameters available for server-side prediction evaluation and saving.")
            return

        log.info(f"Server-side evaluation for prediction dashboard (Round {server_round})...")

        # Discover all cities for which data files exist
        all_data_files = list(DATA_DIR.glob(f"*_*.csv"))
        cities_to_eval = set()
        for f_path in all_data_files:
            try:
                city_name = f_path.name.split('_')[0]
                cities_to_eval.add(city_name)
            except IndexError:
                log.warning(f"Skipping malformed data file: {f_path.name}")
                continue

        if not cities_to_eval:
            log.error("No cities found for server-side evaluation. Cannot infer model dimensions or save model.")
            return

        # Use the first discovered city's dataset to infer model dimensions for XFedFormer instantiation.
        # This assumes all city datasets have consistent feature dimensions and route counts.
        first_city = sorted(list(cities_to_eval))[0]
        # Assuming a consistent naming convention for data files, e.g., city_Xdays_routesY.csv
        possible_files = list(DATA_DIR.glob(f"{first_city}_*days_routes*.csv"))
        if not possible_files:
            log.error(f"No data file found for server-side evaluation of {first_city}. Cannot infer model dimensions.")
            return
        data_file_path_eval = possible_files[0]

        eval_model = None # Initialize eval_model to None
        try:
            df_eval_city = pd.read_csv(data_file_path_eval, parse_dates=["datetime"])
            temp_eval_dataset = TransitDataset(
                df_eval_city, city_name=first_city, seq_len=SEQ_LEN, horizon=HORIZON)

            if len(temp_eval_dataset) == 0:
                log.error(f"Server-side eval dataset for {first_city} is empty. Cannot infer model dimensions.")
                return

            # --- Instantiate XFedFormer model ---
            # Model dimensions (input_dim, n_routes, n_static_feats) are inferred from the dataset.
            # This is crucial for matching the model's architecture to the loaded parameters.
            eval_model = XFedFormer(
                input_dim=temp_eval_dataset.input_dim,
                n_routes=temp_eval_dataset.n_routes,
                n_static_feats=temp_eval_dataset.static_features_norm.shape[1] if temp_eval_dataset.static_features_norm is not None else 0,
                seq_len=SEQ_LEN,
                horizon=HORIZON
            ).to(DEVICE)

            # Load the aggregated global parameters into the instantiated model
            current_model_state_dict = eval_model.state_dict()
            weights_list = fl.common.parameters_to_ndarrays(global_parameters)

            # Critical check: Ensure the number of parameters from Flower matches the model's state_dict keys
            if len(weights_list) != len(current_model_state_dict):
                log.error(f"Mismatched parameter count ({len(weights_list)}) and state_dict keys ({len(current_model_state_dict)}) for XFedFormer. "
                          f"This indicates a mismatch between the model architecture inferred from the dataset "
                          f"and the model that produced the aggregated parameters. Cannot load/save model professionally.")
                # This error is severe, so we return early to prevent further issues.
                return

            # Create a new state_dict from the received parameters
            new_state_dict = OrderedDict()
            for i, key in enumerate(current_model_state_dict.keys()):
                # Ensure the tensor is created on the correct device (CPU or GPU)
                new_state_dict[key] = torch.tensor(weights_list[i], device=DEVICE)

            eval_model.load_state_dict(new_state_dict)
            eval_model.eval() # Set model to evaluation mode (e.g., disable dropout)

            # --- Save the global model's state_dict and its configuration ---
            # This saved config will be used by the Flask dashboard to load the model correctly.
            model_config = {
                "input_dim": temp_eval_dataset.input_dim,
                "n_routes": temp_eval_dataset.n_routes,
                "n_static_feats": temp_eval_dataset.static_features_norm.shape[1] if temp_eval_dataset.static_features_norm is not None else 0,
                "seq_len": SEQ_LEN,
                "horizon": HORIZON
            }
            model_save_data = {
                "model_state_dict": eval_model.state_dict(),
                "model_config": model_config
            }

            CKPT_DIR.mkdir(parents=True, exist_ok=True) # Ensure checkpoint directory exists
            model_path = CKPT_DIR / f"global_model_round_{server_round}.pt"
            latest_model_path = CKPT_DIR / "global_model.pt" # Always save the latest model to a fixed path

            torch.save(model_save_data, model_path)
            torch.save(model_save_data, latest_model_path)
            log.info(f"Global model state_dict and config saved to {model_path} and {latest_model_path}")

        except Exception as e:
            log.error(f"Error during server-side model setup or saving: {e}", exc_info=True)
            eval_model = None # Set to None if any error occurs to skip further steps

        if eval_model is None:
            log.warning("Skipping server-side prediction evaluation due to model setup failure.")
            return

        # --- Perform predictions and save to DB for each city ---
        for city_to_eval in sorted(list(cities_to_eval)):
            log.info(f"Performing server-side predictions for city: {city_to_eval}")
            possible_files = list(DATA_DIR.glob(f"{city_to_eval}_*days_routes*.csv"))
            if not possible_files:
                log.warning(f"No data file found for server-side evaluation of {city_to_eval}. Skipping predictions for this city.")
                continue

            data_file_path_eval = possible_files[0]
            try:
                df_eval_city = pd.read_csv(data_file_path_eval, parse_dates=["datetime"])
            except FileNotFoundError:
                log.error(f"Data file not found for server-side eval: {data_file_path_eval}. Skipping.")
                continue

            current_city_dataset = TransitDataset(
                df_eval_city, city_name=city_to_eval, seq_len=SEQ_LEN, horizon=HORIZON)

            if len(current_city_dataset) == 0:
                log.warning(f"Server-side eval dataset for {city_to_eval} is empty. Skipping predictions for this city.")
                continue

            # Ensure the model is on the correct device and in eval mode
            eval_model.to(DEVICE)
            eval_model.eval()

            # Create DataLoader for batch processing
            data_loader = torch.utils.data.DataLoader(
                current_city_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
            )

            # Get scalers for denormalizing predictions back to original scale
            try:
                # Assuming scaler_means and scaler_stds are pandas Series with route_ids as index
                scaler_means_inflow = current_city_dataset.scaler_means[current_city_dataset.route_ids].values
                scaler_stds_inflow = current_city_dataset.scaler_stds[current_city_dataset.route_ids].values
            except KeyError as e:
                log.error(f"Error getting scaler means/stds for route_ids in {city_to_eval}: {e}. Ensure scaler indices match route_ids in TransitDataset.")
                continue

            predictions_to_save = []
            routes_metadata_to_save = []

            # Collect route metadata for this city (route_id, zone, city)
            # This loop ensures that route metadata is saved for all routes present in the dataset.
            # It assumes `TransitDataset` or `df_eval_city` can provide the `zone` for each `route_id`.
            # If your `TransitDataset` does not expose this, you might need to adjust `data_utils.py`.
            
            # A more robust way would be to query `df_eval_city` directly for unique route_id-zone pairs
            unique_route_zone_pairs = df_eval_city[['route_id', 'zone']].drop_duplicates().to_dict('records')

            for pair in unique_route_zone_pairs:
                routes_metadata_to_save.append({
                    "route_id": pair['route_id'],
                    "zone": pair['zone'],
                    "city": city_to_eval
                })


            with torch.no_grad(): # Disable gradient calculation for inference
                for xb, yb, static_fb, original_datetimes_batch in data_loader:
                    xb = xb.to(DEVICE) # Move input features to the correct device
                    static_fb = static_fb.to(DEVICE) if static_fb is not None else None # Move static features

                    # Perform inference
                    preds_s = eval_model(xb, static_feats=static_fb)
                    
                    # Adjust predictions if the model outputs for a horizon > 1
                    # We are interested in the prediction for the immediate next step (horizon=1)
                    if preds_s.ndim == 3: # If shape is (batch_size, horizon, n_routes)
                        preds_s = preds_s[:, 0, :] # Take predictions for the first horizon step

                    # Get actual values from the target tensor (yb)
                    # Assuming yb is (batch_size, seq_len, n_routes) or (batch_size, horizon, n_routes)
                    # We need the actual value corresponding to the prediction point.
                    # If horizon=1, it's typically the last element of the input sequence or the first of the target sequence.
                    targets_s = yb[:, -1, :] # Assuming target is the last element of the sequence for prediction

                    # Denormalize predictions and actuals back to original scale
                    preds_orig = (preds_s.cpu().numpy() * scaler_stds_inflow) + scaler_means_inflow
                    targets_orig = (targets_s.cpu().numpy() * scaler_stds_inflow) + scaler_means_inflow

                    # Iterate through each sample in the batch and each route to save predictions
                    for sample_idx in range(xb.size(0)):
                        current_datetime = original_datetimes_batch[sample_idx]
                        for r_idx, route_id in enumerate(current_city_dataset.route_ids):
                            actual_val = targets_orig[sample_idx, r_idx]
                            predicted_val = preds_orig[sample_idx, r_idx]
                            predictions_to_save.append({
                                "route_id": route_id,
                                "datetime": current_datetime.isoformat(),
                                "actual": float(actual_val), # Ensure float type for DB
                                "predicted": float(predicted_val), # Ensure float type for DB
                                "city": city_to_eval # Add city to predictions record
                            })

            # Save collected route metadata and predictions to the database
            if routes_metadata_to_save:
                monitor_db.save_route_metadata_batch(routes_metadata_to_save)
                log.info(f"Route metadata for {city_to_eval} saved to DB.")
            else:
                log.warning(f"No route metadata to save for {city_to_eval}.")

            if predictions_to_save:
                monitor_db.save_predictions(server_round, city_to_eval, predictions_to_save)
                log.info(f"Predictions for {city_to_eval} (Round {server_round}) saved to DB.")
            else:
                log.warning(f"No predictions to save for {city_to_eval} (Round {server_round}).")
