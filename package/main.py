#!/usr/bin/env python3
"""
X-FedFormer — Cross-City Federated Transformer with Differential Privacy
Refactored with modular layers, synthetic data generator, FedProx, and DP.
"""

import argparse
import json
import flwr as fl
import pandas as pd
import torch
from rich.table import Table
# Removed: import time # Import the time module for delays

from config import log, console, DATA_DIR, CKPT_DIR, RESULT_DIR, DEVICE, SEQ_LEN, HORIZON
from data_utils import generate_synthetic_kz, TransitDataset
from xfedformer_model import XFedFormer
from client import FedProxClient
from server import FedProxStrategy
from eval_utils import quick_metrics
from pathlib import Path
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CLI Entrypoint
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
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=args.rounds),
            strategy=strategy,
        )
        log.info("Flower server finished.")

    elif args.cmd == "client":
        # Removed: Add a small delay before starting the client to allow the server to fully initialize
        # Removed: log.info("Waiting for 5 seconds to ensure server is ready...")
        # Removed: time.sleep(5) # Wait for 5 seconds

        possible_files = list(DATA_DIR.glob(
            f"{args.city}_{args.days_data}days_routes*.csv"))
        if not possible_files:
            log.error(f"No data file found for {args.city} with {args.days_data} days. "
                      f"Looked for: {DATA_DIR}/{args.city}_{args.days_data}days_routes*.csv")
            return

        data_file_path = possible_files[0]
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
            # Load the dictionary containing model_state_dict and model_config
            loaded_data = torch.load(model_file_path, map_location=DEVICE)
            global_model_state_dict = loaded_data.get('model_state_dict')
            model_config = loaded_data.get('model_config', {})

            if global_model_state_dict is None:
                log.error(f"Loaded data from {args.model_path} does not contain 'model_state_dict'.")
                return

        except Exception as e:
            log.error(
                f"Error loading model from {args.model_path}: {e}")
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

            # Instantiate model using the loaded config
            eval_model = XFedFormer(
                input_dim=model_config.get('input_dim', temp_eval_dataset.input_dim),
                n_routes=model_config.get('n_routes', temp_eval_dataset.n_routes),
                n_static_feats=model_config.get('n_static_feats', temp_eval_dataset.static_features_norm.shape[1] if temp_eval_dataset.static_features_norm is not None else 0),
                seq_len=model_config.get('seq_len', SEQ_LEN),
                horizon=model_config.get('horizon', HORIZON)
            ).to(DEVICE)

            try:
                eval_model.load_state_dict(global_model_state_dict)
                eval_model.eval() # Set to evaluation mode
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
        report_path = RESULT_DIR / args.report_file
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(results_summary, f, indent=2)
        log.info(f"Evaluation report saved to → {report_path}")

    else:
        log.error(f"Unknown command: {args.cmd}")
        ap.print_help()


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
