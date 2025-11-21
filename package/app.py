import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict
from flask import Flask, render_template, jsonify, request
import monitor_db
from datetime import datetime, timedelta
import os
from xfedformer_model import XFedFormer

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
    def debug(self, message):
        print(f"DEBUG: {message}")

log = Log()
DP_ENABLED = False

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# ── Initialize Database ─────────────────────────────────────────────────────────
monitor_db.init_db()

# ── Global Model Loading ─────────────────────────────────────────────────────────
GLOBAL_MODEL = None
GLOBAL_MODEL_CONFIG = None
MODEL_PATH = './checkpoints/global_model.pt'

def load_global_model():
    global GLOBAL_MODEL, GLOBAL_MODEL_CONFIG
    if os.path.exists(MODEL_PATH):
        try:
            loaded_data = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
            if isinstance(loaded_data, dict) and 'model_state_dict' in loaded_data:
                model_config = loaded_data.get('model_config', {})
                GLOBAL_MODEL_CONFIG = model_config

                n_routes_from_config = model_config.get('n_routes', 10)
                log.info(f"Loading model with n_routes from config: {n_routes_from_config}")
                log.info(f"Model config: {model_config}")

                GLOBAL_MODEL = XFedFormer(
                    input_dim=model_config.get('input_dim', 19),
                    n_routes=n_routes_from_config,
                    n_static_feats=model_config.get('n_static_feats', 0),
                    seq_len=model_config.get('seq_len', 1),
                    horizon=model_config.get('horizon', 1)
                )

                GLOBAL_MODEL.load_state_dict(loaded_data['model_state_dict'])
                GLOBAL_MODEL.eval()
                log.info(f"Global model loaded successfully from {MODEL_PATH}")
            else:
                log.error(f"Loaded object from {MODEL_PATH} does not contain 'model_state_dict' or is not a dictionary. Invalid format.")
                GLOBAL_MODEL = None
                GLOBAL_MODEL_CONFIG = None
        except Exception as e:
            log.error(f"Error loading global model from {MODEL_PATH}: {e}", exc_info=True)
            GLOBAL_MODEL = None
            GLOBAL_MODEL_CONFIG = None
    else:
        log.warning(f"Global model not found at {MODEL_PATH}. Prediction functionality will be disabled.")
        GLOBAL_MODEL = None
        GLOBAL_MODEL_CONFIG = None

with app.app_context():
    load_global_model()


# ── Helper Functions for Model Interaction ──────────────────────────────────────
BUSY_THRESHOLD = 50

def _get_route_zone_combinations(city: str, round_num: int) -> List[Dict]:
    """
    Retrieves all unique route-zone combinations for a given city.
    Used for populating dropdowns on the front end.
    """
    combinations = monitor_db.get_route_zone_combinations(city=city, round_num=round_num)
    if combinations is None:
        combinations = []
    log.debug(f"[_get_route_zone_combinations] City: {city}, Round: {round_num}")
    log.debug(f"[_get_route_zone_combinations] Final generated combinations: {combinations}")
    return combinations


def prepare_model_input_from_filters(city: str, single_datetime: datetime):
    """
    Prepares model input for a single-datetime prediction for *all* routes in the specified city.
    Returns:
      - model_input_tensor: a torch.Tensor of shape (1, seq_len, input_dim)
      - prediction_context: a dict { 'city': <str>, 'datetime': <ISO-str> }
      - all_routes_in_order: List[str] of route IDs (in the exact order the model expects)
    """
    if not all([city, single_datetime]):
        log.warning("Missing city or datetime for model input preparation.")
        return None, None, None

    latest_round = monitor_db.get_latest_round()
    all_routes_in_order = monitor_db.get_unique_routes(city=city, round_num=latest_round)
    if not all_routes_in_order:
        log.warning(f"No routes found for city {city}. Cannot prepare model input.")
        return None, None, None

    log.debug(f"[prepare_model_input_from_filters] all_routes_in_order: {all_routes_in_order}")

    try:
        city_id = abs(hash(city)) % 1000  # Placeholder mapping
        
        # 6 dynamic features + 13 dummy features = total 19 (assuming input_dim=19)
        features = [
            float(city_id),
            float(single_datetime.hour),
            float(single_datetime.weekday()),
            float(single_datetime.day),
            float(single_datetime.month),
            float(single_datetime.year),
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ]

        if GLOBAL_MODEL_CONFIG:
            expected_input_dim = GLOBAL_MODEL_CONFIG.get('input_dim', 19)
            if len(features) != expected_input_dim:
                log.error(f"[prepare_model_input_from_filters] Generated feature count ({len(features)}) != model's expected input_dim ({expected_input_dim}).")
                return None, None, None
        else:
            log.warning("[prepare_model_input_from_filters] GLOBAL_MODEL_CONFIG not loaded. Cannot verify input_dim.")

        input_tensor = torch.tensor([features], dtype=torch.float32)
        if input_tensor.ndim == 2:
            input_tensor = input_tensor.unsqueeze(1)

        log.debug(f"[prepare_model_input_from_filters] Final model input tensor shape: {input_tensor.shape}")

        prediction_context = {
            'city': city,
            'datetime': single_datetime.isoformat()
        }
        return input_tensor, prediction_context, all_routes_in_order

    except Exception as e:
        log.error(f"Error preparing model input: {e}", exc_info=True)
        return None, None, None


def process_model_output(
    raw_predictions: torch.Tensor,
    prediction_context: Dict,
    all_routes_in_order: List[str]
):
    """
    Converts the raw model output (tensor of shape [1, n_routes]) into
    a dict keyed by "route_id::zone", fetching actuals from the DB.
    """
    processed_data_by_combo = {}

    if raw_predictions is None or not prediction_context or not all_routes_in_order:
        log.warning("Received incomplete data for model output processing.")
        return {}

    try:
        if not isinstance(raw_predictions, torch.Tensor):
            log.error(f"Expected torch.Tensor for raw_predictions, got {type(raw_predictions)}")
            return {}

        # E.g. raw_predictions.squeeze().tolist() → [val_route0, val_route1, ...]
        predictions_for_all_routes = raw_predictions.squeeze().tolist()
        if not isinstance(predictions_for_all_routes, list):
            predictions_for_all_routes = [predictions_for_all_routes]

        log.debug(f"[process_model_output] Raw predictions (list): {predictions_for_all_routes}")

        if len(predictions_for_all_routes) != len(all_routes_in_order):
            log.error(f"Model predicted {len(predictions_for_all_routes)} values, but expected {len(all_routes_in_order)}.")
            return {}

        city = prediction_context['city']
        datetime_str = prediction_context['datetime']
        single_dt_obj = datetime.fromisoformat(datetime_str)

        for i, predicted_val in enumerate(predictions_for_all_routes):
            route_id = all_routes_in_order[i]
            zone_for_route = monitor_db.get_route_zone(route_id) or "Unknown Zone"
            combo_key = f"{route_id}::{zone_for_route}"

            log.debug(
                f"[process_model_output] Iteration {i}: route_id={route_id}, "
                f"zone_for_route={zone_for_route}, combo_key={combo_key}, predicted_val={predicted_val}"
            )

            actual_val = monitor_db.get_actual_passenger_count(city, route_id, datetime_str)
            log.debug(f"[process_model_output] Actual value for {combo_key}: {actual_val}")

            mae = None
            accuracy_percentage = "N/A"
            if actual_val is not None:
                mae = abs(predicted_val - actual_val)
                if actual_val != 0:
                    accuracy_percentage = f"{((1 - (mae / actual_val)) * 100):.1f}"
                else:
                    accuracy_percentage = "100.0" if mae == 0 else "0.0"

            is_busy = (predicted_val >= BUSY_THRESHOLD)
            processed_data_by_combo[combo_key] = {
                'route_id': route_id,
                'zone': zone_for_route,
                'labels': [single_dt_obj.strftime('%Y-%m-%d %H:%M')],
                'actual': [actual_val] if actual_val is not None else [None],
                'predicted': [max(0, round(predicted_val))],
                'is_busy': is_busy,
                'mae': mae,
                'accuracy_percentage': accuracy_percentage
            }

        log.debug(f"[process_model_output] Full processed_data_by_combo: {processed_data_by_combo}")

    except Exception as e:
        log.error(
            f"Error processing model output: {e}. "
            f"Type={type(raw_predictions)}, value-snippet={str(raw_predictions)[:50]}"
        )
        return {}

    return processed_data_by_combo


# ── Routes ───────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    latest_round = monitor_db.get_latest_round()
    cities_with_preds = monitor_db.get_cities_with_predictions(latest_round) if latest_round else []
    initial_routes = []
    initial_zones = []

    if cities_with_preds and latest_round:
        first_city = cities_with_preds[0]
        combos = _get_route_zone_combinations(city=first_city, round_num=latest_round)
        initial_routes = sorted({combo['route_id'] for combo in combos})
        initial_zones  = sorted({combo['zone']     for combo in combos})

    return render_template(
        'index.html',
        cities= cities_with_preds,
        latest_round= latest_round,
        DP_ENABLED= DP_ENABLED,
        initial_routes= initial_routes,
        initial_zones= initial_zones
    )


@app.route('/metrics')
def get_metrics():
    server_metrics = monitor_db.get_server_metrics()
    client_metrics = monitor_db.get_client_metrics()
    return jsonify(server_metrics=server_metrics, client_metrics=client_metrics)


@app.route('/predictions_data')
def get_predictions_data():
    if GLOBAL_MODEL is None:
        return jsonify({"error": "Prediction model not loaded. Cannot generate predictions."}), 503

    city             = request.args.get('city')
    selected_route_id = request.args.get('route_id')
    selected_zone     = request.args.get('zone')
    datetime_str      = request.args.get('datetime')
    latest_round      = monitor_db.get_latest_round()

    # If no city provided, pick the first available one
    if not city and latest_round:
        cities = monitor_db.get_cities_with_predictions(latest_round)
        if cities:
            city = cities[0]
        else:
            return jsonify({"error": "No cities with history found."}), 404

    if not city:
        return jsonify({"error": "City parameter is required."}), 400

    # Parse datetime
    single_datetime_obj = None
    if datetime_str:
        try:
            single_datetime_obj = datetime.fromisoformat(datetime_str)
        except ValueError:
            return jsonify({"error": "Invalid datetime format. Use YYYY-MM-DDTHH:MM."}), 400
    if not single_datetime_obj:
        return jsonify({"error": "A specific datetime is required."}), 400

    log.debug(
        f"[get_predictions_data] Requesting city={city}, "
        f"route_id={selected_route_id}, zone={selected_zone}, datetime={datetime_str}"
    )

    # Build model input
    model_input_tensor, prediction_context, all_routes_in_order = prepare_model_input_from_filters(
        city=city,
        single_datetime=single_datetime_obj
    )
    if model_input_tensor is None or not prediction_context or not all_routes_in_order:
        return jsonify({"error": "Could not prepare model input for selected filters."}), 400

    # Run inference
    try:
        with torch.no_grad():
            raw_model_output = GLOBAL_MODEL(model_input_tensor)
            log.debug(
                f"[get_predictions_data] Raw model output shape={raw_model_output.shape}, "
                f"values={raw_model_output.flatten()[:5].tolist()}..."
            )
            predictions_data_by_combo = process_model_output(
                raw_model_output, prediction_context, all_routes_in_order
            )
    except Exception as e:
        log.error(f"Error during model inference: {e}", exc_info=True)
        return jsonify({"error": f"Failed to generate predictions: {e}"}), 500

    if not predictions_data_by_combo:
        return jsonify({"error": "No prediction data after model processing."}), 404

    # ── FILTERING LOGIC ────────────────────────────────────────────────────────────
    filtered_chart_data = {}

    # If the user selected a specific route, we ignore zone filtering.
    # Otherwise (route_id == "all"), we apply the zone filter.
    for combo_key, data in predictions_data_by_combo.items():
        route_id = data['route_id']
        zone     = data['zone']

        if selected_route_id and selected_route_id != 'all':
            # User wants one route: include it no matter what the zone is
            if route_id == selected_route_id:
                filtered_chart_data[combo_key] = data
        else:
            # Route == "all", so filter by zone if zone != "all"
            zone_matches = (selected_zone == 'all') or (selected_zone == zone)
            if zone_matches:
                filtered_chart_data[combo_key] = data

    log.debug(f"[get_predictions_data] Final filtered_chart_data keys: {list(filtered_chart_data.keys())}")

    if not filtered_chart_data:
        return jsonify({"error": "No prediction data found for the selected filters. Try broadening your selection."}), 404

    return jsonify(filtered_chart_data)


@app.route('/routes_for_city')
def get_routes_for_city():
    city      = request.args.get('city')
    round_num = monitor_db.get_latest_round()
    combos    = _get_route_zone_combinations(city=city, round_num=round_num)
    routes    = sorted({ combo['route_id'] for combo in combos })
    return jsonify(routes)


@app.route('/zones_for_city')
def get_zones_for_city():
    city      = request.args.get('city')
    round_num = monitor_db.get_latest_round()
    combos    = _get_route_zone_combinations(city=city, round_num=round_num)
    zones     = sorted({ combo['zone'] for combo in combos })
    return jsonify(zones)


@app.route('/clients')
def clients_page():
    latest_round_clients = monitor_db.get_client_metrics(round_num=monitor_db.get_latest_round())
    return render_template('clients.html', clients=latest_round_clients, DP_ENABLED=DP_ENABLED)


@app.context_processor
def inject_now():
    """Injects the current datetime into all templates."""
    return {'now': datetime.now}


if __name__ == '__main__':
    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')
        log.info("Created 'checkpoints' directory.")
    app.run(host='0.0.0.0', port=5123, debug=True)
    log.info("Flask monitoring dashboard started on http://0.0.0.0:5123")
