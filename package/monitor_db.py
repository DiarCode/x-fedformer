import sqlite3
import json
from pathlib import Path
from datetime import datetime
import os

# ── Configuration ────────────────────────────────────────────────────────────────
DATABASE_PATH = Path("monitor_data.db")


def init_db():
    """
    Initializes (or re‐initializes) the SQLite database with the required tables.
    If the database file already exists, it will be deleted to ensure a clean schema
    and to prevent "no such column" errors after schema changes.
    """
    # 1) If the file exists, delete it so that a fresh DB is created
    # try:
    #     if DATABASE_PATH.exists() and deleteAll is :
    #         os.remove(DATABASE_PATH)
    #         print(f"INFO: Existing database file '{DATABASE_PATH}' removed.")
    # except OSError as e:
    #     print(f"ERROR: Could not remove existing database file '{DATABASE_PATH}': {e}")

    # 2) Create a new SQLite connection (this will create an empty file)
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        # ── Table: server_metrics ─────────────────────────────────────────────────
        # Stores aggregated, server‐side metrics (evaluation loss, MAE, ε-values) per round.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS server_metrics (
                round          INTEGER PRIMARY KEY,
                avg_eval_loss  REAL,    -- Aggregated evaluation loss for the round
                avg_mae        REAL,
                max_epsilon    REAL,
                avg_epsilon    REAL,
                timestamp      TEXT     -- ISO‐formatted insertion timestamp
            )
        """)

        # ── Table: client_round_metrics ─────────────────────────────────────────────
        # Stores per‐client metrics (fit_loss, eval_loss, epsilon, MAE) for each round.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS client_round_metrics (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                round        INTEGER NOT NULL,
                client_id    TEXT    NOT NULL,
                num_examples INTEGER,
                fit_loss     REAL,
                eval_loss    REAL,
                epsilon      REAL,
                mae          REAL,
                timestamp    TEXT,
                UNIQUE(round, client_id) ON CONFLICT REPLACE
            )
        """)

        # ── Table: predictions ───────────────────────────────────────────────────────
        # Stores predicted vs. actual passenger counts per (round, city, route_id, datetime).
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                round      INTEGER NOT NULL,
                city       TEXT    NOT NULL,
                route_id   TEXT    NOT NULL,
                datetime   TEXT    NOT NULL,   -- ISO‐formatted date/time string
                actual     REAL,
                predicted  REAL,
                timestamp  TEXT,               -- ISO‐formatted insertion timestamp
                UNIQUE(round, city, route_id, datetime) ON CONFLICT REPLACE
            )
        """)

        # ── Table: routes_metadata ──────────────────────────────────────────────────
        # Stores metadata for each route: the zone and city it belongs to.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS routes_metadata (
                route_id TEXT PRIMARY KEY,
                zone     TEXT NOT NULL,
                city     TEXT NOT NULL
                -- Add more columns here if needed, e.g., route_type, length_km, etc.
            )
        """)

        conn.commit()
        print("INFO: Database initialized successfully with all tables.")
    except sqlite3.Error as e:
        print(f"ERROR: SQLite database initialization failed: {e}")
    finally:
        if conn:
            conn.close()


# ── INSERT / REPLACE HELPERS ─────────────────────────────────────────────────────

def insert_server_metrics(round_num, avg_eval_loss, avg_mae, max_epsilon, avg_epsilon):
    """
    Inserts or replaces aggregated server metrics for a given round.
    Parameters:
      - round_num     (int): Round number
      - avg_eval_loss (float): Aggregated evaluation loss
      - avg_mae       (float): Average MAE across clients
      - max_epsilon   (float): Maximum ε encountered
      - avg_epsilon   (float): Average ε across clients
    """
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    timestamp = datetime.now().isoformat()
    try:
        cursor.execute("""
            INSERT OR REPLACE INTO server_metrics
                (round, avg_eval_loss, avg_mae, max_epsilon, avg_epsilon, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (round_num, avg_eval_loss, avg_mae, max_epsilon, avg_epsilon, timestamp))
        conn.commit()
    except sqlite3.Error as e:
        print(f"ERROR: Failed to insert server metrics for round {round_num}: {e}")
    finally:
        conn.close()


def insert_client_round_metrics(round_num, client_id, num_examples, fit_loss, eval_loss, epsilon, mae):
    """
    Inserts or replaces individual client‐side metrics for a specific round.
    Parameters:
      - round_num    (int)
      - client_id    (str)
      - num_examples (int)
      - fit_loss     (float)
      - eval_loss    (float)
      - epsilon      (float)
      - mae          (float)
    """
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    timestamp = datetime.now().isoformat()
    try:
        cursor.execute("""
            INSERT OR REPLACE INTO client_round_metrics
                (round, client_id, num_examples, fit_loss, eval_loss, epsilon, mae, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (round_num, client_id, num_examples, fit_loss, eval_loss, epsilon, mae, timestamp))
        conn.commit()
    except sqlite3.Error as e:
        print(f"ERROR: Failed to insert client metrics for client '{client_id}' (round {round_num}): {e}")
    finally:
        conn.close()


def insert_prediction(round_num, city, route_id, datetime_str, actual, predicted):
    """
    Inserts or replaces a single prediction record.
    Parameters:
      - round_num    (int)
      - city         (str)
      - route_id     (str)
      - datetime_str (str): ISO‐formatted date/time (e.g. "2025-06-04T08:00")
      - actual       (float or None)
      - predicted    (float or None)
    """
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    timestamp = datetime.now().isoformat()
    try:
        cursor.execute("""
            INSERT OR REPLACE INTO predictions
                (round, city, route_id, datetime, actual, predicted, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (round_num, city, route_id, datetime_str, actual, predicted, timestamp))
        conn.commit()
    except sqlite3.Error as e:
        print(f"ERROR: Failed to insert prediction for {city}/{route_id}/{datetime_str}: {e}")
    finally:
        conn.close()


def insert_route_metadata(route_id, zone, city):
    """
    Inserts or replaces metadata for a single route.
    Parameters:
      - route_id (str)
      - zone     (str)
      - city     (str)
    """
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT OR REPLACE INTO routes_metadata (route_id, zone, city)
            VALUES (?, ?, ?)
        """, (route_id, zone, city))
        conn.commit()
    except sqlite3.Error as e:
        print(f"ERROR: Failed to insert route metadata for route '{route_id}': {e}")
    finally:
        conn.close()


# ── BATCH INSERT HELPERS ─────────────────────────────────────────────────────────

def save_client_metrics_batch(metrics_list):
    """
    Batch‐inserts client_round_metrics from a list of dicts.
    Each dict must contain keys: round, client_id, num_examples, fit_loss, eval_loss, epsilon, mae
    """
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    timestamp = datetime.now().isoformat()
    try:
        for m in metrics_list:
            cursor.execute("""
                INSERT OR REPLACE INTO client_round_metrics
                    (round, client_id, num_examples, fit_loss, eval_loss, epsilon, mae, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                m['round'],
                m['client_id'],
                m['num_examples'],
                m['fit_loss'],
                m['eval_loss'],
                m['epsilon'],
                m['mae'],
                timestamp
            ))
        conn.commit()
    except sqlite3.Error as e:
        print(f"ERROR: Failed to batch save client metrics: {e}")
    finally:
        conn.close()


def save_server_metrics(round_num, metrics_dict):
    """
    Batch‐inserts aggregated server metrics for one round.
    metrics_dict must contain keys: loss_aggregated, avg_mae, max_epsilon, avg_epsilon
    (These correspond to avg_eval_loss, avg_mae, max_epsilon, avg_epsilon in the table.)
    """
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    timestamp = datetime.now().isoformat()
    avg_eval_loss = metrics_dict.get("loss_aggregated")
    avg_mae       = metrics_dict.get("avg_mae")
    max_epsilon   = metrics_dict.get("max_epsilon")
    avg_epsilon   = metrics_dict.get("avg_epsilon")
    try:
        cursor.execute("""
            INSERT OR REPLACE INTO server_metrics
                (round, avg_eval_loss, avg_mae, max_epsilon, avg_epsilon, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (round_num, avg_eval_loss, avg_mae, max_epsilon, avg_epsilon, timestamp))
        conn.commit()
    except sqlite3.Error as e:
        print(f"ERROR: Failed to save server metrics for round {round_num}: {e}")
    finally:
        conn.close()


def save_predictions(round_num, city, predictions_list):
    """
    Batch‐inserts prediction records for a given round and city.
    Each element in predictions_list must be a dict with:
        route_id  (str)
        datetime  (str, ISO‐formatted)
        actual    (float or None)
        predicted (float or None)
    """
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    timestamp = datetime.now().isoformat()
    try:
        for p in predictions_list:
            cursor.execute("""
                INSERT OR REPLACE INTO predictions
                    (round, city, route_id, datetime, actual, predicted, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                round_num,
                city,
                p['route_id'],
                p['datetime'],
                p.get('actual', None),
                p.get('predicted', None),
                timestamp
            ))
        conn.commit()
    except sqlite3.Error as e:
        print(f"ERROR: Failed to batch save predictions for city '{city}' (round {round_num}): {e}")
    finally:
        conn.close()


def save_route_metadata_batch(metadata_list):
    """
    Batch‐inserts route_metadata from a list of dicts.
    Each dict must contain: route_id, zone, city
    """
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    try:
        for m in metadata_list:
            cursor.execute("""
                INSERT OR REPLACE INTO routes_metadata (route_id, zone, city)
                VALUES (?, ?, ?)
            """, (m['route_id'], m['zone'], m['city']))
        conn.commit()
    except sqlite3.Error as e:
        print(f"ERROR: Failed to batch save route metadata: {e}")
    finally:
        conn.close()


# ── QUERY / FETCH HELPERS ─────────────────────────────────────────────────────────

def get_server_metrics():
    """
    Retrieves all server_metrics rows, ordered by round ascending.
    Returns a list of dicts in the form:
      {
        "round": int,
        "avg_client_loss": float,  # This is actually avg_eval_loss in the table
        "avg_mae": float,
        "max_epsilon": float,
        "avg_epsilon": float,
        "timestamp": str
      }
    """
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    metrics = []
    try:
        cursor.execute("""
            SELECT round, avg_eval_loss, avg_mae, max_epsilon, avg_epsilon, timestamp
            FROM server_metrics
            ORDER BY round ASC
        """)
        for row in cursor.fetchall():
            metrics.append({
                "round": row[0],
                "avg_client_loss": row[1],  # Key kept for compatibility with front‐end
                "avg_mae": row[2],
                "max_epsilon": row[3],
                "avg_epsilon": row[4],
                "timestamp": row[5]
            })
    except sqlite3.Error as e:
        print(f"ERROR: Failed to retrieve server metrics: {e}")
    finally:
        conn.close()
    return metrics


def get_client_metrics(round_num=None):
    """
    Retrieves client_round_metrics, optionally filtered by round number.
    Returns a list of dicts:
      {
        "round": int,
        "client_id": str,
        "num_examples": int,
        "fit_loss": float,
        "eval_loss": float,
        "epsilon": float,
        "mae": float,
        "timestamp": str
      }
    """
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    metrics = []
    query = """
        SELECT round, client_id, num_examples, fit_loss, eval_loss, epsilon, mae, timestamp
        FROM client_round_metrics
        WHERE 1=1
    """
    params = []
    if round_num is not None:
        query += " AND round = ?"
        params.append(round_num)
    query += " ORDER BY round ASC, client_id ASC"
    try:
        cursor.execute(query, params)
        for row in cursor.fetchall():
            metrics.append({
                "round": row[0],
                "client_id": row[1],
                "num_examples": row[2],
                "fit_loss": row[3],
                "eval_loss": row[4],
                "epsilon": row[5],
                "mae": row[6],
                "timestamp": row[7]
            })
    except sqlite3.Error as e:
        print(f"ERROR: Failed to retrieve client metrics: {e}")
    finally:
        conn.close()
    return metrics


def get_predictions(city=None, route_id=None, datetime_str=None, round_num=None):
    """
    Retrieves prediction records, with optional filters for city, route_id, datetime, and round.
    Returns a list of dicts:
      {
        "round": int,
        "city": str,
        "route_id": str,
        "datetime": str,
        "actual": float,
        "predicted": float
      }
    """
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    predictions = []
    query = "SELECT round, city, route_id, datetime, actual, predicted FROM predictions WHERE 1=1"
    params = []
    if city:
        query += " AND city = ?"
        params.append(city)
    if route_id:
        query += " AND route_id = ?"
        params.append(route_id)
    if datetime_str:
        query += " AND datetime = ?"
        params.append(datetime_str)
    if round_num is not None:
        query += " AND round = ?"
        params.append(round_num)
    query += " ORDER BY datetime ASC, route_id ASC"
    try:
        cursor.execute(query, params)
        for row in cursor.fetchall():
            predictions.append({
                "round": row[0],
                "city": row[1],
                "route_id": row[2],
                "datetime": row[3],
                "actual": row[4],
                "predicted": row[5]
            })
    except sqlite3.Error as e:
        print(f"ERROR: Failed to retrieve predictions: {e}")
    finally:
        conn.close()
    return predictions


def get_actual_passenger_count(city, route_id, datetime_str):
    """
    Retrieves the actual passenger count for a specific (city, route_id, datetime).
    If multiple rounds exist, returns the most recent (highest round). Returns None if not found.
    """
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    actual_value = None
    try:
        cursor.execute("""
            SELECT actual
            FROM predictions
            WHERE city = ? AND route_id = ? AND datetime = ?
            ORDER BY round DESC
            LIMIT 1
        """, (city, route_id, datetime_str))
        result = cursor.fetchone()
        if result:
            actual_value = result[0]
    except sqlite3.Error as e:
        print(f"ERROR: Failed to retrieve actual passenger count for {city}/{route_id}/{datetime_str}: {e}")
    finally:
        conn.close()
    return actual_value


def get_latest_round():
    """
    Retrieves the maximum (latest) round number from server_metrics.
    Returns None if there are no rows.
    """
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    latest_round = None
    try:
        cursor.execute("SELECT MAX(round) FROM server_metrics")
        latest_round = cursor.fetchone()[0]
    except sqlite3.Error as e:
        print(f"ERROR: Failed to retrieve latest round: {e}")
    finally:
        conn.close()
    return latest_round


def get_cities_with_predictions(round_num=None):
    """
    Retrieves a sorted list of distinct cities that have entries in 'predictions',
    optionally filtered by round number.
    Returns: [ "Almaty", "Astana", ... ]
    """
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cities = []
    query = "SELECT DISTINCT city FROM predictions"
    params = []
    if round_num is not None:
        query += " WHERE round = ?"
        params.append(round_num)
    query += " ORDER BY city ASC"
    try:
        cursor.execute(query, params)
        cities = [row[0] for row in cursor.fetchall()]
    except sqlite3.Error as e:
        print(f"ERROR: Failed to retrieve cities with predictions: {e}")
    finally:
        conn.close()
    return cities


def get_unique_routes(city=None, round_num=None):
    """
    Retrieves a sorted list of distinct route_id’s from routes_metadata,
    optionally filtered by city. (round_num is not used here but kept for API compatibility.)
    Returns: [ "R1", "R2", ... ]
    """
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    routes = []
    query = "SELECT DISTINCT route_id FROM routes_metadata WHERE 1=1"
    params = []
    if city:
        query += " AND city = ?"
        params.append(city)
    query += " ORDER BY route_id ASC"
    try:
        cursor.execute(query, params)
        routes = [row[0] for row in cursor.fetchall()]
    except sqlite3.Error as e:
        print(f"ERROR: Failed to retrieve unique routes: {e}")
    finally:
        conn.close()
    return routes


def get_unique_zones(city=None, round_num=None):
    """
    Retrieves a sorted list of distinct zone names from routes_metadata,
    optionally filtered by city. (round_num is not used here but kept for compatibility.)
    Returns: [ "Center", "North", ... ]
    """
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    zones = []
    query = "SELECT DISTINCT zone FROM routes_metadata WHERE 1=1"
    params = []
    if city:
        query += " AND city = ?"
        params.append(city)
    query += " ORDER BY zone ASC"
    try:
        cursor.execute(query, params)
        zones = [row[0] for row in cursor.fetchall()]
    except sqlite3.Error as e:
        print(f"ERROR: Failed to retrieve unique zones: {e}")
    finally:
        conn.close()
    return zones


def get_route_zone(route_id):
    """
    Retrieves the zone string for a given route_id from routes_metadata.
    Returns the zone (str) if found, otherwise None.
    """
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    zone = None
    try:
        cursor.execute("SELECT zone FROM routes_metadata WHERE route_id = ?", (route_id,))
        result = cursor.fetchone()
        if result:
            zone = result[0]
    except sqlite3.Error as e:
        print(f"ERROR: Failed to retrieve zone for route '{route_id}': {e}")
    finally:
        conn.close()
    return zone


def get_route_zone_combinations(city=None, round_num=None):
    """
    Retrieves a list of distinct {route_id, zone} dicts from routes_metadata,
    optionally filtered by city. This is used to populate the "Routes" and "Zones"
    dropdown menus on the front end.
    Example return value:
      [
        {"route_id": "R1", "zone": "Center"},
        {"route_id": "R2", "zone": "North"},
        ...
      ]
    """
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    combinations = []
    query = "SELECT route_id, zone FROM routes_metadata WHERE 1=1"
    params = []
    if city:
        query += " AND city = ?"
        params.append(city)
    query += " ORDER BY route_id ASC, zone ASC"
    try:
        cursor.execute(query, params)
        for row in cursor.fetchall():
            combinations.append({"route_id": row[0], "zone": row[1]})
    except sqlite3.Error as e:
        print(f"ERROR: Failed to retrieve route-zone combinations: {e}")
    finally:
        conn.close()
    return combinations
