#!/bin/bash

# This script automates the process of running X-FedFormer federated learning:
# 1. Generates synthetic data for specified cities.
# 2. Starts the Flower server in the background.
# 3. Starts multiple Flower clients, one for each city.
# 4. Evaluates the global model after training.

# --- Configuration ---
# Define cities for data generation and client participation
CITIES=("Almaty" "Astana" "Karaganda")
DAYS_DATA=30           # Number of days for synthetic data per city
ROUTES_PER_CITY=10     # Number of routes per city
NUM_ROUNDS=5           # Number of federated learning rounds
SERVER_ADDRESS="127.0.0.1:8080" # Server address for clients to connect to
GLOBAL_MODEL_PATH="./checkpoints/global_model.pt" # Path where the global model will be saved
EVAL_REPORT_PATH="./results/evaluation_report.json" # Path for evaluation report

# --- Setup ---
echo "--- Setting up environment ---"
# Create necessary directories if they don't exist
mkdir -p data checkpoints results

# --- Step 1: Generate Synthetic Data ---
echo "--- Step 1: Generating synthetic data ---"
python3 main.py generate-data --cities "${CITIES[@]}" --days "$DAYS_DATA" --routes-per-city "$ROUTES_PER_CITY"

# Check if data generation was successful
if [ $? -ne 0 ]; then
    echo "Error: Data generation failed. Exiting."
    exit 1
fi
echo "Data generation complete."

# --- Step 2: Start Flower Server ---
echo "--- Step 2: Starting Flower server in background ---"
# Using nohup to run in background and redirect output to a log file
nohup python3 main.py server --rounds "$NUM_ROUNDS" > server.log 2>&1 &
SERVER_PID=$!
echo "Flower server started with PID: $SERVER_PID. Check server.log for output."

# Give the server a more generous moment to start up and stabilize
# This initial sleep is crucial for the gRPC server to bind to the port before clients connect.
echo "Waiting for server to fully start and stabilize (10 seconds)..."
sleep 10

# --- Step 3: Start Flower Clients ---
echo "--- Step 3: Starting Flower clients ---"
CLIENT_PIDS=() # Initialize array to store client PIDs
for CITY in "${CITIES[@]}"; do
    echo "Starting client for city: $CITY"
    # Run clients in the background
    nohup python3 main.py client --city "$CITY" --days-data "$DAYS_DATA" --server_address "$SERVER_ADDRESS" > "client_${CITY}.log" 2>&1 &
    CLIENT_PIDS+=($!) # Store client PIDs
    sleep 1 # Small delay between client starts to avoid overwhelming the server initially
done
echo "All clients launched. Check client_*.log files for their output."

# --- Dynamic Waiting for Federated Learning Completion ---
echo "Waiting for federated learning rounds to complete (waiting for server process to exit)..."
# Wait for the server process to finish. The server is designed to exit after NUM_ROUNDS.
wait "$SERVER_PID"
SERVER_EXIT_STATUS=$?
echo "Server process (PID: $SERVER_PID) exited with status: $SERVER_EXIT_STATUS"

# Wait for all client processes to finish
echo "Waiting for client processes to terminate..."
# Using 'wait -n' in a loop for more robust waiting for multiple background processes
# This ensures we wait for all clients, even if some exit earlier.
for CLIENT_PID in "${CLIENT_PIDS[@]}"; do
    if ps -p "$CLIENT_PID" > /dev/null; then # Check if process still exists
        wait "$CLIENT_PID" || true # Wait for client, '|| true' prevents script from exiting on non-zero status
        echo "Client process (PID: $CLIENT_PID) exited."
    fi
done

echo "Federated learning process completed."

# --- Cleanup: Ensure all processes are terminated ---
echo "Attempting to gracefully terminate any remaining server and client processes..."

# Kill the server process if it's still running (should not be if 'wait' succeeded)
if ps -p "$SERVER_PID" > /dev/null; then
    echo "Server process (PID: $SERVER_PID) is still running. Attempting to terminate..."
    kill "$SERVER_PID"
    sleep 2
    if ps -p "$SERVER_PID" > /dev/null; then
        echo "Server process did not terminate gracefully. Forcing kill."
        kill -9 "$SERVER_PID"
    fi
fi

# Kill any remaining client processes
for CLIENT_PID in "${CLIENT_PIDS[@]}"; do
    if ps -p "$CLIENT_PID" > /dev/null; then
        echo "Client process (PID: $CLIENT_PID) is still running. Terminating..."
        kill "$CLIENT_PID"
        sleep 1
        if ps -p "$CLIENT_PID" > /dev/null; then
            echo "Client process did not terminate gracefully. Forcing kill."
            kill -9 "$CLIENT_PID"
        fi
    fi
done

# --- Step 4: Evaluate Global Model ---
echo "--- Step 4: Evaluating the global model ---"
if [ -f "$GLOBAL_MODEL_PATH" ]; then
    python3 main.py evaluate --cities "${CITIES[@]}" --days-data "$DAYS_DATA" --model-path "$GLOBAL_MODEL_PATH" --report-file "$EVAL_REPORT_PATH"
    if [ $? -ne 0 ]; then
        echo "Error: Model evaluation failed. Check logs."
    else
        echo "Model evaluation complete. Report saved to: $EVAL_REPORT_PATH"
        cat "$EVAL_REPORT_PATH" # Display the evaluation report
    fi
else
    echo "Error: Global model not found at $GLOBAL_MODEL_PATH. Skipping evaluation."
fi

echo "--- Script finished ---"
