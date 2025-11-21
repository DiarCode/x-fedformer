import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

def analyze_and_plot_busyness(data_dir: Path, output_dir: Path, top_n_routes: int = 5):
    """
    Analyzes transit data for each city, calculates busyness, and generates plots.

    Args:
        data_dir (Path): The directory containing the city CSV data files.
                         Expected format: 'CityName_Xdays_routesY.csv'
        output_dir (Path): The directory to save the generated plots.
        top_n_routes (int): The number of top busiest routes to plot in time series.
    """
    if not data_dir.exists():
        print(f"Error: Data directory '{data_dir}' does not exist.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output plots will be saved to: {output_dir}")

    all_data_files = list(data_dir.glob(f"*_*.csv"))

    if not all_data_files:
        print(f"No CSV data files found in '{data_dir}'. Please check the path and file naming convention.")
        return

    cities_processed = set()

    for data_file in all_data_files:
        try:
            city_name = data_file.name.split('_')[0]
            if city_name in cities_processed:
                continue # Skip if already processed for this city (e.g., multiple files for same city)
            
            print(f"Processing data for city: {city_name} from file: {data_file.name}")
            df = pd.read_csv(data_file, parse_dates=["datetime"])
            
            if df.empty:
                print(f"Warning: Data file for {city_name} is empty. Skipping.")
                continue

            # Calculate total traffic
            df['total_traffic'] = df['inflow_count'] + df['outflow_count']

            # --- Plot 1: Busiest Routes Over Time (Top N) ---
            plt.figure(figsize=(15, 8))
            
            # Calculate average traffic per route
            avg_traffic_per_route = df.groupby('route_id')['total_traffic'].mean().sort_values(ascending=False)
            top_routes = avg_traffic_per_route.head(top_n_routes).index.tolist()

            print(f"Top {top_n_routes} busiest routes for {city_name}: {top_routes}")

            for route_id in top_routes:
                route_df = df[df['route_id'] == route_id]
                # Plot daily average to avoid overly dense plots for long periods
                daily_traffic = route_df.set_index('datetime')['total_traffic'].resample('D').mean().reset_index()
                sns.lineplot(data=daily_traffic, x='datetime', y='total_traffic', label=route_id)
            
            plt.title(f'Top {top_n_routes} Busiest Routes Over Time in {city_name}')
            plt.xlabel('Date')
            plt.ylabel('Average Daily Total Traffic')
            plt.legend(title='Route ID')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(output_dir / f'{city_name}_top_{top_n_routes}_busiest_routes_time_series.png')
            plt.close()
            print(f"  Saved {city_name}_top_{top_n_routes}_busiest_routes_time_series.png")

            # --- Plot 2: Average Hourly Traffic by Route Type ---
            plt.figure(figsize=(12, 7))
            avg_traffic_by_route_type = df.groupby('route_type')['total_traffic'].mean().sort_values(ascending=False)
            sns.barplot(x=avg_traffic_by_route_type.index, y=avg_traffic_by_route_type.values, palette='viridis')
            plt.title(f'Average Hourly Traffic by Route Type in {city_name}')
            plt.xlabel('Route Type')
            plt.ylabel('Average Hourly Total Traffic')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(output_dir / f'{city_name}_avg_traffic_by_route_type.png')
            plt.close()
            print(f"  Saved {city_name}_avg_traffic_by_route_type.png")

            # --- Plot 3: Average Hourly Traffic by Zone ---
            if 'zone' in df.columns:
                plt.figure(figsize=(12, 7))
                avg_traffic_by_zone = df.groupby('zone')['total_traffic'].mean().sort_values(ascending=False)
                sns.barplot(x=avg_traffic_by_zone.index, y=avg_traffic_by_zone.values, palette='magma')
                plt.title(f'Average Hourly Traffic by Zone in {city_name}')
                plt.xlabel('Zone')
                plt.ylabel('Average Hourly Total Traffic')
                plt.xticks(rotation=45, ha='right')
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(output_dir / f'{city_name}_avg_traffic_by_zone.png')
                plt.close()
                print(f"  Saved {city_name}_avg_traffic_by_zone.png")
            else:
                print(f"  'zone' column not found in data for {city_name}. Skipping zone-based plot.")
            
            cities_processed.add(city_name)

        except Exception as e:
            print(f"Error processing file {data_file.name}: {e}")

if __name__ == "__main__":
    # Define your data directory and desired output directory
    DATA_CACHE_DIR = Path('data_cache') # Assuming 'data_cache' is in the same directory as this script
    PLOTS_OUTPUT_DIR = Path('traffic_plots')

    # Run the analysis and plotting
    analyze_and_plot_busyness(DATA_CACHE_DIR, PLOTS_OUTPUT_DIR)

    print("\nAnalysis complete. Check the 'traffic_plots' directory for generated graphs.")