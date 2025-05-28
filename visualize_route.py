#!/usr/bin/env python3
# visualize_all_routes.py
"""
Generate a suite of descriptive analytics plots for each route in
data_cache/Astana_90days_routes30.csv, sized at 300dpi for publication.

Dependencies:
    pip install pandas matplotlib
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ——— Helper to make a high-res figure —————————————————————
def new_fig(title):
    fig = plt.figure(figsize=(10, 6), dpi=300)
    ax = fig.add_subplot(1,1,1)
    ax.set_title(title, fontsize=12)
    ax.tick_params(labelsize=10)
    return fig, ax

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def plot_time_series(df, outdir, route_id):
    fig, ax = new_fig(f"{route_id}: Inflow vs. Outflow over Time")
    ax.plot(df['datetime'], df['inflow_count'], label='Inflow')
    ax.plot(df['datetime'], df['outflow_count'], label='Outflow')
    ax.set_xlabel('Datetime')
    ax.set_ylabel('Passenger Count')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{outdir}/01_time_series.png")
    plt.close(fig)

def plot_scatter_temp(df, outdir, route_id):
    fig, ax = new_fig(f"{route_id}: Inflow vs. Temperature (precip flagged)")
    sc = ax.scatter(df['temperature'], df['inflow_count'],
                    c=df['precip_flag'], cmap='viridis', alpha=0.7)
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Inflow Count')
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label('Precipitation Flag')
    fig.tight_layout()
    fig.savefig(f"{outdir}/02_scatter_temp.png")
    plt.close(fig)

def plot_box_inflow(df, outdir, route_id):
    fig, ax = new_fig(f"{route_id}: Inflow Distribution")
    ax.boxplot(df['inflow_count'], vert=True)
    ax.set_ylabel('Inflow Count')
    fig.tight_layout()
    fig.savefig(f"{outdir}/03_boxplot_inflow.png")
    plt.close(fig)

def plot_diurnal(df, outdir, route_id):
    df['hour'] = df['datetime'].dt.hour
    hourly = df.groupby('hour')[['inflow_count','outflow_count']].mean()
    fig, ax = new_fig(f"{route_id}: Average by Hour of Day")
    ax.plot(hourly.index, hourly['inflow_count'], label='Inflow')
    ax.plot(hourly.index, hourly['outflow_count'], label='Outflow')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Average Count')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{outdir}/04_diurnal.png")
    plt.close(fig)

def plot_weekly(df, outdir, route_id):
    df['weekday'] = df['datetime'].dt.day_name()
    order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    weekly = df.groupby('weekday')[['inflow_count','outflow_count']].mean().reindex(order)
    fig, ax = new_fig(f"{route_id}: Average by Day of Week")
    ax.bar(weekly.index, weekly['inflow_count'], alpha=0.7, label='Inflow')
    ax.bar(weekly.index, weekly['outflow_count'], alpha=0.7, bottom=weekly['inflow_count'], label='Outflow')
    ax.set_xlabel('Day of Week')
    ax.set_ylabel('Average Count')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{outdir}/05_weekly.png")
    plt.close(fig)

def plot_corr_matrix(df, outdir, route_id):
    numeric = df[['inflow_count','outflow_count','temperature',
                  'precip_flag','route_length_km','num_stops']]
    corr = numeric.corr()
    fig, ax = new_fig(f"{route_id}: Feature Correlation Matrix")
    im = ax.imshow(corr, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right')
    ax.set_yticklabels(corr.index)
    for (i, j), val in np.ndenumerate(corr.values):
        ax.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Correlation Coefficient')
    fig.tight_layout()
    fig.savefig(f"{outdir}/06_corr_matrix.png")
    plt.close(fig)

def main():
    # 1) Load and parse
    df = pd.read_csv(
        'data_cache/Astana_90days_routes30.csv',
        parse_dates=['datetime']
    )

    # 2) Iterate over every route_id
    for route_id in df['route_id'].unique():
        df_r = df[df['route_id'] == route_id].sort_values('datetime')
        if df_r.empty:
            continue

        # 3) Set up output folder
        outdir = Path('figures') / route_id
        ensure_dir(outdir)

        # 4) Metadata header
        meta = df_r.iloc[0]
        with open(outdir / '00_metadata.txt', 'w') as f:
            f.write(f"Route: {route_id}\n")
            f.write(f"Type: {meta.route_type}\n")
            f.write(f"Zone: {meta.zone}\n")
            f.write(f"Length (km): {meta.route_length_km:.2f}\n")
            f.write(f"Stops: {meta.num_stops}\n")

        # 5) Produce the six figures
        plot_time_series(df_r, outdir, route_id)
        plot_scatter_temp(df_r, outdir, route_id)
        plot_box_inflow(df_r, outdir, route_id)
        plot_diurnal(df_r, outdir, route_id)
        plot_weekly(df_r, outdir, route_id)
        plot_corr_matrix(df_r, outdir, route_id)

    print("All routes processed. Figures are in ./figures/<route_id>/")

if __name__ == '__main__':
    main()
