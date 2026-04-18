#!/usr/bin/env python3
"""
scripts/plot_sweep_results.py — v1.1
Automatically loads the latest epoch_sweep CSV and generates:
• Stability islands heatmap
• Braiding phase distribution histogram
• Parameter vs stability scatter plots
• Top-10 table visualization
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from datetime import datetime
import ast

plt.style.use('seaborn-v0_8')
sns.set_palette("viridis")


def get_latest_csv():
    """Find the most recent epoch_sweep_*.csv"""
    output_dir = Path("outputs")
    csv_files = list(output_dir.glob("epoch_sweep_*.csv"))
    if not csv_files:
        raise FileNotFoundError("No epoch_sweep_*.csv found in outputs/")
    latest = max(csv_files, key=lambda x: x.stat().st_mtime)
    print(f"📊 Loading latest results: {latest.name}")
    return latest


def safe_parse_params(p):
    """Robustly convert params string from CSV into dict"""
    if isinstance(p, str):
        cleaned = p.strip().strip('"').strip("'")
        try:
            return ast.literal_eval(cleaned)
        except:
            # fallback for any odd quoting
            try:
                return ast.literal_eval(cleaned.replace("'", '"'))
            except:
                print(f"⚠️  Could not parse params: {p[:100]}...")
                return {}
    return p


def main():
    csv_path = get_latest_csv()
    df = pd.read_csv(csv_path)

    # === Robust params parsing (this was the failing part) ===
    print("   Parsing params column...")
    df['params'] = df['params'].apply(safe_parse_params)
    param_df = pd.json_normalize(df['params'])
    df = pd.concat([df.drop(columns=['params']), param_df], axis=1)
    print(f"   ✅ Parsed {len(df):,} trials successfully\n")

    # Create output directory
    plots_dir = Path("outputs/plots")
    plots_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Generating visualizations for {len(df):,} trials...\n")

    # 1. Stability Islands Heatmap (gauge_strength × omega_R)
    pivot = df.pivot_table(values='stability_score',
                           index='gauge_strength',
                           columns='omega_R',
                           aggfunc='mean')
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", ax=ax)
    ax.set_title('Stability Islands: Mean Stability Score\n(gauge_strength × ω_R)', fontsize=14)
    plt.tight_layout()
    heatmap_path = plots_dir / f"stability_islands_heatmap_{timestamp}.png"
    plt.savefig(heatmap_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Stability islands heatmap → {heatmap_path}")

    # 2. Braiding Phase Distribution Histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x='braiding_phase', bins=60, kde=True, ax=ax)
    ax.axvline(df['braiding_phase'].mean(), color='red', linestyle='--',
               label=f"Mean = {df['braiding_phase'].mean():.5f}")
    ax.set_title('Braiding Phase Distribution (Attractor Analysis)', fontsize=14)
    ax.set_xlabel('Braiding Phase')
    ax.set_ylabel('Count')
    ax.legend()
    plt.tight_layout()
    hist_path = plots_dir / f"braiding_phase_histogram_{timestamp}.png"
    plt.savefig(hist_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Braiding phase histogram → {hist_path}")

    # 3. Parameter vs Stability Scatter Plots (multi-panel)
    params = ['num_layers', 'num_polarities', 'max_facts', 'gauge_strength', 'omega_R']
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for i, param in enumerate(params):
        sns.scatterplot(data=df, x=param, y='stability_score',
                        hue='stability_score', size='active_cubes',
                        palette="viridis", alpha=0.7, ax=axes[i])
        axes[i].set_title(f'Stability Score vs {param}')
    axes[-1].axis('off')
    plt.tight_layout()
    scatter_path = plots_dir / f"param_vs_stability_scatter_{timestamp}.png"
    plt.savefig(scatter_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Parameter vs stability scatter plots → {scatter_path}")

    # 4. Top-10 Stability Islands Table
    top10 = df.nlargest(10, 'stability_score')[[
        'num_layers', 'num_polarities', 'max_facts',
        'gauge_strength', 'omega_R',
        'stability_score', 'active_cubes', 'braiding_phase'
    ]]
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=top10.round(5).values,
                     colLabels=top10.columns,
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 2.2)
    ax.set_title('Top 10 Stability Islands', pad=20, fontsize=16)
    table_path = plots_dir / f"top10_stability_table_{timestamp}.png"
    plt.savefig(table_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Top-10 table visualization → {table_path}")

    print(f"\n🎉 All plots saved to: {plots_dir}/")
    print(f"   Analyzed {len(df):,} trials from {csv_path.name}")


if __name__ == "__main__":
    main()