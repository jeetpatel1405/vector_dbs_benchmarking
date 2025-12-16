#!/usr/bin/env python3
"""
3D Surface Plot for Ingestion Time Comparison Across All Vector Databases

This script creates a single multiseries 3D surface plot combining the ingestion
time heatmaps of all databases (Qdrant, FAISS, Chroma, PGVector, Milvus, OpenSearch).
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import Dict, List, Tuple

# Configuration
RESULTS_DIR = Path("/Users/rezarassool/Source/vector_dbs_benchmarking/results")
OUTPUT_DIR = Path("/Users/rezarassool/Source/vector_dbs_benchmarking/results")

# Database names and their display labels
DATABASES = {
    'qdrant': 'Qdrant',
    'faiss': 'FAISS',
    'chroma': 'Chroma',
    'pgvector': 'PGVector',
    'milvus': 'Milvus',
    'opensearch': 'OpenSearch'
}

# Color scheme for each database
COLORS = {
    'qdrant': '#FF6B6B',      # Red
    'faiss': '#4ECDC4',       # Teal
    'chroma': '#45B7D1',      # Blue
    'pgvector': '#96CEB4',    # Green
    'milvus': '#FFEAA7',      # Yellow
    'opensearch': '#DDA15E'   # Orange
}


def load_ingestion_results(db_name: str) -> pd.DataFrame:
    """Load ingestion results for a specific database."""
    result_file = RESULTS_DIR / f"{db_name}_ingestion_experiment_001" / "results.json"

    if not result_file.exists():
        print(f"Warning: Results file not found for {db_name}: {result_file}")
        return pd.DataFrame()

    with open(result_file, 'r') as f:
        data = json.load(f)

    # Extract raw results
    raw_results = data.get('raw_results', [])

    if not raw_results:
        print(f"Warning: No raw results found for {db_name}")
        return pd.DataFrame()

    df = pd.DataFrame(raw_results)
    df['database'] = DATABASES[db_name]

    return df


def prepare_surface_data(df: pd.DataFrame, metric='total_time') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare data for 3D surface plotting.

    Returns:
        X, Y, Z arrays for surface plotting
    """
    # Get unique values for each dimension
    chunk_sizes = sorted(df['chunk_size'].unique())
    batch_sizes = sorted(df['batch_size'].unique())

    # Create pivot table
    pivot = df.pivot_table(
        values=metric,
        index='chunk_size',
        columns='batch_size',
        aggfunc='mean'
    )

    # Create meshgrid
    X, Y = np.meshgrid(batch_sizes, chunk_sizes)
    Z = pivot.values

    return X, Y, Z


def create_3d_surface_plot(all_data: Dict[str, pd.DataFrame], metric='total_time',
                           metric_label='Total Ingestion Time (seconds)',
                           use_log_scale=False):
    """Create a single 3D surface plot with all databases."""

    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Order databases by mean performance (fastest to slowest)
    db_order = sorted(
        [(db_key, all_data[db_key][metric].mean())
         for db_key in all_data.keys()],
        key=lambda x: x[1]
    )

    # Track min/max for consistent scaling
    z_min, z_max = float('inf'), float('-inf')
    all_surfaces = []

    # Plot each database as a separate surface (in performance order)
    for idx, (db_key, mean_time) in enumerate(db_order):
        df = all_data[db_key]
        X, Y, Z = prepare_surface_data(df, metric)

        # Apply log scale if requested
        if use_log_scale:
            Z_plot = np.log10(Z)
            z_min = min(z_min, np.min(Z_plot))
            z_max = max(z_max, np.max(Z_plot))
        else:
            Z_plot = Z
            z_min = min(z_min, np.min(Z))
            z_max = max(z_max, np.max(Z))

        # Add slight offset for visual separation (proportional to log scale)
        if use_log_scale:
            z_offset = idx * 0.02  # Smaller offset for log scale
        else:
            z_offset = idx * 0.02

        # Plot surface with transparency for better visibility
        surf = ax.plot_surface(
            X, Y, Z_plot + z_offset,
            alpha=0.75,
            color=COLORS[db_key],
            label=f"{DATABASES[db_key]} (avg: {mean_time:.3f}s)",
            edgecolors='black',
            linewidth=0.3,
            antialiased=True
        )
        all_surfaces.append((db_key, mean_time))

    # Set axis limits with padding
    ax.set_xlim([45, 105])  # Batch size range with padding
    ax.set_ylim([200, 1100])  # Chunk size range with padding
    z_range = z_max - z_min
    ax.set_zlim([z_min - 0.1 * z_range, z_max + 0.2 * z_range])

    # Customize the plot
    ax.set_xlabel('Batch Size', fontsize=13, fontweight='bold', labelpad=12)
    ax.set_ylabel('Chunk Size', fontsize=13, fontweight='bold', labelpad=12)

    if use_log_scale:
        ax.set_zlabel(f'{metric_label} (log₁₀ scale)', fontsize=13, fontweight='bold', labelpad=12)
        title_suffix = '(Log Scale, Ordered by Performance)'
    else:
        ax.set_zlabel(metric_label, fontsize=13, fontweight='bold', labelpad=12)
        title_suffix = '(Linear Scale, Ordered by Performance)'

    ax.set_title(f'3D Ingestion Time Comparison Across Vector Databases\n'
                 f'Surface View: Chunk Size × Batch Size × Time {title_suffix}',
                 fontsize=15, fontweight='bold', pad=25)

    # Set viewing angle for better perspective
    ax.view_init(elev=20, azim=135)

    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--')

    # Create custom legend (ordered by performance)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS[db_key], alpha=0.75,
              label=f"{DATABASES[db_key]}: {mean_time:.3f}s")
        for db_key, mean_time in db_order
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11,
              title='Databases (Avg Time)', title_fontsize=12)

    # Adjust layout
    plt.tight_layout()

    return fig


def create_3d_wireframe_plot(all_data: Dict[str, pd.DataFrame], metric='total_time',
                              metric_label='Total Ingestion Time (seconds)',
                              use_log_scale=False):
    """Create a 3D wireframe plot for better clarity."""

    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Order databases by mean performance (fastest to slowest)
    db_order = sorted(
        [(db_key, all_data[db_key][metric].mean())
         for db_key in all_data.keys()],
        key=lambda x: x[1]
    )

    # Track min/max for consistent scaling
    z_min, z_max = float('inf'), float('-inf')

    # Plot each database as a separate wireframe (in performance order)
    for idx, (db_key, mean_time) in enumerate(db_order):
        df = all_data[db_key]
        X, Y, Z = prepare_surface_data(df, metric)

        # Apply log scale if requested
        if use_log_scale:
            Z_plot = np.log10(Z)
            z_min = min(z_min, np.min(Z_plot))
            z_max = max(z_max, np.max(Z_plot))
            z_offset = idx * 0.02
        else:
            Z_plot = Z
            z_min = min(z_min, np.min(Z))
            z_max = max(z_max, np.max(Z))
            z_offset = idx * 0.03

        # Plot wireframe with offset for visual separation
        ax.plot_wireframe(
            X, Y, Z_plot + z_offset,
            color=COLORS[db_key],
            label=f"{DATABASES[db_key]}: {mean_time:.3f}s",
            linewidth=2.5,
            alpha=0.85
        )

    # Set axis limits with padding
    ax.set_xlim([45, 105])
    ax.set_ylim([200, 1100])
    z_range = z_max - z_min
    ax.set_zlim([z_min - 0.1 * z_range, z_max + 0.2 * z_range])

    # Customize the plot
    ax.set_xlabel('Batch Size', fontsize=13, fontweight='bold', labelpad=12)
    ax.set_ylabel('Chunk Size', fontsize=13, fontweight='bold', labelpad=12)

    if use_log_scale:
        ax.set_zlabel(f'{metric_label} (log₁₀ scale)', fontsize=13, fontweight='bold', labelpad=12)
        title_suffix = '(Log Scale, Ordered by Performance)'
    else:
        ax.set_zlabel(metric_label, fontsize=13, fontweight='bold', labelpad=12)
        title_suffix = '(Linear Scale, Ordered by Performance)'

    ax.set_title(f'3D Ingestion Time Comparison Across Vector Databases\n'
                 f'Wireframe View: Chunk Size × Batch Size × Time {title_suffix}',
                 fontsize=15, fontweight='bold', pad=25)

    # Set viewing angle
    ax.view_init(elev=20, azim=135)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')

    # Legend
    ax.legend(loc='upper left', fontsize=11, title='Databases (Avg Time)', title_fontsize=12)

    # Adjust layout
    plt.tight_layout()

    return fig


def create_combined_subplots(all_data: Dict[str, pd.DataFrame]):
    """Create a figure with multiple 3D views."""

    fig = plt.figure(figsize=(20, 12))

    # Surface plot
    ax1 = fig.add_subplot(221, projection='3d')
    plot_surface_on_axis(ax1, all_data, 'Surface View')

    # Wireframe plot
    ax2 = fig.add_subplot(222, projection='3d')
    plot_wireframe_on_axis(ax2, all_data, 'Wireframe View')

    # Different angle - surface
    ax3 = fig.add_subplot(223, projection='3d')
    plot_surface_on_axis(ax3, all_data, 'Surface View (Alt Angle)', elev=15, azim=-60)

    # Different angle - wireframe
    ax4 = fig.add_subplot(224, projection='3d')
    plot_wireframe_on_axis(ax4, all_data, 'Wireframe View (Alt Angle)', elev=15, azim=-60)

    plt.suptitle('3D Ingestion Time Comparison - Multiple Perspectives',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    return fig


def plot_surface_on_axis(ax, all_data, title, elev=25, azim=45):
    """Helper to plot surface on a specific axis."""
    # Order databases by performance
    db_order = sorted(
        [(db_key, all_data[db_key]['total_time'].mean())
         for db_key in all_data.keys()],
        key=lambda x: x[1]
    )

    z_min, z_max = float('inf'), float('-inf')

    for idx, (db_key, mean_time) in enumerate(db_order):
        df = all_data[db_key]
        X, Y, Z = prepare_surface_data(df, 'total_time')

        z_min = min(z_min, np.min(Z))
        z_max = max(z_max, np.max(Z))

        z_offset = idx * 0.02

        ax.plot_surface(
            X, Y, Z + z_offset,
            alpha=0.7,
            color=COLORS[db_key],
            edgecolors='black',
            linewidth=0.2
        )

    ax.set_xlim([45, 105])
    ax.set_ylim([200, 1100])
    z_range = z_max - z_min
    ax.set_zlim([z_min - 0.1 * z_range, z_max + 0.2 * z_range])

    ax.set_xlabel('Batch Size', fontsize=9)
    ax.set_ylabel('Chunk Size', fontsize=9)
    ax.set_zlabel('Time (s)', fontsize=9)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.view_init(elev=elev, azim=azim)
    ax.grid(True, alpha=0.3, linestyle='--')


def plot_wireframe_on_axis(ax, all_data, title, elev=25, azim=45):
    """Helper to plot wireframe on a specific axis."""
    # Order databases by performance
    db_order = sorted(
        [(db_key, all_data[db_key]['total_time'].mean())
         for db_key in all_data.keys()],
        key=lambda x: x[1]
    )

    z_min, z_max = float('inf'), float('-inf')

    for idx, (db_key, mean_time) in enumerate(db_order):
        df = all_data[db_key]
        X, Y, Z = prepare_surface_data(df, 'total_time')

        z_min = min(z_min, np.min(Z))
        z_max = max(z_max, np.max(Z))

        z_offset = idx * 0.03

        ax.plot_wireframe(
            X, Y, Z + z_offset,
            color=COLORS[db_key],
            linewidth=1.5,
            alpha=0.85
        )

    ax.set_xlim([45, 105])
    ax.set_ylim([200, 1100])
    z_range = z_max - z_min
    ax.set_zlim([z_min - 0.1 * z_range, z_max + 0.2 * z_range])

    ax.set_xlabel('Batch Size', fontsize=9)
    ax.set_ylabel('Chunk Size', fontsize=9)
    ax.set_zlabel('Time (s)', fontsize=9)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.view_init(elev=elev, azim=azim)
    ax.grid(True, alpha=0.3, linestyle='--')


def print_summary_statistics(all_data: Dict[str, pd.DataFrame]):
    """Print summary statistics for all databases."""
    print("\n" + "="*80)
    print("INGESTION TIME SUMMARY STATISTICS")
    print("="*80 + "\n")

    summary_data = []

    for db_key, db_name in DATABASES.items():
        if db_key not in all_data or all_data[db_key].empty:
            continue

        df = all_data[db_key]

        summary_data.append({
            'Database': db_name,
            'Min Time (s)': df['total_time'].min(),
            'Max Time (s)': df['total_time'].max(),
            'Mean Time (s)': df['total_time'].mean(),
            'Std Dev (s)': df['total_time'].std(),
            'Configurations': len(df)
        })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    print("\n" + "="*80 + "\n")


def main():
    """Main execution function."""
    print("Loading ingestion results from all databases...")

    # Load data from all databases
    all_data = {}
    for db_key in DATABASES.keys():
        df = load_ingestion_results(db_key)
        if not df.empty:
            all_data[db_key] = df
            print(f"✓ Loaded {len(df)} configurations for {DATABASES[db_key]}")
        else:
            print(f"✗ No data for {DATABASES[db_key]}")

    if not all_data:
        print("Error: No data loaded from any database!")
        return

    print(f"\nSuccessfully loaded data from {len(all_data)} databases")

    # Print summary statistics
    print_summary_statistics(all_data)

    # Create visualizations
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80 + "\n")

    # Linear scale plots
    print("Creating 3D surface plot (linear scale)...")
    fig_surface = create_3d_surface_plot(all_data, use_log_scale=False)
    surface_path = OUTPUT_DIR / "ingestion_3d_surface_comparison_linear.png"
    fig_surface.savefig(surface_path, dpi=300, bbox_inches='tight')
    plt.close(fig_surface)
    print(f"✓ Saved: {surface_path}")

    print("Creating 3D wireframe plot (linear scale)...")
    fig_wireframe = create_3d_wireframe_plot(all_data, use_log_scale=False)
    wireframe_path = OUTPUT_DIR / "ingestion_3d_wireframe_comparison_linear.png"
    fig_wireframe.savefig(wireframe_path, dpi=300, bbox_inches='tight')
    plt.close(fig_wireframe)
    print(f"✓ Saved: {wireframe_path}")

    # Logarithmic scale plots
    print("\nCreating 3D surface plot (log scale)...")
    fig_surface_log = create_3d_surface_plot(all_data, use_log_scale=True)
    surface_log_path = OUTPUT_DIR / "ingestion_3d_surface_comparison_log.png"
    fig_surface_log.savefig(surface_log_path, dpi=300, bbox_inches='tight')
    plt.close(fig_surface_log)
    print(f"✓ Saved: {surface_log_path}")

    print("Creating 3D wireframe plot (log scale)...")
    fig_wireframe_log = create_3d_wireframe_plot(all_data, use_log_scale=True)
    wireframe_log_path = OUTPUT_DIR / "ingestion_3d_wireframe_comparison_log.png"
    fig_wireframe_log.savefig(wireframe_log_path, dpi=300, bbox_inches='tight')
    plt.close(fig_wireframe_log)
    print(f"✓ Saved: {wireframe_log_path}")

    print("\nCreating combined perspective plot...")
    fig_combined = create_combined_subplots(all_data)
    combined_path = OUTPUT_DIR / "ingestion_3d_combined_perspectives.png"
    fig_combined.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close(fig_combined)
    print(f"✓ Saved: {combined_path}")

    print("\n" + "="*80)
    print("3D VISUALIZATION COMPLETE!")
    print("="*80)
    print(f"\nGenerated files:")
    print(f"\nLinear Scale:")
    print(f"  1. {surface_path.name}")
    print(f"  2. {wireframe_path.name}")
    print(f"\nLogarithmic Scale:")
    print(f"  3. {surface_log_path.name}")
    print(f"  4. {wireframe_log_path.name}")
    print(f"\nMulti-Perspective:")
    print(f"  5. {combined_path.name}")
    print(f"\nAll files saved to: {OUTPUT_DIR}")
    print("\nDone! Visualizations have been saved.")


if __name__ == "__main__":
    main()
