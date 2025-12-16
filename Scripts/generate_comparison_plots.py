#!/usr/bin/env python3
"""
Generate comparison plots from all database ingestion benchmark results.
"""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def load_ingestion_results():
    """Load results from all database ingestion experiments."""
    results_dir = project_root / "results"
    databases = []
    all_data = []

    # Database folders to check
    db_folders = [
        "qdrant_ingestion_experiment_001",
        "faiss_ingestion_experiment_001",
        "chroma_ingestion_experiment_001",
        "milvus_ingestion_experiment_001",
        "weaviate_ingestion_experiment_001",
        "pgvector_ingestion_experiment_001",
        "opensearch_ingestion_experiment_001"
    ]

    for folder in db_folders:
        results_file = results_dir / folder / "results.json"
        if results_file.exists():
            try:
                with open(results_file) as f:
                    data = json.load(f)
                    db_name = folder.split('_')[0].capitalize()
                    databases.append(db_name)

                    # Extract aggregated results if available
                    if 'aggregated_results' in data:
                        for result in data['aggregated_results']:
                            result['database'] = db_name
                            all_data.append(result)
                    print(f"✓ Loaded {db_name}: {len(data.get('aggregated_results', []))} results")
            except Exception as e:
                print(f"✗ Error loading {folder}: {e}")

    return pd.DataFrame(all_data), databases


def generate_comparison_plots(df, databases, output_dir):
    """Generate comparison plots across databases."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating plots in {output_dir}...")

    # Filter for standard configuration (chunk_size=512, batch_size=100, max docs)
    standard_config = df[(df['chunk_size'] == 512) & (df['batch_size'] == 100)]
    if standard_config.empty:
        # Try any configuration if standard not available
        standard_config = df.groupby('database').first().reset_index()

    # Plot 1: Total Ingestion Time Comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    db_times = standard_config.groupby('database')['total_time'].mean().sort_values()
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(db_times)))

    bars = ax.barh(range(len(db_times)), db_times.values, color=colors)
    ax.set_yticks(range(len(db_times)))
    ax.set_yticklabels(db_times.index)
    ax.set_xlabel('Total Ingestion Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Database Ingestion Performance Comparison\n(chunk_size=512, batch_size=100)',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, db_times.values)):
        ax.text(val, i, f'  {val:.1f}s', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'ingestion_time_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Generated: ingestion_time_comparison.png")
    plt.close()

    # Plot 2: Throughput Comparison (chunks/second)
    fig, ax = plt.subplots(figsize=(12, 6))

    db_throughput = standard_config.groupby('database')['chunks_per_second'].mean().sort_values(ascending=False)
    colors = plt.cm.plasma(np.linspace(0, 0.9, len(db_throughput)))

    bars = ax.barh(range(len(db_throughput)), db_throughput.values, color=colors)
    ax.set_yticks(range(len(db_throughput)))
    ax.set_yticklabels(db_throughput.index)
    ax.set_xlabel('Throughput (chunks/second)', fontsize=12, fontweight='bold')
    ax.set_title('Database Ingestion Throughput Comparison\n(chunk_size=512, batch_size=100)',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, db_throughput.values)):
        ax.text(val, i, f'  {val:.1f}', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'throughput_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Generated: throughput_comparison.png")
    plt.close()

    # Plot 3: Phase Breakdown Comparison
    fig, ax = plt.subplots(figsize=(14, 8))

    phases = ['parsing_time', 'embedding_time', 'insertion_time']
    phase_labels = ['Parsing', 'Embedding', 'Insertion']

    db_names = standard_config['database'].unique()
    x = np.arange(len(db_names))
    width = 0.25

    colors_phase = ['#2E86AB', '#A23B72', '#F18F01']

    for i, (phase, label) in enumerate(zip(phases, phase_labels)):
        values = [standard_config[standard_config['database'] == db][phase].mean()
                  for db in db_names]
        ax.bar(x + i*width, values, width, label=label, color=colors_phase[i], alpha=0.8)

    ax.set_xlabel('Database', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Ingestion Phase Breakdown by Database', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(db_names, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'phase_breakdown_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Generated: phase_breakdown_comparison.png")
    plt.close()

    # Plot 4: Scaling with Document Count
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Filter for chunk_size=512, batch_size=100
    scaling_data = df[(df['chunk_size'] == 512) & (df['batch_size'] == 100)]

    # Time scaling
    ax = axes[0]
    for db in databases:
        db_data = scaling_data[scaling_data['database'] == db].sort_values('num_docs')
        if not db_data.empty:
            ax.plot(db_data['num_docs'], db_data['total_time'],
                   marker='o', linewidth=2, markersize=8, label=db)

    ax.set_xlabel('Number of Documents', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Ingestion Time (s)', fontsize=12, fontweight='bold')
    ax.set_title('Scaling: Time vs Document Count', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Throughput scaling
    ax = axes[1]
    for db in databases:
        db_data = scaling_data[scaling_data['database'] == db].sort_values('num_docs')
        if not db_data.empty:
            ax.plot(db_data['num_docs'], db_data['chunks_per_second'],
                   marker='s', linewidth=2, markersize=8, label=db)

    ax.set_xlabel('Number of Documents', fontsize=12, fontweight='bold')
    ax.set_ylabel('Throughput (chunks/s)', fontsize=12, fontweight='bold')
    ax.set_title('Scaling: Throughput vs Document Count', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'scaling_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Generated: scaling_comparison.png")
    plt.close()

    # Plot 5: Summary Dashboard
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Top-left: Time comparison
    ax1 = fig.add_subplot(gs[0, 0])
    db_times_sorted = standard_config.groupby('database')['total_time'].mean().sort_values()
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(db_times_sorted)))
    ax1.barh(range(len(db_times_sorted)), db_times_sorted.values, color=colors)
    ax1.set_yticks(range(len(db_times_sorted)))
    ax1.set_yticklabels(db_times_sorted.index)
    ax1.set_xlabel('Time (s)', fontsize=10, fontweight='bold')
    ax1.set_title('Ingestion Time', fontsize=11, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)

    # Top-right: Throughput comparison
    ax2 = fig.add_subplot(gs[0, 1])
    db_throughput_sorted = standard_config.groupby('database')['chunks_per_second'].mean().sort_values(ascending=False)
    colors = plt.cm.plasma(np.linspace(0, 0.9, len(db_throughput_sorted)))
    ax2.barh(range(len(db_throughput_sorted)), db_throughput_sorted.values, color=colors)
    ax2.set_yticks(range(len(db_throughput_sorted)))
    ax2.set_yticklabels(db_throughput_sorted.index)
    ax2.set_xlabel('Chunks/s', fontsize=10, fontweight='bold')
    ax2.set_title('Throughput', fontsize=11, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)

    # Bottom-left: Phase breakdown for top 3 fastest
    ax3 = fig.add_subplot(gs[1, 0])
    top3_dbs = db_times_sorted.head(3).index
    x_pos = np.arange(len(top3_dbs))
    width = 0.25

    for i, (phase, label) in enumerate(zip(phases, phase_labels)):
        values = [standard_config[standard_config['database'] == db][phase].mean()
                  for db in top3_dbs]
        ax3.bar(x_pos + i*width, values, width, label=label, color=colors_phase[i], alpha=0.8)

    ax3.set_ylabel('Time (s)', fontsize=10, fontweight='bold')
    ax3.set_title('Phase Breakdown (Top 3 Fastest)', fontsize=11, fontweight='bold')
    ax3.set_xticks(x_pos + width)
    ax3.set_xticklabels(top3_dbs, rotation=45, ha='right')
    ax3.legend(fontsize=9)
    ax3.grid(axis='y', alpha=0.3)

    # Bottom-right: Performance metrics table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('tight')
    ax4.axis('off')

    summary_data = []
    for db in databases:
        db_data = standard_config[standard_config['database'] == db]
        if not db_data.empty:
            summary_data.append([
                db,
                f"{db_data['total_time'].mean():.1f}s",
                f"{db_data['chunks_per_second'].mean():.1f}",
                f"{db_data['num_chunks'].mean():.0f}"
            ])

    table = ax4.table(cellText=summary_data,
                     colLabels=['Database', 'Time', 'Chunks/s', 'Total Chunks'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.3, 0.2, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Color header
    for i in range(4):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax4.set_title('Performance Summary', fontsize=11, fontweight='bold', pad=20)

    fig.suptitle('Database Ingestion Benchmark - Complete Summary',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(output_dir / 'summary_dashboard.png', dpi=300, bbox_inches='tight')
    print("✓ Generated: summary_dashboard.png")
    plt.close()

    print(f"\n✅ All plots generated successfully in {output_dir}")


def main():
    """Main execution function."""

    print("="*70)
    print("Database Ingestion Benchmark - Comparison Plot Generator")
    print("="*70)

    # Load data
    print("\nLoading benchmark results...")
    df, databases = load_ingestion_results()

    if df.empty:
        print("\n❌ No benchmark data found!")
        return 1

    print(f"\nLoaded data for {len(databases)} databases: {', '.join(databases)}")
    print(f"Total data points: {len(df)}")

    # Determine output directory
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        output_dir = "results/database_comparison_plots"

    # Generate plots
    generate_comparison_plots(df, databases, output_dir)

    return 0


if __name__ == '__main__':
    sys.exit(main())
