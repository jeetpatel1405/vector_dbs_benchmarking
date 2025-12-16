#!/usr/bin/env python3
"""
Generate comprehensive comparison plot for Nov 24 ingestion benchmark data.
Similar to all_databases_comparison but for ingestion metrics.
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
    all_data = []

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
                    db_name = folder.split('_')[0].upper()

                    if 'aggregated_results' in data:
                        for result in data['aggregated_results']:
                            result['database'] = db_name
                            all_data.append(result)
                    print(f"✓ Loaded {db_name}")
            except Exception as e:
                print(f"✗ Error loading {folder}: {e}")

    return pd.DataFrame(all_data)


def generate_comparison_plot(df, output_path):
    """Generate 4-panel comparison plot."""

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Vector Database Ingestion Benchmark Comparison (7 Databases)',
                 fontsize=16, fontweight='bold', y=0.995)

    databases = sorted(df['database'].unique())
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(databases)))
    db_colors = {db: colors[i] for i, db in enumerate(databases)}

    # Panel 1 (Top-left): Ingestion Time by Chunk Size
    ax1 = axes[0, 0]
    for db in databases:
        db_data = df[(df['database'] == db) & (df['batch_size'] == 100) & (df['num_docs'] == 20)]
        db_data = db_data.sort_values('chunk_size')
        if not db_data.empty:
            ax1.plot(db_data['chunk_size'], db_data['total_time'],
                    marker='o', linewidth=2.5, markersize=8,
                    label=db, color=db_colors[db])

    ax1.set_xlabel('Chunk Size (characters)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Total Ingestion Time (seconds)', fontsize=11, fontweight='bold')
    ax1.set_title('Ingestion Time by Chunk Size', fontsize=12, fontweight='bold', pad=10)
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xticks([256, 512, 1024])

    # Panel 2 (Top-right): Throughput by Chunk Size
    ax2 = axes[0, 1]
    for db in databases:
        db_data = df[(df['database'] == db) & (df['batch_size'] == 100) & (df['num_docs'] == 20)]
        db_data = db_data.sort_values('chunk_size')
        if not db_data.empty:
            ax2.plot(db_data['chunk_size'], db_data['chunks_per_second'],
                    marker='s', linewidth=2.5, markersize=8,
                    label=db, color=db_colors[db])

    ax2.set_xlabel('Chunk Size (characters)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Throughput (chunks/second)', fontsize=11, fontweight='bold')
    ax2.set_title('Ingestion Throughput by Chunk Size', fontsize=12, fontweight='bold', pad=10)
    ax2.legend(fontsize=9, loc='best')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xticks([256, 512, 1024])

    # Panel 3 (Bottom-left): Phase Breakdown at Standard Config
    ax3 = axes[1, 0]
    standard_config = df[(df['chunk_size'] == 512) & (df['batch_size'] == 100) & (df['num_docs'] == 20)]

    phases = ['parsing_time', 'embedding_time', 'insertion_time']
    phase_labels = ['Parsing', 'Embedding', 'Insertion']
    phase_colors = ['#2E86AB', '#A23B72', '#F18F01']

    x_pos = np.arange(len(databases))
    width = 0.25

    for i, (phase, label) in enumerate(zip(phases, phase_labels)):
        values = [standard_config[standard_config['database'] == db][phase].mean()
                  if not standard_config[standard_config['database'] == db].empty else 0
                  for db in databases]
        ax3.bar(x_pos + i*width, values, width, label=label,
               color=phase_colors[i], alpha=0.85, edgecolor='black', linewidth=0.5)

    ax3.set_ylabel('Time (seconds)', fontsize=11, fontweight='bold')
    ax3.set_title('Ingestion Phase Breakdown (chunk=512, batch=100)',
                  fontsize=12, fontweight='bold', pad=10)
    ax3.set_xticks(x_pos + width)
    ax3.set_xticklabels(databases, rotation=45, ha='right', fontsize=10)
    ax3.legend(fontsize=10, loc='upper left', framealpha=0.95)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')

    # Panel 4 (Bottom-right): Normalized Throughput Only (no table overlay)
    ax4 = axes[1, 1]
    standard_config = df[(df['chunk_size'] == 512) & (df['batch_size'] == 100) & (df['num_docs'] == 20)]

    # Get throughput values and normalize to 0-100 scale
    throughput_values = []
    for db in databases:
        db_data = standard_config[standard_config['database'] == db]
        if not db_data.empty:
            throughput_values.append(db_data['chunks_per_second'].mean())
        else:
            throughput_values.append(0)

    max_throughput = max(throughput_values)
    normalized_throughput = [(v / max_throughput * 100) if max_throughput > 0 else 0
                             for v in throughput_values]

    # Create bar chart with gradient colors
    bars = ax4.bar(databases, normalized_throughput,
                   color=[db_colors[db] for db in databases],
                   alpha=0.85, edgecolor='black', linewidth=1.5)

    # Add value labels on top of bars
    for i, (bar, val, raw_val) in enumerate(zip(bars, normalized_throughput, throughput_values)):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{int(val)}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
        # Add actual throughput value inside or below bar
        y_pos = height/2 if height > 15 else -8
        color = 'white' if height > 15 else 'black'
        ax4.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{raw_val:.1f} ch/s',
                ha='center', va='center', fontsize=9, fontweight='bold', color=color)

    ax4.set_ylabel('Normalized Throughput (0-100)', fontsize=11, fontweight='bold')
    ax4.set_title('Ingestion Throughput (Normalized)', fontsize=12, fontweight='bold', pad=10)
    ax4.set_ylim(0, 110)
    ax4.set_xticklabels(databases, rotation=45, ha='right', fontsize=10)
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    ax4.axhline(y=100, color='red', linestyle='--', linewidth=1, alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {output_path}")
    plt.close()


def main():
    """Main execution function."""
    print("="*70)
    print("Nov 24 Database Comparison Plot Generator")
    print("="*70)

    print("\nLoading benchmark results...")
    df = load_ingestion_results()

    if df.empty:
        print("\n❌ No benchmark data found!")
        return 1

    print(f"\nLoaded data for {len(df['database'].unique())} databases")

    output_path = project_root / "results/full_suite_20251124_plots/all_databases_comparison.png"

    print("\nGenerating comparison plot...")
    generate_comparison_plot(df, output_path)

    return 0


if __name__ == '__main__':
    sys.exit(main())
